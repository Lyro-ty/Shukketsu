"""BaseAgent with ReAct (Reason + Act) loop."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any, NamedTuple

from langfuse import observe

from code.shukketsu import config
from code.shukketsu.agents.guardrails import LoopDetector
from code.shukketsu.agents.tasks import AgentResult, AgentRole, AgentTask, TaskStatus, ToolCallRecord
from code.shukketsu.llm.schemas import ActionType, AgentStep
from code.shukketsu.llm.structured import get_structured_output
from code.shukketsu.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

StatusCallback = Callable[[str | dict[str, Any]], Awaitable[None]]

_REACT_INSTRUCTIONS = """You have access to the following tools:

{tool_descriptions}

When you need information to answer the question, use a tool by responding with action "tool_call".
When you have enough information to answer, respond with action "final_answer".
Always think step by step about what you need to do.
If a tool returns no results or an error, try a DIFFERENT tool or different query — never repeat the same tool call.
If rag_search finds nothing, try web_search. If web_search finds relevant pages, use web_ingest to store them.
Once you have useful information from any source, provide your final_answer — do not keep searching."""


class _RunOutcome(NamedTuple):
    """Internal return type from the ReAct loop.

    Carries the output string, execution status, and the full scratchpad
    so that specialist agents (e.g. Researcher) can access tool observations
    for post-processing.
    """

    output: str
    status: TaskStatus
    scratchpad: list[dict[str, Any]]


class BaseAgent:
    """Agent that uses a ReAct loop to answer questions with tools.

    Iterates: think -> act (call tool) -> observe (read result)
    until it reaches a final answer or hits the iteration limit.

    Two public entry points share the same loop:
    - run(query) -> str           — Phase 1 backward-compat interface
    - execute(task) -> AgentResult — Phase 2 task-based interface
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        role: AgentRole | None = None,
        max_iterations: int = config.AGENT_MAX_ITERATIONS,
        system_prompt: str = config.SYSTEM_PROMPT,
    ) -> None:
        self.role = role
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self._system_prompt = system_prompt
        self._loop_detector = LoopDetector()

    @observe(as_type="agent")
    async def run(
        self,
        query: str,
        *,
        on_status: StatusCallback | None = None,
        memory_context: str | None = None,
    ) -> str:
        """Run the ReAct loop to answer a query.

        Args:
            query: The user's question.
            on_status: Optional async callback for progress updates.
            memory_context: Optional context from recalled session memories.

        Returns:
            The agent's final answer, or a graceful failure message.

        Raises:
            LLMUnavailableError: If the LLM backend is unreachable.
            StructuredOutputError: If structured output validation fails.
        """
        outcome = await self._run_loop(query, on_status=on_status, memory_context=memory_context)
        return outcome.output

    @observe(as_type="agent")
    async def execute(self, task: AgentTask, *, on_status: StatusCallback | None = None) -> AgentResult:
        """Execute a typed task and return a structured result.

        This is the task-based interface used by the multi-agent system.
        Calls the same ReAct loop as run() but wraps the result in a
        typed AgentResult with status tracking.

        Reads memory_context from task.context if present, and passes it
        through to the ReAct loop for injection into the system prompt.

        Args:
            task: The task to execute.
            on_status: Optional async callback for progress updates.

        Returns:
            AgentResult with the output and execution status.

        Raises:
            ValueError: If no role is set on this agent.
            LLMUnavailableError: If the LLM backend is unreachable.
            StructuredOutputError: If structured output validation fails.
        """
        if self.role is None:
            raise ValueError("Cannot execute() without a role. Use AgentFactory or set role in constructor.")

        memory_ctx = task.context.get("memory_context") if task.context else None
        outcome = await self._run_loop(task.query, on_status=on_status, memory_context=memory_ctx)

        trajectory = [
            ToolCallRecord(tool_name=entry["tool_name"], tool_input=entry["tool_input"]) for entry in outcome.scratchpad
        ]

        return AgentResult(
            task_id=task.task_id,
            agent_role=self.role,
            status=outcome.status,
            output=outcome.output,
            trajectory=trajectory,
        )

    async def _run_loop(
        self,
        query: str,
        *,
        on_status: StatusCallback | None = None,
        memory_context: str | None = None,
    ) -> _RunOutcome:
        """The core ReAct loop. Shared by run() and execute().

        Returns a _RunOutcome with the output string and status, so callers
        can decide how to surface the result without relying on mutable state.
        """
        scratchpad: list[dict[str, Any]] = []

        for iteration in range(self.max_iterations):
            messages = self._build_messages(query, scratchpad, memory_context=memory_context)

            logger.info("Agent iteration %d/%d", iteration + 1, self.max_iterations)
            if on_status:
                await on_status(f"thinking ({iteration + 1}/{self.max_iterations})...")
            step: AgentStep = await get_structured_output(
                response_model=AgentStep,
                messages=messages,
            )

            if step.action == ActionType.FINAL_ANSWER:
                logger.info("Agent reached final answer after %d iteration(s)", iteration + 1)
                return _RunOutcome(output=step.answer, status=TaskStatus.SUCCESS, scratchpad=scratchpad)  # type: ignore[arg-type]

            tool_call = step.tool_call
            if tool_call is None:
                logger.warning("AgentStep has action=tool_call but no tool_call object; treating as final answer")
                return _RunOutcome(output=step.reasoning, status=TaskStatus.PARTIAL, scratchpad=scratchpad)
            logger.info("Agent calling tool: %s", tool_call.tool_name)
            if on_status:
                await on_status(f"using {tool_call.tool_name}...")

            observation = await self.tool_registry.execute(tool_call.tool_name, tool_call.tool_input)

            if on_status:
                await on_status(
                    {
                        "type": "step",
                        "agent": self.role or "agent",
                        "action": "tool_call",
                        "tool": tool_call.tool_name,
                    }
                )

            # Truncate large observations to keep context manageable for GB10
            if len(observation) > config.OBSERVATION_MAX_CHARS:
                half = config.OBSERVATION_MAX_CHARS // 2
                observation = observation[:half] + "\n\n[... truncated ...]\n\n" + observation[-half:]

            scratchpad.append(
                {
                    "reasoning": step.reasoning,
                    "tool_name": tool_call.tool_name,
                    "tool_input": tool_call.tool_input,
                    "observation": observation,
                }
            )

            # Check for loops after each tool call
            loop_msg = self._loop_detector.check(scratchpad)
            if loop_msg:
                logger.warning("Loop detected: %s", loop_msg)
                return _RunOutcome(
                    output=self._synthesize_partial_answer(scratchpad),
                    status=TaskStatus.PARTIAL,
                    scratchpad=scratchpad,
                )

        logger.warning("Agent reached max iterations (%d) without final answer", self.max_iterations)
        return _RunOutcome(output=config.AGENT_GRACEFUL_FAILURE, status=TaskStatus.FAILED, scratchpad=scratchpad)

    def _synthesize_partial_answer(self, scratchpad: list[dict[str, Any]]) -> str:
        """Build an answer from partial observations when a loop is detected."""
        observations = [e["observation"] for e in scratchpad if e.get("observation")]
        if observations:
            unique = list(dict.fromkeys(observations))
            joined = "\n\n".join(unique)
            return f"Based on partial results:\n\n{joined}"
        return config.AGENT_GRACEFUL_FAILURE

    def _estimate_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimate token count from messages using char/4 heuristic."""
        return sum(len(m.get("content", "")) for m in messages) // 4

    def _compact_scratchpad(
        self,
        scratchpad: list[dict[str, Any]],
        keep_recent: int = 2,
    ) -> list[dict[str, Any]]:
        """Return a compacted copy of the scratchpad.

        Keeps the most recent `keep_recent` entries verbatim. Older entries
        have their observations truncated to a summarized format.
        """
        if len(scratchpad) <= keep_recent:
            return list(scratchpad)

        compacted: list[dict[str, Any]] = []
        cutoff = len(scratchpad) - keep_recent

        for i, entry in enumerate(scratchpad):
            if i < cutoff:
                obs = str(entry.get("observation", ""))
                tool = entry.get("tool_name", "unknown")
                if len(obs) > 200:
                    head = obs[:100]
                    tail = obs[-100:]
                    summarized = f"[Summarized] [{tool}] {head}... [truncated] ...{tail}"
                else:
                    summarized = f"[Summarized] [{tool}] {obs}"
                compacted.append({**entry, "observation": summarized})
            else:
                compacted.append(entry)

        return compacted

    def _build_messages(
        self,
        query: str,
        scratchpad: list[dict[str, Any]],
        *,
        memory_context: str | None = None,
    ) -> list[dict[str, str]]:
        """Build the messages array for the LLM.

        If the estimated token count exceeds COMPACTION_THRESHOLD_TOKENS,
        older scratchpad entries are compacted to reduce context size.

        Args:
            query: The user's question.
            scratchpad: ReAct loop scratchpad entries.
            memory_context: Optional context from recalled session memories.
        """
        tool_descriptions = self.tool_registry.get_tool_descriptions()
        system_content = self._system_prompt + "\n\n" + _REACT_INSTRUCTIONS.format(tool_descriptions=tool_descriptions)

        if memory_context:
            system_content += "\n\n## Relevant Context from Previous Sessions\n\n" + memory_context

        messages = self._assemble_messages(system_content, query, scratchpad)

        if self._estimate_tokens(messages) > config.COMPACTION_THRESHOLD_TOKENS:
            compacted = self._compact_scratchpad(scratchpad)
            messages = self._assemble_messages(system_content, query, compacted)

        return messages

    @staticmethod
    def _assemble_messages(
        system_content: str,
        query: str,
        scratchpad: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Assemble system + user + scratchpad entries into a messages array."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]
        for entry in scratchpad:
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Thought: {entry['reasoning']}\n"
                        f"Action: tool_call\n"
                        f"Tool: {entry['tool_name']}\n"
                        f"Input: {entry['tool_input']}"
                    ),
                }
            )
            messages.append({"role": "user", "content": f"Observation: {entry['observation']}"})
        return messages
