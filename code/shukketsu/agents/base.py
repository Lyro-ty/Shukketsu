"""BaseAgent with ReAct (Reason + Act) loop."""

import logging
from typing import Any

from code.shukketsu import config
from code.shukketsu.llm.schemas import ActionType, AgentStep
from code.shukketsu.llm.structured import get_structured_output
from code.shukketsu.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_REACT_INSTRUCTIONS = """You have access to the following tools:

{tool_descriptions}

When you need information to answer the question, use a tool by responding with action "tool_call".
When you have enough information to answer, respond with action "final_answer".
Always think step by step about what you need to do.
If a tool returns an error, try a different approach or answer with what you know."""


class BaseAgent:
    """Agent that uses a ReAct loop to answer questions with tools.

    Iterates: think -> act (call tool) -> observe (read result)
    until it reaches a final answer or hits the iteration limit.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        max_iterations: int = config.AGENT_MAX_ITERATIONS,
        system_prompt: str = config.SYSTEM_PROMPT,
    ) -> None:
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self._system_prompt = system_prompt

    async def run(self, query: str) -> str:
        """Run the ReAct loop to answer a query.

        Args:
            query: The user's question.

        Returns:
            The agent's final answer, or a graceful failure message.

        Raises:
            LLMUnavailableError: If the LLM backend is unreachable.
            StructuredOutputError: If structured output validation fails.
        """
        scratchpad: list[dict[str, Any]] = []

        for iteration in range(self.max_iterations):
            messages = self._build_messages(query, scratchpad)

            logger.info("Agent iteration %d/%d", iteration + 1, self.max_iterations)
            step: AgentStep = await get_structured_output(
                response_model=AgentStep,
                messages=messages,
            )

            if step.action == ActionType.FINAL_ANSWER:
                logger.info("Agent reached final answer after %d iteration(s)", iteration + 1)
                return step.answer  # type: ignore[return-value]

            tool_call = step.tool_call
            assert tool_call is not None  # guaranteed by AgentStep validator
            logger.info("Agent calling tool: %s", tool_call.tool_name)

            observation = await self.tool_registry.execute(tool_call.tool_name, tool_call.tool_input)

            scratchpad.append(
                {
                    "reasoning": step.reasoning,
                    "tool_name": tool_call.tool_name,
                    "tool_input": tool_call.tool_input,
                    "observation": observation,
                }
            )

        logger.warning("Agent reached max iterations (%d) without final answer", self.max_iterations)
        return config.AGENT_GRACEFUL_FAILURE

    def _build_messages(self, query: str, scratchpad: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Build the messages array for the LLM."""
        tool_descriptions = self.tool_registry.get_tool_descriptions()
        system_content = self._system_prompt + "\n\n" + _REACT_INSTRUCTIONS.format(tool_descriptions=tool_descriptions)

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
