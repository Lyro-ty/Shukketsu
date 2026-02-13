"""Researcher specialist agent.

Overrides execute() to run the standard ReAct loop and then structure
the output into a typed ResearchResult via a Llama 70B structuring pass.
"""

import logging
from typing import Any

from langfuse import observe
from pydantic import BaseModel, Field

from code.shukketsu import config
from code.shukketsu.agents.base import BaseAgent, StatusCallback
from code.shukketsu.agents.tasks import AgentTask, Finding, ResearchResult, TaskStatus, ToolCallRecord
from code.shukketsu.llm.prompts.researcher import REFLECTION_PROMPT, STRUCTURING_PROMPT
from code.shukketsu.llm.schemas import ReflectionResult
from code.shukketsu.llm.structured import get_structured_output

logger = logging.getLogger(__name__)


class StructuredFindings(BaseModel):
    """Schema for the Llama 70B structuring pass.

    Parsed from the agent's output + scratchpad, then mapped
    to ResearchResult fields.
    """

    findings: list[Finding] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    sufficient: bool = False


class Researcher(BaseAgent):
    """Research specialist agent.

    Uses the same ReAct loop as BaseAgent, then runs a second Llama 70B
    call to structure the free-text output into a typed ResearchResult
    with findings, evidence, gaps, and sufficiency assessment.
    """

    @observe(as_type="agent")
    async def execute(
        self, task: AgentTask, *, on_status: StatusCallback | None = None, model_name: str | None = None
    ) -> ResearchResult:
        """Execute a research task and return structured findings.

        Runs the ReAct loop, then structures the output via a second
        LLM call. Falls back to a basic result if structuring fails.

        Args:
            task: The research task to execute.
            on_status: Optional async callback for progress updates.

        Returns:
            ResearchResult with structured findings and evidence.

        Raises:
            ValueError: If no role is set on this agent.
            LLMUnavailableError: If the LLM backend is unreachable.
        """
        if self.role is None:
            raise ValueError("Researcher requires a role. Use AgentFactory or set role in constructor.")

        outcome = await self._run_loop(task.query, on_status=on_status, model_name=model_name)

        trajectory = [ToolCallRecord(tool_name=e["tool_name"], tool_input=e["tool_input"]) for e in outcome.scratchpad]

        # Skip structuring for failed loops — no useful output to parse
        if outcome.status == TaskStatus.FAILED:
            logger.info("ReAct loop failed; skipping structuring pass")
            return ResearchResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=outcome.status,
                output=outcome.output,
                sufficient=False,
                trajectory=trajectory,
            )

        # Structure the output via Llama 70B
        if on_status:
            await on_status("structuring findings...")

        try:
            structured = await self._structure_findings(
                query=task.query,
                output=outcome.output,
                scratchpad=outcome.scratchpad,
                model_name=model_name,
            )
        except Exception as exc:
            logger.warning("Structuring pass failed, returning basic result: %s", exc)
            return ResearchResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=outcome.status,
                output=outcome.output,
                strategies_used=self._extract_strategies(outcome.scratchpad),
                sufficient=False,
                trajectory=trajectory,
            )

        # Derive metadata from structuring output + scratchpad
        sources_used = list(dict.fromkeys(e for f in structured.findings for e in f.evidence))
        strategies_used = self._extract_strategies(outcome.scratchpad)

        # Reflection pass for complex queries
        final_output = outcome.output
        is_complex = task.context.get("complexity") == "complex"

        if is_complex and config.REFLECTION_ENABLED and outcome.status == TaskStatus.SUCCESS:
            if on_status:
                await on_status("reflecting on answer...")

            try:
                reflection = await self._reflect(
                    query=task.query,
                    output=outcome.output,
                    scratchpad=outcome.scratchpad,
                )
                if not reflection.supported and reflection.revised_answer:
                    logger.info(
                        "Reflection found %d issues; using revised answer",
                        len(reflection.issues),
                    )
                    final_output = reflection.revised_answer
            except Exception as exc:
                logger.warning("Reflection failed, keeping original answer: %s", exc)

        return ResearchResult(
            task_id=task.task_id,
            agent_role=self.role,
            status=outcome.status,
            output=final_output,
            evidence=sources_used,
            findings=structured.findings,
            sources_used=sources_used,
            strategies_used=strategies_used,
            gaps=structured.gaps,
            sufficient=structured.sufficient,
            trajectory=trajectory,
        )

    async def _structure_findings(
        self,
        query: str,
        output: str,
        scratchpad: list[dict[str, Any]],
        *,
        model_name: str | None = None,
    ) -> StructuredFindings:
        """Run the structuring pass to convert free-text into typed findings.

        Takes the agent's free-text output and tool observations,
        returns structured findings with evidence and confidence.
        Uses the same model as the ReAct loop when a model override is set.
        """
        observations_text = self._format_scratchpad(scratchpad)

        messages = [
            {"role": "system", "content": STRUCTURING_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\nResearcher's answer:\n{output}\n\nTool observations:\n{observations_text}"
                ),
            },
        ]

        result: StructuredFindings = await get_structured_output(
            response_model=StructuredFindings,
            messages=messages,
            model=model_name,
        )
        return result

    def _extract_strategies(self, scratchpad: list[dict[str, Any]]) -> list[str]:
        """Extract unique tool names used during research."""
        return list(dict.fromkeys(e["tool_name"] for e in scratchpad))

    def _format_scratchpad(self, scratchpad: list[dict[str, Any]]) -> str:
        """Format scratchpad entries for the structuring prompt.

        Truncates each observation to 500 chars to fit within the
        structuring call's token budget.
        """
        if not scratchpad:
            return "(no tool calls made)"

        parts: list[str] = []
        for i, entry in enumerate(scratchpad, 1):
            observation = str(entry.get("observation", ""))
            truncated = observation[:500] + "..." if len(observation) > 500 else observation
            parts.append(f"[{i}] Tool: {entry['tool_name']}\n    Input: {entry['tool_input']}\n    Result: {truncated}")
        return "\n\n".join(parts)

    async def _reflect(
        self,
        query: str,
        output: str,
        scratchpad: list[dict[str, Any]],
    ) -> ReflectionResult:
        """Run a reflection pass to verify the answer against evidence."""
        observations_text = self._format_scratchpad(scratchpad)

        messages = [
            {"role": "system", "content": REFLECTION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\nResearcher's answer:\n{output}\n\nTool observations:\n{observations_text}"
                ),
            },
        ]

        result: ReflectionResult = await get_structured_output(
            response_model=ReflectionResult,
            messages=messages,
            temperature=config.REFLECTION_TEMPERATURE,
        )
        return result
