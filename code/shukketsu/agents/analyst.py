"""Analyst specialist agent for quantitative DPS analysis.

Overrides execute() to run the standard ReAct loop with sim tools,
then post-processes the output to extract DPS values and recommendations
into a typed AnalysisResult.
"""

import logging
import re

from langfuse import observe

from code.shukketsu.agents.base import BaseAgent, StatusCallback
from code.shukketsu.agents.tasks import AgentRole, AgentTask, AnalysisResult, TaskStatus, ToolCallRecord

logger = logging.getLogger(__name__)

# Patterns for extracting DPS numbers from agent output
_DPS_PATTERNS = [
    re.compile(r"(?:mean\s+)?DPS[:\s]+(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s+DPS", re.IGNORECASE),
]


class Analyst(BaseAgent):
    """Specialist agent for quantitative DPS analysis using the sim engine.

    Uses the standard ReAct loop to run simulations and analyze results.
    Post-processes the output to extract DPS values and recommendations.
    """

    @observe(as_type="agent")
    async def execute(
        self, task: AgentTask, *, on_status: StatusCallback | None = None, model_name: str | None = None
    ) -> AnalysisResult:
        """Execute an analysis task and return structured DPS results.

        Runs the ReAct loop with sim tools, then extracts DPS numbers
        from the output text.

        Args:
            task: The analysis task to execute.
            on_status: Optional async callback for progress updates.
            model_name: Optional model override for the ReAct loop LLM calls.

        Returns:
            AnalysisResult with DPS mean and any extracted recommendations.

        Raises:
            ValueError: If no role is set on this agent.
            LLMUnavailableError: If the LLM backend is unreachable.
        """
        if self.role is None:
            raise ValueError("Analyst requires a role. Use AgentFactory or set role in constructor.")

        outcome = await self._run_loop(task.query, on_status=on_status, model_name=model_name)

        trajectory = [
            ToolCallRecord(tool_name=entry["tool_name"], tool_input=entry["tool_input"]) for entry in outcome.scratchpad
        ]

        dps_mean = self._extract_dps(outcome.output)

        return AnalysisResult(
            task_id=task.task_id,
            agent_role=AgentRole.ANALYST,
            status=TaskStatus.SUCCESS if outcome.output else TaskStatus.FAILED,
            output=outcome.output,
            trajectory=trajectory,
            dps_mean=dps_mean,
        )

    @staticmethod
    def _extract_dps(text: str) -> float | None:
        """Extract a DPS number from the agent's output text.

        Searches for patterns like "mean DPS: 1234.5", "DPS: 1234",
        or "1234.5 DPS" in the text.

        Args:
            text: The agent's output text to search.

        Returns:
            The first DPS value found, or None if no match.
        """
        for pattern in _DPS_PATTERNS:
            match = pattern.search(text)
            if match:
                return float(match.group(1))
        return None
