"""Orchestrator specialist agent.

Overrides execute() with a three-phase flow: decompose → dispatch → synthesize.
Does not use the ReAct loop. Creates an execution plan via LLM, dispatches
typed tasks to specialist agents, and synthesizes results.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING

from code.shukketsu.agents.base import BaseAgent, StatusCallback
from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    ArticleType,
    EditTask,
    Finding,
    OrchestratorPlan,
    OrchestratorResult,
    ResearchResult,
    ResearchTask,
    SubTask,
    TaskStatus,
    WriteResult,
    WriteTask,
)
from code.shukketsu.knowledge.manager import KnowledgeManager
from code.shukketsu.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from code.shukketsu.agents.factory import AgentFactory

logger = logging.getLogger(__name__)

# Roles that require knowledge_manager kwarg
_KM_ROLES = frozenset({AgentRole.WRITER, AgentRole.EDITOR})


class Orchestrator(BaseAgent):
    """Multi-agent coordinator that decomposes complex queries into specialist tasks."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        role: AgentRole | None = None,
        max_iterations: int | None = None,
        system_prompt: str | None = None,
        factory: AgentFactory,
        knowledge_manager: KnowledgeManager | None = None,
    ) -> None:
        from code.shukketsu import config

        super().__init__(
            tool_registry=tool_registry,
            role=role,
            max_iterations=max_iterations or config.AGENT_MAX_ITERATIONS,
            system_prompt=system_prompt or config.SYSTEM_PROMPT,
        )
        self._factory = factory
        self._km = knowledge_manager

    async def execute(
        self,
        task: AgentTask,
        *,
        on_status: StatusCallback | None = None,
    ) -> OrchestratorResult:
        """Execute a complex task by decomposing, dispatching, and synthesizing.

        Placeholder — dispatch and synthesis added in later tasks.
        """
        raise NotImplementedError("execute() implemented in Task 5")

    def _validate_plan(self, plan: OrchestratorPlan) -> list[str]:
        """Validate an OrchestratorPlan for structural issues.

        Returns a list of error strings. Empty list means valid.
        """
        from code.shukketsu import config

        errors: list[str] = []

        if len(plan.subtasks) > config.ORCHESTRATOR_MAX_SUBTASKS:
            errors.append(f"Too many subtasks: {len(plan.subtasks)} (max {config.ORCHESTRATOR_MAX_SUBTASKS})")

        for i, st in enumerate(plan.subtasks):
            if st.agent_role == AgentRole.ORCHESTRATOR:
                errors.append(f"Subtask {i}: ORCHESTRATOR role not allowed (no recursion)")

            for dep in st.depends_on:
                if dep < 0 or dep >= len(plan.subtasks):
                    errors.append(f"Subtask {i}: dependency index {dep} out of range")

            if i in st.depends_on:
                errors.append(f"Subtask {i}: self-dependency")

            valid_deps = {d for d in st.depends_on if 0 <= d < len(plan.subtasks)}
            dep_roles = {plan.subtasks[d].agent_role for d in valid_deps}

            if st.agent_role == AgentRole.WRITER:
                if AgentRole.RESEARCHER not in dep_roles:
                    errors.append(f"Subtask {i}: WRITER must depend on a RESEARCHER")

            if st.agent_role == AgentRole.EDITOR:
                if AgentRole.WRITER not in dep_roles:
                    errors.append(f"Subtask {i}: EDITOR must depend on a WRITER")

        try:
            self._topological_sort(plan.subtasks)
        except ValueError:
            errors.append("Dependency cycle detected")

        return errors

    def _topological_sort(self, subtasks: list[SubTask]) -> list[int]:
        """Return subtask indices in topological order.

        Uses Kahn's algorithm. Raises ValueError if a cycle exists.
        """
        n = len(subtasks)
        if n == 0:
            return []

        adj: dict[int, list[int]] = defaultdict(list)
        in_degree = [0] * n

        for i, st in enumerate(subtasks):
            for dep in st.depends_on:
                if 0 <= dep < n:
                    adj[dep].append(i)
                    in_degree[i] += 1

        queue: deque[int] = deque(i for i in range(n) if in_degree[i] == 0)
        order: list[int] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != n:
            raise ValueError("Dependency cycle detected")

        return order

    def _build_task(
        self,
        subtask: SubTask,
        results: list[AgentResult | None],
        trace_id: str,
    ) -> AgentTask:
        """Build a typed task from a SubTask and completed dependency results."""
        if subtask.agent_role == AgentRole.RESEARCHER:
            return ResearchTask(query=subtask.description, trace_id=trace_id)

        if subtask.agent_role == AgentRole.WRITER:
            research_results = [
                results[d]
                for d in subtask.depends_on
                if results[d] is not None and isinstance(results[d], ResearchResult)
            ]
            if not research_results:
                raise ValueError("WRITER subtask has no ResearchResult dependencies")

            research = self._merge_research(research_results) if len(research_results) > 1 else research_results[0]

            spec = subtask.task_params.get("spec", "general")
            category = subtask.task_params.get("category", "general")
            article_type_str = subtask.task_params.get("article_type", "guide")
            try:
                article_type = ArticleType(article_type_str)
            except ValueError:
                article_type = ArticleType.GUIDE

            return WriteTask(
                query=subtask.description,
                trace_id=trace_id,
                research=research,
                article_type=article_type,
                spec=spec,
                category=category,
            )

        if subtask.agent_role == AgentRole.EDITOR:
            write_result = next(
                (
                    results[d]
                    for d in subtask.depends_on
                    if results[d] is not None and isinstance(results[d], WriteResult)
                ),
                None,
            )
            if write_result is None:
                raise ValueError("EDITOR subtask has no WriteResult dependency")

            return EditTask(
                query=subtask.description,
                trace_id=trace_id,
                article_path=write_result.article_path,
                claims=write_result.claims,
            )

        raise ValueError(f"Unsupported agent role: {subtask.agent_role}")

    def _merge_research(
        self,
        results: list[ResearchResult],
    ) -> ResearchResult:
        """Merge multiple ResearchResults into one for the Writer."""
        seen_claims: set[str] = set()
        merged_findings: list[Finding] = []
        for r in results:
            for f in r.findings:
                if f.claim not in seen_claims:
                    seen_claims.add(f.claim)
                    merged_findings.append(f)

        return ResearchResult(
            task_id=results[0].task_id,
            agent_role=AgentRole.RESEARCHER,
            status=(TaskStatus.SUCCESS if all(r.status == TaskStatus.SUCCESS for r in results) else TaskStatus.PARTIAL),
            output="\n\n---\n\n".join(r.output for r in results),
            findings=merged_findings,
            sources_used=list(dict.fromkeys(s for r in results for s in r.sources_used)),
            strategies_used=list(dict.fromkeys(s for r in results for s in r.strategies_used)),
            gaps=list(dict.fromkeys(g for r in results for g in r.gaps)),
            sufficient=all(r.sufficient for r in results),
        )
