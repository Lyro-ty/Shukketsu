"""Orchestrator specialist agent.

Overrides execute() with a three-phase flow: decompose → dispatch → synthesize.
Does not use the ReAct loop. Creates an execution plan via LLM, dispatches
typed tasks to specialist agents, and synthesizes results.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from code.shukketsu.agents.base import BaseAgent, StatusCallback
from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    ArticleType,
    EditResult,
    EditTask,
    Finding,
    OrchestratorPlan,
    OrchestratorResult,
    ResearchResult,
    ResearchTask,
    SubTask,
    TaskStatus,
    ToolCallRecord,
    WriteResult,
    WriteTask,
)
from code.shukketsu.knowledge.manager import KnowledgeManager
from code.shukketsu.llm.prompts.orchestrator import (
    DECOMPOSITION_PROMPT,
    DECOMPOSITION_PROMPT_WITH_HINTS,
    ORCHESTRATOR_SYSTEM_PROMPT,
    SYNTHESIS_PROMPT,
)
from code.shukketsu.llm.structured import get_structured_output
from code.shukketsu.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from code.shukketsu.agents.factory import AgentFactory

logger = logging.getLogger(__name__)

# Roles that require knowledge_manager kwarg
_KM_ROLES = frozenset({AgentRole.WRITER, AgentRole.EDITOR})


class _SynthesisOutput(BaseModel):
    """Schema for the synthesis LLM call."""

    response: str


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
        model_name: str | None = None,
    ) -> OrchestratorResult:
        """Execute a complex task by decomposing, dispatching, and synthesizing."""
        if self.role is None:
            raise ValueError("Orchestrator requires a role. Use AgentFactory or set role in constructor.")

        # Phase 1: Decompose
        if on_status:
            await on_status("planning...")

        try:
            strategy_hints = task.context.get("strategy_hints", "") if task.context else ""
            plan = await self._decompose(task.query, strategy_hints=strategy_hints)
        except Exception as exc:
            logger.warning("Decomposition failed: %s", exc)
            return OrchestratorResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=TaskStatus.FAILED,
                output=f"Failed to decompose query: {exc}",
                plan=OrchestratorPlan(
                    reasoning="Decomposition failed",
                    subtasks=[],
                ),
            )

        # Handle direct answer
        if plan.can_answer_directly and plan.direct_answer:
            return OrchestratorResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=TaskStatus.SUCCESS,
                output=plan.direct_answer,
                plan=plan,
            )

        # Phase 2: Dispatch
        if on_status:
            await on_status(f"executing {len(plan.subtasks)} sub-tasks...")

        results, skipped = await self._dispatch(plan, task, on_status)

        # Build trajectory from dispatched subtasks
        trajectory: list[ToolCallRecord] = []
        for i, subtask in enumerate(plan.subtasks):
            if results[i] is not None:
                trajectory.append(
                    ToolCallRecord(
                        tool_name=f"dispatch:{subtask.agent_role}",
                        tool_input={"description": subtask.description},
                    )
                )

        # Phase 3: Synthesize
        if on_status:
            await on_status("synthesizing results...")

        orch_result = await self._synthesize(task, plan, results, skipped)
        orch_result.trajectory = trajectory
        return orch_result

    async def _decompose(self, query: str, *, strategy_hints: str = "") -> OrchestratorPlan:
        """Phase 1: Decompose query into an execution plan via Llama 70B.

        Args:
            query: The user's question to decompose.
            strategy_hints: Optional strategy hints from past successful sessions.
                Lines beyond 3 are truncated.
        """
        if strategy_hints.strip():
            hint_lines = [line for line in strategy_hints.strip().split("\n") if line.strip()]
            capped = "\n".join(hint_lines[:3])
            user_content = DECOMPOSITION_PROMPT_WITH_HINTS.format(query=query, hints=capped)
        else:
            user_content = DECOMPOSITION_PROMPT.format(query=query)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        plan: OrchestratorPlan = await get_structured_output(
            response_model=OrchestratorPlan,
            messages=messages,
        )

        errors = self._validate_plan(plan)
        if errors:
            logger.info(
                "Plan validation failed (%d errors), retrying",
                len(errors),
            )
            feedback = "Your plan had errors:\n" + "\n".join(f"- {e}" for e in errors) + "\n\nPlease fix and try again."
            messages.append({"role": "assistant", "content": plan.model_dump_json()})
            messages.append({"role": "user", "content": feedback})

            plan = await get_structured_output(
                response_model=OrchestratorPlan,
                messages=messages,
            )

            errors = self._validate_plan(plan)
            if errors:
                raise ValueError(f"Plan validation failed after retry: {'; '.join(errors)}")

        return plan

    async def _execute_single(
        self,
        idx: int,
        subtask: SubTask,
        results: list[AgentResult | None],
        skipped: list[str],
        task: AgentTask,
        on_status: StatusCallback | None = None,
    ) -> None:
        """Execute a single subtask and store its result in the results list.

        Handles failed dependencies, missing knowledge_manager, task building
        errors, and agent execution failures. On any failure, records a FAILED
        AgentResult or appends to skipped list.
        """
        # Check failed dependencies
        failed_deps = [d for d in subtask.depends_on if (r := results[d]) is not None and r.status == TaskStatus.FAILED]
        if failed_deps:
            skipped.append(subtask.description)
            logger.info(
                "Skipping subtask %d (%s): failed dependencies",
                idx,
                subtask.description,
            )
            return

        # Check if Writer/Editor needs knowledge_manager
        if subtask.agent_role in _KM_ROLES and self._km is None:
            skipped.append(f"{subtask.description} (no knowledge_manager)")
            logger.warning(
                "Skipping %s subtask: no knowledge_manager",
                subtask.agent_role,
            )
            return

        # Build typed task
        try:
            typed_task = self._build_task(
                subtask,
                results,
                task.trace_id,
            )
        except (ValueError, KeyError) as exc:
            logger.warning(
                "Failed to build task for subtask %d: %s",
                idx,
                exc,
            )
            skipped.append(subtask.description)
            return

        # Create specialist with role-conditional kwargs
        extra_kwargs: dict[str, Any] = {}
        if subtask.agent_role in _KM_ROLES:
            extra_kwargs["knowledge_manager"] = self._km

        try:
            agent = self._factory.create(
                subtask.agent_role,
                tool_registry=self.tool_registry,
                **extra_kwargs,
            )
        except Exception as exc:
            logger.warning(
                "Failed to create agent for subtask %d (%s): %s",
                idx,
                subtask.agent_role,
                exc,
            )
            results[idx] = AgentResult(
                task_id=typed_task.task_id,
                agent_role=subtask.agent_role,
                status=TaskStatus.FAILED,
                output=f"Agent creation failed: {exc}",
            )
            return

        if on_status:
            await on_status(f"{subtask.agent_role}: {subtask.description[:50]}...")

        try:
            results[idx] = await agent.execute(
                typed_task,
                on_status=on_status,
            )
        except Exception as exc:
            logger.warning("Subtask %d failed: %s", idx, exc)
            results[idx] = AgentResult(
                task_id=typed_task.task_id,
                agent_role=subtask.agent_role,
                status=TaskStatus.FAILED,
                output=f"Specialist failed: {exc}",
            )

    async def _dispatch(
        self,
        plan: OrchestratorPlan,
        task: AgentTask,
        on_status: StatusCallback | None = None,
    ) -> tuple[list[AgentResult | None], list[str]]:
        """Phase 2: Execute subtasks in topological order, parallelizing independent tasks."""
        results: list[AgentResult | None] = [None] * len(plan.subtasks)
        skipped: list[str] = []

        levels = self._group_by_level(plan.subtasks)

        for level in levels:
            if len(level) == 1:
                # Single subtask — run directly (no gather overhead)
                idx = level[0]
                await self._execute_single(idx, plan.subtasks[idx], results, skipped, task, on_status)
            else:
                # Multiple independent subtasks — run in parallel
                coros = [
                    self._execute_single(idx, plan.subtasks[idx], results, skipped, task, on_status) for idx in level
                ]
                gather_results = await asyncio.gather(*coros, return_exceptions=True)
                for i, res in enumerate(gather_results):
                    if isinstance(res, BaseException):
                        idx = level[i]
                        logger.error("Unhandled exception in subtask %d: %s", idx, res)
                        if results[idx] is None:
                            results[idx] = AgentResult(
                                task_id=task.task_id,
                                agent_role=plan.subtasks[idx].agent_role,
                                status=TaskStatus.FAILED,
                                output=f"Unexpected error: {res}",
                            )

        return results, skipped

    async def _synthesize(
        self,
        task: AgentTask,
        plan: OrchestratorPlan,
        results: list[AgentResult | None],
        skipped: list[str],
    ) -> OrchestratorResult:
        """Phase 3: Synthesize specialist results into a final response."""
        completed = [r for r in results if r is not None]

        # Extract article info
        article_path: str | None = None
        needs_human_review = False
        for r in completed:
            if isinstance(r, WriteResult) and r.article_path:
                article_path = r.article_path
            if isinstance(r, EditResult) and r.approved_for_review:
                needs_human_review = True

        # Determine overall status
        if not completed:
            status = TaskStatus.FAILED
        elif all(r.status == TaskStatus.SUCCESS for r in completed) and not skipped:
            status = TaskStatus.SUCCESS
        elif all(r.status == TaskStatus.FAILED for r in completed):
            status = TaskStatus.FAILED
        else:
            status = TaskStatus.PARTIAL

        # Synthesize output
        is_article_workflow = any(isinstance(r, WriteResult) for r in completed)

        if is_article_workflow:
            output = self._synthesize_article_template(completed, skipped)
        elif completed:
            output = await self._synthesize_research(task.query, completed)
        else:
            output = "No specialist results available."
            if skipped:
                output += f" Skipped tasks: {', '.join(skipped)}"

        # Aggregate evidence
        evidence = list(dict.fromkeys(e for r in completed for e in r.evidence))

        assert self.role is not None  # Guaranteed by execute() guard
        return OrchestratorResult(
            task_id=task.task_id,
            agent_role=self.role,
            status=status,
            output=output,
            evidence=evidence,
            plan=plan,
            specialist_results=completed,
            article_path=article_path,
            needs_human_review=needs_human_review,
            skipped_tasks=skipped,
        )

    def _synthesize_article_template(
        self,
        results: list[AgentResult],
        skipped: list[str],
    ) -> str:
        """Template-based synthesis for article workflows."""
        parts: list[str] = []
        for r in results:
            if isinstance(r, WriteResult):
                parts.append(f"Article written: **{r.title}**")
                parts.append(f"Path: `{r.article_path}`")
                parts.append(f"Claims: {len(r.claims)}")
                if r.research_gaps:
                    parts.append(f"Research gaps: {', '.join(r.research_gaps)}")
            elif isinstance(r, EditResult):
                parts.append(f"Verification: confidence {r.overall_confidence:.0%}")
                if r.approved_for_review:
                    parts.append("Status: approved for human review")
                if r.corrections:
                    parts.append(f"Corrections needed: {len(r.corrections)}")
        if skipped:
            parts.append(f"Skipped: {', '.join(skipped)}")
        return "\n".join(parts)

    async def _synthesize_research(
        self,
        query: str,
        results: list[AgentResult],
    ) -> str:
        """LLM-based synthesis for research-only workflows."""
        findings_text = "\n\n".join(r.output for r in results if r.output)

        if not findings_text.strip():
            return "I wasn't able to find relevant information to answer your question. Please try rephrasing."

        messages = [
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {
                "role": "user",
                "content": f"Original question: {query}\n\nResearch findings:\n{findings_text}",
            },
        ]

        try:
            result: _SynthesisOutput = await get_structured_output(
                response_model=_SynthesisOutput,
                messages=messages,
            )
            return result.response
        except Exception as exc:
            logger.warning(
                "Synthesis LLM call failed, falling back to raw findings: %s",
                exc,
            )
            return findings_text

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
            if st.agent_role == AgentRole.ANALYST:
                errors.append(f"Subtask {i}: ANALYST must be dispatched directly, not via Orchestrator")

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

    def _group_by_level(self, subtasks: list[SubTask]) -> list[list[int]]:
        """Group subtask indices by dependency depth.

        Level 0 = no dependencies, level 1 = depends only on level 0, etc.
        Used to identify which subtasks can run in parallel within each level.

        Args:
            subtasks: The list of subtasks to group.

        Returns:
            A list of levels, where each level is a list of subtask indices.
        """
        n = len(subtasks)
        if n == 0:
            return []

        # Compute depth for each node via BFS
        depth = [0] * n
        adj: dict[int, list[int]] = defaultdict(list)
        in_degree = [0] * n

        for i, st in enumerate(subtasks):
            for dep in st.depends_on:
                if 0 <= dep < n:
                    adj[dep].append(i)
                    in_degree[i] += 1

        queue: deque[int] = deque(i for i in range(n) if in_degree[i] == 0)

        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                depth[neighbor] = max(depth[neighbor], depth[node] + 1)
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Group by depth
        max_depth = max(depth) if depth else 0
        levels: list[list[int]] = [[] for _ in range(max_depth + 1)]
        for i, d in enumerate(depth):
            levels[d].append(i)

        return levels

    def _build_task(
        self,
        subtask: SubTask,
        results: list[AgentResult | None],
        trace_id: str,
    ) -> AgentTask:
        """Build a typed task from a SubTask and completed dependency results."""
        if subtask.agent_role == AgentRole.RESEARCHER:
            return ResearchTask(
                query=subtask.description,
                trace_id=trace_id,
                context={"complexity": "complex"},
            )

        if subtask.agent_role == AgentRole.WRITER:
            research_results: list[ResearchResult] = [
                r for d in subtask.depends_on if (r := results[d]) is not None and isinstance(r, ResearchResult)
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
            write_result: WriteResult | None = next(
                (r for d in subtask.depends_on if (r := results[d]) is not None and isinstance(r, WriteResult)),
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
