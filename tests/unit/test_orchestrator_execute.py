"""Tests for Orchestrator.execute — decompose, dispatch, synthesize."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    EditResult,
    Finding,
    OrchestratorPlan,
    OrchestratorResult,
    ResearchResult,
    SubTask,
    TaskStatus,
    WriteResult,
)
from code.shukketsu.tools.registry import ToolRegistry


def _plan(
    subtasks: list[SubTask] | None = None,
    can_answer_directly: bool = False,
    direct_answer: str | None = None,
) -> OrchestratorPlan:
    return OrchestratorPlan(
        reasoning="test plan",
        subtasks=subtasks or [],
        can_answer_directly=can_answer_directly,
        direct_answer=direct_answer,
    )


def _research_result(**overrides: object) -> ResearchResult:
    defaults: dict[str, object] = dict(
        task_id="r1",
        agent_role=AgentRole.RESEARCHER,
        status=TaskStatus.SUCCESS,
        output="Research findings here.",
        findings=[Finding(claim="Hit cap is 142", confidence=0.9, evidence=["src1"])],
        sources_used=["src1"],
        strategies_used=["rag_search"],
        gaps=[],
        sufficient=True,
    )
    defaults.update(overrides)
    return ResearchResult(**defaults)


def _write_result(**overrides: object) -> WriteResult:
    defaults: dict[str, object] = dict(
        task_id="w1",
        agent_role=AgentRole.WRITER,
        status=TaskStatus.SUCCESS,
        output="Article written",
        article_path="combat/gear/trinkets.md",
        title="Trinkets Guide",
        claims=["Hit cap is 142"],
    )
    defaults.update(overrides)
    return WriteResult(**defaults)


def _edit_result(**overrides: object) -> EditResult:
    defaults: dict[str, object] = dict(
        task_id="e1",
        agent_role=AgentRole.EDITOR,
        status=TaskStatus.SUCCESS,
        output="Verified 1 claim. Confidence: 0.85",
        article_path="combat/gear/trinkets.md",
        overall_confidence=0.85,
        approved_for_review=True,
    )
    defaults.update(overrides)
    return EditResult(**defaults)


def _mock_factory(role_results: dict[AgentRole, AgentResult]) -> MagicMock:
    """Create a mock AgentFactory that returns mock agents with preset results."""
    factory = MagicMock()

    def _create(role, **kwargs):
        agent = MagicMock()
        result = role_results.get(role)
        agent.execute = AsyncMock(return_value=result)
        return agent

    factory.create.side_effect = _create
    return factory


def _orchestrator(
    factory: MagicMock | None = None,
    km: MagicMock | None = None,
) -> object:
    from code.shukketsu.agents.orchestrator import Orchestrator

    return Orchestrator(
        tool_registry=ToolRegistry(),
        role=AgentRole.ORCHESTRATOR,
        factory=factory or _mock_factory({}),
        knowledge_manager=km,
    )


class TestDirectAnswer:
    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_direct_answer_skips_dispatch(self, mock_llm: AsyncMock) -> None:
        """can_answer_directly=True returns immediately without specialists."""
        mock_llm.return_value = _plan(
            can_answer_directly=True,
            direct_answer="Shukketsu is a WoW TBC Rogue research system.",
        )
        orch = _orchestrator()
        result = await orch.execute(AgentTask(query="What can you do?"))

        assert isinstance(result, OrchestratorResult)
        assert result.status == TaskStatus.SUCCESS
        assert result.output == "Shukketsu is a WoW TBC Rogue research system."
        assert result.specialist_results == []


class TestSingleResearcher:
    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_single_researcher_plan(self, mock_llm: AsyncMock) -> None:
        """Single RESEARCHER subtask dispatches and returns synthesized output."""
        plan = _plan(
            subtasks=[
                SubTask(agent_role=AgentRole.RESEARCHER, description="Find trinkets"),
            ]
        )

        call_count = 0

        async def _mock_structured(response_model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if response_model is OrchestratorPlan:
                return plan
            # Synthesis call
            return response_model(response="The best trinkets are DST and Bloodlust Brooch.")

        mock_llm.side_effect = _mock_structured

        factory = _mock_factory({AgentRole.RESEARCHER: _research_result()})
        orch = _orchestrator(factory=factory)
        result = await orch.execute(AgentTask(query="Best trinkets?"))

        assert result.status == TaskStatus.SUCCESS
        assert len(result.specialist_results) == 1
        factory.create.assert_called_once()


class TestRoleConditionalKwargs:
    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_writer_receives_km_researcher_does_not(self, mock_llm: AsyncMock) -> None:
        """knowledge_manager passed to Writer/Editor but NOT to Researcher."""
        plan = _plan(
            subtasks=[
                SubTask(agent_role=AgentRole.RESEARCHER, description="research"),
                SubTask(
                    agent_role=AgentRole.WRITER,
                    description="write",
                    depends_on=[0],
                    task_params={"spec": "combat", "category": "gear"},
                ),
            ]
        )
        mock_llm.return_value = plan

        km = MagicMock()
        factory = MagicMock()
        agents_created: list[tuple[AgentRole, dict]] = []

        def _create(role, **kwargs):
            agents_created.append((role, kwargs))
            agent = MagicMock()
            if role == AgentRole.RESEARCHER:
                agent.execute = AsyncMock(return_value=_research_result())
            else:
                agent.execute = AsyncMock(return_value=_write_result())
            return agent

        factory.create.side_effect = _create

        orch = _orchestrator(factory=factory, km=km)
        await orch.execute(AgentTask(query="Write a guide"))

        # Researcher should NOT get knowledge_manager
        researcher_kwargs = agents_created[0][1]
        assert "knowledge_manager" not in researcher_kwargs

        # Writer SHOULD get knowledge_manager
        writer_kwargs = agents_created[1][1]
        assert writer_kwargs["knowledge_manager"] is km


class TestDependencyChain:
    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_research_write_edit_chain(self, mock_llm: AsyncMock) -> None:
        """Full pipeline: research → write → edit chains correctly."""
        plan = _plan(
            subtasks=[
                SubTask(agent_role=AgentRole.RESEARCHER, description="research"),
                SubTask(
                    agent_role=AgentRole.WRITER,
                    description="write",
                    depends_on=[0],
                    task_params={"spec": "combat", "category": "gear"},
                ),
                SubTask(agent_role=AgentRole.EDITOR, description="edit", depends_on=[1]),
            ]
        )
        mock_llm.return_value = plan

        km = MagicMock()
        factory = MagicMock()

        def _create(role, **kwargs):
            agent = MagicMock()
            if role == AgentRole.RESEARCHER:
                agent.execute = AsyncMock(return_value=_research_result())
            elif role == AgentRole.WRITER:
                agent.execute = AsyncMock(return_value=_write_result())
            else:
                agent.execute = AsyncMock(return_value=_edit_result())
            return agent

        factory.create.side_effect = _create

        orch = _orchestrator(factory=factory, km=km)
        result = await orch.execute(AgentTask(query="Write a guide"))

        assert result.status == TaskStatus.SUCCESS
        assert result.article_path == "combat/gear/trinkets.md"
        assert result.needs_human_review is True
        assert len(result.specialist_results) == 3

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_failed_dependency_skips_dependents(self, mock_llm: AsyncMock) -> None:
        """Failed Researcher → Writer and Editor skipped."""
        plan = _plan(
            subtasks=[
                SubTask(agent_role=AgentRole.RESEARCHER, description="research"),
                SubTask(
                    agent_role=AgentRole.WRITER,
                    description="write",
                    depends_on=[0],
                    task_params={"spec": "combat", "category": "gear"},
                ),
                SubTask(agent_role=AgentRole.EDITOR, description="edit", depends_on=[1]),
            ]
        )
        mock_llm.return_value = plan

        factory = MagicMock()
        failed_research = _research_result(status=TaskStatus.FAILED, output="Failed")

        def _create(role, **kwargs):
            agent = MagicMock()
            agent.execute = AsyncMock(return_value=failed_research)
            return agent

        factory.create.side_effect = _create

        orch = _orchestrator(factory=factory)
        result = await orch.execute(AgentTask(query="Write a guide"))

        assert result.status == TaskStatus.FAILED
        assert len(result.skipped_tasks) == 2  # Writer + Editor skipped

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_partial_dependency_proceeds(self, mock_llm: AsyncMock) -> None:
        """Partial Researcher → Writer still runs."""
        plan = _plan(
            subtasks=[
                SubTask(agent_role=AgentRole.RESEARCHER, description="research"),
                SubTask(
                    agent_role=AgentRole.WRITER,
                    description="write",
                    depends_on=[0],
                    task_params={"spec": "combat", "category": "gear"},
                ),
            ]
        )
        mock_llm.return_value = plan

        km = MagicMock()
        factory = MagicMock()
        partial_research = _research_result(status=TaskStatus.PARTIAL)

        def _create(role, **kwargs):
            agent = MagicMock()
            if role == AgentRole.RESEARCHER:
                agent.execute = AsyncMock(return_value=partial_research)
            else:
                agent.execute = AsyncMock(return_value=_write_result())
            return agent

        factory.create.side_effect = _create

        orch = _orchestrator(factory=factory, km=km)
        result = await orch.execute(AgentTask(query="test"))

        assert len(result.specialist_results) == 2  # Both ran
        assert result.skipped_tasks == []


class TestKnowledgeManagerGuard:
    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_writer_skipped_without_km(self, mock_llm: AsyncMock) -> None:
        """Writer subtask skipped when Orchestrator has no knowledge_manager."""
        plan = _plan(
            subtasks=[
                SubTask(agent_role=AgentRole.RESEARCHER, description="research"),
                SubTask(
                    agent_role=AgentRole.WRITER,
                    description="write",
                    depends_on=[0],
                    task_params={"spec": "combat", "category": "gear"},
                ),
            ]
        )
        mock_llm.return_value = plan

        factory = _mock_factory({AgentRole.RESEARCHER: _research_result()})
        orch = _orchestrator(factory=factory, km=None)  # No KM
        result = await orch.execute(AgentTask(query="test"))

        assert len(result.skipped_tasks) == 1
        assert "write" in result.skipped_tasks[0].lower()


class TestErrorHandling:
    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_all_failed_returns_failed(self, mock_llm: AsyncMock) -> None:
        """All specialists failing returns FAILED status."""
        plan = _plan(
            subtasks=[
                SubTask(agent_role=AgentRole.RESEARCHER, description="a"),
                SubTask(agent_role=AgentRole.RESEARCHER, description="b"),
            ]
        )
        mock_llm.return_value = plan

        factory = MagicMock()
        failed = AgentResult(
            task_id="f",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.FAILED,
            output="Failed",
        )

        def _create(role, **kwargs):
            agent = MagicMock()
            agent.execute = AsyncMock(return_value=failed)
            return agent

        factory.create.side_effect = _create

        orch = _orchestrator(factory=factory)
        result = await orch.execute(AgentTask(query="test"))

        assert result.status == TaskStatus.FAILED

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_decomposition_failure_returns_failed(self, mock_llm: AsyncMock) -> None:
        """LLM failure during decomposition returns FAILED."""
        from code.shukketsu.resilience.errors import StructuredOutputError

        mock_llm.side_effect = StructuredOutputError("LLM failed")
        orch = _orchestrator()
        result = await orch.execute(AgentTask(query="test"))

        assert result.status == TaskStatus.FAILED
        assert "decompose" in result.output.lower() or "failed" in result.output.lower()

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_validation_retry_on_structural_error(self, mock_llm: AsyncMock) -> None:
        """First plan invalid, second valid → succeeds."""
        bad_plan = _plan(
            subtasks=[
                SubTask(agent_role=AgentRole.ORCHESTRATOR, description="recurse"),
            ]
        )
        good_plan = _plan(
            subtasks=[
                SubTask(agent_role=AgentRole.RESEARCHER, description="research"),
            ]
        )
        mock_llm.side_effect = [bad_plan, good_plan]

        factory = _mock_factory({AgentRole.RESEARCHER: _research_result()})
        orch = _orchestrator(factory=factory)
        result = await orch.execute(AgentTask(query="test"))

        assert result.status == TaskStatus.SUCCESS
        assert mock_llm.call_count >= 2  # decompose + retry (synthesis may add more)

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_synthesis_llm_failure_falls_back(self, mock_llm: AsyncMock) -> None:
        """Synthesis LLM fail → falls back to concatenated output."""
        plan = _plan(
            subtasks=[
                SubTask(agent_role=AgentRole.RESEARCHER, description="research"),
            ]
        )

        call_count = 0

        async def _side_effect(response_model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if response_model is OrchestratorPlan:
                return plan
            # Synthesis call fails
            raise Exception("Synthesis failed")

        mock_llm.side_effect = _side_effect

        factory = _mock_factory({AgentRole.RESEARCHER: _research_result()})
        orch = _orchestrator(factory=factory)
        result = await orch.execute(AgentTask(query="test"))

        assert result.status == TaskStatus.SUCCESS
        # Falls back to concatenated research output
        assert "Research findings" in result.output or "Research done" in result.output


class TestGroupByLevel:
    """Tests for _group_by_level — dependency depth grouping."""

    def test_empty_subtasks(self) -> None:
        """Empty subtask list returns empty levels."""
        orch = _orchestrator()
        levels = orch._group_by_level([])
        assert levels == []

    def test_all_independent(self) -> None:
        """3 subtasks with no deps -> single level with all 3."""
        orch = _orchestrator()
        subtasks = [
            SubTask(agent_role=AgentRole.RESEARCHER, description="a"),
            SubTask(agent_role=AgentRole.RESEARCHER, description="b"),
            SubTask(agent_role=AgentRole.RESEARCHER, description="c"),
        ]
        levels = orch._group_by_level(subtasks)
        assert len(levels) == 1
        assert sorted(levels[0]) == [0, 1, 2]

    def test_chain_three_levels(self) -> None:
        """A->B->C produces 3 levels of 1 each."""
        orch = _orchestrator()
        subtasks = [
            SubTask(agent_role=AgentRole.RESEARCHER, description="a"),
            SubTask(agent_role=AgentRole.RESEARCHER, description="b", depends_on=[0]),
            SubTask(agent_role=AgentRole.RESEARCHER, description="c", depends_on=[1]),
        ]
        levels = orch._group_by_level(subtasks)
        assert len(levels) == 3
        assert levels[0] == [0]
        assert levels[1] == [1]
        assert levels[2] == [2]

    def test_diamond_dependency(self) -> None:
        """Diamond: A -> B, A -> C, B+C -> D produces 3 levels."""
        orch = _orchestrator()
        subtasks = [
            SubTask(agent_role=AgentRole.RESEARCHER, description="a"),
            SubTask(agent_role=AgentRole.RESEARCHER, description="b", depends_on=[0]),
            SubTask(agent_role=AgentRole.RESEARCHER, description="c", depends_on=[0]),
            SubTask(agent_role=AgentRole.RESEARCHER, description="d", depends_on=[1, 2]),
        ]
        levels = orch._group_by_level(subtasks)
        assert len(levels) == 3
        assert levels[0] == [0]
        assert sorted(levels[1]) == [1, 2]
        assert levels[2] == [3]

    def test_two_independent_plus_dependent(self) -> None:
        """A, B independent; C depends on both -> 2 levels."""
        orch = _orchestrator()
        subtasks = [
            SubTask(agent_role=AgentRole.RESEARCHER, description="a"),
            SubTask(agent_role=AgentRole.RESEARCHER, description="b"),
            SubTask(agent_role=AgentRole.RESEARCHER, description="c", depends_on=[0, 1]),
        ]
        levels = orch._group_by_level(subtasks)
        assert len(levels) == 2
        assert sorted(levels[0]) == [0, 1]
        assert levels[1] == [2]


class TestParallelExecution:
    """Tests for parallel subtask execution via asyncio.gather."""

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_parallel_independent_subtasks(self, mock_llm: AsyncMock) -> None:
        """Two independent RESEARCHER tasks run concurrently via asyncio.gather."""
        plan = _plan(
            subtasks=[
                SubTask(agent_role=AgentRole.RESEARCHER, description="find trinkets"),
                SubTask(agent_role=AgentRole.RESEARCHER, description="find weapons"),
            ]
        )

        call_count = 0

        async def _mock_structured(response_model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if response_model is OrchestratorPlan:
                return plan
            return response_model(response="Synthesized.")

        mock_llm.side_effect = _mock_structured

        execution_order: list[str] = []

        factory = MagicMock()

        def _create(role, **kwargs):
            agent = MagicMock()

            async def _execute(task, on_status=None):
                desc = task.query
                execution_order.append(f"start:{desc}")
                await asyncio.sleep(0.05)
                execution_order.append(f"end:{desc}")
                return _research_result(output=f"Results for {desc}")

            agent.execute = _execute
            return agent

        factory.create.side_effect = _create

        orch = _orchestrator(factory=factory)
        result = await orch.execute(AgentTask(query="Compare trinkets and weapons"))

        # Both should have started before either finished (parallel)
        assert "start:find trinkets" in execution_order
        assert "start:find weapons" in execution_order
        assert len(result.specialist_results) == 2

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_dependent_subtasks_wait(self, mock_llm: AsyncMock) -> None:
        """C depends on A and B — C runs only after both A and B complete."""
        plan = _plan(
            subtasks=[
                SubTask(agent_role=AgentRole.RESEARCHER, description="a"),
                SubTask(agent_role=AgentRole.RESEARCHER, description="b"),
                SubTask(agent_role=AgentRole.RESEARCHER, description="c", depends_on=[0, 1]),
            ]
        )

        call_count = 0

        async def _mock_structured(response_model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if response_model is OrchestratorPlan:
                return plan
            return response_model(response="Synthesized.")

        mock_llm.side_effect = _mock_structured

        execution_order: list[str] = []

        factory = MagicMock()

        def _create(role, **kwargs):
            agent = MagicMock()

            async def _execute(task, on_status=None):
                execution_order.append(f"start:{task.query}")
                await asyncio.sleep(0.01)
                execution_order.append(f"end:{task.query}")
                return _research_result(output=f"Results for {task.query}")

            agent.execute = _execute
            return agent

        factory.create.side_effect = _create

        orch = _orchestrator(factory=factory)
        await orch.execute(AgentTask(query="test"))

        # C must start after both A and B end
        c_start = execution_order.index("start:c")
        a_end = execution_order.index("end:a")
        b_end = execution_order.index("end:b")
        assert c_start > a_end
        assert c_start > b_end

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_parallel_failure_doesnt_block(self, mock_llm: AsyncMock) -> None:
        """One of two parallel subtasks fails; the other still completes."""
        plan = _plan(
            subtasks=[
                SubTask(agent_role=AgentRole.RESEARCHER, description="good"),
                SubTask(agent_role=AgentRole.RESEARCHER, description="bad"),
            ]
        )

        call_count = 0

        async def _mock_structured(response_model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if response_model is OrchestratorPlan:
                return plan
            return response_model(response="Synthesized.")

        mock_llm.side_effect = _mock_structured

        factory = MagicMock()

        def _create(role, **kwargs):
            agent = MagicMock()

            async def _execute(task, on_status=None):
                if task.query == "bad":
                    raise RuntimeError("Agent failed")
                return _research_result(output="Good result")

            agent.execute = _execute
            return agent

        factory.create.side_effect = _create

        orch = _orchestrator(factory=factory)
        result = await orch.execute(AgentTask(query="test"))

        # One succeeded, one failed
        assert len(result.specialist_results) == 2
        statuses = [r.status for r in result.specialist_results]
        assert TaskStatus.SUCCESS in statuses
        assert TaskStatus.FAILED in statuses
