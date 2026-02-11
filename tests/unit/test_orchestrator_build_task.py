"""Tests for Orchestrator._build_task and _merge_research."""

import pytest

from code.shukketsu.agents.tasks import (
    AgentRole,
    AgentTask,
    ArticleType,
    EditTask,
    Finding,
    ResearchResult,
    ResearchTask,
    SubTask,
    TaskStatus,
    WriteResult,
    WriteTask,
)


def _sub(
    role: AgentRole = AgentRole.RESEARCHER,
    desc: str = "test",
    depends_on: list[int] | None = None,
    task_params: dict | None = None,
) -> SubTask:
    """Build a SubTask with defaults."""
    return SubTask(
        agent_role=role,
        description=desc,
        depends_on=depends_on or [],
        task_params=task_params or {},
    )


def _research_result(**overrides: object) -> ResearchResult:
    """Build a ResearchResult with defaults."""
    defaults: dict[str, object] = dict(
        task_id="r1",
        agent_role=AgentRole.RESEARCHER,
        status=TaskStatus.SUCCESS,
        output="Research done",
        findings=[Finding(claim="Hit cap is 142", confidence=0.9, evidence=["src1"])],
        sources_used=["src1"],
        strategies_used=["rag_search"],
        gaps=[],
        sufficient=True,
    )
    defaults.update(overrides)
    return ResearchResult(**defaults)


def _write_result(**overrides: object) -> WriteResult:
    """Build a WriteResult with defaults."""
    defaults: dict[str, object] = dict(
        task_id="w1",
        agent_role=AgentRole.WRITER,
        status=TaskStatus.SUCCESS,
        output="Article written",
        article_path="combat/gear/trinkets.md",
        title="Trinkets Guide",
        claims=["Hit cap is 142", "DST is BiS"],
    )
    defaults.update(overrides)
    return WriteResult(**defaults)


def _build(subtask: SubTask, results: list, trace_id: str = "trace-1") -> AgentTask:
    """Call Orchestrator._build_task on a minimal instance."""
    from code.shukketsu.agents.orchestrator import Orchestrator

    orch = object.__new__(Orchestrator)
    return orch._build_task(subtask, results, trace_id)


class TestBuildResearchTask:
    def test_researcher_creates_research_task(self) -> None:
        """RESEARCHER subtask builds a ResearchTask with description as query."""
        task = _build(_sub(AgentRole.RESEARCHER, "Find trinkets"), [])
        assert isinstance(task, ResearchTask)
        assert task.query == "Find trinkets"

    def test_propagates_trace_id(self) -> None:
        """trace_id flows from parent task to built task."""
        task = _build(_sub(AgentRole.RESEARCHER, "test"), [], trace_id="my-trace")
        assert task.trace_id == "my-trace"


class TestBuildWriteTask:
    def test_pulls_research_from_dependency(self) -> None:
        """WRITER subtask extracts ResearchResult from completed dependency."""
        results = [_research_result(), None]
        subtask = _sub(
            AgentRole.WRITER,
            "write guide",
            depends_on=[0],
            task_params={"spec": "combat", "category": "gear"},
        )
        task = _build(subtask, results)
        assert isinstance(task, WriteTask)
        assert task.research.task_id == "r1"

    def test_reads_task_params(self) -> None:
        """spec, category, article_type read from task_params."""
        results = [_research_result()]
        subtask = _sub(
            AgentRole.WRITER,
            "write",
            depends_on=[0],
            task_params={"spec": "assassination", "category": "rotation", "article_type": "analysis"},
        )
        task = _build(subtask, results)
        assert isinstance(task, WriteTask)
        assert task.spec == "assassination"
        assert task.category == "rotation"
        assert task.article_type == ArticleType.ANALYSIS

    def test_defaults_on_missing_params(self) -> None:
        """Missing task_params fall back to general/general/GUIDE."""
        results = [_research_result()]
        subtask = _sub(AgentRole.WRITER, "write", depends_on=[0])
        task = _build(subtask, results)
        assert isinstance(task, WriteTask)
        assert task.spec == "general"
        assert task.category == "general"
        assert task.article_type == ArticleType.GUIDE

    def test_merges_multiple_research(self) -> None:
        """Two Researcher deps are merged into one ResearchResult."""
        r1 = _research_result(
            task_id="r1",
            findings=[Finding(claim="Claim A", confidence=0.9, evidence=["s1"])],
            sources_used=["s1"],
            strategies_used=["rag_search"],
            gaps=["gap1"],
        )
        r2 = _research_result(
            task_id="r2",
            output="Second research",
            findings=[Finding(claim="Claim B", confidence=0.8, evidence=["s2"])],
            sources_used=["s2"],
            strategies_used=["graph_search"],
            gaps=["gap2"],
        )
        results = [r1, r2]
        subtask = _sub(AgentRole.WRITER, "write", depends_on=[0, 1])
        task = _build(subtask, results)
        assert isinstance(task, WriteTask)
        assert len(task.research.findings) == 2
        assert set(task.research.sources_used) == {"s1", "s2"}
        assert set(task.research.strategies_used) == {"rag_search", "graph_search"}
        assert set(task.research.gaps) == {"gap1", "gap2"}

    def test_no_research_dependency_raises(self) -> None:
        """WRITER with no ResearchResult dependency raises ValueError."""
        subtask = _sub(AgentRole.WRITER, "write", depends_on=[0])
        results: list = [None]
        with pytest.raises(ValueError, match="no ResearchResult"):
            _build(subtask, results)


class TestBuildEditTask:
    def test_pulls_write_result(self) -> None:
        """EDITOR subtask extracts article_path and claims from WriteResult."""
        results = [None, _write_result()]
        subtask = _sub(AgentRole.EDITOR, "verify", depends_on=[1])
        task = _build(subtask, results)
        assert isinstance(task, EditTask)
        assert task.article_path == "combat/gear/trinkets.md"
        assert task.claims == ["Hit cap is 142", "DST is BiS"]

    def test_no_write_dependency_raises(self) -> None:
        """EDITOR with no WriteResult dependency raises ValueError."""
        subtask = _sub(AgentRole.EDITOR, "verify", depends_on=[0])
        results: list = [_research_result()]
        with pytest.raises(ValueError, match="no WriteResult"):
            _build(subtask, results)
