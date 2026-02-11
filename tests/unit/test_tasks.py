"""Tests for the Structured Task Protocol models."""

import uuid

import pytest
from pydantic import ValidationError

from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    ArticleType,
    EditTask,
    OrchestratorPlan,
    ResearchTask,
    SearchStrategy,
    SubTask,
    TaskStatus,
    WriteTask,
)


class TestAgentRole:
    def test_all_roles_exist(self) -> None:
        assert AgentRole.RESEARCHER == "researcher"
        assert AgentRole.WRITER == "writer"
        assert AgentRole.EDITOR == "editor"
        assert AgentRole.ORCHESTRATOR == "orchestrator"

    def test_role_is_str(self) -> None:
        assert isinstance(AgentRole.RESEARCHER, str)


class TestTaskStatus:
    def test_all_statuses_exist(self) -> None:
        assert TaskStatus.SUCCESS == "success"
        assert TaskStatus.PARTIAL == "partial"
        assert TaskStatus.FAILED == "failed"


class TestAgentTask:
    def test_minimal_creation(self) -> None:
        task = AgentTask(query="What is the hit cap?")
        assert task.query == "What is the hit cap?"
        assert task.context == {}

    def test_auto_generates_task_id(self) -> None:
        task = AgentTask(query="test")
        uuid.UUID(task.task_id)  # raises if not valid UUID

    def test_auto_generates_trace_id(self) -> None:
        task = AgentTask(query="test")
        uuid.UUID(task.trace_id)  # raises if not valid UUID

    def test_unique_ids_per_instance(self) -> None:
        t1 = AgentTask(query="a")
        t2 = AgentTask(query="b")
        assert t1.task_id != t2.task_id
        assert t1.trace_id != t2.trace_id

    def test_explicit_ids(self) -> None:
        task = AgentTask(task_id="my-id", trace_id="my-trace", query="q")
        assert task.task_id == "my-id"
        assert task.trace_id == "my-trace"

    def test_context_dict(self) -> None:
        task = AgentTask(query="q", context={"phase": 1, "spec": "combat"})
        assert task.context["phase"] == 1

    def test_rejects_missing_query(self) -> None:
        with pytest.raises(ValidationError):
            AgentTask()  # type: ignore[call-arg]


class TestAgentResult:
    def test_minimal_creation(self) -> None:
        result = AgentResult(
            task_id="t1",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="The hit cap is 142 rating.",
        )
        assert result.output == "The hit cap is 142 rating."
        assert result.evidence == []
        assert result.metadata == {}

    def test_with_evidence(self) -> None:
        result = AgentResult(
            task_id="t1",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="answer",
            evidence=["chunk:42", "https://example.com"],
        )
        assert len(result.evidence) == 2

    def test_with_metadata(self) -> None:
        result = AgentResult(
            task_id="t1",
            agent_role=AgentRole.WRITER,
            status=TaskStatus.PARTIAL,
            output="draft",
            metadata={"word_count": 500},
        )
        assert result.metadata["word_count"] == 500

    def test_rejects_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            AgentResult(task_id="t1")  # type: ignore[call-arg]

    def test_all_statuses_valid(self) -> None:
        for status in TaskStatus:
            r = AgentResult(
                task_id="t",
                agent_role=AgentRole.EDITOR,
                status=status,
                output="x",
            )
            assert r.status == status


class TestSearchStrategy:
    def test_all_strategies(self) -> None:
        assert SearchStrategy.HYBRID == "hybrid"
        assert SearchStrategy.GRAPH == "graph"
        assert SearchStrategy.WEB == "web"
        assert SearchStrategy.AUTO == "auto"


class TestArticleType:
    def test_all_types(self) -> None:
        assert ArticleType.GUIDE == "guide"
        assert ArticleType.REFERENCE == "reference"
        assert ArticleType.ANALYSIS == "analysis"


class TestResearchTask:
    def test_inherits_agent_task(self) -> None:
        task = ResearchTask(query="What drops from Gruul?")
        assert isinstance(task, AgentTask)
        uuid.UUID(task.task_id)

    def test_defaults(self) -> None:
        task = ResearchTask(query="q")
        assert task.search_strategy is None
        assert task.max_sources == 10

    def test_explicit_strategy(self) -> None:
        task = ResearchTask(query="q", search_strategy=SearchStrategy.GRAPH)
        assert task.search_strategy == SearchStrategy.GRAPH

    def test_custom_max_sources(self) -> None:
        task = ResearchTask(query="q", max_sources=5)
        assert task.max_sources == 5


class TestWriteTask:
    def test_requires_research_result(self) -> None:
        research = AgentResult(
            task_id="r1",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="findings",
        )
        task = WriteTask(query="Write a gear guide", research=research, article_type=ArticleType.GUIDE)
        assert task.research.task_id == "r1"
        assert task.article_type == ArticleType.GUIDE

    def test_rejects_missing_research(self) -> None:
        with pytest.raises(ValidationError):
            WriteTask(query="Write", article_type=ArticleType.GUIDE)  # type: ignore[call-arg]

    def test_rejects_missing_article_type(self) -> None:
        research = AgentResult(task_id="r1", agent_role=AgentRole.RESEARCHER, status=TaskStatus.SUCCESS, output="x")
        with pytest.raises(ValidationError):
            WriteTask(query="Write", research=research)  # type: ignore[call-arg]


class TestEditTask:
    def test_creation(self) -> None:
        task = EditTask(
            query="Verify claims",
            article_path="knowledge/combat/gear/trinkets.md",
            claims=["DST drops from Gruul", "Hit cap is 142"],
        )
        assert task.article_path == "knowledge/combat/gear/trinkets.md"
        assert len(task.claims) == 2

    def test_inherits_agent_task(self) -> None:
        task = EditTask(query="q", article_path="p", claims=[])
        assert isinstance(task, AgentTask)

    def test_rejects_missing_fields(self) -> None:
        with pytest.raises(ValidationError):
            EditTask(query="q")  # type: ignore[call-arg]


class TestSubTask:
    def test_creation(self) -> None:
        st = SubTask(
            agent_role=AgentRole.RESEARCHER,
            description="Find Phase 1 trinkets",
        )
        assert st.depends_on == []
        assert st.task_params == {}

    def test_with_dependencies(self) -> None:
        st = SubTask(
            agent_role=AgentRole.WRITER,
            description="Write article",
            depends_on=[0, 1],
        )
        assert st.depends_on == [0, 1]


class TestOrchestratorPlan:
    def test_creation(self) -> None:
        plan = OrchestratorPlan(
            reasoning="Need research then writing",
            subtasks=[
                SubTask(agent_role=AgentRole.RESEARCHER, description="research"),
                SubTask(agent_role=AgentRole.WRITER, description="write", depends_on=[0]),
            ],
        )
        assert len(plan.subtasks) == 2
        assert plan.can_answer_directly is False
        assert plan.direct_answer is None

    def test_direct_answer(self) -> None:
        plan = OrchestratorPlan(
            reasoning="Simple question",
            subtasks=[],
            can_answer_directly=True,
            direct_answer="Sinister Strike costs 40 energy.",
        )
        assert plan.can_answer_directly is True
        assert plan.direct_answer is not None
