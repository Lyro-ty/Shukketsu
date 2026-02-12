"""Tests for the Structured Task Protocol models."""

import uuid

import pytest
from pydantic import ValidationError

from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    ArticleType,
    ClaimJudgment,
    ClaimVerification,
    EditResult,
    EditTask,
    Finding,
    OrchestratorPlan,
    OrchestratorResult,
    ResearchResult,
    ResearchTask,
    SearchStrategy,
    SubTask,
    TaskStatus,
    VerificationStatus,
    WriteResult,
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
        research = ResearchResult(
            task_id="r1",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="findings",
            findings=[Finding(claim="Hit cap is 142", confidence=0.9)],
            sufficient=True,
        )
        task = WriteTask(
            query="Write a gear guide",
            research=research,
            article_type=ArticleType.GUIDE,
            spec="combat",
            category="gear",
        )
        assert task.research.task_id == "r1"
        assert task.article_type == ArticleType.GUIDE
        assert task.spec == "combat"
        assert task.category == "gear"

    def test_rejects_plain_agent_result(self) -> None:
        """WriteTask.research no longer accepts plain AgentResult."""
        research = AgentResult(
            task_id="r1",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="findings",
        )
        with pytest.raises(ValidationError):
            WriteTask(
                query="Write",
                research=research,  # type: ignore[arg-type]
                article_type=ArticleType.GUIDE,
                spec="combat",
                category="gear",
            )

    def test_rejects_missing_research(self) -> None:
        with pytest.raises(ValidationError):
            WriteTask(query="Write", article_type=ArticleType.GUIDE, spec="combat", category="gear")  # type: ignore[call-arg]

    def test_rejects_missing_article_type(self) -> None:
        research = ResearchResult(task_id="r1", agent_role=AgentRole.RESEARCHER, status=TaskStatus.SUCCESS, output="x")
        with pytest.raises(ValidationError):
            WriteTask(query="Write", research=research, spec="combat", category="gear")  # type: ignore[call-arg]

    def test_rejects_missing_spec(self) -> None:
        research = ResearchResult(task_id="r1", agent_role=AgentRole.RESEARCHER, status=TaskStatus.SUCCESS, output="x")
        with pytest.raises(ValidationError):
            WriteTask(query="Write", research=research, article_type=ArticleType.GUIDE, category="gear")  # type: ignore[call-arg]

    def test_rejects_missing_category(self) -> None:
        research = ResearchResult(task_id="r1", agent_role=AgentRole.RESEARCHER, status=TaskStatus.SUCCESS, output="x")
        with pytest.raises(ValidationError):
            WriteTask(query="Write", research=research, article_type=ArticleType.GUIDE, spec="combat")  # type: ignore[call-arg]


class TestWriteResult:
    def test_creation(self) -> None:
        result = WriteResult(
            task_id="w1",
            agent_role=AgentRole.WRITER,
            status=TaskStatus.SUCCESS,
            output="Article written",
            article_path="combat/gear/trinkets.md",
            title="Combat Trinkets",
        )
        assert result.article_path == "combat/gear/trinkets.md"
        assert result.claims == []
        assert result.research_gaps == []
        assert result.word_count == 0
        assert result.entity_refs == []

    def test_with_all_fields(self) -> None:
        result = WriteResult(
            task_id="w1",
            agent_role=AgentRole.WRITER,
            status=TaskStatus.SUCCESS,
            output="Article written",
            article_path="combat/gear/trinkets.md",
            title="Combat Trinkets",
            claims=["DST is BiS"],
            research_gaps=["proc rates"],
            word_count=500,
            entity_refs=["Dragonspine Trophy"],
        )
        assert result.claims == ["DST is BiS"]
        assert result.word_count == 500


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


class TestVerificationStatus:
    def test_all_values_exist(self) -> None:
        assert VerificationStatus.VERIFIED == "verified"
        assert VerificationStatus.UNCERTAIN == "uncertain"
        assert VerificationStatus.CONTRADICTED == "contradicted"
        assert VerificationStatus.UNSUPPORTED == "unsupported"

    def test_is_str_enum(self) -> None:
        assert isinstance(VerificationStatus.VERIFIED, str)


class TestClaimVerification:
    def test_valid_creation(self) -> None:
        cv = ClaimVerification(
            claim="DST drops from Gruul",
            status=VerificationStatus.VERIFIED,
            supporting_evidence=["Source A"],
            confidence=0.9,
            note="Confirmed",
        )
        assert cv.claim == "DST drops from Gruul"
        assert cv.status == VerificationStatus.VERIFIED

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ClaimVerification(claim="c", status=VerificationStatus.VERIFIED, confidence=1.5)
        with pytest.raises(ValidationError):
            ClaimVerification(claim="c", status=VerificationStatus.VERIFIED, confidence=-0.1)


class TestClaimJudgment:
    def test_valid_creation(self) -> None:
        j = ClaimJudgment(
            status=VerificationStatus.UNCERTAIN,
            confidence=0.5,
            supporting=["Partial match"],
            note="Not conclusive",
        )
        assert j.status == VerificationStatus.UNCERTAIN
        assert j.confidence == 0.5


class TestEditResult:
    def test_valid_creation(self) -> None:
        result = EditResult(
            task_id="e1",
            agent_role=AgentRole.EDITOR,
            status=TaskStatus.SUCCESS,
            output="Verified 3 claims",
            article_path="combat/gear/trinkets.md",
            claim_results=[],
            overall_confidence=0.75,
            approved_for_review=True,
        )
        assert result.article_path == "combat/gear/trinkets.md"
        assert result.approved_for_review is True

    def test_defaults(self) -> None:
        result = EditResult(
            task_id="e1",
            agent_role=AgentRole.EDITOR,
            status=TaskStatus.SUCCESS,
            output="Done",
            article_path="p.md",
        )
        assert result.claim_results == []
        assert result.overall_confidence == 0.0
        assert result.approved_for_review is False
        assert result.internal_consistency is True
        assert result.corrections == []
        assert result.needs_more_research == []


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


class TestOrchestratorResult:
    def test_validates_with_all_fields(self) -> None:
        """OrchestratorResult round-trips with all fields populated."""
        plan = OrchestratorPlan(
            reasoning="Simple research",
            subtasks=[SubTask(agent_role=AgentRole.RESEARCHER, description="Find info")],
        )
        result = OrchestratorResult(
            task_id="t1",
            agent_role=AgentRole.ORCHESTRATOR,
            status=TaskStatus.SUCCESS,
            output="Here are the findings.",
            plan=plan,
            specialist_results=[
                AgentResult(
                    task_id="r1",
                    agent_role=AgentRole.RESEARCHER,
                    status=TaskStatus.SUCCESS,
                    output="Found it.",
                ),
            ],
            article_path="combat/gear/trinkets.md",
            needs_human_review=True,
            skipped_tasks=["skipped task"],
        )
        assert result.plan.reasoning == "Simple research"
        assert len(result.specialist_results) == 1
        assert result.article_path == "combat/gear/trinkets.md"
        assert result.needs_human_review is True
        assert result.skipped_tasks == ["skipped task"]

    def test_defaults(self) -> None:
        """Default values for optional fields."""
        plan = OrchestratorPlan(reasoning="test", subtasks=[])
        result = OrchestratorResult(
            task_id="t1",
            agent_role=AgentRole.ORCHESTRATOR,
            status=TaskStatus.SUCCESS,
            output="answer",
            plan=plan,
        )
        assert result.specialist_results == []
        assert result.article_path is None
        assert result.needs_human_review is False
        assert result.skipped_tasks == []

    def test_with_skipped_tasks(self) -> None:
        """skipped_tasks stores descriptions of tasks skipped due to failed deps."""
        plan = OrchestratorPlan(reasoning="test", subtasks=[])
        result = OrchestratorResult(
            task_id="t1",
            agent_role=AgentRole.ORCHESTRATOR,
            status=TaskStatus.PARTIAL,
            output="partial",
            plan=plan,
            skipped_tasks=["Write article", "Verify claims"],
        )
        assert len(result.skipped_tasks) == 2

    def test_inherits_agent_result(self) -> None:
        """OrchestratorResult has all AgentResult fields."""
        plan = OrchestratorPlan(reasoning="test", subtasks=[])
        result = OrchestratorResult(
            task_id="t1",
            agent_role=AgentRole.ORCHESTRATOR,
            status=TaskStatus.SUCCESS,
            output="answer",
            evidence=["src1", "src2"],
            plan=plan,
        )
        assert isinstance(result, AgentResult)
        assert result.evidence == ["src1", "src2"]
        assert result.metadata == {}
