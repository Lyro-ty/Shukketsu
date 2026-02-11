"""Tests for Editor agent."""

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from code.shukketsu.agents.editor import (
    Editor,
    _match_entities,
    _update_claims,
    compute_article_confidence,
)
from code.shukketsu.agents.tasks import (
    AgentRole,
    AgentTask,
    ClaimJudgment,
    ClaimVerification,
    EditResult,
    EditTask,
    TaskStatus,
    VerificationStatus,
)
from code.shukketsu.knowledge.manager import (
    ArticleMeta,
    ArticleStatus,
    ClaimRef,
    KnowledgeManager,
    Spec,
)
from code.shukketsu.tools.registry import ToolRegistry
from code.shukketsu.tools.schemas import Tool


class MockTool(Tool):
    """Mock tool for testing. Tracks calls."""

    def __init__(self, tool_name: str, return_value: str = "mock result") -> None:
        self.name = tool_name
        self.description = f"Mock {tool_name}"
        self.parameters_schema: dict[str, Any] = {"query": {"type": "string", "description": "test"}}
        self._return_value = return_value
        self.call_count = 0
        self.call_args: list[dict[str, Any]] = []

    async def execute(self, tool_input: dict[str, Any]) -> str:
        self.call_count += 1
        self.call_args.append(tool_input)
        return self._return_value


@pytest.fixture
def knowledge_dir(tmp_path: Path) -> Path:
    """Create a temporary knowledge directory."""
    d = tmp_path / "knowledge"
    d.mkdir()
    return d


@pytest.fixture
def km(test_db: sqlite3.Connection, knowledge_dir: Path) -> KnowledgeManager:
    """Create a KnowledgeManager backed by test DB and tmp knowledge dir."""
    return KnowledgeManager(test_db, knowledge_dir)


@pytest.fixture
def editor(km: KnowledgeManager) -> Editor:
    """Create an Editor agent with a test KnowledgeManager."""
    return Editor(
        tool_registry=ToolRegistry(),
        role=AgentRole.EDITOR,
        knowledge_manager=km,
    )


def _create_draft_article(km: KnowledgeManager) -> tuple[str, ArticleMeta]:
    """Create a draft article for Editor tests. Returns (path, meta)."""
    from datetime import UTC, datetime

    now = datetime.now(tz=UTC)
    meta = ArticleMeta(
        title="Combat Trinkets Guide",
        spec=Spec.COMBAT,
        category="gear",
        status=ArticleStatus.DRAFT,
        confidence=0.5,
        created_at=now,
        updated_at=now,
        claims=[
            ClaimRef(text="DST drops from Gruul", verified=False),
            ClaimRef(text="Hit cap is 142 rating", verified=False),
        ],
        entity_refs=["Dragonspine Trophy", "Gruul", "Hit Rating"],
        tags=["combat", "gear"],
    )
    content = "# Combat Trinkets Guide\n\nDST drops from Gruul. Hit cap is 142 rating."
    path = km.create_draft(meta, content)
    return path, meta


def _edit_task(path: str, claims: list[str] | None = None) -> EditTask:
    """Build an EditTask with sensible defaults."""
    return EditTask(
        query="Verify combat trinkets claims",
        article_path=path,
        claims=claims or ["DST drops from Gruul", "Hit cap is 142 rating"],
    )


def _mock_judgment(
    status: VerificationStatus = VerificationStatus.VERIFIED,
    confidence: float = 0.9,
    **kwargs: object,
) -> ClaimJudgment:
    """Build a ClaimJudgment with defaults."""
    defaults: dict[str, object] = dict(
        status=status,
        confidence=confidence,
        supporting=["Source A confirms this"],
        contradicting=[],
        note="Evidence supports the claim",
    )
    defaults.update(kwargs)
    return ClaimJudgment(**defaults)


def _make_mock_judgment_fn(judgments: list[ClaimJudgment] | None = None):
    """Create a side_effect function that returns sequential judgments."""
    call_idx = 0
    default_judgment = _mock_judgment()

    async def _mock_fn(response_model, messages, **kwargs):
        nonlocal call_idx
        if judgments and call_idx < len(judgments):
            j = judgments[call_idx]
        else:
            j = default_judgment
        call_idx += 1
        return j

    return _mock_fn


# --------------------------------------------------------------------------- #
# Confidence scoring (pure function tests)
# --------------------------------------------------------------------------- #


class TestComputeArticleConfidence:
    def test_basic_average(self) -> None:
        """Confidence is the average of claim confidences."""
        claims = [
            ClaimVerification(claim="a", status=VerificationStatus.VERIFIED, confidence=0.8),
            ClaimVerification(claim="b", status=VerificationStatus.VERIFIED, confidence=0.6),
        ]
        assert compute_article_confidence(claims) == pytest.approx(0.7)

    def test_contradiction_penalty(self) -> None:
        """Each contradicted claim reduces score by 0.15."""
        claims = [
            ClaimVerification(claim="a", status=VerificationStatus.VERIFIED, confidence=0.8),
            ClaimVerification(claim="b", status=VerificationStatus.CONTRADICTED, confidence=0.1),
        ]
        # avg = 0.45, penalty = 0.15, result = 0.30
        assert compute_article_confidence(claims) == pytest.approx(0.30)

    def test_empty_claims_returns_zero(self) -> None:
        assert compute_article_confidence([]) == 0.0

    def test_clamps_to_zero(self) -> None:
        """Many contradictions don't go below 0.0."""
        claims = [
            ClaimVerification(claim=f"c{i}", status=VerificationStatus.CONTRADICTED, confidence=0.1) for i in range(10)
        ]
        result = compute_article_confidence(claims)
        assert result == 0.0

    def test_clamps_to_one(self) -> None:
        """High values don't exceed 1.0."""
        claims = [
            ClaimVerification(claim="a", status=VerificationStatus.VERIFIED, confidence=1.0),
        ]
        assert compute_article_confidence(claims) == 1.0

    def test_single_claim(self) -> None:
        claims = [
            ClaimVerification(claim="a", status=VerificationStatus.UNCERTAIN, confidence=0.5),
        ]
        assert compute_article_confidence(claims) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Entity matching (pure function tests)
# --------------------------------------------------------------------------- #


class TestMatchEntities:
    def test_exact_match(self) -> None:
        result = _match_entities("Dragonspine Trophy drops from Gruul", ["Dragonspine Trophy"])
        assert result == ["Dragonspine Trophy"]

    def test_case_insensitive(self) -> None:
        result = _match_entities("dragonspine trophy drops from gruul", ["Dragonspine Trophy"])
        assert result == ["Dragonspine Trophy"]

    def test_no_match(self) -> None:
        result = _match_entities("Hit cap is 142 rating", ["Dragonspine Trophy"])
        assert result == []

    def test_multiple_matches(self) -> None:
        result = _match_entities(
            "Dragonspine Trophy drops from Gruul",
            ["Dragonspine Trophy", "Gruul", "Tsunami Talisman"],
        )
        assert set(result) == {"Dragonspine Trophy", "Gruul"}


# --------------------------------------------------------------------------- #
# Frontmatter update (pure function tests)
# --------------------------------------------------------------------------- #


class TestUpdateClaims:
    def test_marks_verified(self) -> None:
        """VERIFIED claim sets claim_ref.verified = True."""
        from datetime import UTC, datetime

        now = datetime.now(tz=UTC)
        meta = ArticleMeta(
            title="Test",
            spec=Spec.COMBAT,
            category="gear",
            created_at=now,
            updated_at=now,
            claims=[ClaimRef(text="DST drops from Gruul", verified=False)],
        )
        results = [
            ClaimVerification(
                claim="DST drops from Gruul",
                status=VerificationStatus.VERIFIED,
                confidence=0.9,
                supporting_evidence=["Source A"],
            )
        ]
        _update_claims(meta, results)
        assert meta.claims[0].verified is True

    def test_marks_unverified(self) -> None:
        """CONTRADICTED claim sets claim_ref.verified = False."""
        from datetime import UTC, datetime

        now = datetime.now(tz=UTC)
        meta = ArticleMeta(
            title="Test",
            spec=Spec.COMBAT,
            category="gear",
            created_at=now,
            updated_at=now,
            claims=[ClaimRef(text="DST drops from Gruul", verified=True)],
        )
        results = [
            ClaimVerification(
                claim="DST drops from Gruul",
                status=VerificationStatus.CONTRADICTED,
                confidence=0.1,
                contradicting_evidence=["Source B says no"],
            )
        ]
        _update_claims(meta, results)
        assert meta.claims[0].verified is False

    def test_merges_evidence(self) -> None:
        """Evidence lists are combined and deduplicated."""
        from datetime import UTC, datetime

        now = datetime.now(tz=UTC)
        meta = ArticleMeta(
            title="Test",
            spec=Spec.COMBAT,
            category="gear",
            created_at=now,
            updated_at=now,
            claims=[ClaimRef(text="c1", verified=False, evidence=["existing"])],
        )
        results = [
            ClaimVerification(
                claim="c1",
                status=VerificationStatus.VERIFIED,
                confidence=0.8,
                supporting_evidence=["existing", "new_source"],
            )
        ]
        _update_claims(meta, results)
        assert meta.claims[0].evidence == ["existing", "new_source"]

    def test_unmatched_ignored(self) -> None:
        """Claims not in frontmatter are skipped gracefully."""
        from datetime import UTC, datetime

        now = datetime.now(tz=UTC)
        meta = ArticleMeta(
            title="Test",
            spec=Spec.COMBAT,
            category="gear",
            created_at=now,
            updated_at=now,
            claims=[ClaimRef(text="c1", verified=False)],
        )
        results = [ClaimVerification(claim="unknown claim", status=VerificationStatus.VERIFIED, confidence=0.9)]
        _update_claims(meta, results)
        assert meta.claims[0].verified is False  # unchanged


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


class TestEditorValidation:
    async def test_execute_requires_role(self, km: KnowledgeManager) -> None:
        """Editor without a role raises ValueError."""
        editor = Editor(tool_registry=ToolRegistry(), role=None, knowledge_manager=km)
        task = EditTask(query="verify", article_path="x.md", claims=["c1"])
        with pytest.raises(ValueError, match="role"):
            await editor.execute(task)

    async def test_execute_requires_edit_task(self, editor: Editor) -> None:
        """Non-EditTask input returns FAILED AgentResult."""
        task = AgentTask(query="Not an edit task")
        result = await editor.execute(task)
        assert result.status == TaskStatus.FAILED

    async def test_execute_empty_claims_returns_failed(self, editor: Editor, km: KnowledgeManager) -> None:
        """EditTask with empty claims returns FAILED EditResult."""
        path, _ = _create_draft_article(km)
        task = EditTask(query="verify", article_path=path, claims=[])
        result = await editor.execute(task)
        assert result.status == TaskStatus.FAILED
        assert isinstance(result, EditResult)
        assert result.overall_confidence == 0.0


# --------------------------------------------------------------------------- #
# Verification flow (mocked LLM + tools)
# --------------------------------------------------------------------------- #


class TestEditorVerificationFlow:
    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_calls_rag_search_per_claim(self, mock_llm: AsyncMock, editor: Editor, km: KnowledgeManager) -> None:
        """rag_search is called once per claim."""
        mock_llm.side_effect = _make_mock_judgment_fn()

        mock_rag = MockTool("rag_search", return_value="Some evidence")
        editor.tool_registry.register(mock_rag)

        path, _ = _create_draft_article(km)
        task = _edit_task(path)
        await editor.execute(task)

        assert mock_rag.call_count == 2  # one per claim

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_calls_graph_search_for_matched_entities(
        self, mock_llm: AsyncMock, editor: Editor, km: KnowledgeManager
    ) -> None:
        """graph_search is called for entities found in claim text."""
        mock_llm.side_effect = _make_mock_judgment_fn()

        mock_graph = MockTool("graph_search", return_value="Graph data")
        editor.tool_registry.register(mock_graph)

        path, _ = _create_draft_article(km)
        # "Dragonspine Trophy drops from Gruul" matches both "Dragonspine Trophy" and "Gruul" → 2 calls
        task = _edit_task(path, claims=["Dragonspine Trophy drops from Gruul"])
        await editor.execute(task)

        assert mock_graph.call_count == 2

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_skips_graph_search_no_entity_match(
        self, mock_llm: AsyncMock, editor: Editor, km: KnowledgeManager
    ) -> None:
        """No graph_search call when claim has no matching entities."""
        mock_llm.side_effect = _make_mock_judgment_fn()

        mock_graph = MockTool("graph_search", return_value="Graph data")
        editor.tool_registry.register(mock_graph)

        path, _ = _create_draft_article(km)
        # Claim about "expertise" doesn't match any entity_refs in the article
        task = _edit_task(path, claims=["Expertise soft cap is 23"])
        await editor.execute(task)

        assert mock_graph.call_count == 0

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_returns_edit_result(self, mock_llm: AsyncMock, editor: Editor, km: KnowledgeManager) -> None:
        """Successful execution returns an EditResult."""
        mock_llm.side_effect = _make_mock_judgment_fn()

        path, _ = _create_draft_article(km)
        result = await editor.execute(_edit_task(path))
        assert isinstance(result, EditResult)
        assert result.article_path == path
        assert len(result.claim_results) == 2

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_verified_claim(self, mock_llm: AsyncMock, editor: Editor, km: KnowledgeManager) -> None:
        """VERIFIED claim with supporting evidence."""
        judgment = _mock_judgment(VerificationStatus.VERIFIED, 0.9, supporting=["Source confirms"])
        mock_llm.side_effect = _make_mock_judgment_fn([judgment])

        path, _ = _create_draft_article(km)
        task = _edit_task(path, claims=["DST drops from Gruul"])
        result = await editor.execute(task)

        cv = result.claim_results[0]
        assert cv.status == VerificationStatus.VERIFIED
        assert cv.confidence == 0.9

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_contradicted_claim_populates_corrections(
        self, mock_llm: AsyncMock, editor: Editor, km: KnowledgeManager
    ) -> None:
        """CONTRADICTED claim populates corrections list."""
        judgment = _mock_judgment(
            VerificationStatus.CONTRADICTED,
            0.1,
            contradicting=["Source says DST drops from Netherspite"],
            note="Evidence conflicts",
        )
        mock_llm.side_effect = _make_mock_judgment_fn([judgment])

        path, _ = _create_draft_article(km)
        task = _edit_task(path, claims=["DST drops from Gruul"])
        result = await editor.execute(task)

        assert len(result.corrections) == 1
        assert "contradicted" in result.corrections[0].lower()

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_unsupported_claim_populates_needs_more_research(
        self, mock_llm: AsyncMock, editor: Editor, km: KnowledgeManager
    ) -> None:
        """UNSUPPORTED claim populates needs_more_research list."""
        judgment = _mock_judgment(VerificationStatus.UNSUPPORTED, 0.3, note="No evidence found")
        mock_llm.side_effect = _make_mock_judgment_fn([judgment])

        path, _ = _create_draft_article(km)
        task = _edit_task(path, claims=["Proc rate is 1 PPM"])
        result = await editor.execute(task)

        assert "Proc rate is 1 PPM" in result.needs_more_research

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_mixed_claims(self, mock_llm: AsyncMock, editor: Editor, km: KnowledgeManager) -> None:
        """Mix of VERIFIED, UNCERTAIN, CONTRADICTED, UNSUPPORTED."""
        judgments = [
            _mock_judgment(VerificationStatus.VERIFIED, 0.9),
            _mock_judgment(VerificationStatus.UNCERTAIN, 0.4),
            _mock_judgment(VerificationStatus.CONTRADICTED, 0.1, note="Wrong"),
            _mock_judgment(VerificationStatus.UNSUPPORTED, 0.3, note="No evidence"),
        ]
        mock_llm.side_effect = _make_mock_judgment_fn(judgments)

        path, _ = _create_draft_article(km)
        task = _edit_task(path, claims=["c1", "c2", "c3", "c4"])
        result = await editor.execute(task)

        statuses = [cv.status for cv in result.claim_results]
        assert VerificationStatus.VERIFIED in statuses
        assert VerificationStatus.UNCERTAIN in statuses
        assert VerificationStatus.CONTRADICTED in statuses
        assert VerificationStatus.UNSUPPORTED in statuses
        assert len(result.corrections) == 1
        assert len(result.needs_more_research) == 1


# --------------------------------------------------------------------------- #
# Confidence and approval
# --------------------------------------------------------------------------- #


class TestEditorApproval:
    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_above_threshold_approves(self, mock_llm: AsyncMock, editor: Editor, km: KnowledgeManager) -> None:
        """Confidence >= 0.6 sets approved_for_review=True."""
        judgment = _mock_judgment(VerificationStatus.VERIFIED, 0.8)
        mock_llm.side_effect = _make_mock_judgment_fn([judgment, judgment])

        path, _ = _create_draft_article(km)
        result = await editor.execute(_edit_task(path))
        assert result.approved_for_review is True

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_below_threshold_rejects(self, mock_llm: AsyncMock, editor: Editor, km: KnowledgeManager) -> None:
        """Confidence < 0.6 sets approved_for_review=False."""
        judgment = _mock_judgment(VerificationStatus.UNCERTAIN, 0.3)
        mock_llm.side_effect = _make_mock_judgment_fn([judgment, judgment])

        path, _ = _create_draft_article(km)
        result = await editor.execute(_edit_task(path))
        assert result.approved_for_review is False

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_updates_frontmatter_confidence(
        self, mock_llm: AsyncMock, editor: Editor, km: KnowledgeManager
    ) -> None:
        """Article meta.confidence is updated after verification."""
        judgment = _mock_judgment(VerificationStatus.VERIFIED, 0.85)
        mock_llm.side_effect = _make_mock_judgment_fn([judgment, judgment])

        path, _ = _create_draft_article(km)
        await editor.execute(_edit_task(path))

        meta, _ = km.read_article(path)
        assert meta.confidence == pytest.approx(0.85, abs=0.01)

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_transitions_to_review(
        self, mock_llm: AsyncMock, editor: Editor, km: KnowledgeManager, test_db: sqlite3.Connection
    ) -> None:
        """Article status changes from draft to review when approved."""
        judgment = _mock_judgment(VerificationStatus.VERIFIED, 0.9)
        mock_llm.side_effect = _make_mock_judgment_fn([judgment, judgment])

        path, _ = _create_draft_article(km)
        result = await editor.execute(_edit_task(path))

        assert result.approved_for_review is True
        row = test_db.execute("SELECT status FROM articles WHERE path = ?", (path,)).fetchone()
        assert row["status"] == ArticleStatus.REVIEW


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


class TestEditorErrors:
    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_llm_failure_marks_unsupported(
        self, mock_llm: AsyncMock, editor: Editor, km: KnowledgeManager
    ) -> None:
        """LLM error on one claim marks it as UNSUPPORTED, continues to next."""
        from code.shukketsu.resilience.errors import StructuredOutputError

        call_idx = 0

        async def _side_effect(response_model, messages, **kwargs):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                raise StructuredOutputError("LLM failed")
            return _mock_judgment()

        mock_llm.side_effect = _side_effect

        path, _ = _create_draft_article(km)
        result = await editor.execute(_edit_task(path))

        assert result.claim_results[0].status == VerificationStatus.UNSUPPORTED
        assert result.claim_results[1].status == VerificationStatus.VERIFIED
        assert result.status == TaskStatus.SUCCESS

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_all_failures_returns_partial(
        self, mock_llm: AsyncMock, editor: Editor, km: KnowledgeManager
    ) -> None:
        """All LLM calls fail -> status=PARTIAL."""
        from code.shukketsu.resilience.errors import StructuredOutputError

        mock_llm.side_effect = StructuredOutputError("LLM unavailable")

        path, _ = _create_draft_article(km)
        result = await editor.execute(_edit_task(path))

        assert result.status == TaskStatus.PARTIAL
        assert all(cv.status == VerificationStatus.UNSUPPORTED for cv in result.claim_results)

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_missing_tool_skips_search(self, mock_llm: AsyncMock, editor: Editor, km: KnowledgeManager) -> None:
        """Missing rag_search tool -> still verifies with whatever evidence is available."""
        mock_llm.side_effect = _make_mock_judgment_fn()
        # Empty registry — no tools available
        editor.tool_registry = ToolRegistry()

        path, _ = _create_draft_article(km)
        result = await editor.execute(_edit_task(path))

        assert isinstance(result, EditResult)
        assert result.status == TaskStatus.SUCCESS

    @patch("code.shukketsu.agents.editor.get_structured_output", new_callable=AsyncMock)
    async def test_non_draft_skips_update(
        self, mock_llm: AsyncMock, editor: Editor, km: KnowledgeManager, test_db: sqlite3.Connection
    ) -> None:
        """Article in review -> skip frontmatter update, still return results."""
        mock_llm.side_effect = _make_mock_judgment_fn([_mock_judgment(VerificationStatus.UNCERTAIN, 0.4)] * 2)

        path, _ = _create_draft_article(km)
        # Manually set article to review status
        test_db.execute("UPDATE articles SET status = 'review' WHERE path = ?", (path,))
        test_db.commit()

        result = await editor.execute(_edit_task(path))

        # Should still return results even if it couldn't update frontmatter
        assert isinstance(result, EditResult)
        assert len(result.claim_results) == 2


# --------------------------------------------------------------------------- #
# Factory integration
# --------------------------------------------------------------------------- #


class TestEditorFactory:
    def test_factory_creates_editor(self, km: KnowledgeManager) -> None:
        """AgentFactory creates an Editor instance for EDITOR role."""
        from code.shukketsu.agents.factory import AgentFactory

        factory = AgentFactory()
        agent = factory.create(AgentRole.EDITOR, knowledge_manager=km)
        assert isinstance(agent, Editor)

    def test_factory_editor_requires_knowledge_manager(self) -> None:
        """Editor via factory without knowledge_manager raises TypeError."""
        from code.shukketsu.agents.factory import AgentFactory

        factory = AgentFactory()
        with pytest.raises(TypeError):
            factory.create(AgentRole.EDITOR)

    def test_factory_editor_in_role_classes(self) -> None:
        """Editor is registered in _ROLE_CLASSES dict."""
        from code.shukketsu.agents.factory import _ROLE_CLASSES

        assert AgentRole.EDITOR in _ROLE_CLASSES
