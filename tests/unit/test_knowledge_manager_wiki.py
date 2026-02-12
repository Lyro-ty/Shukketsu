"""Tests for KnowledgeManager wiki UI extensions."""

import sqlite3
from datetime import UTC, datetime

import pytest

from code.shukketsu.knowledge.manager import (
    ArticleMeta,
    ArticleStatus,
    ClaimRef,
    KnowledgeManager,
    SourceRef,
    Spec,
)


@pytest.fixture
def km(test_db: sqlite3.Connection, tmp_path) -> KnowledgeManager:
    """KnowledgeManager with test DB and temp knowledge dir."""
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    return KnowledgeManager(test_db, knowledge_dir)


def _sample_meta(**overrides) -> ArticleMeta:
    """Create a sample ArticleMeta for testing."""
    now = datetime.now(tz=UTC)
    defaults = dict(
        title="Phase 1 Trinkets",
        spec=Spec.COMBAT,
        category="gear",
        status=ArticleStatus.DRAFT,
        confidence=0.85,
        created_at=now,
        updated_at=now,
        sources=[SourceRef(url="https://example.com", trust=0.7)],
        claims=[ClaimRef(text="DST is BiS", verified=False, evidence=["chunk:42"])],
        entity_refs=["Dragonspine Trophy"],
        tags=["combat", "trinkets"],
    )
    defaults.update(overrides)
    return ArticleMeta(**defaults)


class TestArticleSummaryFields:
    def test_list_articles_includes_id(
        self, km: KnowledgeManager, test_db: sqlite3.Connection
    ) -> None:
        """ArticleSummary should include the integer id from the DB."""
        km.create_draft(_sample_meta(), "Content")
        articles = km.list_articles()
        assert len(articles) == 1
        assert isinstance(articles[0].id, int)
        assert articles[0].id > 0

    def test_list_articles_includes_claim_counts(
        self, km: KnowledgeManager, test_db: sqlite3.Connection
    ) -> None:
        """ArticleSummary should include verified_claims and unverified_claims."""
        path = km.create_draft(_sample_meta(), "Content")
        # Set claim counts directly in DB (Editor sets these during verification)
        test_db.execute(
            "UPDATE articles SET verified_claims = 3, unverified_claims = 1 WHERE path = ?",
            (path,),
        )
        test_db.commit()
        articles = km.list_articles()
        assert articles[0].verified_claims == 3
        assert articles[0].unverified_claims == 1

    def test_list_articles_claim_counts_default_zero(
        self, km: KnowledgeManager
    ) -> None:
        """Claim counts should default to 0 for new articles."""
        km.create_draft(_sample_meta(), "Content")
        articles = km.list_articles()
        assert articles[0].verified_claims == 0
        assert articles[0].unverified_claims == 0
