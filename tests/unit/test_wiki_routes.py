"""Tests for wiki routes."""

import sqlite3
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from code.shukketsu.knowledge.manager import (
    ArticleMeta,
    ArticleStatus,
    ClaimRef,
    KnowledgeManager,
    SourceRef,
    Spec,
)


def _get_app():
    from code.shukketsu.web.app import app

    return app


@pytest.fixture
def wiki_db(tmp_path) -> sqlite3.Connection:
    """SQLite connection that allows cross-thread access for TestClient.

    TestClient runs async handlers in a separate thread, so the default
    check_same_thread=True would raise ProgrammingError. This fixture
    creates a fresh DB with check_same_thread=False for route testing.
    """
    import sqlite_vec

    from code.shukketsu.db.connection import init_db

    db_path = tmp_path / "wiki_test.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def km(wiki_db: sqlite3.Connection, tmp_path) -> KnowledgeManager:
    """KnowledgeManager with cross-thread-safe DB and temp knowledge dir."""
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    return KnowledgeManager(wiki_db, knowledge_dir)


def _sample_meta(**overrides) -> ArticleMeta:
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
        claims=[ClaimRef(text="DST is BiS", verified=True, evidence=["chunk:42"])],
        entity_refs=["Dragonspine Trophy"],
        tags=["combat", "trinkets"],
    )
    defaults.update(overrides)
    return ArticleMeta(**defaults)


@pytest.fixture
def client(km: KnowledgeManager):
    """TestClient with KnowledgeManager dependency overridden."""
    from code.shukketsu.web.routers.wiki import _get_km

    app = _get_app()
    app.dependency_overrides[_get_km] = lambda: km
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestBrowser:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/wiki/")
        assert resp.status_code == 200
        assert "Wiki Articles" in resp.text

    def test_empty_state(self, client: TestClient) -> None:
        resp = client.get("/wiki/")
        assert "No articles found" in resp.text

    def test_lists_articles(self, client: TestClient, km: KnowledgeManager) -> None:
        km.create_draft(_sample_meta(), "# Test\n\nBody")
        resp = client.get("/wiki/")
        assert "Phase 1 Trinkets" in resp.text


class TestReview:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/wiki/review")
        assert resp.status_code == 200
        assert "Pending Review" in resp.text

    def test_empty_state(self, client: TestClient) -> None:
        resp = client.get("/wiki/review")
        assert "No articles pending review" in resp.text


class TestArticleDetail:
    def test_returns_200(self, client: TestClient, km: KnowledgeManager) -> None:
        path = km.create_draft(_sample_meta(), "# Trinkets\n\n**Bold text**")
        view_path = path.replace(".md", "")
        resp = client.get(f"/wiki/view/{view_path}")
        assert resp.status_code == 200

    def test_not_found_returns_404(self, client: TestClient) -> None:
        resp = client.get("/wiki/view/nonexistent/path")
        assert resp.status_code == 404

    def test_renders_markdown(self, client: TestClient, km: KnowledgeManager) -> None:
        path = km.create_draft(_sample_meta(), "# Title\n\n**Bold text**")
        view_path = path.replace(".md", "")
        resp = client.get(f"/wiki/view/{view_path}")
        assert "<strong>Bold text</strong>" in resp.text

    def test_shows_sidebar_data(self, client: TestClient, km: KnowledgeManager) -> None:
        path = km.create_draft(_sample_meta(), "# Test\n\nBody")
        view_path = path.replace(".md", "")
        resp = client.get(f"/wiki/view/{view_path}")
        assert "85%" in resp.text  # confidence
        assert "draft" in resp.text  # status
        assert "example.com" in resp.text  # source URL


class TestApprove:
    def test_approve_review_article(
        self, client: TestClient, km: KnowledgeManager, wiki_db: sqlite3.Connection
    ) -> None:
        path = km.create_draft(_sample_meta(), "Content")
        km.set_status(path, ArticleStatus.REVIEW)
        article_id = wiki_db.execute("SELECT id FROM articles WHERE path = ?", (path,)).fetchone()["id"]
        resp = client.post(f"/wiki/approve/{article_id}")
        assert resp.status_code == 200
        row = wiki_db.execute("SELECT status FROM articles WHERE path = ?", (path,)).fetchone()
        assert row["status"] == "published"

    def test_approve_draft_returns_400(
        self, client: TestClient, km: KnowledgeManager, wiki_db: sqlite3.Connection
    ) -> None:
        path = km.create_draft(_sample_meta(), "Content")
        article_id = wiki_db.execute("SELECT id FROM articles WHERE path = ?", (path,)).fetchone()["id"]
        resp = client.post(f"/wiki/approve/{article_id}")
        assert resp.status_code == 400


class TestReject:
    def test_reject_review_article(self, client: TestClient, km: KnowledgeManager, wiki_db: sqlite3.Connection) -> None:
        path = km.create_draft(_sample_meta(), "Content")
        km.set_status(path, ArticleStatus.REVIEW)
        article_id = wiki_db.execute("SELECT id FROM articles WHERE path = ?", (path,)).fetchone()["id"]
        resp = client.post(f"/wiki/reject/{article_id}", data={"reason": "Needs work"})
        assert resp.status_code == 200
        row = wiki_db.execute("SELECT status FROM articles WHERE path = ?", (path,)).fetchone()
        assert row["status"] == "draft"

    def test_reject_stores_reason(self, client: TestClient, km: KnowledgeManager, wiki_db: sqlite3.Connection) -> None:
        path = km.create_draft(_sample_meta(), "Content")
        km.set_status(path, ArticleStatus.REVIEW)
        article_id = wiki_db.execute("SELECT id FROM articles WHERE path = ?", (path,)).fetchone()["id"]
        client.post(f"/wiki/reject/{article_id}", data={"reason": "Bad sources"})
        meta, _ = km.read_article(path)
        assert meta.rejection_reason == "Bad sources"


class TestHtmxFragments:
    def test_articles_returns_fragment(self, client: TestClient) -> None:
        """Fragment should NOT contain base template elements like <html>."""
        resp = client.get("/wiki/htmx/articles")
        assert resp.status_code == 200
        assert "<html" not in resp.text
        assert "No articles found" in resp.text

    def test_articles_filters_by_spec(self, client: TestClient, km: KnowledgeManager) -> None:
        km.create_draft(_sample_meta(title="Combat Guide", spec=Spec.COMBAT), "A")
        km.create_draft(_sample_meta(title="Mut Guide", spec=Spec.ASSASSINATION), "B")
        resp = client.get("/wiki/htmx/articles?spec=combat")
        assert "Combat Guide" in resp.text
        assert "Mut Guide" not in resp.text

    def test_articles_filters_by_title_search(self, client: TestClient, km: KnowledgeManager) -> None:
        km.create_draft(_sample_meta(title="Phase 1 Trinkets"), "A")
        km.create_draft(_sample_meta(title="Rotation Guide"), "B")
        resp = client.get("/wiki/htmx/articles?q=trinket")
        assert "Phase 1 Trinkets" in resp.text
        assert "Rotation Guide" not in resp.text
