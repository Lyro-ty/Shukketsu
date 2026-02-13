"""Tests for KnowledgeManager article CRUD."""

import sqlite3
import time
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from code.shukketsu.knowledge.manager import (
    ArticleMeta,
    ArticleStatus,
    ClaimRef,
    KnowledgeManager,
    SourceRef,
    Spec,
    derive_path,
    slugify,
)


class TestSlugify:
    def test_basic(self) -> None:
        assert slugify("Phase 1 BiS Gear Guide") == "phase-1-bis-gear-guide"

    def test_special_characters(self) -> None:
        assert slugify("Combat Swords: Rotation Priority") == "combat-swords-rotation-priority"
        assert slugify("What's BiS? (Phase 1)") == "whats-bis-phase-1"

    def test_multiple_hyphens_collapsed(self) -> None:
        assert slugify("foo---bar") == "foo-bar"
        assert slugify("hello   world") == "hello-world"

    def test_leading_trailing_stripped(self) -> None:
        assert slugify("--hello--") == "hello"
        assert slugify("  spaced  ") == "spaced"


class TestDerivePath:
    def test_basic(self) -> None:
        assert derive_path("combat", "gear", "Phase 1 BiS Gear Guide") == "combat/gear/phase-1-bis-gear-guide.md"

    def test_with_special_title(self) -> None:
        assert (
            derive_path("assassination", "rotation", "Mutilate: Priority List")
            == "assassination/rotation/mutilate-priority-list.md"
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


class TestCreateDraft:
    def test_writes_file(self, km: KnowledgeManager, tmp_path) -> None:
        meta = _sample_meta()
        path = km.create_draft(meta, "# Article\n\nBody text.")
        full_path = tmp_path / "knowledge" / path
        assert full_path.exists()
        assert "Body text." in full_path.read_text()

    def test_inserts_db_row(self, km: KnowledgeManager, test_db: sqlite3.Connection) -> None:
        meta = _sample_meta()
        path = km.create_draft(meta, "Content")
        row = test_db.execute("SELECT * FROM articles WHERE path = ?", (path,)).fetchone()
        assert row is not None
        assert row["title"] == "Phase 1 Trinkets"
        assert row["spec"] == "combat"
        assert row["category"] == "gear"
        assert row["status"] == "draft"

    def test_frontmatter_roundtrip(self, km: KnowledgeManager) -> None:
        meta = _sample_meta()
        path = km.create_draft(meta, "# Test\n\nContent here.")
        read_meta, read_content = km.read_article(path)
        assert read_meta.title == meta.title
        assert read_meta.spec == Spec.COMBAT
        assert read_meta.confidence == 0.85
        assert "Content here." in read_content

    def test_creates_parent_dirs(self, km: KnowledgeManager, tmp_path) -> None:
        meta = _sample_meta(spec=Spec.ASSASSINATION, category="rotation")
        path = km.create_draft(meta, "Content")
        full_path = tmp_path / "knowledge" / path
        assert full_path.exists()
        assert full_path.parent.name == "rotation"

    def test_duplicate_raises(self, km: KnowledgeManager) -> None:
        meta = _sample_meta()
        km.create_draft(meta, "First version")
        with pytest.raises(ValueError, match="already exists"):
            km.create_draft(meta, "Second version")

    def test_datetime_serialization(self, km: KnowledgeManager, tmp_path) -> None:
        """Datetimes should be ISO 8601 strings in YAML, not !!python/object."""
        meta = _sample_meta()
        path = km.create_draft(meta, "Content")
        raw_text = (tmp_path / "knowledge" / path).read_text()
        assert "!!python" not in raw_text
        assert "created_at:" in raw_text


class TestReadArticle:
    def test_parses_frontmatter(self, km: KnowledgeManager) -> None:
        meta = _sample_meta()
        path = km.create_draft(meta, "# Article\n\nBody.")
        read_meta, read_content = km.read_article(path)
        assert read_meta.title == "Phase 1 Trinkets"
        assert read_meta.status == ArticleStatus.DRAFT
        assert "Body." in read_content

    def test_missing_file_raises(self, km: KnowledgeManager) -> None:
        with pytest.raises(FileNotFoundError):
            km.read_article("nonexistent/path.md")


class TestUpdateDraft:
    def test_overwrites_content(self, km: KnowledgeManager) -> None:
        meta = _sample_meta()
        path = km.create_draft(meta, "Original content")
        updated_meta = _sample_meta()
        km.update_draft(path, updated_meta, "Updated content")
        _, content = km.read_article(path)
        assert "Updated content" in content
        assert "Original content" not in content

    def test_bumps_timestamp(self, km: KnowledgeManager, test_db: sqlite3.Connection) -> None:
        meta = _sample_meta()
        path = km.create_draft(meta, "Content")
        original_updated = test_db.execute("SELECT last_updated FROM articles WHERE path = ?", (path,)).fetchone()[0]
        time.sleep(0.01)  # Ensure time difference
        km.update_draft(path, _sample_meta(confidence=0.9), "New content")
        new_updated = test_db.execute("SELECT last_updated FROM articles WHERE path = ?", (path,)).fetchone()[0]
        assert new_updated > original_updated

    def test_refuses_review(self, km: KnowledgeManager, test_db: sqlite3.Connection) -> None:
        meta = _sample_meta()
        path = km.create_draft(meta, "Content")
        test_db.execute("UPDATE articles SET status = 'review' WHERE path = ?", (path,))
        test_db.commit()
        with pytest.raises(ValueError, match="Cannot update"):
            km.update_draft(path, _sample_meta(), "New")

    def test_refuses_published(self, km: KnowledgeManager, test_db: sqlite3.Connection) -> None:
        meta = _sample_meta()
        path = km.create_draft(meta, "Content")
        test_db.execute("UPDATE articles SET status = 'published' WHERE path = ?", (path,))
        test_db.commit()
        with pytest.raises(ValueError, match="Cannot update"):
            km.update_draft(path, _sample_meta(), "New")

    def test_path_traversal_blocked(self, km: KnowledgeManager, test_db: sqlite3.Connection) -> None:
        """If a path traversal bypasses the DB check, the resolve check blocks it."""
        # Inject a malicious path directly into the DB to bypass the first check
        test_db.execute(
            """INSERT INTO articles (path, title, spec, category, status,
               confidence_score, created_at, last_updated)
               VALUES ('../../etc/passwd', 'Evil', 'combat', 'gear', 'draft',
               0.5, '2025-01-01', '2025-01-01')"""
        )
        test_db.commit()
        with pytest.raises((FileNotFoundError, ValueError)):
            km.update_draft("../../etc/passwd", _sample_meta(), "malicious")

    def test_db_failure_restores_file(self, km: KnowledgeManager, tmp_path, test_db: sqlite3.Connection) -> None:
        """If DB UPDATE fails, original file content should be restored."""
        meta = _sample_meta()
        path = km.create_draft(meta, "Original content")
        full_path = tmp_path / "knowledge" / path

        # Close the connection to force a DB error on update
        test_db.close()

        with pytest.raises(Exception):
            km.update_draft(path, _sample_meta(confidence=0.9), "New content")

        # File should still have original content
        content = full_path.read_text(encoding="utf-8")
        assert "Original content" in content
        assert "New content" not in content


class TestCreateDraftConsistency:
    """Tests for create_draft file-DB consistency on errors."""

    def test_db_error_cleans_up_file(self, km: KnowledgeManager, tmp_path, test_db: sqlite3.Connection) -> None:
        """If DB INSERT fails (non-IntegrityError), the file should be cleaned up."""
        meta = _sample_meta()
        path = derive_path(meta.spec, meta.category, meta.title)
        full_path = tmp_path / "knowledge" / path

        # Close the DB connection to force a generic DB error
        test_db.close()

        with pytest.raises(Exception):
            km.create_draft(meta, "Content")

        # File should NOT exist (cleaned up after DB error)
        assert not full_path.exists(), "Orphaned file left after DB error"


class TestUpdateDraftConsistency:
    """Tests for update_draft file-DB consistency on write errors."""

    def test_file_write_failure_restores_original(self, km: KnowledgeManager, tmp_path) -> None:
        """If file write fails, original content should be restored."""
        meta = _sample_meta()
        path = km.create_draft(meta, "Original content")
        full_path = tmp_path / "knowledge" / path

        # Patch write_text to fail on the UPDATE call (second call)
        original_write = full_path.__class__.write_text
        call_count = [0]

        def _failing_write(self_path, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 0 and str(self_path) == str(full_path):
                raise OSError("Disk full")
            return original_write(self_path, *args, **kwargs)

        with patch.object(full_path.__class__, "write_text", _failing_write):
            with pytest.raises(OSError, match="Disk full"):
                km.update_draft(path, _sample_meta(confidence=0.9), "New content")

        # File should still have original content (restored)
        content = full_path.read_text(encoding="utf-8")
        assert "Original content" in content


class TestRejectArticleConsistency:
    """Tests for reject_article file-DB consistency."""

    def test_db_failure_restores_file(self, km: KnowledgeManager, tmp_path, test_db: sqlite3.Connection) -> None:
        """If DB UPDATE fails during reject, original file should be restored."""
        meta = _sample_meta()
        path = km.create_draft(meta, "Content")
        km.set_status(path, ArticleStatus.REVIEW)
        full_path = tmp_path / "knowledge" / path
        original_content = full_path.read_text(encoding="utf-8")

        # Close DB to force failure
        test_db.close()

        with pytest.raises(Exception):
            km.reject_article(path, "Bad article")

        # File should still have the pre-rejection content
        restored = full_path.read_text(encoding="utf-8")
        assert restored == original_content


class TestListArticles:
    def test_no_filter(self, km: KnowledgeManager) -> None:
        km.create_draft(_sample_meta(title="Article 1"), "A")
        km.create_draft(_sample_meta(title="Article 2"), "B")
        articles = km.list_articles()
        assert len(articles) == 2
        assert all(hasattr(a, "path") for a in articles)  # ArticleSummary objects

    def test_filter_by_status(self, km: KnowledgeManager, test_db: sqlite3.Connection) -> None:
        path1 = km.create_draft(_sample_meta(title="Draft One"), "A")
        km.create_draft(_sample_meta(title="Draft Two"), "B")
        test_db.execute("UPDATE articles SET status = 'review' WHERE path = ?", (path1,))
        test_db.commit()
        drafts = km.list_articles(status=ArticleStatus.DRAFT)
        assert len(drafts) == 1
        assert drafts[0].title == "Draft Two"

    def test_filter_by_spec(self, km: KnowledgeManager) -> None:
        km.create_draft(_sample_meta(title="Combat Guide", spec=Spec.COMBAT), "A")
        km.create_draft(_sample_meta(title="Mut Guide", spec=Spec.ASSASSINATION), "B")
        combat = km.list_articles(spec=Spec.COMBAT)
        assert len(combat) == 1
        assert combat[0].spec == "combat"

    def test_empty(self, km: KnowledgeManager) -> None:
        assert km.list_articles() == []


class TestSetStatus:
    def test_draft_to_review(self, km: KnowledgeManager, test_db: sqlite3.Connection) -> None:
        path = km.create_draft(_sample_meta(), "Content")
        km.set_status(path, ArticleStatus.REVIEW)
        row = test_db.execute("SELECT status FROM articles WHERE path = ?", (path,)).fetchone()
        assert row["status"] == "review"

    def test_review_to_published(self, km: KnowledgeManager, test_db: sqlite3.Connection) -> None:
        path = km.create_draft(_sample_meta(), "Content")
        km.set_status(path, ArticleStatus.REVIEW)
        km.set_status(path, ArticleStatus.PUBLISHED)
        row = test_db.execute("SELECT status FROM articles WHERE path = ?", (path,)).fetchone()
        assert row["status"] == "published"

    def test_invalid_transition_raises(self, km: KnowledgeManager) -> None:
        path = km.create_draft(_sample_meta(), "Content")
        with pytest.raises(ValueError, match="Invalid.*transition"):
            km.set_status(path, ArticleStatus.PUBLISHED)  # draft -> published not allowed

    def test_backward_raises(self, km: KnowledgeManager) -> None:
        path = km.create_draft(_sample_meta(), "Content")
        km.set_status(path, ArticleStatus.REVIEW)
        with pytest.raises(ValueError, match="Invalid.*transition"):
            km.set_status(path, ArticleStatus.DRAFT)  # review -> draft not allowed


class TestExists:
    def test_returns_path_when_found(self, km: KnowledgeManager) -> None:
        km.create_draft(_sample_meta(), "Content")
        result = km.exists("combat", "gear", "Phase 1 Trinkets")
        assert result is not None
        assert result.endswith(".md")

    def test_returns_none_when_missing(self, km: KnowledgeManager) -> None:
        result = km.exists("combat", "gear", "Nonexistent Article")
        assert result is None
