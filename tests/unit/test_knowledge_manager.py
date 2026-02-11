"""Tests for KnowledgeManager article CRUD."""

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
