# Writer Agent + Wiki Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the Writer agent and KnowledgeManager so research findings become git-tracked wiki articles with YAML frontmatter.

**Architecture:** Writer subclass overrides `execute()` with direct LLM generation (no ReAct loop). Two-pass: generate article, then extract claims/entities. KnowledgeManager handles file + DB CRUD with `python-frontmatter` for YAML serialization. Schema v3 adds article lifecycle columns.

**Tech Stack:** Python 3.12, Pydantic v2, python-frontmatter, SQLite, Instructor/Llama 70B

**Design doc:** `docs/plans/2026-02-11-phase2-step5-writer-wiki.md`

---

## Task 1: Install python-frontmatter + Clean knowledge/ Directory

**Files:**
- Modify: `requirements.txt:26` (after sqlite-vec line)
- Delete: `knowledge/encounters/`, `knowledge/fundamentals/`, `knowledge/gearing/`, `knowledge/pvp/`, `knowledge/specs/`
- Create: `knowledge/.gitkeep`

**Step 1: Add python-frontmatter to requirements.txt**

Add after the `sqlparse` line (line 27):

```
python-frontmatter>=1.1
```

**Step 2: Install the dependency**

Run: `pip install python-frontmatter --break-system-packages`
Expected: Successfully installed python-frontmatter and pyyaml

**Step 3: Remove old empty knowledge dirs and create .gitkeep**

Run:
```bash
rm -rf knowledge/encounters knowledge/fundamentals knowledge/gearing knowledge/pvp knowledge/specs
touch knowledge/.gitkeep
```

**Step 4: Verify**

Run: `ls knowledge/`
Expected: Only `.gitkeep`

Run: `python3 -c "import frontmatter; print(frontmatter.__version__)"`
Expected: Prints version number

**Step 5: Commit**

```bash
git add requirements.txt knowledge/
git commit -m "chore: add python-frontmatter dep, clean knowledge/ layout"
```

---

## Task 2: Schema v3 — Articles Table + Migration

**Files:**
- Modify: `code/shukketsu/db/schema.sql:15,78-92`
- Modify: `code/shukketsu/db/connection.py:73-89`
- Modify: `tests/unit/test_db.py:82-85,149-152`

**Step 1: Write the failing schema version test**

In `tests/unit/test_db.py`, update existing test and add new one. First update `test_sets_schema_version` (line 82-85):

```python
    def test_sets_schema_version(self, db: sqlite3.Connection) -> None:
        """Schema version should be 3 after initialization."""
        version = db.execute("SELECT version FROM schema_version").fetchone()[0]
        assert version == 3
```

Update `test_schema_version_is_2` (line 149-152) to test v3:

```python
    def test_schema_version_is_3(self, db: sqlite3.Connection) -> None:
        """Schema version should be 3 after fresh initialization."""
        version = db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert version == 3
```

Add new tests at the end of `TestInitDb`:

```python
    def test_articles_table_v3_columns(self, db: sqlite3.Connection) -> None:
        """articles table should have v3 columns: status, spec, category, created_at."""
        columns = {row[1] for row in db.execute("PRAGMA table_info(articles)").fetchall()}
        for col in ("status", "spec", "category", "created_at", "last_updated"):
            assert col in columns, f"Missing column: {col}"
        assert "needs_review" not in columns, "needs_review should be removed in v3"

    def test_articles_status_index_exists(self, db: sqlite3.Connection) -> None:
        """articles table should have indexes on status and spec."""
        indexes = {row[1] for row in db.execute("PRAGMA index_list(articles)").fetchall()}
        assert "idx_articles_status" in indexes
        assert "idx_articles_spec" in indexes
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_db.py -v -k "schema_version or v3_columns or status_index"`
Expected: FAIL — schema version is 2, columns don't exist

**Step 3: Update schema.sql**

In `code/shukketsu/db/schema.sql`, change line 15:

```sql
INSERT INTO schema_version (version) VALUES (3);
```

Replace lines 78-92 (the articles table) with:

```sql
-- Wiki article metadata (articles themselves are Markdown in knowledge/)
CREATE TABLE articles (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,                     -- "combat/gear/trinkets.md"
    title TEXT NOT NULL,
    spec TEXT NOT NULL,                            -- "combat", "assassination", "subtlety", "general"
    category TEXT NOT NULL,                        -- "gear", "rotation", "mechanics", etc.
    status TEXT NOT NULL DEFAULT 'draft',           -- "draft", "review", "published"
    confidence_score REAL NOT NULL DEFAULT 0.0,    -- 0.0-1.0, average of finding confidences
    verified_claims INTEGER NOT NULL DEFAULT 0,
    unverified_claims INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_updated TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_articles_status ON articles(status);
CREATE INDEX idx_articles_spec ON articles(spec);
```

**Step 4: Add migration to connection.py**

In `code/shukketsu/db/connection.py`, add the v3 migration SQL constant after `_GRAPH_TABLES_SQL` (after line 70):

```python
_ARTICLES_V3_SQL = """
DROP TABLE IF EXISTS articles;

CREATE TABLE articles (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    spec TEXT NOT NULL,
    category TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    confidence_score REAL NOT NULL DEFAULT 0.0,
    verified_claims INTEGER NOT NULL DEFAULT 0,
    unverified_claims INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_updated TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);
CREATE INDEX IF NOT EXISTS idx_articles_spec ON articles(spec);
"""
```

Add the migration function after `_migrate_v1_to_v2`:

```python
def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """Migrate v2 schema to v3: replace articles table with richer columns."""
    conn.executescript(_ARTICLES_V3_SQL)
    conn.execute("INSERT INTO schema_version (version) VALUES (3)")
    conn.commit()
    logger.info("Database migrated from v2 to v3 (articles table v3)")
```

Update `init_db` (lines 78-89):

```python
def init_db(conn: sqlite3.Connection) -> None:
    """Initialize database schema from schema.sql. Idempotent.

    Checks schema_version table before applying. Safe to call multiple times.
    """
    try:
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        if version is not None:
            if version < 2:
                _migrate_v1_to_v2(conn)
            if version < 3:
                _migrate_v2_to_v3(conn)
            return
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet — need to initialize

    schema_sql = (_DB_DIR / "schema.sql").read_text()
    conn.executescript(schema_sql)
    logger.info("Database schema initialized (version 3)")
```

**Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_db.py -v`
Expected: ALL PASS (including updated version checks)

**Step 6: Commit**

```bash
git add code/shukketsu/db/schema.sql code/shukketsu/db/connection.py tests/unit/test_db.py
git commit -m "feat(db): upgrade articles table to schema v3"
```

---

## Task 3: WriteTask Type Tightening + WriteResult Model

**Files:**
- Modify: `code/shukketsu/agents/tasks.py:110-117`
- Modify: `code/shukketsu/agents/__init__.py`
- Modify: `tests/unit/test_tasks.py:157-176`

**Step 1: Write/update the failing tests**

In `tests/unit/test_tasks.py`, update the imports to include `ResearchResult`, `Finding`, and `WriteResult` (add to the existing import block at the top).

Replace the entire `TestWriteTask` class (lines 157-176):

```python
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
                research=research,
                article_type=ArticleType.GUIDE,
                spec="combat",
                category="gear",
            )

    def test_rejects_missing_research(self) -> None:
        with pytest.raises(ValidationError):
            WriteTask(query="Write", article_type=ArticleType.GUIDE, spec="combat", category="gear")  # type: ignore[call-arg]

    def test_rejects_missing_article_type(self) -> None:
        research = ResearchResult(
            task_id="r1", agent_role=AgentRole.RESEARCHER, status=TaskStatus.SUCCESS, output="x"
        )
        with pytest.raises(ValidationError):
            WriteTask(query="Write", research=research, spec="combat", category="gear")  # type: ignore[call-arg]

    def test_rejects_missing_spec(self) -> None:
        research = ResearchResult(
            task_id="r1", agent_role=AgentRole.RESEARCHER, status=TaskStatus.SUCCESS, output="x"
        )
        with pytest.raises(ValidationError):
            WriteTask(query="Write", research=research, article_type=ArticleType.GUIDE, category="gear")  # type: ignore[call-arg]

    def test_rejects_missing_category(self) -> None:
        research = ResearchResult(
            task_id="r1", agent_role=AgentRole.RESEARCHER, status=TaskStatus.SUCCESS, output="x"
        )
        with pytest.raises(ValidationError):
            WriteTask(query="Write", research=research, article_type=ArticleType.GUIDE, spec="combat")  # type: ignore[call-arg]
```

Add a new `TestWriteResult` class after `TestWriteTask`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_tasks.py -v -k "WriteTask or WriteResult"`
Expected: FAIL — `WriteResult` doesn't exist, `WriteTask` type mismatch

**Step 3: Update tasks.py**

In `code/shukketsu/agents/tasks.py`, replace the `WriteTask` class (lines 110-117):

```python
class WriteTask(AgentTask):
    """Task for the Writer agent.

    Requires research results, an article type, and target spec/category.
    The Orchestrator populates spec and category when dispatching.
    """

    research: ResearchResult
    article_type: ArticleType
    spec: str
    category: str
```

Add `WriteResult` after `WriteTask` (before `EditTask`):

```python
class WriteResult(AgentResult):
    """Structured output from the Writer agent."""

    article_path: str
    title: str
    claims: list[str] = Field(default_factory=list)
    research_gaps: list[str] = Field(default_factory=list)
    word_count: int = 0
    entity_refs: list[str] = Field(default_factory=list)
```

**Step 4: Update agents/__init__.py**

Add `WriteResult` to both the import and `__all__`:

```python
from code.shukketsu.agents.tasks import (
    ...
    WriteResult,
    WriteTask,
)

__all__ = [
    ...
    "WriteResult",
    "WriteTask",
]
```

**Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_tasks.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add code/shukketsu/agents/tasks.py code/shukketsu/agents/__init__.py tests/unit/test_tasks.py
git commit -m "feat(agents): add WriteResult model, tighten WriteTask types"
```

---

## Task 4: KnowledgeManager Models + Slug/Path Utilities

**Files:**
- Create: `code/shukketsu/knowledge/__init__.py`
- Create: `code/shukketsu/knowledge/manager.py`
- Create: `tests/unit/test_knowledge_manager.py`

**Step 1: Write the failing tests for slugify and derive_path**

Create `tests/unit/test_knowledge_manager.py`:

```python
"""Tests for KnowledgeManager article CRUD."""

from code.shukketsu.knowledge.manager import derive_path, slugify


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
        assert derive_path("assassination", "rotation", "Mutilate: Priority List") == "assassination/rotation/mutilate-priority-list.md"
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_knowledge_manager.py -v -k "Slugify or DerivePath"`
Expected: FAIL — module doesn't exist

**Step 3: Create the knowledge package and manager module with models + utilities**

Create `code/shukketsu/knowledge/__init__.py`:

```python
"""Wiki article management: KnowledgeManager and supporting models."""

from code.shukketsu.knowledge.manager import (
    ArticleMeta,
    ArticleStatus,
    ArticleSummary,
    ClaimRef,
    KnowledgeManager,
    SourceRef,
    Spec,
)

__all__ = [
    "ArticleMeta",
    "ArticleStatus",
    "ArticleSummary",
    "ClaimRef",
    "KnowledgeManager",
    "SourceRef",
    "Spec",
]
```

Create `code/shukketsu/knowledge/manager.py`:

```python
"""KnowledgeManager for wiki article CRUD.

Handles both filesystem (Markdown + YAML frontmatter) and database
(articles table) operations. Articles are the source of truth on disk;
the DB row is for querying and status tracking.
"""

import logging
import re
import sqlite3
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import frontmatter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ArticleStatus(StrEnum):
    """Lifecycle status of a wiki article."""

    DRAFT = "draft"
    REVIEW = "review"
    PUBLISHED = "published"


class Spec(StrEnum):
    """Valid spec values for article classification."""

    COMBAT = "combat"
    ASSASSINATION = "assassination"
    SUBTLETY = "subtlety"
    GENERAL = "general"


class SourceRef(BaseModel):
    """A source referenced in an article."""

    url: str
    trust: float = Field(ge=0.0, le=1.0)


class ClaimRef(BaseModel):
    """A factual claim in an article, tracked for verification."""

    text: str
    verified: bool = False
    evidence: list[str] = Field(default_factory=list)


class ArticleSummary(BaseModel):
    """Lightweight article record from DB queries."""

    path: str
    title: str
    spec: str
    category: str
    status: ArticleStatus
    confidence_score: float
    last_updated: str


class ArticleMeta(BaseModel):
    """Pydantic model for article YAML frontmatter."""

    title: str
    spec: Spec
    category: str
    status: ArticleStatus = ArticleStatus.DRAFT
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    created_at: datetime
    updated_at: datetime
    sources: list[SourceRef] = Field(default_factory=list)
    claims: list[ClaimRef] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    """Convert a title to a URL-safe slug.

    Lowercase, replace non-alphanumeric runs with hyphens,
    strip leading/trailing hyphens.
    """
    return _SLUG_RE.sub("-", title.lower()).strip("-")


def derive_path(spec: str, category: str, title: str) -> str:
    """Derive the relative article path from spec, category, and title."""
    return f"{spec}/{category}/{slugify(title)}.md"


class KnowledgeManager:
    """Article CRUD: filesystem (Markdown) + database (articles table)."""

    def __init__(self, conn: sqlite3.Connection, knowledge_dir: Path) -> None:
        self._conn = conn
        self._knowledge_dir = knowledge_dir
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_knowledge_manager.py -v -k "Slugify or DerivePath"`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add code/shukketsu/knowledge/ tests/unit/test_knowledge_manager.py
git commit -m "feat(knowledge): add KnowledgeManager models and slug utilities"
```

---

## Task 5: KnowledgeManager.create_draft + read_article

**Files:**
- Modify: `code/shukketsu/knowledge/manager.py`
- Modify: `tests/unit/test_knowledge_manager.py`

**Step 1: Write the failing tests**

Add to `tests/unit/test_knowledge_manager.py`:

```python
import sqlite3
from datetime import datetime, timezone

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
    now = datetime.now(tz=timezone.utc)
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
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_knowledge_manager.py -v -k "CreateDraft or ReadArticle"`
Expected: FAIL — methods don't exist

**Step 3: Implement create_draft and read_article**

Add to `KnowledgeManager` in `code/shukketsu/knowledge/manager.py`:

```python
    def create_draft(self, meta: ArticleMeta, content: str) -> str:
        """Write markdown file + insert DB row. Returns relative path.

        Checks DB uniqueness before writing to prevent orphaned files.
        """
        path = derive_path(meta.spec, meta.category, meta.title)

        # Check DB first to prevent orphaned files
        existing = self._conn.execute("SELECT id FROM articles WHERE path = ?", (path,)).fetchone()
        if existing is not None:
            raise ValueError(f"Article already exists at path: {path}")

        # Write file
        full_path = self._knowledge_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        post = frontmatter.Post(content, **meta.model_dump(mode="json"))
        full_path.write_text(frontmatter.dumps(post), encoding="utf-8")

        # Insert DB row
        try:
            self._conn.execute(
                """INSERT INTO articles (path, title, spec, category, status, confidence_score, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    path,
                    meta.title,
                    meta.spec,
                    meta.category,
                    meta.status,
                    meta.confidence,
                    meta.created_at.isoformat() if isinstance(meta.created_at, datetime) else meta.created_at,
                    meta.updated_at.isoformat() if isinstance(meta.updated_at, datetime) else meta.updated_at,
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            # Race condition: another process inserted between check and insert
            full_path.unlink(missing_ok=True)
            raise ValueError(f"Article already exists at path: {path}")

        logger.info("Created draft article: %s", path)
        return path

    def read_article(self, path: str) -> tuple[ArticleMeta, str]:
        """Read and parse frontmatter + content from a markdown file."""
        full_path = self._knowledge_dir / path
        if not full_path.exists():
            raise FileNotFoundError(f"Article not found: {path}")

        post = frontmatter.loads(full_path.read_text(encoding="utf-8"))
        meta = ArticleMeta(**post.metadata)
        return meta, post.content
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_knowledge_manager.py -v -k "CreateDraft or ReadArticle"`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add code/shukketsu/knowledge/manager.py tests/unit/test_knowledge_manager.py
git commit -m "feat(knowledge): implement create_draft and read_article"
```

---

## Task 6: KnowledgeManager.update_draft, list_articles, set_status, exists

**Files:**
- Modify: `code/shukketsu/knowledge/manager.py`
- Modify: `tests/unit/test_knowledge_manager.py`

**Step 1: Write the failing tests**

Add to `tests/unit/test_knowledge_manager.py`:

```python
import time


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
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_knowledge_manager.py -v -k "UpdateDraft or ListArticles or SetStatus or Exists"`
Expected: FAIL — methods don't exist

**Step 3: Implement the remaining methods**

Add to `KnowledgeManager` class in `code/shukketsu/knowledge/manager.py`:

```python
    def update_draft(self, path: str, meta: ArticleMeta, content: str) -> None:
        """Overwrite a draft article's content and metadata.

        Only works on articles with status 'draft'. Raises ValueError
        for review or published articles.
        """
        row = self._conn.execute("SELECT status FROM articles WHERE path = ?", (path,)).fetchone()
        if row is None:
            raise ValueError(f"Article not found: {path}")

        status = row["status"]
        if status != ArticleStatus.DRAFT:
            raise ValueError(f"Cannot update article with status '{status}'")

        # Write file
        full_path = self._knowledge_dir / path
        now = datetime.now(tz=meta.updated_at.tzinfo)
        meta_updated = meta.model_copy(update={"updated_at": now})

        post = frontmatter.Post(content, **meta_updated.model_dump(mode="json"))
        full_path.write_text(frontmatter.dumps(post), encoding="utf-8")

        # Update DB row
        self._conn.execute(
            """UPDATE articles SET title = ?, confidence_score = ?, last_updated = ?
            WHERE path = ?""",
            (meta_updated.title, meta_updated.confidence, now.isoformat(), path),
        )
        self._conn.commit()
        logger.info("Updated draft article: %s", path)

    def list_articles(
        self,
        *,
        status: ArticleStatus | None = None,
        spec: "Spec | None" = None,
    ) -> list[ArticleSummary]:
        """Query articles with optional filters. Returns typed ArticleSummary objects."""
        query = "SELECT path, title, spec, category, status, confidence_score, last_updated FROM articles"
        conditions: list[str] = []
        params: list[str] = []

        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if spec is not None:
            conditions.append("spec = ?")
            params.append(spec)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY last_updated DESC"

        rows = self._conn.execute(query, params).fetchall()
        return [
            ArticleSummary(
                path=row["path"],
                title=row["title"],
                spec=row["spec"],
                category=row["category"],
                status=ArticleStatus(row["status"]),
                confidence_score=row["confidence_score"],
                last_updated=row["last_updated"],
            )
            for row in rows
        ]

    def set_status(self, path: str, new_status: ArticleStatus) -> None:
        """Transition article status. Enforces draft->review->published order."""
        row = self._conn.execute("SELECT status FROM articles WHERE path = ?", (path,)).fetchone()
        if row is None:
            raise ValueError(f"Article not found: {path}")

        current = ArticleStatus(row["status"])
        valid_transitions = {
            ArticleStatus.DRAFT: ArticleStatus.REVIEW,
            ArticleStatus.REVIEW: ArticleStatus.PUBLISHED,
        }

        if valid_transitions.get(current) != new_status:
            raise ValueError(f"Invalid status transition: {current} -> {new_status}")

        self._conn.execute(
            "UPDATE articles SET status = ?, last_updated = ? WHERE path = ?",
            (new_status, datetime.now().isoformat(), path),
        )
        self._conn.commit()
        logger.info("Article %s status: %s -> %s", path, current, new_status)

    def exists(self, spec: str, category: str, title: str) -> str | None:
        """Check if an article exists at the derived path. Returns path or None."""
        path = derive_path(spec, category, title)
        row = self._conn.execute("SELECT path FROM articles WHERE path = ?", (path,)).fetchone()
        return row["path"] if row else None
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_knowledge_manager.py -v`
Expected: ALL PASS

**Step 5: Run full test suite for regressions**

Run: `python3 -m pytest tests/unit/ -v`
Expected: ALL PASS (473 existing + new KnowledgeManager tests)

**Step 6: Commit**

```bash
git add code/shukketsu/knowledge/manager.py tests/unit/test_knowledge_manager.py
git commit -m "feat(knowledge): implement update_draft, list_articles, set_status, exists"
```

---

## Task 7: Writer System Prompt + Config Update

**Files:**
- Create: `code/shukketsu/llm/prompts/writer.py`
- Modify: `code/shukketsu/config.py:117-124`

**Step 1: Create the writer prompt module**

Create `code/shukketsu/llm/prompts/writer.py`:

```python
"""Writer agent system prompt.

This module is a pure string constant with zero imports.
config.py imports from here — never the reverse (prevents circular imports).
"""

WRITER_SYSTEM_PROMPT = """\
You are a Wiki Writer for WoW: The Burning Crusade (TBC) Rogue content. You produce \
clear, accurate, well-structured Markdown articles from research findings.

## Writing Rules

STRUCTURE every article with:
  - A title that clearly identifies the topic and scope
  - A concise overview paragraph (2-3 sentences)
  - Logical sections with ## headers
  - Specific numbers and values (never vague)
  - Source attribution for key claims

TONE: Authoritative but approachable. Like an experienced raider explaining to a guild \
member. Use "you" to address the reader.

CLAIMS: Every factual claim must trace back to a research finding. Never invent facts \
the research didn't provide. If a finding has low confidence (< 0.5), hedge with \
"likely" or "appears to".

COMPLETENESS: If the research has gaps, insert a <!-- NEEDS RESEARCH: [topic] --> \
HTML comment at the relevant location rather than guessing.

FORMAT:
  - Use tables for gear comparisons, stat breakdowns, priority lists
  - Use **bold** for item names, spell names, stat values
  - Use ordered lists for priorities/rankings
  - Keep paragraphs short (3-5 sentences max)
  - Use > blockquotes for important callouts or tips

## Article Types

GUIDE: Step-by-step or priority-based. "How to gear your combat rogue in Phase 1."
  - Structure: Overview → Priority List → Detailed Breakdown → Tips

REFERENCE: Factual lookup. "Stat caps and hit table mechanics."
  - Structure: Overview → Core Mechanics → Values Table → Edge Cases

ANALYSIS: Comparative or analytical. "Combat vs Mutilate in Phase 2."
  - Structure: Overview → Methodology → Comparison → Conclusion

## Example 1: GUIDE from Research Findings

Input findings:
  1. "DST is BiS trinket for combat" (confidence: 0.9, evidence: [chunk:42, chunk:78])
  2. "Hit cap is 142 rating (9%)" (confidence: 0.95, evidence: [chunk:15, chunk:91])
  3. "Expertise soft cap is 23" (confidence: 0.7, evidence: [chunk:91])

Output:

# Combat Swords: Phase 1 Stat Priority

## Overview

Combat Swords is the dominant PvE rogue spec in Phase 1 of TBC. Reaching your hit and \
expertise caps before stacking attack power and crit is essential for maximizing DPS.

## Stat Priority

1. **Hit Rating** to cap (142 rating / 9%)
2. **Expertise** to soft cap (23 rating)
3. **Attack Power**
4. **Crit Rating**

## Example 2: REFERENCE from Research Findings

Input findings:
  1. "Dual wield miss penalty is 19%" (confidence: 0.85, evidence: [chunk:15])
  2. "Special attacks use single-roll hit table" (confidence: 0.8, evidence: [chunk:203])

Output:

# Hit Table Mechanics for Rogues

## Overview

Understanding the hit table is fundamental to gearing decisions. Rogues face a dual wield \
miss penalty that makes hit rating the most valuable stat until capped.

## How the Hit Table Works

Auto-attacks use a **two-roll system** where miss, dodge, parry, glancing, block, crit, \
and hit are resolved separately for each roll.\
"""

EXTRACTION_PROMPT = """\
You are an extraction assistant. Given a wiki article about WoW TBC Rogue content, \
extract:

1. CLAIMS: Every factual claim in the article. Each claim should be a single, \
verifiable statement. Examples: "Hit cap is 142 rating", "DST drops from Gruul", \
"Combat Swords is the highest DPS spec in Phase 1".

2. ENTITY_REFS: Every WoW game entity mentioned in the article. Include item names, \
spell names, talent names, boss names, instance names, stat names, spec names. \
Use the canonical form (e.g., "Dragonspine Trophy" not "DST").

Be exhaustive. Extract every claim and every entity, not just the main ones.\
"""
```

**Step 2: Update config.py to import from writer prompt module**

In `code/shukketsu/config.py`, replace lines 121-124:

```python
from code.shukketsu.llm.prompts.writer import (  # noqa: E402
    WRITER_SYSTEM_PROMPT as WRITER_SYSTEM_PROMPT,
)
```

**Step 3: Verify imports work**

Run: `python3 -c "from code.shukketsu.config import WRITER_SYSTEM_PROMPT; print(len(WRITER_SYSTEM_PROMPT))"`
Expected: Prints a number > 1000 (the full prompt length)

**Step 4: Run existing tests for regressions**

Run: `python3 -m pytest tests/unit/ -v --timeout=10 -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add code/shukketsu/llm/prompts/writer.py code/shukketsu/config.py
git commit -m "feat(llm): add Writer system prompt and extraction prompt"
```

---

## Task 8: Writer Agent + Factory Integration

**Files:**
- Create: `code/shukketsu/agents/writer.py`
- Modify: `code/shukketsu/agents/factory.py`
- Modify: `code/shukketsu/agents/__init__.py`
- Create: `tests/unit/test_writer.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_writer.py`:

```python
"""Tests for Writer agent."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from code.shukketsu.agents.tasks import (
    AgentRole,
    AgentTask,
    ArticleType,
    Finding,
    ResearchResult,
    TaskStatus,
    WriteResult,
    WriteTask,
)
from code.shukketsu.agents.writer import Writer
from code.shukketsu.knowledge.manager import KnowledgeManager, Spec
from code.shukketsu.tools.registry import ToolRegistry


@pytest.fixture
def knowledge_dir(tmp_path) -> Path:
    d = tmp_path / "knowledge"
    d.mkdir()
    return d


@pytest.fixture
def km(test_db: sqlite3.Connection, knowledge_dir: Path) -> KnowledgeManager:
    return KnowledgeManager(test_db, knowledge_dir)


@pytest.fixture
def writer(km: KnowledgeManager) -> Writer:
    return Writer(
        tool_registry=ToolRegistry(),
        role=AgentRole.WRITER,
        knowledge_manager=km,
    )


def _research_result(**overrides) -> ResearchResult:
    defaults = dict(
        task_id="r1",
        agent_role=AgentRole.RESEARCHER,
        status=TaskStatus.SUCCESS,
        output="Research complete",
        findings=[
            Finding(claim="Hit cap is 142 rating", confidence=0.9, evidence=["https://example.com/guide"]),
            Finding(claim="DST is BiS trinket", confidence=0.8, evidence=["chunk:42"]),
        ],
        gaps=["proc rate details"],
        sufficient=True,
    )
    defaults.update(overrides)
    return ResearchResult(**defaults)


def _write_task(**overrides) -> WriteTask:
    defaults = dict(
        query="Write about combat trinkets",
        research=_research_result(),
        article_type=ArticleType.GUIDE,
        spec="combat",
        category="gear",
    )
    defaults.update(overrides)
    return WriteTask(**defaults)


# Mock structured output to return predictable results
def _mock_structured_output(response_model, messages, **kwargs):
    """Return a mock structured response based on the model type."""
    if response_model.__name__ == "GeneratedArticle":
        return response_model(
            title="Combat Trinkets Guide",
            content="# Combat Trinkets Guide\n\n## Overview\n\nTrinkets are important.\n\n## BiS List\n\n**Dragonspine Trophy** is the best.",
        )
    elif response_model.__name__ == "ArticleExtraction":
        return response_model(
            claims=["Hit cap is 142 rating", "DST is BiS trinket"],
            entity_refs=["Dragonspine Trophy", "Hit Rating"],
        )
    raise ValueError(f"Unexpected model: {response_model.__name__}")


class TestWriterValidation:
    async def test_execute_requires_role(self, km: KnowledgeManager) -> None:
        writer = Writer(tool_registry=ToolRegistry(), role=None, knowledge_manager=km)
        task = _write_task()
        with pytest.raises(ValueError, match="role"):
            await writer.execute(task)

    async def test_execute_requires_write_task(self, writer: Writer) -> None:
        task = AgentTask(query="Not a write task")
        result = await writer.execute(task)
        assert result.status == TaskStatus.FAILED

    async def test_execute_empty_findings_returns_failed(self, writer: Writer) -> None:
        task = _write_task(research=_research_result(findings=[]))
        result = await writer.execute(task)
        assert result.status == TaskStatus.FAILED
        assert "No findings" in result.output


class TestWriterGeneration:
    @patch("code.shukketsu.agents.writer.get_structured_output", new_callable=AsyncMock, side_effect=_mock_structured_output)
    async def test_execute_produces_article_file(self, mock_llm, writer: Writer, knowledge_dir: Path) -> None:
        task = _write_task()
        result = await writer.execute(task)
        assert result.status == TaskStatus.SUCCESS
        full_path = knowledge_dir / result.article_path
        assert full_path.exists()

    @patch("code.shukketsu.agents.writer.get_structured_output", new_callable=AsyncMock, side_effect=_mock_structured_output)
    async def test_execute_returns_write_result(self, mock_llm, writer: Writer) -> None:
        task = _write_task()
        result = await writer.execute(task)
        assert isinstance(result, WriteResult)
        assert result.article_path.endswith(".md")
        assert result.title == "Combat Trinkets Guide"

    @patch("code.shukketsu.agents.writer.get_structured_output", new_callable=AsyncMock, side_effect=_mock_structured_output)
    async def test_execute_article_has_frontmatter(self, mock_llm, writer: Writer, knowledge_dir: Path) -> None:
        import frontmatter

        task = _write_task()
        result = await writer.execute(task)
        raw = (knowledge_dir / result.article_path).read_text()
        post = frontmatter.loads(raw)
        assert post.metadata["title"] == "Combat Trinkets Guide"
        assert post.metadata["spec"] == "combat"

    @patch("code.shukketsu.agents.writer.get_structured_output", new_callable=AsyncMock, side_effect=_mock_structured_output)
    async def test_execute_claims_extracted(self, mock_llm, writer: Writer) -> None:
        result = await writer.execute(_write_task())
        assert len(result.claims) == 2
        assert "Hit cap is 142 rating" in result.claims

    @patch("code.shukketsu.agents.writer.get_structured_output", new_callable=AsyncMock, side_effect=_mock_structured_output)
    async def test_execute_entity_refs_extracted(self, mock_llm, writer: Writer) -> None:
        result = await writer.execute(_write_task())
        assert "Dragonspine Trophy" in result.entity_refs

    @patch("code.shukketsu.agents.writer.get_structured_output", new_callable=AsyncMock, side_effect=_mock_structured_output)
    async def test_execute_word_count_accurate(self, mock_llm, writer: Writer) -> None:
        result = await writer.execute(_write_task())
        assert result.word_count > 0

    @patch("code.shukketsu.agents.writer.get_structured_output", new_callable=AsyncMock, side_effect=_mock_structured_output)
    async def test_execute_gaps_carried_through(self, mock_llm, writer: Writer) -> None:
        result = await writer.execute(_write_task())
        assert result.research_gaps == ["proc rate details"]

    @patch("code.shukketsu.agents.writer.get_structured_output", new_callable=AsyncMock, side_effect=_mock_structured_output)
    async def test_execute_confidence_is_average(self, mock_llm, writer: Writer) -> None:
        result = await writer.execute(_write_task())
        # Findings have confidence 0.9 and 0.8, average is 0.85
        assert result.evidence is not None
        meta, _ = writer._km.read_article(result.article_path)
        assert meta.confidence == pytest.approx(0.85)

    @patch("code.shukketsu.agents.writer.get_structured_output", new_callable=AsyncMock, side_effect=_mock_structured_output)
    async def test_execute_uses_task_spec_category(self, mock_llm, writer: Writer) -> None:
        task = _write_task(spec="assassination", category="rotation")
        result = await writer.execute(task)
        assert result.article_path.startswith("assassination/rotation/")


class TestWriterExistingArticle:
    @patch("code.shukketsu.agents.writer.get_structured_output", new_callable=AsyncMock, side_effect=_mock_structured_output)
    async def test_execute_overwrites_draft(self, mock_llm, writer: Writer, knowledge_dir: Path) -> None:
        task = _write_task()
        await writer.execute(task)  # First write
        result = await writer.execute(task)  # Should overwrite draft
        assert result.status == TaskStatus.SUCCESS

    @patch("code.shukketsu.agents.writer.get_structured_output", new_callable=AsyncMock, side_effect=_mock_structured_output)
    async def test_execute_refuses_published(self, mock_llm, writer: Writer, test_db: sqlite3.Connection) -> None:
        task = _write_task()
        result1 = await writer.execute(task)
        test_db.execute("UPDATE articles SET status = 'published' WHERE path = ?", (result1.article_path,))
        test_db.commit()
        result2 = await writer.execute(task)
        assert result2.status == TaskStatus.FAILED

    @patch("code.shukketsu.agents.writer.get_structured_output", new_callable=AsyncMock, side_effect=_mock_structured_output)
    async def test_execute_refuses_review(self, mock_llm, writer: Writer, test_db: sqlite3.Connection) -> None:
        task = _write_task()
        result1 = await writer.execute(task)
        test_db.execute("UPDATE articles SET status = 'review' WHERE path = ?", (result1.article_path,))
        test_db.commit()
        result2 = await writer.execute(task)
        assert result2.status == TaskStatus.FAILED


class TestWriterErrors:
    @patch("code.shukketsu.agents.writer.get_structured_output", new_callable=AsyncMock)
    async def test_execute_generation_failure_returns_failed(self, mock_llm, writer: Writer) -> None:
        from code.shukketsu.resilience.errors import StructuredOutputError

        mock_llm.side_effect = StructuredOutputError("LLM failed")
        result = await writer.execute(_write_task())
        assert result.status == TaskStatus.FAILED

    @patch("code.shukketsu.agents.writer.get_structured_output", new_callable=AsyncMock)
    async def test_execute_extraction_failure_still_writes(self, mock_llm, writer: Writer, knowledge_dir: Path) -> None:
        from code.shukketsu.resilience.errors import StructuredOutputError

        call_count = 0

        async def _side_effect(response_model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # Generation pass succeeds
                return response_model(
                    title="Test Article",
                    content="# Test\n\nContent here.",
                )
            else:  # Extraction pass fails
                raise StructuredOutputError("Extraction failed")

        mock_llm.side_effect = _side_effect
        result = await writer.execute(_write_task())
        assert result.status == TaskStatus.SUCCESS
        assert result.claims == []
        assert (knowledge_dir / result.article_path).exists()


class TestWriterFactory:
    def test_factory_creates_writer(self, km: KnowledgeManager) -> None:
        from code.shukketsu.agents.factory import AgentFactory

        factory = AgentFactory()
        agent = factory.create(AgentRole.WRITER, knowledge_manager=km)
        assert isinstance(agent, Writer)

    def test_factory_writer_requires_knowledge_manager(self) -> None:
        from code.shukketsu.agents.factory import AgentFactory

        factory = AgentFactory()
        with pytest.raises(TypeError):
            factory.create(AgentRole.WRITER)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_writer.py -v`
Expected: FAIL — Writer class doesn't exist

**Step 3: Implement the Writer agent**

Create `code/shukketsu/agents/writer.py`:

```python
"""Writer specialist agent.

Overrides execute() with direct LLM generation (no ReAct loop).
Two-pass: generate article from findings, then extract claims/entities.
"""

import logging
import statistics
from datetime import datetime, timezone

from langfuse import observe
from pydantic import BaseModel, Field

from code.shukketsu.agents.base import BaseAgent, StatusCallback
from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    TaskStatus,
    WriteResult,
    WriteTask,
)
from code.shukketsu.knowledge.manager import (
    ArticleMeta,
    ArticleStatus,
    ClaimRef,
    KnowledgeManager,
    SourceRef,
    Spec,
)
from code.shukketsu.llm.prompts.writer import EXTRACTION_PROMPT, WRITER_SYSTEM_PROMPT
from code.shukketsu.llm.structured import get_structured_output
from code.shukketsu.resilience.errors import LLMUnavailableError, StructuredOutputError
from code.shukketsu.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class GeneratedArticle(BaseModel):
    """Schema for the article generation pass."""

    title: str
    content: str


class ArticleExtraction(BaseModel):
    """Schema for the claims/entity extraction pass."""

    claims: list[str] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)


class Writer(BaseAgent):
    """Wiki article writer agent.

    Uses direct LLM generation (no ReAct loop). Receives a ResearchResult,
    generates an article via Llama 70B, extracts claims/entities in a
    second pass, and writes the result via KnowledgeManager.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        role: AgentRole | None = None,
        max_iterations: int | None = None,
        system_prompt: str | None = None,
        knowledge_manager: KnowledgeManager,
    ) -> None:
        from code.shukketsu import config

        super().__init__(
            tool_registry=tool_registry,
            role=role,
            max_iterations=max_iterations or config.AGENT_MAX_ITERATIONS,
            system_prompt=system_prompt or config.SYSTEM_PROMPT,
        )
        self._km = knowledge_manager

    @observe(as_type="agent")
    async def execute(self, task: AgentTask, *, on_status: StatusCallback | None = None) -> WriteResult | AgentResult:
        """Execute a write task: generate article from research findings.

        Args:
            task: Must be a WriteTask with ResearchResult.
            on_status: Optional async callback for progress updates.

        Returns:
            WriteResult on success, AgentResult with FAILED status on error.
        """
        # Step 1: Validate input
        if self.role is None:
            raise ValueError("Writer requires a role. Use AgentFactory or set role in constructor.")

        if not isinstance(task, WriteTask):
            return AgentResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=TaskStatus.FAILED,
                output="Writer requires a WriteTask",
            )

        # Step 2: Guard — empty findings
        if not task.research.findings:
            return WriteResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=TaskStatus.FAILED,
                output="No findings to write about",
                article_path="",
                title="",
            )

        # Step 3: Generate article content (Pass 1)
        if on_status:
            await on_status("generating article...")

        try:
            generated = await self._generate_article(task)
        except (StructuredOutputError, LLMUnavailableError, Exception) as exc:
            logger.warning("Article generation failed: %s", exc)
            return WriteResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=TaskStatus.FAILED,
                output=f"Article generation failed: {exc}",
                article_path="",
                title="",
            )

        # Step 4: Check for existing article (using LLM-generated title)
        existing_path = self._km.exists(task.spec, task.category, generated.title)
        if existing_path is not None:
            meta, _ = self._km.read_article(existing_path)
            if meta.status != ArticleStatus.DRAFT:
                return WriteResult(
                    task_id=task.task_id,
                    agent_role=self.role,
                    status=TaskStatus.FAILED,
                    output=f"Cannot overwrite {meta.status} article: {existing_path}",
                    article_path=existing_path,
                    title=generated.title,
                )

        # Step 5: Extract claims and entities (Pass 2)
        if on_status:
            await on_status("extracting claims...")

        try:
            extraction = await self._extract_claims(generated.content)
        except (StructuredOutputError, LLMUnavailableError, Exception) as exc:
            logger.warning("Extraction pass failed, continuing with empty claims: %s", exc)
            extraction = ArticleExtraction()

        # Step 6: Build metadata and write
        now = datetime.now(tz=timezone.utc)
        confidence = statistics.mean(f.confidence for f in task.research.findings)

        # Deduplicate source URLs from findings evidence
        seen_urls: set[str] = set()
        sources: list[SourceRef] = []
        for finding in task.research.findings:
            for ev in finding.evidence:
                if ev.startswith("http") and ev not in seen_urls:
                    seen_urls.add(ev)
                    sources.append(SourceRef(url=ev, trust=finding.confidence))

        claims = [ClaimRef(text=c, verified=False) for c in extraction.claims]

        article_meta = ArticleMeta(
            title=generated.title,
            spec=Spec(task.spec),
            category=task.category,
            status=ArticleStatus.DRAFT,
            confidence=round(confidence, 4),
            created_at=now,
            updated_at=now,
            sources=sources,
            claims=claims,
            entity_refs=extraction.entity_refs,
            tags=[task.spec, task.category],
        )

        if existing_path is not None:
            self._km.update_draft(existing_path, article_meta, generated.content)
            path = existing_path
        else:
            path = self._km.create_draft(article_meta, generated.content)

        # Step 7: Return WriteResult
        return WriteResult(
            task_id=task.task_id,
            agent_role=self.role,
            status=TaskStatus.SUCCESS,
            output=f"Article written: {generated.title}",
            evidence=list(seen_urls),
            article_path=path,
            title=generated.title,
            claims=extraction.claims,
            research_gaps=task.research.gaps,
            word_count=len(generated.content.split()),
            entity_refs=extraction.entity_refs,
        )

    async def _generate_article(self, task: WriteTask) -> GeneratedArticle:
        """Pass 1: Generate article content from research findings."""
        findings_text = "\n".join(
            f"- {f.claim} (confidence: {f.confidence}, evidence: {f.evidence})" for f in task.research.findings
        )
        gaps_text = "\n".join(f"- {g}" for g in task.research.gaps) if task.research.gaps else "(none)"

        messages = [
            {"role": "system", "content": WRITER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Write a {task.article_type.value} article about: {task.query}\n\n"
                    f"Spec: {task.spec}\nCategory: {task.category}\n\n"
                    f"Research findings:\n{findings_text}\n\n"
                    f"Research gaps:\n{gaps_text}"
                ),
            },
        ]

        return await get_structured_output(response_model=GeneratedArticle, messages=messages)

    async def _extract_claims(self, content: str) -> ArticleExtraction:
        """Pass 2: Extract claims and entity references from generated article."""
        messages = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": content},
        ]

        return await get_structured_output(response_model=ArticleExtraction, messages=messages)
```

**Step 4: Update factory.py**

In `code/shukketsu/agents/factory.py`, add Writer import and registry entry. Add `**kwargs` to `create()`:

```python
from code.shukketsu.agents.writer import Writer

_ROLE_CLASSES: dict[AgentRole, type[BaseAgent]] = {
    AgentRole.RESEARCHER: Researcher,
    AgentRole.WRITER: Writer,
}
```

Change the `create` method signature and body to forward `**kwargs`:

```python
    def create(
        self,
        role: AgentRole,
        *,
        tool_registry: ToolRegistry | None = None,
        **kwargs,
    ) -> BaseAgent:
        """Create an agent configured for the given role.

        Args:
            role: The specialist role to configure.
            tool_registry: Optional pre-configured tool registry.
                If None, a new empty registry is created.
            **kwargs: Role-specific dependencies forwarded to the
                agent constructor (e.g., knowledge_manager for Writer).

        Returns:
            An agent configured with the role's prompt, limits, and class.
        """
        registry = tool_registry if tool_registry is not None else ToolRegistry()
        prompt = _ROLE_PROMPTS.get(role, config.SYSTEM_PROMPT)
        max_iter = _ROLE_MAX_ITERATIONS.get(role, config.AGENT_MAX_ITERATIONS)
        cls = _ROLE_CLASSES.get(role, BaseAgent)

        agent = cls(
            tool_registry=registry,
            role=role,
            max_iterations=max_iter,
            system_prompt=prompt,
            **kwargs,
        )

        logger.info("Created %s agent (%s, max_iter=%d)", role.value, cls.__name__, max_iter)
        return agent
```

**Step 5: Update agents/__init__.py**

Add `Writer` to the imports and `__all__`.

**Step 6: Fix existing test that expects Writer to be BaseAgent**

In `tests/unit/test_researcher.py`, the test `test_factory_other_roles_return_base_agent` (line 419) creates a Writer via factory and asserts `type(writer) is BaseAgent`. This will fail now. Update it to test with a role that still falls back to BaseAgent (like `EDITOR`):

```python
    def test_factory_other_roles_return_base_agent(self) -> None:
        """Non-specialized roles still return BaseAgent."""
        from code.shukketsu.agents.base import BaseAgent
        from code.shukketsu.agents.factory import AgentFactory

        factory = AgentFactory()
        editor = factory.create(AgentRole.EDITOR)
        assert type(editor) is BaseAgent
```

**Step 7: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_writer.py tests/unit/test_researcher.py -v`
Expected: ALL PASS

**Step 8: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v`
Expected: ALL PASS

**Step 9: Commit**

```bash
git add code/shukketsu/agents/writer.py code/shukketsu/agents/factory.py code/shukketsu/agents/__init__.py tests/unit/test_writer.py tests/unit/test_researcher.py
git commit -m "feat(agents): add Writer agent with two-pass generation"
```

---

## Task 9: Linting, Type Checking, Final Verification

**Files:**
- All modified/created files

**Step 1: Run ruff check and format**

Run: `ruff check code/ tests/ --fix && ruff format code/ tests/`
Expected: Clean or auto-fixed

**Step 2: Run mypy**

Run: `python3 -m mypy code/shukketsu/`
Expected: No errors (or only pre-existing ones)

**Step 3: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v`
Expected: ALL PASS, count should be ~515-520

**Step 4: Fix any issues found**

Address any linting, type, or test failures.

**Step 5: Commit fixes if any**

```bash
git add -u
git commit -m "style: apply ruff formatting and fix mypy errors for Phase 2 Step 5"
```

---

## Task 10: Update CLAUDE.md + Memory

**Files:**
- Modify: `CLAUDE.md`
- Modify: `/home/lyro/.claude/projects/-home-lyro-nvidia-workbench-Shukettsu/memory/MEMORY.md`

**Step 1: Update CLAUDE.md**

Update the Phase 2 step 5 line from "DESIGN COMPLETE" to "COMPLETE" with test count. Add `knowledge/manager.py` to the Key files list. Update the project overview to mention `knowledge/manager.py`.

**Step 2: Update MEMORY.md**

Add the new test count line:
```
- **Phase 2 Step 5 complete: N tests** (+ M knowledge_manager + K writer + J schema + ...)
```

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Phase 2 Step 5 completion (N tests)"
```
