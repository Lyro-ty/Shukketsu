# Step 2: Database Foundation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up SQLite with WAL mode, sqlite-vec for vector search, FTS5 for keyword search, and a connection factory — the storage foundation for all RAG and knowledge features.

**Architecture:** Single SQLite file (`data/shukketsu.db`) with two extensions: sqlite-vec (768-dim cosine similarity for semantic search) and FTS5 (BM25 keyword matching). Connection factory applies WAL pragmas and loads extensions. Schema covers Phase 1 tables only: `sources`, `chunks` (with vector and FTS virtual tables), and `articles`. FTS stays in sync with chunks via triggers.

**Tech Stack:** SQLite 3 (stdlib), sqlite-vec, FTS5 (built into SQLite 3.9+)

---

## Task 0: Add DatabaseError Exception

Add a database-specific error class following the established pattern in `resilience/errors.py`.

**Files:**
- Modify: `code/shukketsu/resilience/errors.py` (append after line 58)

**Step 1: Add the class**

Append after the `LLMUnavailableError` class:

```python


class DatabaseError(ShukketsuError):
    """Raised when a database operation fails."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.DB_ERROR)
```

**Step 2: Verify import**

Run: `python3 -c "from code.shukketsu.resilience.errors import DatabaseError; print(DatabaseError.__name__)"`
Expected: `DatabaseError`

**Step 3: Lint**

Run: `ruff check code/shukketsu/resilience/errors.py`
Expected: No errors

**Step 4: Commit**

```bash
git add code/shukketsu/resilience/errors.py
git commit -m "feat: add DatabaseError exception class"
```

---

## Task 1: Write Schema SQL

Create the DDL file for Phase 1 tables. This schema is loaded by `connection.py` (Task 2) and requires sqlite-vec to be loaded first.

**Files:**
- Create: `code/shukketsu/db/schema.sql`

**Step 1: Write the schema**

Create `code/shukketsu/db/schema.sql`:

```sql
-- Shukketsu Database Schema (Phase 1)
--
-- Prerequisites:
--   - sqlite-vec extension loaded (for vec0 virtual table)
--   - FTS5 available (built into SQLite 3.9+)
--
-- Apply via: code.shukketsu.db.connection.init_db()

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT INTO schema_version (version) VALUES (1);

-- ============================================================
-- Ingested Content
-- ============================================================

-- Web pages, guides, forum posts we've ingested
CREATE TABLE sources (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    source_type TEXT,                              -- "guide", "forum", "wiki", etc.
    trust_score REAL NOT NULL DEFAULT 0.3,         -- 0.0–1.0
    fetched_at TEXT,                                -- ISO 8601
    content_hash TEXT,                              -- SHA-256 for dedup
    chunk_count INTEGER NOT NULL DEFAULT 0,
    -- Freshness tracking
    last_checked TEXT,
    check_interval_hours INTEGER NOT NULL DEFAULT 168,  -- 1 week
    content_hash_previous TEXT,
    change_count INTEGER NOT NULL DEFAULT 0,
    is_stale INTEGER NOT NULL DEFAULT 0            -- SQLite has no BOOLEAN
);

-- Text chunks extracted from sources (semantic boundaries, ~400 tokens)
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,                  -- order within source
    metadata_json TEXT                              -- {"author": "...", "topic": "..."}
);

CREATE INDEX idx_chunks_source ON chunks(source_id, chunk_index);

-- 768-dim embeddings for semantic search (nomic-embed-text-v2, cosine distance)
CREATE VIRTUAL TABLE chunks_vec USING vec0(
    embedding float[768] distance_type=cosine
);

-- Full-text search index (BM25 ranking, content-synced with chunks table)
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    content=chunks,
    content_rowid=id
);

-- Triggers to keep FTS5 in sync with chunks table
CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER chunks_fts_update AFTER UPDATE OF content ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

-- ============================================================
-- Knowledge Management
-- ============================================================

-- Wiki article metadata (articles themselves are Markdown in knowledge/)
CREATE TABLE articles (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,                     -- "specs/combat/overview.md"
    title TEXT NOT NULL,
    last_updated TEXT,                              -- ISO 8601
    confidence_score REAL,                         -- 0.0–1.0
    verified_claims INTEGER NOT NULL DEFAULT 0,
    unverified_claims INTEGER NOT NULL DEFAULT 0,
    needs_review INTEGER NOT NULL DEFAULT 0        -- 0 or 1
);
```

**Step 2: Validate SQL syntax**

Run: `python3 -c "import sqlparse; sql = open('code/shukketsu/db/schema.sql').read(); stmts = [s for s in sqlparse.parse(sql) if s.ttype is not sqlparse.tokens.Whitespace]; print(f'{len(stmts)} statements parsed')"`
Expected: Shows statement count without errors

**Step 3: Commit**

```bash
git add code/shukketsu/db/schema.sql
git commit -m "feat: add Phase 1 database schema with FTS5 and sqlite-vec"
```

---

## Task 2: Connection Factory (TDD)

Build `get_connection()` and `init_db()` — the two functions everything else uses.

**Files:**
- Create: `tests/unit/test_db.py`
- Create: `code/shukketsu/db/connection.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_db.py`:

```python
"""Tests for database connection and schema initialization."""

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def db_conn(tmp_path: Path) -> sqlite3.Connection:
    """Create a configured connection without schema."""
    from code.shukketsu.db.connection import get_connection

    conn = get_connection(tmp_path / "test.db")
    yield conn
    conn.close()


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    """Create a fully initialized test database."""
    from code.shukketsu.db.connection import get_connection, init_db

    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    yield conn
    conn.close()


# --- Connection tests ---


class TestGetConnection:
    """Tests for get_connection factory."""

    def test_returns_connection(self, db_conn: sqlite3.Connection) -> None:
        """get_connection should return a sqlite3.Connection."""
        assert isinstance(db_conn, sqlite3.Connection)

    def test_sets_wal_mode(self, db_conn: sqlite3.Connection) -> None:
        """Connection should use WAL journal mode."""
        mode = db_conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_sets_foreign_keys(self, db_conn: sqlite3.Connection) -> None:
        """Foreign key enforcement should be enabled."""
        fk = db_conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

    def test_sets_row_factory(self, db_conn: sqlite3.Connection) -> None:
        """Connection should use sqlite3.Row for dict-like access."""
        assert db_conn.row_factory is sqlite3.Row

    def test_loads_sqlite_vec(self, db_conn: sqlite3.Connection) -> None:
        """sqlite-vec extension should be loaded and functional."""
        version = db_conn.execute("SELECT vec_version()").fetchone()[0]
        assert version  # non-empty version string


# --- Schema tests ---


class TestInitDb:
    """Tests for init_db schema initialization."""

    def test_creates_core_tables(self, db: sqlite3.Connection) -> None:
        """init_db should create sources, chunks, articles, schema_version."""
        tables = _get_tables(db)
        for name in ("sources", "chunks", "articles", "schema_version"):
            assert name in tables, f"Missing table: {name}"

    def test_creates_fts_virtual_table(self, db: sqlite3.Connection) -> None:
        """init_db should create the FTS5 virtual table for keyword search."""
        tables = _get_tables(db)
        assert "chunks_fts" in tables

    def test_creates_vec_virtual_table(self, db: sqlite3.Connection) -> None:
        """init_db should create the vec0 virtual table for vector search."""
        tables = _get_tables(db)
        assert "chunks_vec" in tables

    def test_sets_schema_version(self, db: sqlite3.Connection) -> None:
        """Schema version should be 1 after initialization."""
        version = db.execute("SELECT version FROM schema_version").fetchone()[0]
        assert version == 1

    def test_is_idempotent(self, db: sqlite3.Connection) -> None:
        """Calling init_db twice should not raise or duplicate data."""
        from code.shukketsu.db.connection import init_db

        init_db(db)  # second call — should be a no-op
        count = db.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        assert count == 1

    def test_foreign_key_enforcement(self, db: sqlite3.Connection) -> None:
        """Inserting a chunk with invalid source_id should raise IntegrityError."""
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO chunks (source_id, content, chunk_index) VALUES (999, 'test', 0)"
            )

    def test_fts_sync_on_insert(self, db: sqlite3.Connection) -> None:
        """Inserting into chunks should auto-populate FTS index via trigger."""
        db.execute("INSERT INTO sources (url, title) VALUES ('https://example.com', 'Test')")
        db.execute(
            "INSERT INTO chunks (source_id, content, chunk_index) "
            "VALUES (1, 'combat rogue hit cap guide for TBC', 0)"
        )
        db.commit()

        rows = db.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'combat rogue'").fetchall()
        assert len(rows) == 1

    def test_fts_sync_on_delete(self, db: sqlite3.Connection) -> None:
        """Deleting a chunk should remove it from FTS index via trigger."""
        db.execute("INSERT INTO sources (url, title) VALUES ('https://example.com', 'Test')")
        db.execute("INSERT INTO chunks (source_id, content, chunk_index) VALUES (1, 'unique search term xyzzy', 0)")
        db.commit()

        db.execute("DELETE FROM chunks WHERE id = 1")
        db.commit()

        rows = db.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'xyzzy'").fetchall()
        assert len(rows) == 0

    def test_cascade_delete_removes_chunks(self, db: sqlite3.Connection) -> None:
        """Deleting a source should cascade-delete its chunks."""
        db.execute("INSERT INTO sources (url, title) VALUES ('https://example.com', 'Test')")
        db.execute("INSERT INTO chunks (source_id, content, chunk_index) VALUES (1, 'test content', 0)")
        db.commit()

        db.execute("DELETE FROM sources WHERE id = 1")
        db.commit()

        count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count == 0

    def test_source_url_unique(self, db: sqlite3.Connection) -> None:
        """Inserting duplicate source URLs should raise IntegrityError."""
        db.execute("INSERT INTO sources (url, title) VALUES ('https://example.com', 'First')")
        db.commit()

        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO sources (url, title) VALUES ('https://example.com', 'Duplicate')")


# --- Helpers ---


def _get_tables(conn: sqlite3.Connection) -> set[str]:
    """Get all table names from the database."""
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'code.shukketsu.db.connection'`

**Step 3: Write the implementation**

Create `code/shukketsu/db/connection.py`:

```python
"""SQLite database connection factory with WAL mode and extensions."""

import logging
import sqlite3
from pathlib import Path

import sqlite_vec

from code.shukketsu import config

logger = logging.getLogger(__name__)

_DB_DIR = Path(__file__).parent


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Create a configured SQLite connection with extensions loaded.

    Args:
        db_path: Path to the database file. Defaults to config.DB_PATH.

    Returns:
        A connection with WAL mode, foreign keys, and sqlite-vec enabled.
    """
    path = str(db_path or config.DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _configure(conn)
    _load_extensions(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize database schema from schema.sql. Idempotent.

    Checks schema_version table before applying. Safe to call multiple times.
    """
    try:
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        if version is not None:
            return
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet — need to initialize

    schema_sql = (_DB_DIR / "schema.sql").read_text()
    conn.executescript(schema_sql)
    logger.info("Database schema initialized (version 1)")


def _configure(conn: sqlite3.Connection) -> None:
    """Apply SQLite pragmas for WAL mode, safety, and performance."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")


def _load_extensions(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec extension for vector similarity search."""
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_db.py -v`
Expected: 12 passed

**Step 5: Lint**

Run: `ruff check code/shukketsu/db/connection.py tests/unit/test_db.py`
Expected: No errors

**Step 6: Commit**

```bash
git add code/shukketsu/db/connection.py tests/unit/test_db.py
git commit -m "feat: add database connection factory with WAL mode and tests"
```

---

## Task 3: Update conftest.py Fixture

Replace the manual conftest fixture with one that uses our connection factory. This ensures test databases match production configuration (WAL, extensions, row factory).

**Files:**
- Modify: `tests/conftest.py`

**Step 1: Replace the fixture**

Replace the entire contents of `tests/conftest.py` with:

```python
"""Shared test fixtures for Shukketsu test suite."""

import sqlite3

import pytest


@pytest.fixture
def test_db(tmp_path) -> sqlite3.Connection:
    """Create a fresh test database with full schema.

    Uses the same connection factory as production, ensuring WAL mode,
    sqlite-vec, and foreign keys are configured identically.
    """
    from code.shukketsu.db.connection import get_connection, init_db

    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    yield conn
    conn.close()
```

**Step 2: Verify all tests pass**

Run: `python3 -m pytest tests/unit/ -v`
Expected: 26 passed (14 chat/LLM + 12 database)

**Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "refactor: use connection factory in test_db fixture"
```

---

## Task 4: Final Verification

Run the full verification suite.

**Files:** None (verification only)

**Step 1: Ruff lint**

Run: `ruff check code/ tests/`
Expected: No errors. Fix any issues before proceeding.

**Step 2: Ruff format check**

Run: `ruff format --check code/ tests/`
Expected: All files formatted. If not, run `ruff format code/ tests/` to fix.

**Step 3: Mypy type check**

Run: `python3 -m mypy code/shukketsu/`
Expected: No errors (or only pre-existing warnings from untyped libraries).

**Step 4: Full test suite**

Run: `python3 -m pytest tests/unit/ -v`
Expected: 26 passed (14 chat/LLM + 12 database)

**Step 5: Verify sqlite-vec version**

Run: `python3 -c "from code.shukketsu.db.connection import get_connection; import tempfile, pathlib; c = get_connection(pathlib.Path(tempfile.mkdtemp()) / 'check.db'); print('vec:', c.execute('SELECT vec_version()').fetchone()[0]); print('WAL:', c.execute('PRAGMA journal_mode').fetchone()[0]); c.close()"`
Expected: Shows sqlite-vec version and `wal` mode

**Step 6: Commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: address lint, format, and mypy issues from step 2 verification"
```

---

## Files Created/Modified Summary

| Action | File | Task |
|--------|------|------|
| Modify | `code/shukketsu/resilience/errors.py` | 0 |
| Create | `code/shukketsu/db/schema.sql` | 1 |
| Create | `code/shukketsu/db/connection.py` | 2 |
| Create | `tests/unit/test_db.py` | 2 |
| Modify | `tests/conftest.py` | 3 |

**Total: 3 files created, 2 files modified, 12 unit tests**
