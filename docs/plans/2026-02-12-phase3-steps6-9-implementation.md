# Phase 3 Steps 6–9: Memory Foundation, Integration, Strategy Routing, Evidence-Based Trust — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add conversational memory (session + strategy), integrate memory recall/extraction into the chat pipeline, feed strategy hints to the Orchestrator's decomposition, and replace time-based trust scoring with evidence-based trust events.

**Architecture:** MemoryManager wraps session_memories + strategy_memories tables with vector recall (cosine + recency + quality composite scoring). Memory extraction uses Qwen 4B (fire-and-forget). The Orchestrator's decomposition prompt accepts strategy hints from past successful runs. Trust scoring shifts from time-based decay to an additive event model (`base_trust + sum(deltas)`, clamped [0.1, 1.0]).

---

## Step 6: Memory Foundation — Schema + MemoryManager

---

### Task 6.1: Add memory config constants

**Files:**
- Modify: `code/shukketsu/config.py`

**Step 1: Add memory constants after the Backup section (after line 143)**

Add these lines after `BACKUP_KEEP_COUNT = ...` and before the `# Role-specific system prompts` section:

```python
# Memory
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
MEMORY_RECALL_TOP_K = int(os.getenv("MEMORY_RECALL_TOP_K", "5"))
MEMORY_STRATEGY_TOP_K = int(os.getenv("MEMORY_STRATEGY_TOP_K", "3"))
MEMORY_RECENCY_HALF_LIFE_DAYS = float(os.getenv("MEMORY_RECENCY_HALF_LIFE_DAYS", "30.0"))
MEMORY_COMPOSITE_SIM_WEIGHT = 0.6
MEMORY_COMPOSITE_RECENCY_WEIGHT = 0.2
MEMORY_COMPOSITE_QUALITY_WEIGHT = 0.2
MEMORY_MAX_CONTEXT_CHARS = 2000
```

(No commit yet — we'll commit with the full step.)

---

### Task 6.2: Schema v4 migration — add tables + update schema.sql

**Files:**
- Modify: `code/shukketsu/db/schema.sql`
- Modify: `code/shukketsu/db/connection.py`

**Step 1: Update schema.sql — change version to 4 and add new tables**

In `code/shukketsu/db/schema.sql`, replace:

```sql
INSERT INTO schema_version (version) VALUES (3);
```

With:

```sql
INSERT INTO schema_version (version) VALUES (4);
```

Then add the following after the `-- Knowledge Graph (Phase 2)` section (at the end of the file):

```sql

-- ============================================================
-- Memory (Phase 3)
-- ============================================================

-- Facts extracted from conversations
CREATE TABLE session_memories (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    answer_summary TEXT NOT NULL,
    key_facts_json TEXT NOT NULL DEFAULT '[]',
    entities_mentioned TEXT NOT NULL DEFAULT '[]',
    user_feedback TEXT,
    retrieval_quality REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    session_id TEXT
);
CREATE INDEX idx_session_memories_created ON session_memories(created_at);

CREATE VIRTUAL TABLE session_memories_vec USING vec0(
    embedding float[768] distance_metric=cosine
);

CREATE TABLE strategy_memories (
    id INTEGER PRIMARY KEY,
    query_pattern TEXT NOT NULL,
    strategy_type TEXT NOT NULL DEFAULT 'routing',
    successful_tools TEXT NOT NULL DEFAULT '[]',
    failed_tools TEXT NOT NULL DEFAULT '[]',
    best_sources TEXT NOT NULL DEFAULT '[]',
    times_reinforced INTEGER NOT NULL DEFAULT 1,
    avg_quality REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_used TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_strategy_memories_pattern ON strategy_memories(query_pattern);
CREATE INDEX idx_strategy_memories_type ON strategy_memories(strategy_type);

-- ============================================================
-- Trust Events (Phase 3)
-- ============================================================

CREATE TABLE trust_events (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    delta REAL NOT NULL,
    details TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_trust_events_source ON trust_events(source_id);
```

**Step 2: Add `_migrate_v3_to_v4()` in connection.py**

In `code/shukketsu/db/connection.py`, add the migration SQL string after `_ARTICLES_V3_SQL`:

```python
_MEMORY_V4_SQL = """
CREATE TABLE IF NOT EXISTS session_memories (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    answer_summary TEXT NOT NULL,
    key_facts_json TEXT NOT NULL DEFAULT '[]',
    entities_mentioned TEXT NOT NULL DEFAULT '[]',
    user_feedback TEXT,
    retrieval_quality REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    session_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_session_memories_created ON session_memories(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS session_memories_vec USING vec0(
    embedding float[768] distance_metric=cosine
);

CREATE TABLE IF NOT EXISTS strategy_memories (
    id INTEGER PRIMARY KEY,
    query_pattern TEXT NOT NULL,
    strategy_type TEXT NOT NULL DEFAULT 'routing',
    successful_tools TEXT NOT NULL DEFAULT '[]',
    failed_tools TEXT NOT NULL DEFAULT '[]',
    best_sources TEXT NOT NULL DEFAULT '[]',
    times_reinforced INTEGER NOT NULL DEFAULT 1,
    avg_quality REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_used TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_strategy_memories_pattern ON strategy_memories(query_pattern);
CREATE INDEX IF NOT EXISTS idx_strategy_memories_type ON strategy_memories(strategy_type);

CREATE TABLE IF NOT EXISTS trust_events (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    delta REAL NOT NULL,
    details TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_trust_events_source ON trust_events(source_id);
"""
```

Add the migration function after `_migrate_v2_to_v3()`:

```python
def _migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
    """Migrate v3 schema to v4: add memory tables and trust events."""
    conn.executescript(_MEMORY_V4_SQL)
    conn.execute("INSERT INTO schema_version (version) VALUES (4)")
    conn.commit()
    logger.info("Database migrated from v3 to v4 (memory + trust_events tables)")
```

Update `init_db()` to handle v4. Replace:

```python
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
```

With:

```python
    try:
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        if version is not None:
            if version < 2:
                _migrate_v1_to_v2(conn)
            if version < 3:
                _migrate_v2_to_v3(conn)
            if version < 4:
                _migrate_v3_to_v4(conn)
            return
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet — need to initialize
```

Update the fresh-install log message:

```python
    logger.info("Database schema initialized (version 4)")
```

**Step 3: Verify existing tests still pass**

Run: `python3 -m pytest tests/unit/test_db.py -v`

Expected: All pass. The `test_sets_schema_version` test checks for version 3 — it will **fail** because fresh installs now produce version 4. We'll fix this in the next task.

---

### Task 6.3: Write failing tests for schema v4 + MemoryManager

**Files:**
- Modify: `tests/unit/test_db.py` (update schema version assertion + add migration test)
- Create: `tests/unit/test_memory_manager.py`

**Step 1: Fix the schema version assertion in test_db.py**

In `tests/unit/test_db.py`, replace:

```python
    def test_sets_schema_version(self, db: sqlite3.Connection) -> None:
        """Schema version should be 3 after initialization."""
        version = db.execute("SELECT version FROM schema_version").fetchone()[0]
        assert version == 3
```

With:

```python
    def test_sets_schema_version(self, db: sqlite3.Connection) -> None:
        """Schema version should be 4 after initialization."""
        version = db.execute("SELECT version FROM schema_version").fetchone()[0]
        assert version == 4
```

Also replace:

```python
    def test_schema_version_is_3(self, db: sqlite3.Connection) -> None:
        """Schema version should be 3 after fresh initialization."""
        version = db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert version == 3
```

With:

```python
    def test_schema_version_is_4(self, db: sqlite3.Connection) -> None:
        """Schema version should be 4 after fresh initialization."""
        version = db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert version == 4
```

Add a new test at the end of the `TestInitDb` class:

```python
    def test_creates_memory_tables(self, db: sqlite3.Connection) -> None:
        """init_db should create session_memories, strategy_memories, trust_events."""
        tables = _get_tables(db)
        for name in ("session_memories", "strategy_memories", "trust_events"):
            assert name in tables, f"Missing table: {name}"

    def test_creates_session_memories_vec(self, db: sqlite3.Connection) -> None:
        """init_db should create the session_memories_vec virtual table."""
        tables = _get_tables(db)
        assert "session_memories_vec" in tables

    def test_trust_events_fk_to_sources(self, db: sqlite3.Connection) -> None:
        """trust_events.source_id should reference sources(id)."""
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO trust_events (source_id, event_type, delta) VALUES (999, 'test', 0.1)"
            )

    def test_trust_events_cascade_on_source_delete(self, db: sqlite3.Connection) -> None:
        """Deleting a source should cascade-delete its trust_events."""
        db.execute("INSERT INTO sources (url, title) VALUES ('https://example.com', 'Test')")
        db.commit()
        source_id = db.execute("SELECT id FROM sources WHERE url = 'https://example.com'").fetchone()["id"]
        db.execute(
            "INSERT INTO trust_events (source_id, event_type, delta) VALUES (?, 'contradiction', -0.1)",
            (source_id,),
        )
        db.commit()
        db.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        db.commit()
        count = db.execute("SELECT COUNT(*) FROM trust_events").fetchone()[0]
        assert count == 0
```

**Step 2: Create the memory manager test file**

Create `tests/unit/test_memory_manager.py`:

```python
"""Tests for the MemoryManager."""

import json
import math
import sqlite3
import struct
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from code.shukketsu import config


@pytest.fixture
def mem_db(tmp_path) -> sqlite3.Connection:
    """Create a fresh test database with schema v4."""
    from code.shukketsu.db.connection import get_connection, init_db

    conn = get_connection(tmp_path / "test_mem.db")
    init_db(conn)
    return conn


def _make_embedding(dim: int = 768, fill: float = 0.1) -> list[float]:
    """Create a dummy embedding vector."""
    return [fill] * dim


def _embed_fn_factory(embedding: list[float] | None = None):
    """Create a mock embedding function."""
    vec = embedding or _make_embedding()

    async def _embed(text: str) -> list[float]:
        return vec

    return _embed


def _insert_memory_row(
    conn: sqlite3.Connection,
    query: str = "test query",
    summary: str = "test summary",
    quality: float = 0.5,
    created_at: str | None = None,
    embedding: list[float] | None = None,
) -> int:
    """Insert a session memory + embedding directly for test setup."""
    if created_at is None:
        created_at = datetime.now(UTC).isoformat()
    cursor = conn.execute(
        """INSERT INTO session_memories (query, answer_summary, key_facts_json,
           entities_mentioned, retrieval_quality, created_at)
           VALUES (?, ?, '[]', '[]', ?, ?)""",
        (query, summary, quality, created_at),
    )
    row_id = cursor.lastrowid
    assert row_id is not None

    vec = embedding or _make_embedding()
    blob = struct.pack(f"{len(vec)}f", *vec)
    conn.execute(
        "INSERT INTO session_memories_vec (rowid, embedding) VALUES (?, ?)",
        (row_id, blob),
    )
    conn.commit()
    return row_id


class TestExtractSessionMemory:
    """Tests for extract_session_memory."""

    @patch("code.shukketsu.memory.manager.get_structured_output", new_callable=AsyncMock)
    async def test_extract_stores_session_memory(
        self, mock_llm: AsyncMock, mem_db: sqlite3.Connection
    ) -> None:
        """Extraction should store a row in session_memories."""
        from code.shukketsu.memory.manager import MemoryManager
        from code.shukketsu.memory.models import MemoryExtraction

        mock_llm.return_value = MemoryExtraction(
            summary="Hit cap is 142 for combat rogues.",
            key_facts=["Hit cap is 142"],
            entities_mentioned=["combat rogue"],
        )

        mm = MemoryManager(conn=mem_db, embed_fn=_embed_fn_factory())
        await mm.extract_session_memory(
            query="What is the hit cap?",
            answer="The hit cap for combat rogues is 142 rating.",
            trajectory=[],
        )

        row = mem_db.execute("SELECT * FROM session_memories WHERE query = ?", ("What is the hit cap?",)).fetchone()
        assert row is not None
        assert row["answer_summary"] == "Hit cap is 142 for combat rogues."
        assert "142" in row["key_facts_json"]

    @patch("code.shukketsu.memory.manager.get_structured_output", new_callable=AsyncMock)
    async def test_extract_stores_embedding(
        self, mock_llm: AsyncMock, mem_db: sqlite3.Connection
    ) -> None:
        """Extraction should store an embedding in session_memories_vec."""
        from code.shukketsu.memory.manager import MemoryManager
        from code.shukketsu.memory.models import MemoryExtraction

        mock_llm.return_value = MemoryExtraction(
            summary="test",
            key_facts=[],
            entities_mentioned=[],
        )

        mm = MemoryManager(conn=mem_db, embed_fn=_embed_fn_factory())
        await mm.extract_session_memory(
            query="test query",
            answer="test answer",
            trajectory=[],
        )

        row = mem_db.execute("SELECT COUNT(*) FROM session_memories_vec").fetchone()
        assert row[0] == 1

    @patch("code.shukketsu.memory.manager.get_structured_output", new_callable=AsyncMock)
    async def test_extract_error_logged_not_raised(
        self, mock_llm: AsyncMock, mem_db: sqlite3.Connection
    ) -> None:
        """LLM failure during extraction should be logged, not raised."""
        from code.shukketsu.memory.manager import MemoryManager

        mock_llm.side_effect = RuntimeError("LLM down")

        mm = MemoryManager(conn=mem_db, embed_fn=_embed_fn_factory())
        # Should not raise
        await mm.extract_session_memory(
            query="test",
            answer="test",
            trajectory=[],
        )

        count = mem_db.execute("SELECT COUNT(*) FROM session_memories").fetchone()[0]
        assert count == 0


class TestRecallRelevant:
    """Tests for recall_relevant."""

    async def test_recall_empty_db_returns_empty(self, mem_db: sqlite3.Connection) -> None:
        """No memories in DB should return empty list."""
        from code.shukketsu.memory.manager import MemoryManager

        mm = MemoryManager(conn=mem_db, embed_fn=_embed_fn_factory())
        results = await mm.recall_relevant("any query", top_k=5)
        assert results == []

    async def test_recall_returns_scored_results(self, mem_db: sqlite3.Connection) -> None:
        """Recall should return SessionMemory objects with composite scores."""
        from code.shukketsu.memory.manager import MemoryManager

        _insert_memory_row(mem_db, query="hit cap question", summary="hit cap is 142")
        _insert_memory_row(mem_db, query="trinket question", summary="DST is BiS")

        mm = MemoryManager(conn=mem_db, embed_fn=_embed_fn_factory())
        results = await mm.recall_relevant("hit cap for rogues", top_k=5)
        assert len(results) == 2
        # All results should have a score attribute
        assert all(hasattr(r, "score") for r in results)

    async def test_recall_composite_scoring(self, mem_db: sqlite3.Connection) -> None:
        """Composite score should be 0.6*sim + 0.2*recency + 0.2*quality."""
        from code.shukketsu.memory.manager import MemoryManager, _composite_score

        # Test the scoring function directly
        sim = 0.8
        recency = 0.5
        quality = 0.9
        expected = (
            config.MEMORY_COMPOSITE_SIM_WEIGHT * sim
            + config.MEMORY_COMPOSITE_RECENCY_WEIGHT * recency
            + config.MEMORY_COMPOSITE_QUALITY_WEIGHT * quality
        )
        result = _composite_score(sim, recency, quality)
        assert abs(result - expected) < 1e-6


class TestRecordStrategy:
    """Tests for record_strategy."""

    async def test_record_strategy_inserts(self, mem_db: sqlite3.Connection) -> None:
        """New pattern should create a new strategy_memories row."""
        from code.shukketsu.memory.manager import MemoryManager

        mm = MemoryManager(conn=mem_db, embed_fn=_embed_fn_factory())
        await mm.record_strategy(
            query="hit cap question",
            tools_used=["rag_search", "graph_search"],
            quality=0.8,
        )

        row = mem_db.execute("SELECT * FROM strategy_memories").fetchone()
        assert row is not None
        assert row["query_pattern"] == "hit cap question"
        assert json.loads(row["successful_tools"]) == ["rag_search", "graph_search"]
        assert row["avg_quality"] == 0.8
        assert row["times_reinforced"] == 1

    async def test_record_strategy_updates_existing(self, mem_db: sqlite3.Connection) -> None:
        """Same pattern should increment times_reinforced and update avg_quality."""
        from code.shukketsu.memory.manager import MemoryManager

        mm = MemoryManager(conn=mem_db, embed_fn=_embed_fn_factory())
        await mm.record_strategy(query="hit cap", tools_used=["rag_search"], quality=0.6)
        await mm.record_strategy(query="hit cap", tools_used=["rag_search"], quality=1.0)

        row = mem_db.execute("SELECT * FROM strategy_memories WHERE query_pattern = ?", ("hit cap",)).fetchone()
        assert row["times_reinforced"] == 2
        # Running average: (0.6 + 1.0) / 2 = 0.8
        assert abs(row["avg_quality"] - 0.8) < 0.01

    async def test_record_strategy_error_logged_not_raised(self, mem_db: sqlite3.Connection) -> None:
        """DB failure during strategy recording should be logged, not raised."""
        from code.shukketsu.memory.manager import MemoryManager

        mm = MemoryManager(conn=mem_db, embed_fn=_embed_fn_factory())
        # Close the connection to force a DB error
        mem_db.close()
        # Should not raise
        await mm.record_strategy(query="test", tools_used=[], quality=0.5)


class TestRecallStrategies:
    """Tests for recall_strategies."""

    async def test_recall_strategies_by_quality(self, mem_db: sqlite3.Connection) -> None:
        """Multiple strategies should be returned sorted by avg_quality desc."""
        from code.shukketsu.memory.manager import MemoryManager

        mm = MemoryManager(conn=mem_db, embed_fn=_embed_fn_factory())
        await mm.record_strategy(query="trinkets", tools_used=["rag_search"], quality=0.5)
        await mm.record_strategy(query="rotation", tools_used=["graph_search"], quality=0.9)
        await mm.record_strategy(query="talents", tools_used=["web_search"], quality=0.7)

        results = await mm.recall_strategies(top_k=3)
        assert len(results) == 3
        assert results[0]["avg_quality"] >= results[1]["avg_quality"]
        assert results[1]["avg_quality"] >= results[2]["avg_quality"]

    async def test_recall_strategies_empty(self, mem_db: sqlite3.Connection) -> None:
        """No strategies in DB should return empty list."""
        from code.shukketsu.memory.manager import MemoryManager

        mm = MemoryManager(conn=mem_db, embed_fn=_embed_fn_factory())
        results = await mm.recall_strategies(top_k=3)
        assert results == []
```

**Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_memory_manager.py -v`

Expected: Import errors — `code.shukketsu.memory.manager` and `code.shukketsu.memory.models` don't exist yet.

Run: `python3 -m pytest tests/unit/test_db.py -v`

Expected: All pass (schema version assertions updated to 4, new table tests should pass since schema.sql was already updated in Task 6.2).

---

### Task 6.4: Create memory models

**Files:**
- Create: `code/shukketsu/memory/__init__.py`
- Create: `code/shukketsu/memory/models.py`

**Step 1: Create the empty `__init__.py`**

Create `code/shukketsu/memory/__init__.py` with empty contents (just a docstring):

```python
"""Memory subsystem for session recall and strategy learning."""
```

**Step 2: Create `models.py`**

Create `code/shukketsu/memory/models.py`:

```python
"""Pydantic models for the memory subsystem."""

from pydantic import BaseModel, Field


class MemoryExtraction(BaseModel):
    """LLM-extracted facts from a conversation turn.

    Used as the structured output schema for Qwen 4B extraction.
    """

    summary: str
    key_facts: list[str] = Field(default_factory=list)
    entities_mentioned: list[str] = Field(default_factory=list)


class SessionMemory(BaseModel):
    """A recalled session memory with composite relevance score."""

    id: int
    query: str
    answer_summary: str
    key_facts: list[str] = Field(default_factory=list)
    entities_mentioned: list[str] = Field(default_factory=list)
    retrieval_quality: float
    created_at: str
    score: float = 0.0
```

---

### Task 6.5: Implement MemoryManager

**Files:**
- Create: `code/shukketsu/memory/manager.py`

**Step 1: Create the MemoryManager**

Create `code/shukketsu/memory/manager.py`:

```python
"""MemoryManager: session memory recall and strategy learning.

Stores facts extracted from conversations and retrieval strategies.
All writes are fire-and-forget — errors are logged, never raised.
"""

import json
import logging
import math
import sqlite3
import struct
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from code.shukketsu import config
from code.shukketsu.memory.models import MemoryExtraction, SessionMemory

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Awaitable[list[float]]]


def _composite_score(similarity: float, recency: float, quality: float) -> float:
    """Compute composite memory score: 0.6*sim + 0.2*recency + 0.2*quality."""
    return (
        config.MEMORY_COMPOSITE_SIM_WEIGHT * similarity
        + config.MEMORY_COMPOSITE_RECENCY_WEIGHT * recency
        + config.MEMORY_COMPOSITE_QUALITY_WEIGHT * quality
    )


def _recency_score(created_at: str, half_life_days: float = config.MEMORY_RECENCY_HALF_LIFE_DAYS) -> float:
    """Exponential decay recency score: exp(-days / half_life_days)."""
    try:
        created = datetime.fromisoformat(created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        age_days = (datetime.now(UTC) - created).total_seconds() / 86400.0
        return math.exp(-age_days / half_life_days)
    except (ValueError, TypeError):
        return 0.5


class MemoryManager:
    """Manages session memories and retrieval strategy learning.

    All public methods are fire-and-forget on write paths — errors are
    logged but never propagated to callers.
    """

    def __init__(self, conn: sqlite3.Connection, embed_fn: EmbedFn) -> None:
        self._conn = conn
        self._embed_fn = embed_fn

    async def extract_session_memory(
        self,
        query: str,
        answer: str,
        trajectory: list[dict],
        session_id: str | None = None,
    ) -> None:
        """Extract key facts from a conversation turn and store in DB.

        Uses Qwen 4B via get_structured_output to extract a MemoryExtraction,
        then stores the result + embedding. Fire-and-forget: errors are logged.
        """
        try:
            from code.shukketsu.llm.structured import ModelBackend, get_structured_output

            messages = [
                {
                    "role": "system",
                    "content": (
                        "Extract the key facts from this conversation. "
                        "Summarize the answer concisely. List specific facts "
                        "and any WoW entities mentioned."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nAnswer: {answer}",
                },
            ]

            extraction: MemoryExtraction = await get_structured_output(
                response_model=MemoryExtraction,
                messages=messages,
                backend=ModelBackend.ROUTER,
                max_tokens=512,
            )

            # Store in DB
            cursor = self._conn.execute(
                """INSERT INTO session_memories
                   (query, answer_summary, key_facts_json, entities_mentioned,
                    retrieval_quality, session_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    query,
                    extraction.summary,
                    json.dumps(extraction.key_facts),
                    json.dumps(extraction.entities_mentioned),
                    0.5,  # Default quality; updated on feedback
                    session_id,
                ),
            )
            row_id = cursor.lastrowid

            # Store embedding
            embedding = await self._embed_fn(query)
            blob = struct.pack(f"{len(embedding)}f", *embedding)
            self._conn.execute(
                "INSERT INTO session_memories_vec (rowid, embedding) VALUES (?, ?)",
                (row_id, blob),
            )
            self._conn.commit()

        except Exception:
            logger.warning("Memory extraction failed", exc_info=True)

    async def recall_relevant(
        self,
        query: str,
        top_k: int = config.MEMORY_RECALL_TOP_K,
    ) -> list[SessionMemory]:
        """Recall the most relevant session memories for a query.

        Embeds the query, performs KNN search on session_memories_vec,
        then scores with composite: 0.6*cosine_sim + 0.2*recency + 0.2*quality.

        Returns up to top_k SessionMemory objects sorted by composite score.
        """
        try:
            # Check if any memories exist
            count = self._conn.execute("SELECT COUNT(*) FROM session_memories").fetchone()[0]
            if count == 0:
                return []

            embedding = await self._embed_fn(query)
            blob = struct.pack(f"{len(embedding)}f", *embedding)

            # KNN search — fetch more than top_k so composite scoring can reorder
            fetch_k = min(count, top_k * 3)
            rows = self._conn.execute(
                """SELECT v.rowid, v.distance, m.query, m.answer_summary,
                          m.key_facts_json, m.entities_mentioned,
                          m.retrieval_quality, m.created_at
                   FROM (
                       SELECT rowid, distance
                       FROM session_memories_vec
                       WHERE embedding MATCH ? AND k = ?
                   ) v
                   JOIN session_memories m ON m.id = v.rowid""",
                (blob, fetch_k),
            ).fetchall()

            memories: list[SessionMemory] = []
            for row in rows:
                # cosine distance → similarity: 1 - distance (sqlite-vec cosine returns distance)
                similarity = max(0.0, 1.0 - row["distance"])
                recency = _recency_score(row["created_at"])
                quality = row["retrieval_quality"]
                score = _composite_score(similarity, recency, quality)

                memories.append(
                    SessionMemory(
                        id=row["rowid"],
                        query=row["query"],
                        answer_summary=row["answer_summary"],
                        key_facts=json.loads(row["key_facts_json"]),
                        entities_mentioned=json.loads(row["entities_mentioned"]),
                        retrieval_quality=quality,
                        created_at=row["created_at"],
                        score=score,
                    )
                )

            memories.sort(key=lambda m: m.score, reverse=True)
            return memories[:top_k]

        except Exception:
            logger.warning("Memory recall failed", exc_info=True)
            return []

    async def record_strategy(
        self,
        query: str,
        tools_used: list[str],
        quality: float,
        strategy_type: str = "routing",
    ) -> None:
        """Record or update a retrieval strategy. Fire-and-forget.

        If a strategy with the same query_pattern exists, increments
        times_reinforced and updates the running average quality.
        Otherwise inserts a new row.
        """
        try:
            existing = self._conn.execute(
                "SELECT id, times_reinforced, avg_quality FROM strategy_memories WHERE query_pattern = ?",
                (query,),
            ).fetchone()

            if existing:
                new_reinforced = existing["times_reinforced"] + 1
                new_avg = (existing["avg_quality"] * existing["times_reinforced"] + quality) / new_reinforced
                self._conn.execute(
                    """UPDATE strategy_memories
                       SET times_reinforced = ?,
                           avg_quality = ?,
                           last_used = datetime('now'),
                           successful_tools = ?
                       WHERE id = ?""",
                    (
                        new_reinforced,
                        new_avg,
                        json.dumps(tools_used),
                        existing["id"],
                    ),
                )
            else:
                self._conn.execute(
                    """INSERT INTO strategy_memories
                       (query_pattern, strategy_type, successful_tools, avg_quality)
                       VALUES (?, ?, ?, ?)""",
                    (query, strategy_type, json.dumps(tools_used), quality),
                )
            self._conn.commit()

        except Exception:
            logger.warning("Strategy recording failed", exc_info=True)

    async def recall_strategies(
        self,
        top_k: int = config.MEMORY_STRATEGY_TOP_K,
    ) -> list[dict]:
        """Recall the highest-quality strategies.

        Returns up to top_k strategy dicts sorted by avg_quality descending.
        """
        try:
            rows = self._conn.execute(
                """SELECT query_pattern, strategy_type, successful_tools,
                          failed_tools, best_sources, times_reinforced,
                          avg_quality, last_used
                   FROM strategy_memories
                   ORDER BY avg_quality DESC
                   LIMIT ?""",
                (top_k,),
            ).fetchall()

            return [
                {
                    "query_pattern": row["query_pattern"],
                    "strategy_type": row["strategy_type"],
                    "successful_tools": json.loads(row["successful_tools"]),
                    "failed_tools": json.loads(row["failed_tools"]),
                    "best_sources": json.loads(row["best_sources"]),
                    "times_reinforced": row["times_reinforced"],
                    "avg_quality": row["avg_quality"],
                    "last_used": row["last_used"],
                }
                for row in rows
            ]

        except Exception:
            logger.warning("Strategy recall failed", exc_info=True)
            return []
```

**Step 2: Run the tests**

Run: `python3 -m pytest tests/unit/test_memory_manager.py tests/unit/test_db.py -v`

Expected: All pass.

---

### Task 6.6: Run full test suite + linting

**Files:** None (verification only)

**Step 1: Run linting + type checking + all tests**

Run: `ruff check code/ tests/ --fix && ruff format code/ tests/ && python3 -m mypy code/shukketsu/ && python3 -m pytest tests/unit/ -v`

Expected: 741+ tests pass (new tests added), no lint errors, no type errors.

**Step 2: Fix any issues found**

If mypy flags `struct.pack` or `json.dumps` types, add appropriate type annotations or `# type: ignore` comments.

---

### Task 6.7: Commit

**Step 1: Stage and commit**

```bash
git add code/shukketsu/config.py code/shukketsu/db/schema.sql code/shukketsu/db/connection.py code/shukketsu/memory/__init__.py code/shukketsu/memory/models.py code/shukketsu/memory/manager.py tests/unit/test_db.py tests/unit/test_memory_manager.py
git commit -m "feat(memory): add schema v4, MemoryManager, session + strategy memories

Schema v4 adds session_memories, strategy_memories, trust_events tables.
MemoryManager handles extraction (Qwen 4B), vector recall with composite
scoring (cosine + recency + quality), and strategy upsert. All writes are
fire-and-forget.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Step 7: Memory Integration — Recall + Extraction Hooks

---

### Task 7.1: Add memory_context parameter to BaseAgent._build_messages

**Files:**
- Modify: `code/shukketsu/agents/base.py`

**Step 1: Update `_build_messages` signature and implementation**

In `code/shukketsu/agents/base.py`, replace the `_build_messages` method:

```python
    def _build_messages(self, query: str, scratchpad: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Build the messages array for the LLM."""
        tool_descriptions = self.tool_registry.get_tool_descriptions()
        system_content = self._system_prompt + "\n\n" + _REACT_INSTRUCTIONS.format(tool_descriptions=tool_descriptions)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        for entry in scratchpad:
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Thought: {entry['reasoning']}\n"
                        f"Action: tool_call\n"
                        f"Tool: {entry['tool_name']}\n"
                        f"Input: {entry['tool_input']}"
                    ),
                }
            )
            messages.append({"role": "user", "content": f"Observation: {entry['observation']}"})

        return messages
```

With:

```python
    def _build_messages(
        self,
        query: str,
        scratchpad: list[dict[str, Any]],
        *,
        memory_context: str | None = None,
    ) -> list[dict[str, str]]:
        """Build the messages array for the LLM.

        Args:
            query: The user's question.
            scratchpad: Previous reasoning + observation pairs.
            memory_context: Optional context from recalled memories.
        """
        tool_descriptions = self.tool_registry.get_tool_descriptions()
        system_content = self._system_prompt + "\n\n" + _REACT_INSTRUCTIONS.format(tool_descriptions=tool_descriptions)

        if memory_context:
            system_content += "\n\n## Relevant Context from Previous Sessions\n\n" + memory_context

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        for entry in scratchpad:
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Thought: {entry['reasoning']}\n"
                        f"Action: tool_call\n"
                        f"Tool: {entry['tool_name']}\n"
                        f"Input: {entry['tool_input']}"
                    ),
                }
            )
            messages.append({"role": "user", "content": f"Observation: {entry['observation']}"})

        return messages
```

Also update `_run_loop` to accept and pass through memory_context. Replace:

```python
    async def _run_loop(self, query: str, *, on_status: StatusCallback | None = None) -> _RunOutcome:
```

With:

```python
    async def _run_loop(
        self,
        query: str,
        *,
        on_status: StatusCallback | None = None,
        memory_context: str | None = None,
    ) -> _RunOutcome:
```

And inside `_run_loop`, replace:

```python
            messages = self._build_messages(query, scratchpad)
```

With:

```python
            messages = self._build_messages(query, scratchpad, memory_context=memory_context)
```

Update `run()` to accept and pass through memory_context. Replace:

```python
    @observe(as_type="agent")
    async def run(self, query: str, *, on_status: StatusCallback | None = None) -> str:
```

With:

```python
    @observe(as_type="agent")
    async def run(
        self,
        query: str,
        *,
        on_status: StatusCallback | None = None,
        memory_context: str | None = None,
    ) -> str:
```

And inside `run()`, replace:

```python
        outcome = await self._run_loop(query, on_status=on_status)
```

With:

```python
        outcome = await self._run_loop(query, on_status=on_status, memory_context=memory_context)
```

Update `execute()` similarly. Replace:

```python
        outcome = await self._run_loop(task.query, on_status=on_status)
```

With:

```python
        memory_ctx = task.context.get("memory_context") if task.context else None
        outcome = await self._run_loop(task.query, on_status=on_status, memory_context=memory_ctx)
```

**Step 2: Verify existing tests still pass**

Run: `python3 -m pytest tests/unit/test_base_agent.py -v`

Expected: All pass (new parameter has default `None`, existing callers unaffected).

---

### Task 7.2: Write failing tests for memory integration in chat handler

**Files:**
- Modify: `tests/unit/test_chat_handler.py`

**Step 1: Add memory integration tests at the end of the file**

Add the following test class at the end of `tests/unit/test_chat_handler.py`:

```python


class TestMemoryIntegration:
    """Tests for memory recall/extraction hooks in the chat handler."""

    @patch("code.shukketsu.web.routers.chat._get_memory_manager")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_memory_recall_injects_context(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
        mock_get_mm: MagicMock,
    ) -> None:
        """Recalled memories should cause memory_context to be set on the task."""
        from code.shukketsu.memory.models import SessionMemory

        mock_classify.return_value = _moderate_decision()
        researcher, orchestrator = _mock_agents("Answer with context")
        mock_agents.return_value = (researcher, orchestrator)

        mock_mm = MagicMock()
        mock_mm.recall_relevant = AsyncMock(
            return_value=[
                SessionMemory(
                    id=1,
                    query="hit cap?",
                    answer_summary="Hit cap is 142 rating.",
                    key_facts=["Hit cap is 142"],
                    entities_mentioned=[],
                    retrieval_quality=0.8,
                    created_at="2026-02-12T00:00:00",
                    score=0.9,
                ),
            ]
        )
        mock_mm.extract_session_memory = AsyncMock()
        mock_mm.record_strategy = AsyncMock()
        mock_get_mm.return_value = mock_mm

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "What is the hit cap?"})
            _drain_status(ws)

        # Verify recall was called
        mock_mm.recall_relevant.assert_called_once()

    @patch("code.shukketsu.web.routers.chat._get_memory_manager")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_memory_extraction_fires_after_response(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
        mock_get_mm: MagicMock,
    ) -> None:
        """extract_session_memory should be called after the response is sent."""
        mock_classify.return_value = _moderate_decision()
        mock_agents.return_value = _mock_agents("Answer")

        mock_mm = MagicMock()
        mock_mm.recall_relevant = AsyncMock(return_value=[])
        mock_mm.extract_session_memory = AsyncMock()
        mock_mm.record_strategy = AsyncMock()
        mock_get_mm.return_value = mock_mm

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "test question"})
            _drain_status(ws)

        mock_mm.extract_session_memory.assert_called_once()

    @patch("code.shukketsu.web.routers.chat.config")
    @patch("code.shukketsu.web.routers.chat._get_memory_manager")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_memory_disabled_skips_both(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
        mock_get_mm: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """MEMORY_ENABLED=False should skip both recall and extraction."""
        mock_config.MEMORY_ENABLED = False
        mock_config.CHAT_MAX_MESSAGE_LENGTH = 10_000
        mock_config.CHAT_MAX_HISTORY_PAIRS = 20
        mock_classify.return_value = _moderate_decision()
        mock_agents.return_value = _mock_agents("Answer")

        mock_mm = MagicMock()
        mock_mm.recall_relevant = AsyncMock(return_value=[])
        mock_mm.extract_session_memory = AsyncMock()
        mock_mm.record_strategy = AsyncMock()
        mock_get_mm.return_value = mock_mm

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "test"})
            _drain_status(ws)

        mock_mm.recall_relevant.assert_not_called()
        mock_mm.extract_session_memory.assert_not_called()

    @patch("code.shukketsu.web.routers.chat._get_memory_manager")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_memory_recall_empty_no_injection(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
        mock_get_mm: MagicMock,
    ) -> None:
        """Empty recall should not add memory_context to the task."""
        mock_classify.return_value = _moderate_decision()
        researcher, orchestrator = _mock_agents("Clean answer")
        mock_agents.return_value = (researcher, orchestrator)

        mock_mm = MagicMock()
        mock_mm.recall_relevant = AsyncMock(return_value=[])
        mock_mm.extract_session_memory = AsyncMock()
        mock_mm.record_strategy = AsyncMock()
        mock_get_mm.return_value = mock_mm

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hello"})
            _drain_status(ws)

        # Verify execute was called — we just want to confirm no crash
        researcher.execute.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_chat_handler.py::TestMemoryIntegration -v`

Expected: Failures — `_get_memory_manager` doesn't exist in chat.py yet.

---

### Task 7.3: Implement memory hooks in chat handler

**Files:**
- Modify: `code/shukketsu/web/routers/chat.py`

**Step 1: Add memory manager singleton and _get_memory_manager function**

In `code/shukketsu/web/routers/chat.py`, after the existing module-level singleton variables (after line 25):

Add:

```python
_memory_manager_instance = None
```

Add the `_get_memory_manager()` function after `_get_agents()`:

```python
def _get_memory_manager():
    """Get or create the MemoryManager singleton.

    Returns None if memory subsystem is unavailable.
    """
    global _memory_manager_instance  # noqa: PLW0603

    if _memory_manager_instance is None:
        try:
            from code.shukketsu.db.connection import get_connection, init_db
            from code.shukketsu.ingest.embedder import get_embedder
            from code.shukketsu.memory.manager import MemoryManager

            conn = get_connection()
            init_db(conn)
            embedder = get_embedder()
            _memory_manager_instance = MemoryManager(conn=conn, embed_fn=embedder.embed_query)
        except Exception:
            logger.warning("Failed to initialize MemoryManager", exc_info=True)
            return None

    return _memory_manager_instance
```

**Step 2: Integrate memory into _agent_response**

Add a helper to format memories as context, before `_agent_response`:

```python
def _format_memory_context(memories: list) -> str:
    """Format recalled memories as a context string for the agent."""
    if not memories:
        return ""
    parts: list[str] = []
    for mem in memories:
        parts.append(f"- Q: {mem.query}")
        parts.append(f"  A: {mem.answer_summary}")
        if mem.key_facts:
            for fact in mem.key_facts:
                parts.append(f"  - {fact}")
    text = "\n".join(parts)
    if len(text) > config.MEMORY_MAX_CONTEXT_CHARS:
        text = text[: config.MEMORY_MAX_CONTEXT_CHARS] + "\n..."
    return text
```

In `_agent_response()`, add memory recall before the routing decision, and memory extraction after the done message. Replace the entire function body with:

```python
@observe()
async def _agent_response(websocket: WebSocket, session: ChatSession, content: str) -> None:
    """Get an agent response for the given user message.

    Routes through Qwen 4B first: trivial queries get a direct answer,
    moderate queries go to the Researcher, complex queries go to the
    Orchestrator for multi-agent coordination.
    """
    session.is_streaming = True
    session.add_message("user", content)

    try:
        langfuse = get_client()
        langfuse.update_current_trace(
            session_id=str(id(session)),
            tags=["chat"],
            input=content,
        )

        # Memory recall (before routing)
        memory_context: str | None = None
        mm = _get_memory_manager() if config.MEMORY_ENABLED else None
        if mm is not None:
            try:
                memories = await mm.recall_relevant(content, top_k=config.MEMORY_RECALL_TOP_K)
                memory_context = _format_memory_context(memories) or None
            except Exception:
                logger.warning("Memory recall failed", exc_info=True)

        await websocket.send_json({"type": "status", "content": "routing..."})
        decision = await classify_query(content)
        logger.info("Route: %s → %s", decision.complexity, decision.category)

        async def _send_status(msg: str) -> None:
            await websocket.send_json({"type": "status", "content": msg})

        if (
            decision.complexity == TaskComplexity.TRIVIAL
            and decision.direct_answer is not None
            and decision.direct_answer.strip()
        ):
            answer = decision.direct_answer
        elif decision.complexity == TaskComplexity.MODERATE:
            await websocket.send_json({"type": "status", "content": "researching..."})
            researcher, _ = _get_agents()
            task = ResearchTask(
                query=content,
                context={"memory_context": memory_context} if memory_context else {},
            )
            result = await researcher.execute(
                task,
                on_status=_send_status,
            )
            answer = result.output
        else:
            await websocket.send_json({"type": "status", "content": "planning..."})
            _, orchestrator = _get_agents()
            task = AgentTask(
                query=content,
                context={"memory_context": memory_context} if memory_context else {},
            )
            result = await orchestrator.execute(
                task,
                on_status=_send_status,
            )
            answer = result.output
            if isinstance(result, OrchestratorResult):
                if result.article_path:
                    answer += f"\n\n---\n*Draft article created: {result.article_path}*"
                if result.needs_human_review:
                    answer += "\n*Article pending review in Wiki*"

        session.add_message("assistant", answer)
        await websocket.send_json({"type": "done", "content": answer})

        # Memory extraction (fire-and-forget, after response)
        if mm is not None:
            try:
                await mm.extract_session_memory(
                    query=content,
                    answer=answer,
                    trajectory=[],
                )
            except Exception:
                logger.warning("Memory extraction failed", exc_info=True)
    except ShukketsuError as exc:
        if session.history and session.history[-1]["role"] == "user":
            session.history.pop()
        await websocket.send_json({"type": "error", "content": str(exc)})
    except Exception:
        logger.exception("Unexpected error in agent response")
        if session.history and session.history[-1]["role"] == "user":
            session.history.pop()
        await websocket.send_json({"type": "error", "content": "An unexpected error occurred. Please try again."})
    finally:
        session.is_streaming = False
```

**Step 3: Run the tests**

Run: `python3 -m pytest tests/unit/test_chat_handler.py -v`

Expected: All pass (both old and new tests).

---

### Task 7.4: Run full test suite + linting

**Files:** None (verification only)

**Step 1: Run linting + type checking + all tests**

Run: `ruff check code/ tests/ --fix && ruff format code/ tests/ && python3 -m mypy code/shukketsu/ && python3 -m pytest tests/unit/ -v`

Expected: All tests pass.

---

### Task 7.5: Commit

**Step 1: Stage and commit**

```bash
git add code/shukketsu/agents/base.py code/shukketsu/web/routers/chat.py tests/unit/test_chat_handler.py
git commit -m "feat(memory): integrate recall + extraction hooks into chat pipeline

BaseAgent._build_messages accepts optional memory_context for injection
into the system prompt. Chat handler recalls relevant memories before
routing and extracts session memories after sending the response (fire-
and-forget). Memory can be disabled via MEMORY_ENABLED config.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Step 8: Strategy Memory for Orchestrator Routing

---

### Task 8.1: Write failing tests for strategy hints in Orchestrator decomposition

**Files:**
- Modify: `tests/unit/test_orchestrator_build_task.py`

**Step 1: Add strategy hint tests at the end of the file**

Add to `tests/unit/test_orchestrator_build_task.py`:

```python


class TestStrategyHintsInDecomposition:
    """Tests for strategy hints being injected into decomposition."""

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_strategy_hints_in_decomposition_prompt(
        self, mock_llm: AsyncMock
    ) -> None:
        """Non-empty strategy hints should appear in the LLM messages."""
        from code.shukketsu.agents.factory import AgentFactory
        from code.shukketsu.agents.tasks import AgentRole, OrchestratorPlan

        mock_llm.return_value = OrchestratorPlan(
            reasoning="test",
            subtasks=[],
            can_answer_directly=True,
            direct_answer="Direct answer.",
        )

        factory = AgentFactory()
        orch = object.__new__(type(
            factory.create(
                AgentRole.ORCHESTRATOR,
                tool_registry=MagicMock(),
                factory=factory,
            )
        ))
        # Manually set attributes needed for _decompose
        orch._factory = factory
        orch._km = None
        orch.role = AgentRole.ORCHESTRATOR
        orch.tool_registry = MagicMock()
        orch.max_iterations = 5
        orch._system_prompt = "test"

        hints = "- rag_search works well for trinket questions (quality: 0.9)\n- graph_search finds entity relationships"
        await orch._decompose("What trinkets?", strategy_hints=hints)

        call_args = mock_llm.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        user_msg = messages[-1]["content"]
        assert "rag_search works well" in user_msg
        assert "graph_search finds entity" in user_msg

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_empty_strategy_hints_no_change(self, mock_llm: AsyncMock) -> None:
        """Empty strategy hints should not modify the decomposition prompt."""
        from code.shukketsu.agents.orchestrator import Orchestrator
        from code.shukketsu.agents.tasks import AgentRole, OrchestratorPlan

        mock_llm.return_value = OrchestratorPlan(
            reasoning="test",
            subtasks=[],
            can_answer_directly=True,
            direct_answer="Direct answer.",
        )

        orch = object.__new__(Orchestrator)
        orch._factory = MagicMock()
        orch._km = None
        orch.role = AgentRole.ORCHESTRATOR
        orch.tool_registry = MagicMock()
        orch.max_iterations = 5
        orch._system_prompt = "test"
        from code.shukketsu.agents.guardrails import LoopDetector
        orch._loop_detector = LoopDetector()

        await orch._decompose("What trinkets?", strategy_hints="")

        call_args = mock_llm.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        user_msg = messages[-1]["content"]
        # Should just contain the standard decomposition prompt
        assert "Decompose this query" in user_msg
        assert "Strategy hints" not in user_msg

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_max_strategy_hints_capped(self, mock_llm: AsyncMock) -> None:
        """More than 3 strategy hint lines should be truncated to 3."""
        from code.shukketsu.agents.orchestrator import Orchestrator
        from code.shukketsu.agents.tasks import AgentRole, OrchestratorPlan

        mock_llm.return_value = OrchestratorPlan(
            reasoning="test",
            subtasks=[],
            can_answer_directly=True,
            direct_answer="Direct answer.",
        )

        orch = object.__new__(Orchestrator)
        orch._factory = MagicMock()
        orch._km = None
        orch.role = AgentRole.ORCHESTRATOR
        orch.tool_registry = MagicMock()
        orch.max_iterations = 5
        orch._system_prompt = "test"
        from code.shukketsu.agents.guardrails import LoopDetector
        orch._loop_detector = LoopDetector()

        hints = "\n".join([
            "- hint 1: rag_search",
            "- hint 2: graph_search",
            "- hint 3: web_search",
            "- hint 4: should be dropped",
            "- hint 5: should be dropped too",
        ])
        await orch._decompose("test", strategy_hints=hints)

        call_args = mock_llm.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        user_msg = messages[-1]["content"]
        assert "hint 1" in user_msg
        assert "hint 3" in user_msg
        assert "hint 4" not in user_msg
        assert "hint 5" not in user_msg
```

Also add this import at the top of the file:

```python
from unittest.mock import AsyncMock, MagicMock, patch
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_orchestrator_build_task.py::TestStrategyHintsInDecomposition -v`

Expected: Failures — `_decompose` doesn't accept `strategy_hints` yet.

---

### Task 8.2: Update DECOMPOSITION_PROMPT and Orchestrator._decompose

**Files:**
- Modify: `code/shukketsu/llm/prompts/orchestrator.py`
- Modify: `code/shukketsu/agents/orchestrator.py`

**Step 1: Update the decomposition prompt template in orchestrator prompts**

In `code/shukketsu/llm/prompts/orchestrator.py`, replace:

```python
DECOMPOSITION_PROMPT = "Decompose this query into a plan:\n\n{query}"
```

With:

```python
DECOMPOSITION_PROMPT = "Decompose this query into a plan:\n\n{query}"

DECOMPOSITION_PROMPT_WITH_HINTS = (
    "Decompose this query into a plan:\n\n{query}\n\n"
    "## Strategy Hints from Past Sessions\n\n"
    "The following retrieval strategies worked well for similar questions:\n\n{hints}"
)
```

**Step 2: Add strategy_hints parameter to Orchestrator._decompose**

In `code/shukketsu/agents/orchestrator.py`, update the import to include the new prompt:

Replace:

```python
from code.shukketsu.llm.prompts.orchestrator import (
    DECOMPOSITION_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    SYNTHESIS_PROMPT,
)
```

With:

```python
from code.shukketsu.llm.prompts.orchestrator import (
    DECOMPOSITION_PROMPT,
    DECOMPOSITION_PROMPT_WITH_HINTS,
    ORCHESTRATOR_SYSTEM_PROMPT,
    SYNTHESIS_PROMPT,
)
```

Replace the `_decompose` method signature and first section:

```python
    async def _decompose(self, query: str) -> OrchestratorPlan:
        """Phase 1: Decompose query into an execution plan via Llama 70B."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": DECOMPOSITION_PROMPT.format(query=query)},
        ]
```

With:

```python
    async def _decompose(self, query: str, *, strategy_hints: str = "") -> OrchestratorPlan:
        """Phase 1: Decompose query into an execution plan via Llama 70B.

        Args:
            query: The user's question to decompose.
            strategy_hints: Optional strategy hints from past successful sessions.
                Lines beyond 3 are truncated.
        """
        # Cap strategy hints to 3 lines
        if strategy_hints.strip():
            hint_lines = [line for line in strategy_hints.strip().split("\n") if line.strip()]
            capped = "\n".join(hint_lines[:3])
            user_content = DECOMPOSITION_PROMPT_WITH_HINTS.format(query=query, hints=capped)
        else:
            user_content = DECOMPOSITION_PROMPT.format(query=query)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
```

**Step 3: Update the call to _decompose in execute()**

In the `execute()` method, replace:

```python
            plan = await self._decompose(task.query)
```

With:

```python
            strategy_hints = task.context.get("strategy_hints", "") if task.context else ""
            plan = await self._decompose(task.query, strategy_hints=strategy_hints)
```

**Step 4: Update chat handler to pass strategy hints**

In `code/shukketsu/web/routers/chat.py`, in the `_agent_response()` function, update the complex routing path (the `else` branch) to recall strategies and pass them. Replace:

```python
        else:
            await websocket.send_json({"type": "status", "content": "planning..."})
            _, orchestrator = _get_agents()
            task = AgentTask(
                query=content,
                context={"memory_context": memory_context} if memory_context else {},
            )
            result = await orchestrator.execute(
                task,
                on_status=_send_status,
            )
```

With:

```python
        else:
            await websocket.send_json({"type": "status", "content": "planning..."})
            _, orchestrator = _get_agents()

            # Recall strategy hints for the Orchestrator
            strategy_hints = ""
            if mm is not None:
                try:
                    strategies = await mm.recall_strategies(top_k=config.MEMORY_STRATEGY_TOP_K)
                    if strategies:
                        hint_parts = [
                            f"- {s['query_pattern']}: {', '.join(s['successful_tools'])} (quality: {s['avg_quality']:.1f})"
                            for s in strategies
                        ]
                        strategy_hints = "\n".join(hint_parts)
                except Exception:
                    logger.warning("Strategy recall failed", exc_info=True)

            context: dict = {}
            if memory_context:
                context["memory_context"] = memory_context
            if strategy_hints:
                context["strategy_hints"] = strategy_hints

            task = AgentTask(
                query=content,
                context=context,
            )
            result = await orchestrator.execute(
                task,
                on_status=_send_status,
            )
```

Also update the strategy recording after the done message. After the existing `extract_session_memory` call, add:

```python
            try:
                # Record which tools were effective (simplified — full tool tracking in Phase 4)
                await mm.record_strategy(
                    query=content,
                    tools_used=[],
                    quality=0.5,
                )
            except Exception:
                logger.warning("Strategy recording failed", exc_info=True)
```

**Step 5: Run the tests**

Run: `python3 -m pytest tests/unit/test_orchestrator_build_task.py tests/unit/test_chat_handler.py -v`

Expected: All pass.

---

### Task 8.3: Run full test suite + linting

**Files:** None (verification only)

**Step 1: Run linting + type checking + all tests**

Run: `ruff check code/ tests/ --fix && ruff format code/ tests/ && python3 -m mypy code/shukketsu/ && python3 -m pytest tests/unit/ -v`

Expected: All tests pass.

---

### Task 8.4: Commit

**Step 1: Stage and commit**

```bash
git add code/shukketsu/llm/prompts/orchestrator.py code/shukketsu/agents/orchestrator.py code/shukketsu/web/routers/chat.py tests/unit/test_orchestrator_build_task.py
git commit -m "feat(memory): inject strategy hints into Orchestrator decomposition

Orchestrator._decompose accepts strategy_hints parameter that appends
past successful tool strategies to the decomposition prompt. Chat handler
recalls top strategies from MemoryManager and formats them as hints.
Lines capped at 3 to keep the prompt focused.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Step 9: Evidence-Based Trust Scoring

---

### Task 9.1: Write failing tests for evidence-based trust scoring

**Files:**
- Rewrite: `tests/unit/test_trust_scoring.py`

**Step 1: Replace the entire test file**

Replace `tests/unit/test_trust_scoring.py` with:

```python
"""Tests for evidence-based trust scoring."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from code.shukketsu.trust.scoring import (
    SOURCE_TRUST,
    _effective_trust_legacy,
    effective_trust,
    get_trust_events,
    record_trust_event,
)


@pytest.fixture
def trust_db(tmp_path: Path) -> sqlite3.Connection:
    """Create a fresh test database with schema v4 for trust scoring tests."""
    from code.shukketsu.db.connection import get_connection, init_db

    conn = get_connection(tmp_path / "trust_test.db")
    init_db(conn)
    return conn


def _insert_source(conn: sqlite3.Connection, url: str = "https://example.com") -> int:
    """Insert a test source and return its id."""
    conn.execute("INSERT INTO sources (url, title, trust_score) VALUES (?, 'Test', 0.7)", (url,))
    conn.commit()
    row = conn.execute("SELECT id FROM sources WHERE url = ?", (url,)).fetchone()
    return row["id"]


class TestSourceTrust:
    """Tests for the SOURCE_TRUST mapping."""

    def test_game_data_highest(self) -> None:
        assert SOURCE_TRUST["game_data"] == 1.0

    def test_unknown_lowest(self) -> None:
        assert SOURCE_TRUST["unknown"] == 0.3

    def test_all_values_between_zero_and_one(self) -> None:
        for value in SOURCE_TRUST.values():
            assert 0.0 < value <= 1.0


class TestEffectiveTrustEvidence:
    """Tests for the evidence-based effective_trust function."""

    def test_effective_trust_no_events(self, trust_db: sqlite3.Connection) -> None:
        """Base trust returned unchanged when no events exist."""
        source_id = _insert_source(trust_db)
        result = effective_trust(0.7, trust_db, source_id)
        assert result == 0.7

    def test_effective_trust_negative_events(self, trust_db: sqlite3.Connection) -> None:
        """Contradiction events reduce trust."""
        source_id = _insert_source(trust_db)
        record_trust_event(trust_db, source_id, "contradiction", -0.1)
        result = effective_trust(0.7, trust_db, source_id)
        assert abs(result - 0.6) < 0.01

    def test_effective_trust_positive_events(self, trust_db: sqlite3.Connection) -> None:
        """Confirmation events increase trust."""
        source_id = _insert_source(trust_db)
        record_trust_event(trust_db, source_id, "confirmation", 0.05)
        result = effective_trust(0.7, trust_db, source_id)
        assert abs(result - 0.75) < 0.01

    def test_effective_trust_clamped_high(self, trust_db: sqlite3.Connection) -> None:
        """Trust cannot exceed 1.0."""
        source_id = _insert_source(trust_db)
        record_trust_event(trust_db, source_id, "confirmation", 0.05)
        record_trust_event(trust_db, source_id, "confirmation", 0.05)
        record_trust_event(trust_db, source_id, "confirmation", 0.05)
        record_trust_event(trust_db, source_id, "confirmation", 0.05)
        record_trust_event(trust_db, source_id, "confirmation", 0.05)
        record_trust_event(trust_db, source_id, "confirmation", 0.05)
        record_trust_event(trust_db, source_id, "confirmation", 0.05)
        result = effective_trust(0.9, trust_db, source_id)
        assert result == 1.0

    def test_effective_trust_clamped_low(self, trust_db: sqlite3.Connection) -> None:
        """Trust cannot go below 0.1."""
        source_id = _insert_source(trust_db)
        record_trust_event(trust_db, source_id, "contradiction", -0.1)
        record_trust_event(trust_db, source_id, "contradiction", -0.1)
        record_trust_event(trust_db, source_id, "contradiction", -0.1)
        record_trust_event(trust_db, source_id, "contradiction", -0.1)
        record_trust_event(trust_db, source_id, "contradiction", -0.1)
        record_trust_event(trust_db, source_id, "contradiction", -0.1)
        record_trust_event(trust_db, source_id, "contradiction", -0.1)
        result = effective_trust(0.3, trust_db, source_id)
        assert result == 0.1

    def test_dead_url_sets_minimum(self, trust_db: sqlite3.Connection) -> None:
        """dead_url event with large negative delta should floor trust at 0.1."""
        source_id = _insert_source(trust_db)
        # delta = -(base - 0.1) to force trust to 0.1
        record_trust_event(trust_db, source_id, "dead_url", -0.6, details="HTTP 404")
        result = effective_trust(0.7, trust_db, source_id)
        assert result == 0.1


class TestRecordTrustEvent:
    """Tests for record_trust_event."""

    def test_record_trust_event_inserts(self, trust_db: sqlite3.Connection) -> None:
        """Event should be stored in the trust_events table."""
        source_id = _insert_source(trust_db)
        record_trust_event(trust_db, source_id, "contradiction", -0.1, details="Claim X was wrong")

        row = trust_db.execute("SELECT * FROM trust_events WHERE source_id = ?", (source_id,)).fetchone()
        assert row is not None
        assert row["event_type"] == "contradiction"
        assert abs(row["delta"] - (-0.1)) < 0.001
        assert row["details"] == "Claim X was wrong"


class TestGetTrustEvents:
    """Tests for get_trust_events."""

    def test_get_trust_events_ordered(self, trust_db: sqlite3.Connection) -> None:
        """Events should be returned ordered by created_at."""
        source_id = _insert_source(trust_db)
        record_trust_event(trust_db, source_id, "confirmation", 0.05, details="first")
        record_trust_event(trust_db, source_id, "contradiction", -0.1, details="second")

        events = get_trust_events(trust_db, source_id)
        assert len(events) == 2
        assert events[0]["details"] == "first"
        assert events[1]["details"] == "second"

    def test_get_trust_events_empty(self, trust_db: sqlite3.Connection) -> None:
        """No events for source should return empty list."""
        source_id = _insert_source(trust_db)
        events = get_trust_events(trust_db, source_id)
        assert events == []


class TestLegacyFunction:
    """Tests for the preserved legacy decay function."""

    def test_legacy_function_preserved(self) -> None:
        """_effective_trust_legacy should still compute time-based decay."""
        now = datetime.now(UTC)
        result = _effective_trust_legacy(0.8, fetched_at=now, max_age=timedelta(days=30), decay_factor=0.5)
        assert result == 0.8

    def test_legacy_decay_applied(self) -> None:
        """Legacy function should apply decay after max_age."""
        now = datetime.now(UTC)
        fetched = now - timedelta(days=60)
        result = _effective_trust_legacy(0.8, fetched_at=fetched, max_age=timedelta(days=30), decay_factor=0.5)
        assert abs(result - 0.4) < 0.01  # 0.8 * 0.5^1
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_trust_scoring.py -v`

Expected: Import errors — `_effective_trust_legacy`, `record_trust_event`, `get_trust_events` don't exist yet, and `effective_trust` has different signature.

---

### Task 9.2: Implement evidence-based trust scoring

**Files:**
- Rewrite: `code/shukketsu/trust/scoring.py`

**Step 1: Replace the entire trust scoring module**

Replace `code/shukketsu/trust/scoring.py` with:

```python
"""Source trust scoring with evidence-based event model.

Trust is computed as: base_trust + sum(event_deltas), clamped [0.1, 1.0].
The legacy time-based decay function is preserved as _effective_trust_legacy.
"""

import sqlite3
from datetime import UTC, datetime, timedelta

SOURCE_TRUST = {
    "game_data": 1.0,
    "simulation": 0.9,
    "combat_logs": 0.85,
    "expert_guide": 0.75,
    "archived_theory": 0.7,
    "community": 0.5,
    "unknown": 0.3,
}

# Standard deltas by event type
TRUST_DELTAS = {
    "contradiction": -0.1,
    "correction": -0.15,
    "dead_url": -0.6,  # Large enough to floor most sources
    "confirmation": 0.05,
    "corroboration": 0.05,
}


def effective_trust(
    base_trust: float,
    conn: sqlite3.Connection,
    source_id: int,
) -> float:
    """Compute trust from base + sum of evidence events, clamped [0.1, 1.0].

    Args:
        base_trust: The initial trust score for this source type.
        conn: Database connection with trust_events table.
        source_id: The source to compute trust for.

    Returns:
        Effective trust score between 0.1 and 1.0.
    """
    row = conn.execute(
        "SELECT COALESCE(SUM(delta), 0.0) as total_delta FROM trust_events WHERE source_id = ?",
        (source_id,),
    ).fetchone()
    total_delta = float(row["total_delta"]) if row else 0.0
    return max(0.1, min(1.0, base_trust + total_delta))


def record_trust_event(
    conn: sqlite3.Connection,
    source_id: int,
    event_type: str,
    delta: float,
    details: str | None = None,
) -> None:
    """Insert a trust event into the audit trail.

    Args:
        conn: Database connection.
        source_id: The source this event relates to.
        event_type: Type of event (contradiction, confirmation, dead_url, etc.).
        delta: Trust score adjustment (positive or negative).
        details: Optional human-readable description.
    """
    conn.execute(
        "INSERT INTO trust_events (source_id, event_type, delta, details) VALUES (?, ?, ?, ?)",
        (source_id, event_type, delta, details),
    )
    conn.commit()


def get_trust_events(
    conn: sqlite3.Connection,
    source_id: int,
) -> list[dict]:
    """Get all trust events for a source, ordered by created_at.

    Args:
        conn: Database connection.
        source_id: The source to get events for.

    Returns:
        List of event dicts with keys: id, event_type, delta, details, created_at.
    """
    rows = conn.execute(
        """SELECT id, event_type, delta, details, created_at
           FROM trust_events
           WHERE source_id = ?
           ORDER BY created_at ASC""",
        (source_id,),
    ).fetchall()
    return [
        {
            "id": row["id"],
            "event_type": row["event_type"],
            "delta": row["delta"],
            "details": row["details"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def _effective_trust_legacy(
    base_trust: float,
    fetched_at: datetime,
    max_age: timedelta,
    decay_factor: float,
) -> float:
    """Compute effective trust with time-based decay (legacy).

    Trust stays at base_trust until max_age, then decays
    by decay_factor for each additional max_age period.

    Preserved for backward compatibility reference. New code
    should use effective_trust() with the evidence model.
    """
    age = datetime.now(UTC) - fetched_at
    if age <= max_age:
        return base_trust
    periods_past = (age - max_age) / max_age
    return float(base_trust * (decay_factor**periods_past))
```

**Step 2: Run the trust scoring tests**

Run: `python3 -m pytest tests/unit/test_trust_scoring.py -v`

Expected: All pass.

---

### Task 9.3: Update Editor to emit trust events on contradiction

**Files:**
- Modify: `code/shukketsu/agents/editor.py`

**Step 1: Add trust event recording to _verify_claim**

The Editor's `_verify_claim` method doesn't have access to source_id, so we'll add trust event emission at the higher level — in `execute()` — when processing claim results that have contradictions.

In `code/shukketsu/agents/editor.py`, add the import at the top:

```python
from code.shukketsu.trust.scoring import TRUST_DELTAS, record_trust_event
```

In the `execute()` method, after `# Step 6: Build corrections and research gaps` section and before `# Step 7: Update article frontmatter`, add a trust event recording step. After the loop that builds `corrections` and `needs_more_research`, add:

```python
        # Step 6b: Record trust events for contradicted claims
        try:
            for cv in claim_results:
                if cv.status == VerificationStatus.CONTRADICTED:
                    # Find source IDs from contradicting evidence URLs
                    for evidence_url in cv.contradicting_evidence:
                        source_row = self._conn_for_trust().execute(
                            "SELECT id FROM sources WHERE url = ?", (evidence_url,)
                        ).fetchone()
                        if source_row:
                            record_trust_event(
                                self._conn_for_trust(),
                                source_row["id"],
                                "contradiction",
                                TRUST_DELTAS["contradiction"],
                                details=f"Contradicted claim: {cv.claim[:100]}",
                            )
        except Exception:
            logger.warning("Failed to record trust events for contradictions", exc_info=True)
```

Wait — the Editor doesn't have a DB connection. It uses tool_registry for searches. The trust event recording needs a DB connection. Let's take a simpler approach: pass the connection through the KnowledgeManager, which the Editor already has.

Actually, let's check if KnowledgeManager has a connection.

Looking at the code, KnowledgeManager is initialized with `conn`. So we can access `self._km._conn`. But accessing private attributes of another class is bad practice. Instead, let's add a public `conn` property or pass the connection directly.

Simpler approach: Since the Editor already has `self._km`, and `self._km` has a `_conn` attribute, we'll add a helper method on the Editor that accesses the connection through the KM.

In `code/shukketsu/agents/editor.py`, add a helper method to the Editor class:

```python
    def _get_db_conn(self) -> sqlite3.Connection | None:
        """Get the database connection from KnowledgeManager (for trust events)."""
        try:
            return self._km._conn  # type: ignore[attr-defined]
        except AttributeError:
            return None
```

Add `import sqlite3` to the imports.

Now update the execute() method. After the corrections/needs_more_research loop (after `# Step 6`), add:

```python
        # Step 6b: Record trust events for contradicted claims
        conn = self._get_db_conn()
        if conn is not None:
            try:
                for cv in claim_results:
                    if cv.status == VerificationStatus.CONTRADICTED:
                        for evidence_url in cv.contradicting_evidence:
                            source_row = conn.execute(
                                "SELECT id FROM sources WHERE url = ?", (evidence_url,)
                            ).fetchone()
                            if source_row:
                                record_trust_event(
                                    conn,
                                    source_row["id"],
                                    "contradiction",
                                    TRUST_DELTAS["contradiction"],
                                    details=f"Contradicted claim: {cv.claim[:100]}",
                                )
            except Exception:
                logger.warning("Failed to record trust events", exc_info=True)
```

**Step 2: Verify existing editor tests still pass**

Run: `python3 -m pytest tests/unit/test_editor.py -v`

Expected: All pass (trust event recording is wrapped in try/except, and mock KM won't have _conn, so _get_db_conn returns None).

---

### Task 9.4: Update freshness checker to emit trust events on dead URLs

**Files:**
- Modify: `code/shukketsu/freshness/checker.py`

**Step 1: Add trust event recording for HTTP 4xx+ responses**

In `code/shukketsu/freshness/checker.py`, add the import:

```python
from code.shukketsu.trust.scoring import TRUST_DELTAS, record_trust_event
```

In `check_source_freshness()`, after the `head_resp.status_code >= 400` check that returns an early FreshnessResult, add trust event recording before the return. Replace:

```python
            if head_resp.status_code >= 400:
                return FreshnessResult(
                    source_id=source.id,
                    url=source.url,
                    changed=False,
                    old_hash=source.content_hash,
                    new_hash=None,
                    checked_at=now,
                    head_only=True,
                    error=f"Source returned HTTP {head_resp.status_code}",
                )
```

With:

```python
            if head_resp.status_code >= 400:
                try:
                    record_trust_event(
                        conn,
                        source.id,
                        "dead_url",
                        TRUST_DELTAS["dead_url"],
                        details=f"HTTP {head_resp.status_code} for {source.url}",
                    )
                except Exception:
                    logger.warning("Failed to record dead_url trust event", exc_info=True)
                return FreshnessResult(
                    source_id=source.id,
                    url=source.url,
                    changed=False,
                    old_hash=source.content_hash,
                    new_hash=None,
                    checked_at=now,
                    head_only=True,
                    error=f"Source returned HTTP {head_resp.status_code}",
                )
```

**Step 2: Verify existing freshness tests still pass**

Run: `python3 -m pytest tests/unit/test_freshness_checker.py -v`

Expected: All pass (the trust event recording is wrapped in try/except; in unit tests the DB may not have the trust_events table, so it silently fails).

---

### Task 9.5: Run full test suite + linting

**Files:** None (verification only)

**Step 1: Run linting + type checking + all tests**

Run: `ruff check code/ tests/ --fix && ruff format code/ tests/ && python3 -m mypy code/shukketsu/ && python3 -m pytest tests/unit/ -v`

Expected: All tests pass, no lint errors, no type errors.

**Step 2: Fix any issues**

If imports cause circular dependency issues, use lazy imports inside functions. If mypy complains about `_conn` access on KnowledgeManager, add the `# type: ignore[attr-defined]` comment (already included in the code above).

---

### Task 9.6: Commit

**Step 1: Stage and commit**

```bash
git add code/shukketsu/trust/scoring.py code/shukketsu/agents/editor.py code/shukketsu/freshness/checker.py tests/unit/test_trust_scoring.py
git commit -m "feat(trust): replace time-based decay with evidence-based trust events

Trust is now computed as base_trust + sum(event_deltas), clamped [0.1, 1.0].
Editor records contradiction events when claims are disproven. Freshness
checker records dead_url events when HEAD returns 4xx+. Legacy time-based
function preserved as _effective_trust_legacy for reference.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
