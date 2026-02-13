"""Tests for the MemoryManager."""

import json
import sqlite3
import struct
from datetime import UTC, datetime
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

    @patch("code.shukketsu.llm.structured.get_structured_output", new_callable=AsyncMock)
    async def test_extract_stores_session_memory(self, mock_llm: AsyncMock, mem_db: sqlite3.Connection) -> None:
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

    @patch("code.shukketsu.llm.structured.get_structured_output", new_callable=AsyncMock)
    async def test_extract_stores_embedding(self, mock_llm: AsyncMock, mem_db: sqlite3.Connection) -> None:
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

    @patch("code.shukketsu.llm.structured.get_structured_output", new_callable=AsyncMock)
    async def test_extract_error_logged_not_raised(self, mock_llm: AsyncMock, mem_db: sqlite3.Connection) -> None:
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

    @patch("code.shukketsu.llm.structured.get_structured_output", new_callable=AsyncMock)
    async def test_extract_embedding_failure_rolls_back(self, mock_llm: AsyncMock, mem_db: sqlite3.Connection) -> None:
        """If embedding fails after session_memories INSERT, the row must be rolled back.

        Without rollback, a subsequent commit (e.g. from record_strategy) would
        persist the orphaned row that has no corresponding embedding.
        """
        from code.shukketsu.memory.manager import MemoryManager
        from code.shukketsu.memory.models import MemoryExtraction

        mock_llm.return_value = MemoryExtraction(
            summary="test summary",
            key_facts=[],
            entities_mentioned=[],
        )

        # Embedding function that raises after the INSERT
        async def _failing_embed(text: str) -> list[float]:
            raise RuntimeError("Embedding service down")

        mm = MemoryManager(conn=mem_db, embed_fn=_failing_embed)
        await mm.extract_session_memory(query="test", answer="test", trajectory=[])

        # The partial INSERT should have been rolled back
        count = mem_db.execute("SELECT COUNT(*) FROM session_memories").fetchone()[0]
        assert count == 0, "Orphaned session_memories row not rolled back"

        # Verify a subsequent strategy recording doesn't accidentally commit the orphan
        await mm.record_strategy(query="test", tools_used=["rag_search"], quality=0.5)
        count = mem_db.execute("SELECT COUNT(*) FROM session_memories").fetchone()[0]
        assert count == 0, "Strategy commit leaked orphaned session_memories row"


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
        from code.shukketsu.memory.manager import _composite_score

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
