"""Tests for the rag_search tool."""

import sqlite3
import struct

from code.shukketsu.tools.knowledge.search import RagSearchTool

EMBEDDING_DIM = 768


def _make_embedding(value: float = 0.1) -> list[float]:
    return [value] * EMBEDDING_DIM


def _pack_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def _seed_test_data(conn: sqlite3.Connection, num_chunks: int = 3) -> None:
    """Insert test source, chunks, and vectors."""
    conn.execute(
        "INSERT INTO sources (url, title, source_type, trust_score) "
        "VALUES ('https://example.com/guide', 'Hit Cap Guide', 'guide', 0.75)"
    )
    for i in range(num_chunks):
        conn.execute(
            "INSERT INTO chunks (source_id, content, chunk_index) VALUES (1, ?, ?)",
            (f"Chunk {i}: The hit cap for combat rogues is {9 + i}%.", i),
        )
    conn.commit()

    for i in range(num_chunks):
        embedding = _make_embedding(0.1 + i * 0.01)
        conn.execute(
            "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
            (i + 1, _pack_embedding(embedding)),
        )
    conn.commit()


async def _mock_embed(text: str) -> list[float]:
    return _make_embedding(0.1)


class TestRagSearchTool:
    """Tests for RagSearchTool."""

    def test_has_correct_name(self) -> None:
        tool = RagSearchTool(conn=None, embed_fn=_mock_embed)  # type: ignore[arg-type]
        assert tool.name == "rag_search"

    def test_has_description(self) -> None:
        tool = RagSearchTool(conn=None, embed_fn=_mock_embed)  # type: ignore[arg-type]
        assert len(tool.description) > 0

    def test_has_parameters_schema(self) -> None:
        tool = RagSearchTool(conn=None, embed_fn=_mock_embed)  # type: ignore[arg-type]
        assert "query" in tool.parameters_schema
        assert "top_k" in tool.parameters_schema

    async def test_empty_db_returns_no_results(self, test_db: sqlite3.Connection) -> None:
        tool = RagSearchTool(conn=test_db, embed_fn=_mock_embed)
        result = await tool.execute({"query": "hit cap"})
        assert "no relevant" in result.lower()

    async def test_returns_matching_chunks(self, test_db: sqlite3.Connection) -> None:
        _seed_test_data(test_db, num_chunks=3)
        tool = RagSearchTool(conn=test_db, embed_fn=_mock_embed)
        result = await tool.execute({"query": "hit cap"})
        assert "Chunk" in result
        assert "Hit Cap Guide" in result

    async def test_respects_top_k(self, test_db: sqlite3.Connection) -> None:
        _seed_test_data(test_db, num_chunks=3)
        tool = RagSearchTool(conn=test_db, embed_fn=_mock_embed)
        result = await tool.execute({"query": "hit cap", "top_k": 1})
        assert "[1]" in result
        assert "[2]" not in result

    async def test_default_top_k(self, test_db: sqlite3.Connection) -> None:
        _seed_test_data(test_db, num_chunks=3)
        tool = RagSearchTool(conn=test_db, embed_fn=_mock_embed)
        result = await tool.execute({"query": "hit cap"})
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    async def test_includes_source_metadata(self, test_db: sqlite3.Connection) -> None:
        _seed_test_data(test_db, num_chunks=1)
        tool = RagSearchTool(conn=test_db, embed_fn=_mock_embed)
        result = await tool.execute({"query": "hit cap"})
        assert "https://example.com/guide" in result
        assert "Hit Cap Guide" in result

    async def test_embedding_failure_falls_back_to_fts(self, test_db: sqlite3.Connection) -> None:
        """When embedding fails, tool should fall back to FTS5-only keyword search."""
        from code.shukketsu.resilience.errors import EmbeddingError

        _seed_test_data(test_db, num_chunks=2)

        async def failing_embed(text: str) -> list[float]:
            raise EmbeddingError("Ollama embed down")

        tool = RagSearchTool(conn=test_db, embed_fn=failing_embed)
        result = await tool.execute({"query": "hit cap"})
        # Should get FTS5 results or a graceful "unavailable" message
        assert "keyword search only" in result.lower() or "unavailable" in result.lower()

    async def test_embedding_failure_empty_db_returns_message(self, test_db: sqlite3.Connection) -> None:
        """When embedding fails on empty DB, tool should return informative message."""
        from code.shukketsu.resilience.errors import EmbeddingError

        async def failing_embed(text: str) -> list[float]:
            raise EmbeddingError("Ollama embed down")

        tool = RagSearchTool(conn=test_db, embed_fn=failing_embed)
        result = await tool.execute({"query": "hit cap"})
        assert "unavailable" in result.lower()
