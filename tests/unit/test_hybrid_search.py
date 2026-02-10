"""Tests for hybrid search (vector + FTS5 + RRF)."""

import sqlite3
import struct

from code.shukketsu.rag.search import SearchResult, hybrid_search

EMBEDDING_DIM = 768


def _make_embedding(value: float = 0.1) -> list[float]:
    return [value] * EMBEDDING_DIM


def _pack_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def _seed_source(
    conn: sqlite3.Connection,
    *,
    url: str = "https://example.com/guide",
    title: str = "Test Guide",
    trust_score: float = 0.75,
) -> int:
    """Insert a source and return its ID."""
    cur = conn.execute(
        "INSERT INTO sources (url, title, source_type, trust_score) VALUES (?, ?, 'guide', ?)",
        (url, title, trust_score),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def _seed_chunk(
    conn: sqlite3.Connection,
    source_id: int,
    content: str,
    chunk_index: int,
    embedding_value: float,
) -> int:
    """Insert a chunk with vector embedding and return its ID."""
    cur = conn.execute(
        "INSERT INTO chunks (source_id, content, chunk_index) VALUES (?, ?, ?)",
        (source_id, content, chunk_index),
    )
    chunk_id = cur.lastrowid
    embedding = _make_embedding(embedding_value)
    conn.execute(
        "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
        (chunk_id, _pack_embedding(embedding)),
    )
    conn.commit()
    return chunk_id  # type: ignore[return-value]


class TestHybridSearch:
    """Tests for the hybrid_search function."""

    async def test_empty_db_returns_empty(self, test_db: sqlite3.Connection) -> None:
        query_embedding = _make_embedding(0.1)
        results = await hybrid_search(test_db, "hit cap", query_embedding)
        assert results == []

    async def test_vector_only_match(self, test_db: sqlite3.Connection) -> None:
        """Query embedding is close but keyword doesn't match."""
        src_id = _seed_source(test_db)
        # Content has no keyword overlap with query "zzznotaword"
        _seed_chunk(test_db, src_id, "The hit cap for combat rogues is 9%.", 0, 0.1)
        query_embedding = _make_embedding(0.1)  # Close to chunk embedding
        results = await hybrid_search(test_db, "zzznotaword", query_embedding)
        assert len(results) >= 1
        assert "hit cap" in results[0].content

    async def test_fts_only_match(self, test_db: sqlite3.Connection) -> None:
        """Keyword matches but embedding is distant."""
        src_id = _seed_source(test_db)
        _seed_chunk(test_db, src_id, "DST proc rate is 1 PPM.", 0, 0.1)
        query_embedding = _make_embedding(0.9)  # Far from chunk embedding (0.1)
        results = await hybrid_search(test_db, "DST", query_embedding)
        assert len(results) >= 1
        assert "DST" in results[0].content

    async def test_both_match_ranks_highest(self, test_db: sqlite3.Connection) -> None:
        """Document matching both vector and keyword ranks above single-match docs."""
        src_id = _seed_source(test_db)
        # Chunk A: matches keyword "DST" AND close embedding
        chunk_a = _seed_chunk(test_db, src_id, "DST Dragonspine Trophy is best in slot.", 0, 0.1)
        # Chunk B: matches keyword "DST" but distant embedding
        _seed_chunk(test_db, src_id, "DST has a 1 PPM proc rate.", 1, 0.5)
        # Chunk C: close embedding but no keyword match
        _seed_chunk(test_db, src_id, "The best haste trinket for rogues.", 2, 0.1)

        query_embedding = _make_embedding(0.1)
        results = await hybrid_search(test_db, "DST", query_embedding)
        assert len(results) >= 2
        # Chunk A should be first (matches both)
        assert results[0].chunk_id == chunk_a

    async def test_respects_top_k(self, test_db: sqlite3.Connection) -> None:
        src_id = _seed_source(test_db)
        for i in range(5):
            _seed_chunk(test_db, src_id, f"Chunk {i} about rogues.", i, 0.1 + i * 0.01)
        query_embedding = _make_embedding(0.1)
        results = await hybrid_search(test_db, "rogues", query_embedding, top_k=2)
        assert len(results) == 2

    async def test_returns_search_result_fields(self, test_db: sqlite3.Connection) -> None:
        """SearchResult has all expected fields populated."""
        src_id = _seed_source(test_db, title="Hit Cap Guide", url="https://example.com/hitcap")
        _seed_chunk(test_db, src_id, "Combat rogue hit cap is 9%.", 0, 0.1)
        query_embedding = _make_embedding(0.1)
        results = await hybrid_search(test_db, "hit cap", query_embedding)
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, SearchResult)
        assert r.content == "Combat rogue hit cap is 9%."
        assert r.source_title == "Hit Cap Guide"
        assert r.source_url == "https://example.com/hitcap"
        assert r.trust_score == 0.75
        assert r.rrf_score > 0.0

    async def test_empty_query_text_still_works(self, test_db: sqlite3.Connection) -> None:
        """Empty FTS query degrades to vector-only results."""
        src_id = _seed_source(test_db)
        _seed_chunk(test_db, src_id, "Rogue hit cap guide.", 0, 0.1)
        query_embedding = _make_embedding(0.1)
        results = await hybrid_search(test_db, "", query_embedding)
        assert len(results) >= 1
