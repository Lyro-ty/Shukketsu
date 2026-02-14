"""Hybrid search combining vector similarity and FTS5 keyword matching."""

import logging
import sqlite3
import struct
from dataclasses import dataclass

from code.shukketsu import config
from code.shukketsu.rag.fusion import escape_fts_query

logger = logging.getLogger(__name__)

# SQL: Single-query RRF fusion via CTEs + FULL OUTER JOIN
# Ranks are 0-based (row_number() - 1). RRF constant k=60.
_HYBRID_SEARCH_SQL = """
WITH vec_matches AS (
    SELECT rowid, row_number() OVER (ORDER BY distance) - 1 AS rank
    FROM chunks_vec
    WHERE embedding MATCH ? AND k = ?
),
fts_matches AS (
    SELECT rowid, row_number() OVER (ORDER BY rank) - 1 AS rank
    FROM chunks_fts
    WHERE chunks_fts MATCH ?
    ORDER BY rank
    LIMIT ?
)
SELECT
    c.id AS chunk_id,
    c.content,
    c.chunk_index,
    s.title AS source_title,
    s.url AS source_url,
    MAX(0.0, MIN(1.0,
        s.trust_score + COALESCE(
            (SELECT SUM(delta) FROM trust_events te WHERE te.source_id = s.id), 0.0
        )
    )) AS trust_score,
    (
        coalesce(1.0 / (60 + f.rank), 0.0)
        + coalesce(1.0 / (60 + v.rank), 0.0)
    ) AS rrf_score
FROM fts_matches f
FULL OUTER JOIN vec_matches v ON v.rowid = f.rowid
JOIN chunks c ON c.id = coalesce(f.rowid, v.rowid)
JOIN sources s ON s.id = c.source_id AND s.is_stale = 0
ORDER BY rrf_score DESC
LIMIT ?
"""

# Fallback: vector-only search when FTS query is empty
_VECTOR_ONLY_SQL = """
SELECT
    c.id AS chunk_id,
    c.content,
    c.chunk_index,
    s.title AS source_title,
    s.url AS source_url,
    MAX(0.0, MIN(1.0,
        s.trust_score + COALESCE(
            (SELECT SUM(delta) FROM trust_events te WHERE te.source_id = s.id), 0.0
        )
    )) AS trust_score,
    (1.0 / (60 + row_number() OVER (ORDER BY v.distance) - 1)) AS rrf_score
FROM (
    SELECT rowid, distance
    FROM chunks_vec
    WHERE embedding MATCH ? AND k = ?
) v
JOIN chunks c ON c.id = v.rowid
JOIN sources s ON s.id = c.source_id AND s.is_stale = 0
ORDER BY rrf_score DESC
LIMIT ?
"""


@dataclass(frozen=True)
class SearchResult:
    """A single hybrid search result with metadata."""

    chunk_id: int
    content: str
    source_title: str
    source_url: str
    trust_score: float
    rrf_score: float
    rerank_score: float | None = None


async def hybrid_search(
    conn: sqlite3.Connection,
    query_text: str,
    query_embedding: list[float],
    top_k: int = config.RAG_SEARCH_TOP_K,
    fetch_k: int = config.RAG_SEARCH_FETCH_K,
) -> list[SearchResult]:
    """Run hybrid vector + FTS5 search, merged via RRF.

    Args:
        conn: SQLite connection with sqlite-vec and FTS5 initialized.
        query_text: Raw search query for keyword matching.
        query_embedding: 768-dim embedding for vector similarity.
        top_k: Number of final results to return.
        fetch_k: Candidates to fetch from each source before fusion.

    Returns:
        List of SearchResult ordered by RRF score (descending).
    """
    if not query_embedding:
        return []
    embedding_blob = struct.pack(f"{len(query_embedding)}f", *query_embedding)
    fts_query = escape_fts_query(query_text)

    if not fts_query:
        # No keyword query — fall back to vector-only
        rows = conn.execute(_VECTOR_ONLY_SQL, (embedding_blob, fetch_k, top_k)).fetchall()
    else:
        rows = conn.execute(
            _HYBRID_SEARCH_SQL,
            (embedding_blob, fetch_k, fts_query, fetch_k, top_k),
        ).fetchall()

    return [
        SearchResult(
            chunk_id=row["chunk_id"],
            content=row["content"],
            source_title=row["source_title"],
            source_url=row["source_url"],
            trust_score=row["trust_score"],
            rrf_score=row["rrf_score"],
        )
        for row in rows
    ]
