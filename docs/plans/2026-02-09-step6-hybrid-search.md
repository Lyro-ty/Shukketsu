# Step 6: Hybrid Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade RAG search from vector-only to hybrid (vector + FTS5 keyword) search using Reciprocal Rank Fusion, so that both semantic and exact keyword queries return relevant results.

**Architecture:** Single-SQL approach using CTEs and `FULL OUTER JOIN` (inspired by [sqlite-vec author's blog](https://alexgarcia.xyz/blog/2024/sqlite-vec-hybrid-search/index.html)). RRF math happens in SQL for the production path. A pure Python `compute_rrf()` function exists in `rag/fusion.py` for testability and potential reuse. `rag/search.py` runs the hybrid query and returns typed `SearchResult` objects. `tools/knowledge/search.py` becomes a thin wrapper.

**Tech Stack:** SQLite 3.45.1, sqlite-vec (cosine KNN), FTS5 (BM25), Python 3.12, pytest

---

## Reference

### Key Files
- `code/shukketsu/rag/__init__.py` — empty stub (scaffolded)
- `code/shukketsu/tools/knowledge/search.py` — current vector-only `RagSearchTool`
- `code/shukketsu/config.py:44` — `RAG_SEARCH_TOP_K = 5`
- `code/shukketsu/db/schema.sql:50-76` — `chunks_vec`, `chunks_fts`, sync triggers
- `tests/unit/test_rag_search.py` — 8 existing tests for `RagSearchTool`
- `tests/conftest.py` — `test_db` fixture (WAL, sqlite-vec, full schema)

### RRF Formula
```
score(doc) = sum(1 / (k + rank)) for each list the doc appears in
```
- `k = 60` (standard constant, prevents top items from dominating)
- `rank` is 0-based ordinal position (lower = better match)
- Higher RRF score = better combined result

### SQL Pattern (Single-Query RRF)
```sql
WITH vec_matches AS (
    SELECT rowid, row_number() OVER (ORDER BY distance) - 1 AS rank
    FROM chunks_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?
),
fts_matches AS (
    SELECT rowid, row_number() OVER (ORDER BY rank) - 1 AS rank
    FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?
)
SELECT
    coalesce(f.rowid, v.rowid) AS chunk_id,
    coalesce(1.0 / (60 + f.rank), 0.0) + coalesce(1.0 / (60 + v.rank), 0.0) AS rrf_score
FROM fts_matches f
FULL OUTER JOIN vec_matches v ON v.rowid = f.rowid
ORDER BY rrf_score DESC
LIMIT ?
```

---

## Task 1: Add `RAG_SEARCH_FETCH_K` config constant

**Files:**
- Modify: `code/shukketsu/config.py:44`

**Step 1: Add the constant**

Add after line 44 (`RAG_SEARCH_TOP_K = 5`):

```python
RAG_SEARCH_FETCH_K = 20  # Candidates per source before RRF fusion
```

**Step 2: Commit**

```bash
git add code/shukketsu/config.py
git commit -m "feat: add RAG_SEARCH_FETCH_K config for hybrid search candidate pool"
```

---

## Task 2: Implement `rag/fusion.py` — pure RRF function + FTS5 escaping

**Files:**
- Create: `code/shukketsu/rag/fusion.py`
- Test: `tests/unit/test_rrf.py`

**Step 1: Write failing tests for `compute_rrf`**

Create `tests/unit/test_rrf.py`:

```python
"""Tests for RRF fusion and FTS5 query escaping."""

from code.shukketsu.rag.fusion import compute_rrf, escape_fts_query


class TestComputeRrf:
    """Tests for Reciprocal Rank Fusion."""

    def test_disjoint_lists(self) -> None:
        """Documents in only one list get single-source RRF scores."""
        vec_ranks = {1: 0, 2: 1}  # doc 1 rank 0, doc 2 rank 1
        fts_ranks = {3: 0, 4: 1}  # doc 3 rank 0, doc 4 rank 1
        scores = compute_rrf([vec_ranks, fts_ranks], k=60)
        assert len(scores) == 4
        # All same-ranked docs have equal scores
        assert scores[1] == scores[3]
        assert scores[2] == scores[4]
        # Rank 0 scores higher than rank 1
        assert scores[1] > scores[2]

    def test_overlapping_ranks_higher(self) -> None:
        """Documents appearing in both lists score higher than single-list docs."""
        vec_ranks = {1: 0, 2: 1}
        fts_ranks = {1: 0, 3: 1}  # doc 1 in both lists
        scores = compute_rrf([vec_ranks, fts_ranks], k=60)
        # Doc 1 (in both) beats doc 2 and doc 3 (in one each)
        assert scores[1] > scores[2]
        assert scores[1] > scores[3]

    def test_one_empty_list(self) -> None:
        """Degrades gracefully to single-source ranking."""
        vec_ranks = {1: 0, 2: 1, 3: 2}
        fts_ranks: dict[int, int] = {}
        scores = compute_rrf([vec_ranks, fts_ranks], k=60)
        assert len(scores) == 3
        assert scores[1] > scores[2] > scores[3]

    def test_both_empty(self) -> None:
        """Empty inputs return empty result."""
        scores = compute_rrf([{}, {}], k=60)
        assert scores == {}

    def test_all_overlap_same_order(self) -> None:
        """Both lists agree on ranking — scores are doubled."""
        vec_ranks = {1: 0, 2: 1}
        fts_ranks = {1: 0, 2: 1}
        scores = compute_rrf([vec_ranks, fts_ranks], k=60)
        # Doc 1 score = 2 * 1/(60+0) = 2/60
        expected_1 = 2.0 / 60.0
        assert abs(scores[1] - expected_1) < 1e-9

    def test_custom_k_value(self) -> None:
        """Different k values change the score distribution."""
        ranks = [{1: 0, 2: 1}]
        scores_k10 = compute_rrf(ranks, k=10)
        scores_k100 = compute_rrf(ranks, k=100)
        # Smaller k = bigger spread between rank 0 and rank 1
        spread_k10 = scores_k10[1] - scores_k10[2]
        spread_k100 = scores_k100[1] - scores_k100[2]
        assert spread_k10 > spread_k100


class TestEscapeFtsQuery:
    """Tests for FTS5 query escaping."""

    def test_simple_query(self) -> None:
        assert escape_fts_query("DST proc rate") == '"DST" "proc" "rate"'

    def test_single_token(self) -> None:
        assert escape_fts_query("DST") == '"DST"'

    def test_empty_string(self) -> None:
        assert escape_fts_query("") == ""

    def test_extra_whitespace(self) -> None:
        assert escape_fts_query("  DST   proc  ") == '"DST" "proc"'

    def test_special_characters_escaped(self) -> None:
        """Quotes and FTS5 operators inside tokens are handled."""
        result = escape_fts_query('rogue "combat" spec')
        # Each token is individually quoted; inner quotes are doubled for FTS5
        assert '"rogue"' in result
        assert '"spec"' in result
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_rrf.py -v
```

Expected: `ModuleNotFoundError: No module named 'code.shukketsu.rag.fusion'`

**Step 3: Implement `rag/fusion.py`**

Create `code/shukketsu/rag/fusion.py`:

```python
"""Reciprocal Rank Fusion and FTS5 query utilities."""

RRF_K = 60


def compute_rrf(ranked_lists: list[dict[int, int]], k: int = RRF_K) -> dict[int, float]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each dict maps document_id to 0-based rank (lower = better).
        k: Smoothing constant (default 60). Prevents top items from dominating.

    Returns:
        Dict mapping document_id to RRF score (higher = better).
    """
    scores: dict[int, float] = {}
    for ranks in ranked_lists:
        for doc_id, rank in ranks.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def escape_fts_query(query: str) -> str:
    """Escape a raw query string for safe FTS5 MATCH.

    Wraps each whitespace-delimited token in double quotes to treat
    them as literals, preventing FTS5 syntax injection.

    Args:
        query: Raw search query string.

    Returns:
        FTS5-safe query with each token quoted.
    """
    tokens = query.split()
    if not tokens:
        return ""
    escaped = []
    for token in tokens:
        # Double any internal quotes for FTS5 escaping
        safe = token.replace('"', '""')
        escaped.append(f'"{safe}"')
    return " ".join(escaped)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_rrf.py -v
```

Expected: All 12 tests PASS.

**Step 5: Lint and commit**

```bash
ruff check code/shukketsu/rag/fusion.py tests/unit/test_rrf.py
ruff format code/shukketsu/rag/fusion.py tests/unit/test_rrf.py
git add code/shukketsu/rag/fusion.py tests/unit/test_rrf.py
git commit -m "feat: add RRF fusion function and FTS5 query escaping"
```

---

## Task 3: Implement `rag/search.py` — hybrid search with single-SQL RRF

**Files:**
- Create: `code/shukketsu/rag/search.py`
- Test: `tests/unit/test_hybrid_search.py`

**Step 1: Write failing tests for `hybrid_search`**

Create `tests/unit/test_hybrid_search.py`:

```python
"""Tests for hybrid search (vector + FTS5 + RRF)."""

import sqlite3
import struct

import pytest

from code.shukketsu.rag.search import SearchResult, hybrid_search

EMBEDDING_DIM = 768


def _make_embedding(value: float = 0.1) -> list[float]:
    return [value] * EMBEDDING_DIM


def _pack_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def _seed_source(conn: sqlite3.Connection, *, url: str = "https://example.com/guide",
                 title: str = "Test Guide", trust_score: float = 0.75) -> int:
    """Insert a source and return its ID."""
    cur = conn.execute(
        "INSERT INTO sources (url, title, source_type, trust_score) VALUES (?, ?, 'guide', ?)",
        (url, title, trust_score),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def _seed_chunk(conn: sqlite3.Connection, source_id: int, content: str,
                chunk_index: int, embedding_value: float) -> int:
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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_hybrid_search.py -v
```

Expected: `ModuleNotFoundError: No module named 'code.shukketsu.rag.search'`

**Step 3: Implement `rag/search.py`**

Create `code/shukketsu/rag/search.py`:

```python
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
    WHERE embedding MATCH ?
    ORDER BY distance
    LIMIT ?
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
    s.trust_score,
    (
        coalesce(1.0 / (60 + f.rank), 0.0)
        + coalesce(1.0 / (60 + v.rank), 0.0)
    ) AS rrf_score
FROM fts_matches f
FULL OUTER JOIN vec_matches v ON v.rowid = f.rowid
JOIN chunks c ON c.id = coalesce(f.rowid, v.rowid)
JOIN sources s ON s.id = c.source_id
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
    s.trust_score,
    (1.0 / (60 + row_number() OVER (ORDER BY v.distance) - 1)) AS rrf_score
FROM (
    SELECT rowid, distance
    FROM chunks_vec
    WHERE embedding MATCH ?
    ORDER BY distance
    LIMIT ?
) v
JOIN chunks c ON c.id = v.rowid
JOIN sources s ON s.id = c.source_id
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
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_hybrid_search.py -v
```

Expected: All 7 tests PASS.

**Step 5: Lint and commit**

```bash
ruff check code/shukketsu/rag/search.py tests/unit/test_hybrid_search.py
ruff format code/shukketsu/rag/search.py tests/unit/test_hybrid_search.py
git add code/shukketsu/rag/search.py tests/unit/test_hybrid_search.py
git commit -m "feat: add hybrid search with single-SQL RRF fusion"
```

---

## Task 4: Refactor `RagSearchTool` to use hybrid search

**Files:**
- Modify: `code/shukketsu/tools/knowledge/search.py` (lines 35-72)
- Modify: `tests/unit/test_rag_search.py` (lines 19-38, update seeding; lines 66-93, update assertions)

**Step 1: Update existing test helpers to seed FTS-compatible data**

The existing `_seed_test_data` function doesn't trigger FTS5 sync because it inserts directly. The `test_db` fixture has the triggers, so data inserted via `INSERT INTO chunks` will auto-populate `chunks_fts`. Verify the existing tests still pass after the refactor — they should, since hybrid search is a superset of vector-only.

No test changes needed if the existing seeding already triggers FTS sync (it does — the `chunks_fts_insert` trigger fires on `INSERT INTO chunks`). The existing tests assert on result format, which stays the same.

**Step 2: Refactor `search.py`**

Replace the full `execute` method in `code/shukketsu/tools/knowledge/search.py`:

```python
"""RAG search tool for hybrid vector + keyword search over the knowledge base."""

import logging
import sqlite3
from collections.abc import Callable, Coroutine
from typing import Any

from code.shukketsu import config
from code.shukketsu.rag.search import hybrid_search
from code.shukketsu.tools.schemas import Tool

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Coroutine[Any, Any, list[float]]]


class RagSearchTool(Tool):
    """Search the knowledge base using hybrid vector + keyword search.

    Combines semantic vector similarity (chunks_vec) with FTS5 keyword
    matching (chunks_fts), merged via Reciprocal Rank Fusion.
    """

    name = "rag_search"
    description = "Search the knowledge base for relevant information about WoW TBC Rogues."
    parameters_schema = {
        "query": {"type": "string", "description": "The search query"},
        "top_k": {"type": "integer", "description": "Number of results to return", "optional": True},
    }

    def __init__(self, conn: sqlite3.Connection, embed_fn: EmbedFn) -> None:
        self._conn = conn
        self._embed_fn = embed_fn

    async def execute(self, tool_input: dict[str, Any]) -> str:
        """Execute a hybrid search (vector + FTS5 + RRF)."""
        query = tool_input.get("query", "")
        top_k = tool_input.get("top_k", config.RAG_SEARCH_TOP_K)

        if not query:
            return "Error: 'query' parameter is required."

        embedding = await self._embed_fn(query)
        results = await hybrid_search(self._conn, query, embedding, top_k=top_k)

        if not results:
            return "No relevant documents found for this query."

        parts = [f"Found {len(results)} result{'s' if len(results) != 1 else ''}:\n"]
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] Source: {r.source_title} ({r.source_url})\n"
                f"Trust: {r.trust_score}\nContent: {r.content}\n"
            )

        return "\n".join(parts)
```

**Step 3: Run existing tests to verify they still pass**

```bash
pytest tests/unit/test_rag_search.py -v
```

Expected: All 8 existing tests PASS (no behavior change for vector-matched data).

**Step 4: Run full test suite**

```bash
pytest tests/unit/ -v
```

Expected: All tests PASS (122 existing + 12 new from Task 2 + 7 new from Task 3 = 141).

**Step 5: Lint and commit**

```bash
ruff check code/shukketsu/tools/knowledge/search.py
ruff format code/shukketsu/tools/knowledge/search.py
git add code/shukketsu/tools/knowledge/search.py
git commit -m "refactor: upgrade RagSearchTool to hybrid search via rag.search"
```

---

## Task 5: Full verification and CLAUDE.md update

**Files:**
- Modify: `CLAUDE.md` (update step status)

**Step 1: Run full test suite with count**

```bash
pytest tests/unit/ -v
```

Expected: ~141 tests PASS.

**Step 2: Lint check**

```bash
ruff check code/shukketsu/ tests/
ruff format --check code/shukketsu/ tests/
```

Expected: No errors.

**Step 3: Update CLAUDE.md**

In `CLAUDE.md`, update the Phase 1 step list:
- Change line `6. **Hybrid search (vector + FTS5 + RRF)** ← current` to `6. ~~Hybrid search (vector + FTS5 + RRF)~~ **DONE**`
- Change line 7 to add `← current`
- Update the "Currently at" line to say "Phase 1, Step 7"

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Step 6 completion"
```

---

## Summary

| Task | What | New Tests | New Files |
|------|------|-----------|-----------|
| 1 | Config constant | 0 | 0 |
| 2 | `rag/fusion.py` — RRF + FTS5 escape | 12 | 2 (`fusion.py`, `test_rrf.py`) |
| 3 | `rag/search.py` — hybrid search | 7 | 2 (`search.py`, `test_hybrid_search.py`) |
| 4 | Refactor `RagSearchTool` | 0 (existing pass) | 0 |
| 5 | Verification + docs | 0 | 0 |

**Total new tests:** ~19
**Expected final count:** ~141
**Files created:** 4 (2 source, 2 test)
**Files modified:** 3 (`config.py`, `search.py`, `CLAUDE.md`)
