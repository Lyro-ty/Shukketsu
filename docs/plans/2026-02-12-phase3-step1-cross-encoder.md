# Phase 3 Step 1: Cross-Encoder Reranker — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the generative Qwen 4B categorical reranker with a cross-encoder model for faster, more accurate continuous relevance scoring.

**Architecture:** Lazy-loaded `CrossEncoder` from `sentence-transformers` scores query-document pairs directly on CPU. Scores are continuous floats (0.0–1.0) instead of categorical (HIGH/MEDIUM/LOW). Circuit breaker wraps the scoring call; fallback returns unreranked results truncated to `top_k`. The CPU-bound `predict()` call runs in a thread executor to avoid blocking the event loop.

**Tech Stack:** `sentence-transformers>=3.0`, `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, ~5ms/pair on CPU)

---

### Task 1: Add dependency + config constants

**Files:**
- Modify: `requirements.txt`
- Modify: `code/shukketsu/config.py:90-91`

**Step 1: Add sentence-transformers to requirements.txt**

Add after the `sqlite-vec` line (in the Database section):

```
# Reranking
sentence-transformers>=3.0
```

**Step 2: Update config.py — replace Qwen reranker constants with cross-encoder constants**

Replace lines 90-91:
```python
CB_QWEN_RERANKER_FAILURE_THRESHOLD = int(os.getenv("CB_QWEN_RERANKER_FAILURE_THRESHOLD", "3"))
CB_QWEN_RERANKER_RECOVERY_TIMEOUT = float(os.getenv("CB_QWEN_RERANKER_RECOVERY_TIMEOUT", "30"))
```

With:
```python
# Cross-encoder reranker
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CB_RERANKER_FAILURE_THRESHOLD = int(os.getenv("CB_RERANKER_FAILURE_THRESHOLD", "3"))
CB_RERANKER_RECOVERY_TIMEOUT = float(os.getenv("CB_RERANKER_RECOVERY_TIMEOUT", "30"))
```

**Step 3: Install the new dependency**

Run: `pip install --break-system-packages sentence-transformers>=3.0`

(No commit yet — we'll commit with the implementation.)

---

### Task 2: Add `rerank_score` to SearchResult

**Files:**
- Modify: `code/shukketsu/rag/search.py:69-78`
- Test: existing tests (verify they still pass)

**Step 1: Add the optional field to the frozen dataclass**

In `code/shukketsu/rag/search.py`, change the `SearchResult` dataclass to:

```python
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
```

**Step 2: Verify existing tests still pass**

Run: `python3 -m pytest tests/unit/test_reranker.py tests/unit/test_search_reranking.py tests/unit/test_hybrid_search.py -v`

Expected: All pass (the new field has a default of `None`, so existing constructors are unaffected).

---

### Task 3: Rename circuit breaker from `qwen_reranker_breaker` to `reranker_breaker`

**Files:**
- Modify: `code/shukketsu/resilience/circuit_breaker.py:140-144,149-156`
- Modify: `code/shukketsu/rag/reranker.py:14` (import)
- Modify: `tests/unit/test_reranker.py:166` (reference in test)

**Step 1: Update circuit_breaker.py**

Replace lines 140-144:
```python
reranker_breaker = CircuitBreaker(
    "reranker",
    failure_threshold=config.CB_RERANKER_FAILURE_THRESHOLD,
    recovery_timeout=config.CB_RERANKER_RECOVERY_TIMEOUT,
)
```

Update `reset_all_breakers()` on line 149-156 — replace `qwen_reranker_breaker` with `reranker_breaker` in the tuple.

**Step 2: Update reranker.py import**

Change line 14:
```python
from code.shukketsu.resilience.circuit_breaker import reranker_breaker
```

And update line 136:
```python
return await reranker_breaker.call(_rerank_impl, query, results, top_k)
```

**Step 3: Update the test reference**

In `tests/unit/test_reranker.py` line 166, change `qwen_reranker_breaker` to `reranker_breaker`.

**Step 4: Verify no other references to old name**

Run: `grep -r "qwen_reranker_breaker" code/ tests/`

Expected: No matches.

---

### Task 4: Write failing tests for cross-encoder reranker

**Files:**
- Rewrite: `tests/unit/test_reranker.py`

**Step 1: Write the new test file**

Replace the entire contents of `tests/unit/test_reranker.py` with:

```python
"""Tests for the cross-encoder reranker."""

from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from code.shukketsu.rag.reranker import rerank
from code.shukketsu.rag.search import SearchResult


def _make_result(chunk_id: int, content: str = "test", rrf_score: float = 0.5) -> SearchResult:
    """Helper to create SearchResult instances."""
    return SearchResult(
        chunk_id=chunk_id,
        content=content,
        source_title=f"Source {chunk_id}",
        source_url=f"https://example.com/{chunk_id}",
        trust_score=0.7,
        rrf_score=rrf_score,
    )


class TestRerankShortCircuit:
    """Tests for short-circuit paths (no model needed)."""

    async def test_empty_results_returns_empty(self) -> None:
        filtered = await rerank("query", [], top_k=5)
        assert filtered == []

    async def test_skips_when_results_lte_top_k(self) -> None:
        results = [_make_result(0), _make_result(1)]
        filtered = await rerank("query", results, top_k=5)
        assert filtered == results


class TestRerankScoring:
    """Tests for cross-encoder scoring and sort order."""

    @patch("code.shukketsu.rag.reranker._get_model")
    async def test_sorts_by_score_descending(self, mock_get_model: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])
        mock_get_model.return_value = mock_model

        results = [_make_result(0), _make_result(1), _make_result(2)]
        filtered = await rerank("query", results, top_k=3)

        assert [r.chunk_id for r in filtered] == [1, 2, 0]

    @patch("code.shukketsu.rag.reranker._get_model")
    async def test_returns_top_k_only(self, mock_get_model: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.1, 0.5, 0.3, 0.7])
        mock_get_model.return_value = mock_model

        results = [_make_result(i) for i in range(5)]
        filtered = await rerank("query", results, top_k=3)

        assert len(filtered) == 3
        assert [r.chunk_id for r in filtered] == [0, 4, 2]

    @patch("code.shukketsu.rag.reranker._get_model")
    async def test_populates_rerank_score(self, mock_get_model: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8, 0.3])
        mock_get_model.return_value = mock_model

        results = [_make_result(0), _make_result(1)]
        # Need more than top_k to trigger reranking
        extra = [_make_result(i) for i in range(2, 8)]
        filtered = await rerank("query", results + extra, top_k=2)

        assert all(r.rerank_score is not None for r in filtered)
        assert filtered[0].rerank_score >= filtered[1].rerank_score  # type: ignore[operator]

    @patch("code.shukketsu.rag.reranker._get_model")
    async def test_builds_query_document_pairs(self, mock_get_model: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.3, 0.8])
        mock_get_model.return_value = mock_model

        results = [
            _make_result(0, content="alpha"),
            _make_result(1, content="beta"),
            _make_result(2, content="gamma"),
        ]
        await rerank("my query", results, top_k=2)

        pairs = mock_model.predict.call_args[0][0]
        assert len(pairs) == 3
        assert pairs[0] == ("my query", "alpha")
        assert pairs[1] == ("my query", "beta")
        assert pairs[2] == ("my query", "gamma")


class TestRerankFallback:
    """Tests for fallback on model failure."""

    @patch("code.shukketsu.rag.reranker._get_model")
    async def test_falls_back_on_model_load_error(self, mock_get_model: MagicMock) -> None:
        mock_get_model.side_effect = RuntimeError("Model not found")

        results = [_make_result(i) for i in range(10)]
        filtered = await rerank("query", results, top_k=3)

        assert len(filtered) == 3
        assert filtered == results[:3]
        assert all(r.rerank_score is None for r in filtered)

    @patch("code.shukketsu.rag.reranker._get_model")
    async def test_falls_back_on_predict_error(self, mock_get_model: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("CUDA OOM")
        mock_get_model.return_value = mock_model

        results = [_make_result(i) for i in range(10)]
        filtered = await rerank("query", results, top_k=3)

        assert len(filtered) == 3
        assert filtered == results[:3]

    @patch("code.shukketsu.rag.reranker._get_model")
    async def test_falls_back_on_circuit_open(self, mock_get_model: MagicMock) -> None:
        from code.shukketsu.resilience.circuit_breaker import reranker_breaker

        # Force breaker open
        reranker_breaker._state = reranker_breaker._state.__class__("open")
        reranker_breaker._failure_count = 100
        import time
        reranker_breaker._last_failure_time = time.monotonic()

        results = [_make_result(i) for i in range(10)]
        filtered = await rerank("query", results, top_k=3)

        assert len(filtered) == 3
        assert filtered == results[:3]
        mock_get_model.assert_not_called()


class TestRerankScoreField:
    """Tests for the rerank_score field on SearchResult."""

    def test_default_is_none(self) -> None:
        r = _make_result(0)
        assert r.rerank_score is None

    def test_replace_sets_score(self) -> None:
        r = _make_result(0)
        scored = replace(r, rerank_score=0.85)
        assert scored.rerank_score == 0.85
        assert scored.chunk_id == 0  # other fields unchanged

    def test_frozen_immutable(self) -> None:
        r = _make_result(0)
        with pytest.raises(AttributeError):
            r.rerank_score = 0.5  # type: ignore[misc]
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_reranker.py -v`

Expected: Multiple failures — `_get_model` doesn't exist yet, reranker still has old interface.

---

### Task 5: Implement cross-encoder reranker

**Files:**
- Rewrite: `code/shukketsu/rag/reranker.py`

**Step 1: Replace the entire reranker module**

Replace `code/shukketsu/rag/reranker.py` with:

```python
"""Cross-encoder reranker for search results.

Uses a lightweight cross-encoder model to score query-document relevance.
Circuit breaker wraps the scoring call — falls back to unreranked on failure.
"""

import asyncio
import logging
from dataclasses import replace

from code.shukketsu import config
from code.shukketsu.rag.search import SearchResult
from code.shukketsu.resilience.circuit_breaker import reranker_breaker
from code.shukketsu.resilience.errors import CircuitOpenError

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Lazily load and cache the cross-encoder model.

    The model is loaded on first call and cached for subsequent calls.
    Runs on CPU (PyTorch sm_121 kernels not compiled, model is tiny anyway).
    """
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder model: %s", config.RERANKER_MODEL)
        _model = CrossEncoder(config.RERANKER_MODEL)
        logger.info("Cross-encoder model loaded")
    return _model


async def _rerank_impl(query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
    """Score query-document pairs with the cross-encoder and return top_k."""
    model = _get_model()
    pairs = [(query, r.content) for r in results]

    # predict() is CPU-bound — run in executor to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    scores = await loop.run_in_executor(None, model.predict, pairs)

    # Pair results with scores, sort by score descending
    scored = sorted(zip(results, scores), key=lambda x: float(x[1]), reverse=True)

    return [
        replace(r, rerank_score=float(s))
        for r, s in scored[:top_k]
    ]


async def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int = 5,
) -> list[SearchResult]:
    """Rerank search results using cross-encoder relevance scoring.

    Short-circuits if results <= top_k. Falls back to truncated unreranked
    results if the circuit breaker is open or the model fails.

    Args:
        query: The original search query.
        results: Search results to rerank.
        top_k: Number of results to return.

    Returns:
        Filtered and reranked list of SearchResult with rerank_score populated.
    """
    if len(results) <= top_k:
        return results

    try:
        return await reranker_breaker.call(_rerank_impl, query, results, top_k)
    except CircuitOpenError:
        logger.warning("Reranker circuit breaker open — returning unreranked results")
        return results[:top_k]
    except Exception:
        logger.warning("Reranker failed — returning unreranked results", exc_info=True)
        return results[:top_k]
```

**Step 2: Run the new tests**

Run: `python3 -m pytest tests/unit/test_reranker.py -v`

Expected: All pass.

**Step 3: Run the search reranking integration tests**

Run: `python3 -m pytest tests/unit/test_search_reranking.py -v`

Expected: All pass (these mock `rerank()` at the call site, so the internal implementation change is transparent).

---

### Task 6: Run full test suite + linting

**Files:** None (verification only)

**Step 1: Run linting + type checking + all tests**

Run: `ruff check code/ tests/ --fix && ruff format code/ tests/ && python3 -m mypy code/shukketsu/ && python3 -m pytest tests/unit/ -v`

Expected: 741+ tests pass, no lint errors, no type errors.

**Step 2: Fix any issues found**

If mypy complains about `CrossEncoder` type (no stubs), add `# type: ignore[import-untyped]` to the import line inside `_get_model()`.

If numpy import in tests needs a type stub, add `types-numpy` or use `# type: ignore` as needed.

---

### Task 7: Commit

**Step 1: Stage and commit**

```bash
git add requirements.txt code/shukketsu/config.py code/shukketsu/rag/search.py code/shukketsu/rag/reranker.py code/shukketsu/resilience/circuit_breaker.py tests/unit/test_reranker.py
git commit -m "feat(rag): replace Qwen 4B reranker with cross-encoder model

Cross-encoder/ms-marco-MiniLM-L-6-v2 scores query-document pairs directly
instead of generative categorical ranking. Produces continuous relevance
scores, runs ~20x faster on CPU, and populates rerank_score on SearchResult.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
