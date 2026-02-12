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
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5, 0.2])
        mock_get_model.return_value = mock_model

        results = [_make_result(i) for i in range(4)]
        filtered = await rerank("query", results, top_k=3)

        assert [r.chunk_id for r in filtered] == [1, 2, 3]

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
