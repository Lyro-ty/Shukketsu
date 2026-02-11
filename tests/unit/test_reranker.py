"""Tests for the Qwen 4B categorical reranker."""

from unittest.mock import AsyncMock, patch

import pytest

from code.shukketsu.rag.reranker import (
    RankedItem,
    RerankerResponse,
    _apply_rankings,
    _build_rerank_messages,
    rerank,
)
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


class TestRerankerSchemas:
    """Tests for Pydantic schemas."""

    def test_ranked_item_validates(self) -> None:
        item = RankedItem(index=0, relevance="HIGH", reason="directly relevant")
        assert item.relevance == "HIGH"

    def test_ranked_item_rejects_invalid_relevance(self) -> None:
        with pytest.raises(Exception):
            RankedItem(index=0, relevance="SUPER_HIGH", reason="nope")

    def test_reranker_response_validates(self) -> None:
        resp = RerankerResponse(rankings=[RankedItem(index=0, relevance="HIGH", reason="good")])
        assert len(resp.rankings) == 1


class TestBuildMessages:
    """Tests for prompt construction."""

    def test_includes_query(self) -> None:
        results = [_make_result(1, "Hit cap is 142")]
        messages = _build_rerank_messages("hit cap", results)
        user_msg = messages[-1]["content"]
        assert "hit cap" in user_msg

    def test_includes_results_indexed(self) -> None:
        results = [_make_result(1, "Content A"), _make_result(2, "Content B")]
        messages = _build_rerank_messages("query", results)
        user_msg = messages[-1]["content"]
        assert "[0]" in user_msg
        assert "[1]" in user_msg

    def test_truncates_long_content(self) -> None:
        long_content = "x" * 500
        results = [_make_result(1, long_content)]
        messages = _build_rerank_messages("query", results)
        user_msg = messages[-1]["content"]
        # Should not contain the full 500 chars
        assert len(user_msg) < 400


class TestApplyRankings:
    """Tests for the index validation + categorical filtering logic."""

    def test_filters_low_keeps_high(self) -> None:
        results = [_make_result(i, rrf_score=0.5 - i * 0.1) for i in range(5)]
        rankings = [
            RankedItem(index=0, relevance="HIGH", reason=""),
            RankedItem(index=1, relevance="HIGH", reason=""),
            RankedItem(index=2, relevance="LOW", reason=""),
            RankedItem(index=3, relevance="LOW", reason=""),
            RankedItem(index=4, relevance="LOW", reason=""),
        ]
        filtered = _apply_rankings(results, rankings, top_k=3)
        assert len(filtered) == 2  # only 2 HIGHs, no MEDIUMs to fill
        assert filtered[0].chunk_id == 0
        assert filtered[1].chunk_id == 1

    def test_fills_medium_up_to_top_k(self) -> None:
        results = [_make_result(i, rrf_score=0.5 - i * 0.1) for i in range(5)]
        rankings = [
            RankedItem(index=0, relevance="HIGH", reason=""),
            RankedItem(index=1, relevance="MEDIUM", reason=""),
            RankedItem(index=2, relevance="MEDIUM", reason=""),
            RankedItem(index=3, relevance="MEDIUM", reason=""),
            RankedItem(index=4, relevance="LOW", reason=""),
        ]
        filtered = _apply_rankings(results, rankings, top_k=3)
        assert len(filtered) == 3
        assert filtered[0].chunk_id == 0  # HIGH first
        assert filtered[1].chunk_id == 1  # then MEDIUMs
        assert filtered[2].chunk_id == 2

    def test_preserves_original_order_within_category(self) -> None:
        results = [_make_result(i, rrf_score=1.0 - i * 0.1) for i in range(4)]
        rankings = [
            RankedItem(index=2, relevance="HIGH", reason=""),
            RankedItem(index=0, relevance="HIGH", reason=""),
            RankedItem(index=3, relevance="HIGH", reason=""),
            RankedItem(index=1, relevance="HIGH", reason=""),
        ]
        # All HIGH — order should follow original rrf_score order (0, 1, 2, 3)
        filtered = _apply_rankings(results, rankings, top_k=4)
        assert [r.chunk_id for r in filtered] == [0, 1, 2, 3]

    def test_all_high_returns_top_k(self) -> None:
        results = [_make_result(i) for i in range(6)]
        rankings = [RankedItem(index=i, relevance="HIGH", reason="") for i in range(6)]
        filtered = _apply_rankings(results, rankings, top_k=3)
        assert len(filtered) == 3

    def test_ignores_out_of_range_indices(self) -> None:
        results = [_make_result(0), _make_result(1)]
        rankings = [
            RankedItem(index=0, relevance="HIGH", reason=""),
            RankedItem(index=99, relevance="HIGH", reason=""),  # out of range
            RankedItem(index=-1, relevance="HIGH", reason=""),  # negative
        ]
        filtered = _apply_rankings(results, rankings, top_k=5)
        assert len(filtered) == 1
        assert filtered[0].chunk_id == 0

    def test_deduplicates_indices(self) -> None:
        results = [_make_result(0), _make_result(1)]
        rankings = [
            RankedItem(index=0, relevance="HIGH", reason=""),
            RankedItem(index=0, relevance="HIGH", reason=""),  # duplicate
            RankedItem(index=1, relevance="MEDIUM", reason=""),
        ]
        filtered = _apply_rankings(results, rankings, top_k=5)
        assert len(filtered) == 2

    def test_unranked_treated_as_low(self) -> None:
        results = [_make_result(0), _make_result(1), _make_result(2)]
        rankings = [
            RankedItem(index=0, relevance="HIGH", reason=""),
            # indices 1 and 2 not ranked — should be treated as LOW
        ]
        filtered = _apply_rankings(results, rankings, top_k=5)
        assert len(filtered) == 1
        assert filtered[0].chunk_id == 0


class TestRerank:
    """Tests for the top-level rerank() function."""

    async def test_skips_when_results_lte_top_k(self) -> None:
        results = [_make_result(0), _make_result(1)]
        filtered = await rerank("query", results, top_k=5)
        assert filtered == results  # unchanged, no LLM call

    async def test_empty_results_returns_empty(self) -> None:
        filtered = await rerank("query", [], top_k=5)
        assert filtered == []

    @patch("code.shukketsu.rag.reranker._rerank_impl")
    async def test_falls_back_on_circuit_open(self, mock_impl: AsyncMock) -> None:
        from code.shukketsu.resilience.circuit_breaker import qwen_reranker_breaker

        # Force breaker open
        qwen_reranker_breaker._state = qwen_reranker_breaker._state.__class__("open")
        qwen_reranker_breaker._failure_count = 100
        import time

        qwen_reranker_breaker._last_failure_time = time.monotonic()

        results = [_make_result(i) for i in range(10)]
        filtered = await rerank("query", results, top_k=3)
        assert len(filtered) == 3
        assert filtered == results[:3]
        mock_impl.assert_not_called()

    @patch("code.shukketsu.rag.reranker._rerank_impl")
    async def test_falls_back_on_llm_error(self, mock_impl: AsyncMock) -> None:
        from code.shukketsu.resilience.errors import LLMUnavailableError

        mock_impl.side_effect = LLMUnavailableError("Qwen down")
        results = [_make_result(i) for i in range(10)]
        filtered = await rerank("query", results, top_k=3)
        # After enough failures, breaker opens — but first call should still fallback
        assert len(filtered) == 3
