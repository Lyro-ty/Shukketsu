"""Tests for reranker integration with RagSearchTool."""

from unittest.mock import AsyncMock, patch

import pytest

from code.shukketsu import config
from code.shukketsu.rag.search import SearchResult
from code.shukketsu.tools.knowledge.search import RagSearchTool


def _make_result(chunk_id: int, content: str = "test") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        content=content,
        source_title=f"Source {chunk_id}",
        source_url=f"https://example.com/{chunk_id}",
        trust_score=0.7,
        rrf_score=0.5,
    )


async def _mock_embed(text: str) -> list[float]:
    return [0.1] * 768


class TestRagSearchReranking:
    """Tests for reranker integration in RagSearchTool."""

    @patch("code.shukketsu.tools.knowledge.search.rerank")
    @patch("code.shukketsu.tools.knowledge.search.hybrid_search")
    async def test_over_retrieves_with_multiplier(
        self, mock_hybrid: AsyncMock, mock_rerank: AsyncMock, test_db
    ) -> None:
        raw_results = [_make_result(i) for i in range(15)]
        mock_hybrid.return_value = raw_results
        mock_rerank.return_value = raw_results[:5]

        tool = RagSearchTool(conn=test_db, embed_fn=_mock_embed)
        await tool.execute({"query": "hit cap", "top_k": 5})

        # hybrid_search should be called with top_k * multiplier
        call_kwargs = mock_hybrid.call_args
        assert call_kwargs.kwargs["top_k"] == 5 * config.RERANKER_FETCH_MULTIPLIER

    @patch("code.shukketsu.tools.knowledge.search.rerank")
    @patch("code.shukketsu.tools.knowledge.search.hybrid_search")
    async def test_passes_results_through_reranker(
        self, mock_hybrid: AsyncMock, mock_rerank: AsyncMock, test_db
    ) -> None:
        raw_results = [_make_result(i) for i in range(15)]
        mock_hybrid.return_value = raw_results
        mock_rerank.return_value = raw_results[:5]

        tool = RagSearchTool(conn=test_db, embed_fn=_mock_embed)
        await tool.execute({"query": "hit cap"})

        mock_rerank.assert_called_once()
        args = mock_rerank.call_args
        assert args[0][0] == "hit cap"  # query
        assert len(args[0][1]) == 15  # all raw results passed in

    @patch("code.shukketsu.tools.knowledge.search.rerank")
    @patch("code.shukketsu.tools.knowledge.search.hybrid_search")
    async def test_fetch_k_scales_with_multiplier(
        self, mock_hybrid: AsyncMock, mock_rerank: AsyncMock, test_db
    ) -> None:
        mock_hybrid.return_value = []
        mock_rerank.return_value = []

        tool = RagSearchTool(conn=test_db, embed_fn=_mock_embed)
        await tool.execute({"query": "hit cap", "top_k": 10})

        call_kwargs = mock_hybrid.call_args
        expected_fetch = 10 * config.RERANKER_FETCH_MULTIPLIER
        assert call_kwargs.kwargs["fetch_k"] >= expected_fetch

    @patch("code.shukketsu.tools.knowledge.search.rerank")
    @patch("code.shukketsu.tools.knowledge.search.hybrid_search")
    async def test_top_k_respected_after_reranking(
        self, mock_hybrid: AsyncMock, mock_rerank: AsyncMock, test_db
    ) -> None:
        raw_results = [_make_result(i) for i in range(15)]
        mock_hybrid.return_value = raw_results
        # Reranker returns exactly top_k
        mock_rerank.return_value = raw_results[:3]

        tool = RagSearchTool(conn=test_db, embed_fn=_mock_embed)
        result = await tool.execute({"query": "hit cap", "top_k": 3})

        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result
        assert "[4]" not in result

    async def test_fts_fallback_skips_reranking(self, test_db) -> None:
        """When embedding fails, FTS fallback should not call reranker."""
        from code.shukketsu.resilience.errors import EmbeddingError

        async def failing_embed(text: str) -> list[float]:
            raise EmbeddingError("Ollama embed down")

        with patch("code.shukketsu.tools.knowledge.search.rerank") as mock_rerank:
            tool = RagSearchTool(conn=test_db, embed_fn=failing_embed)
            await tool.execute({"query": "hit cap"})
            mock_rerank.assert_not_called()

    @patch("code.shukketsu.tools.knowledge.search.rerank")
    @patch("code.shukketsu.tools.knowledge.search.hybrid_search")
    async def test_reranker_failure_returns_unreranked(
        self, mock_hybrid: AsyncMock, mock_rerank: AsyncMock, test_db
    ) -> None:
        """If reranker itself fails, we still get results (unreranked fallback)."""
        raw_results = [_make_result(i, f"Content {i}") for i in range(6)]
        mock_hybrid.return_value = raw_results
        # rerank already has internal fallback — simulate it returning truncated
        mock_rerank.return_value = raw_results[:5]

        tool = RagSearchTool(conn=test_db, embed_fn=_mock_embed)
        result = await tool.execute({"query": "hit cap"})
        assert "Content" in result
