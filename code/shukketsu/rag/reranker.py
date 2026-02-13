"""Cross-encoder reranker for search results.

Uses a lightweight cross-encoder model to score query-document relevance.
Circuit breaker wraps the scoring call — falls back to unreranked on failure.
"""

import asyncio
import logging
from dataclasses import replace
from typing import Any

from code.shukketsu import config
from code.shukketsu.rag.search import SearchResult
from code.shukketsu.resilience.circuit_breaker import reranker_breaker
from code.shukketsu.resilience.errors import CircuitOpenError

logger = logging.getLogger(__name__)

_model = None


def _get_model() -> Any:
    """Lazily load and cache the cross-encoder model.

    The model is loaded on first call and cached for subsequent calls.
    Runs on CPU (PyTorch sm_121 kernels not compiled, model is tiny anyway).
    """
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder model: %s", config.RERANKER_MODEL)
        _model = CrossEncoder(config.RERANKER_MODEL, device="cpu")
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

    return [replace(r, rerank_score=float(s)) for r, s in scored[:top_k]]


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
        result: list[SearchResult] = await reranker_breaker.call(_rerank_impl, query, results, top_k)
        return result
    except CircuitOpenError:
        logger.warning("Reranker circuit breaker open — returning unreranked results")
        return results[:top_k]
    except Exception:
        logger.warning("Reranker failed — returning unreranked results", exc_info=True)
        return results[:top_k]
