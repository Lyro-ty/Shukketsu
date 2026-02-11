"""Qwen 4B categorical relevance reranker for search results.

Scores search results as HIGH/MEDIUM/LOW and filters to top_k.
Circuit breaker wraps the LLM call — falls back to unreranked on failure.
"""

import logging
from typing import Literal

from pydantic import BaseModel

from code.shukketsu.llm.structured import ModelBackend, get_structured_output
from code.shukketsu.rag.search import SearchResult
from code.shukketsu.resilience.circuit_breaker import qwen_reranker_breaker
from code.shukketsu.resilience.errors import CircuitOpenError

logger = logging.getLogger(__name__)

_CONTENT_TRUNCATE = 200

_SYSTEM_PROMPT = (
    "You are a relevance scorer for WoW TBC Rogue search results.\n"
    "Given a query and numbered search results, rate each result's "
    "relevance as HIGH, MEDIUM, or LOW.\n\n"
    "HIGH: Directly answers or is essential to answering the query.\n"
    "MEDIUM: Related and potentially useful but not a direct answer.\n"
    "LOW: Irrelevant or only tangentially related.\n\n"
    "Be strict — only mark HIGH if the result clearly helps answer the query."
)


class RankedItem(BaseModel):
    """A single result's relevance assessment."""

    index: int
    relevance: Literal["HIGH", "MEDIUM", "LOW"]
    reason: str


class RerankerResponse(BaseModel):
    """Batch relevance assessment from Qwen 4B."""

    rankings: list[RankedItem]


def _build_rerank_messages(query: str, results: list[SearchResult]) -> list[dict[str, str]]:
    """Build the prompt messages for the reranker."""
    result_lines = []
    for i, r in enumerate(results):
        content = r.content[:_CONTENT_TRUNCATE]
        result_lines.append(f"[{i}] {r.source_title} — {content}")

    user_content = f"Query: {query}\n\nResults:\n" + "\n".join(result_lines)

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _apply_rankings(
    results: list[SearchResult],
    rankings: list[RankedItem],
    top_k: int,
) -> list[SearchResult]:
    """Apply categorical rankings with index validation.

    - Ignores out-of-range indices
    - Deduplicates indices (keeps first occurrence)
    - Unranked results treated as LOW (dropped)
    - Within each category, preserves original list order (by rrf_score)
    """
    n = len(results)
    seen: set[int] = set()
    buckets: dict[str, set[int]] = {"HIGH": set(), "MEDIUM": set(), "LOW": set()}

    for item in rankings:
        if item.index < 0 or item.index >= n:
            continue
        if item.index in seen:
            continue
        seen.add(item.index)
        buckets[item.relevance].add(item.index)

    # Collect in original order within each bucket
    high = [results[i] for i in range(n) if i in buckets["HIGH"]]
    medium = [results[i] for i in range(n) if i in buckets["MEDIUM"]]

    # All HIGHs, then fill with MEDIUMs up to top_k
    merged = high[:top_k]
    remaining = top_k - len(merged)
    if remaining > 0:
        merged.extend(medium[:remaining])

    return merged


async def _rerank_impl(query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
    """Call Qwen 4B for relevance scoring."""
    messages = _build_rerank_messages(query, results)
    response = await get_structured_output(
        RerankerResponse,
        messages,
        backend=ModelBackend.ROUTER,
        max_tokens=2048,
    )

    if not response.rankings:
        return results[:top_k]

    return _apply_rankings(results, response.rankings, top_k)


async def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int = 5,
) -> list[SearchResult]:
    """Rerank search results using Qwen 4B categorical scoring.

    Short-circuits if results <= top_k. Falls back to truncated unreranked
    results if the circuit breaker is open or the LLM call fails.

    Args:
        query: The original search query.
        results: Search results to rerank.
        top_k: Number of results to return.

    Returns:
        Filtered and reranked list of SearchResult.
    """
    if len(results) <= top_k:
        return results

    try:
        return await qwen_reranker_breaker.call(_rerank_impl, query, results, top_k)
    except CircuitOpenError:
        logger.warning("Reranker circuit breaker open — returning unreranked results")
        return results[:top_k]
    except Exception:
        logger.warning("Reranker failed — returning unreranked results", exc_info=True)
        return results[:top_k]
