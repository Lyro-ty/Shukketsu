"""RAG search tool for hybrid vector + keyword search over the knowledge base."""

import logging
import sqlite3
from collections.abc import Callable, Coroutine
from typing import Any

from code.shukketsu import config
from code.shukketsu.rag.fusion import escape_fts_query
from code.shukketsu.rag.reranker import rerank
from code.shukketsu.rag.search import hybrid_search
from code.shukketsu.resilience.circuit_breaker import ollama_embed_breaker
from code.shukketsu.tools.schemas import Tool

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Coroutine[Any, Any, list[float]]]


class RagSearchTool(Tool):
    """Search the knowledge base using hybrid vector + keyword search.

    Combines semantic vector similarity (chunks_vec) with FTS5 keyword
    matching (chunks_fts), merged via Reciprocal Rank Fusion.
    Falls back to FTS5-only keyword search when embedding is unavailable.
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
        """Execute a hybrid search (vector + FTS5 + RRF) with reranking."""
        query = tool_input.get("query", "")
        top_k = tool_input.get("top_k", config.RAG_SEARCH_TOP_K)

        if not query:
            return "Error: 'query' parameter is required."

        try:
            embedding = await ollama_embed_breaker.call(self._embed_fn, query)
        except Exception as exc:
            logger.warning("Embedding failed, falling back to keyword-only search: %s", exc)
            return self._fts_only_search(query, top_k)

        # Over-retrieve for reranking
        rerank_fetch = top_k * config.RERANKER_FETCH_MULTIPLIER
        effective_fetch_k = max(config.RAG_SEARCH_FETCH_K, rerank_fetch)
        raw_results = await hybrid_search(self._conn, query, embedding, top_k=rerank_fetch, fetch_k=effective_fetch_k)

        # Rerank (falls back to truncated on failure internally)
        results = await rerank(query, raw_results, top_k)

        if not results:
            return "No relevant documents found for this query."

        parts = [f"Found {len(results)} result{'s' if len(results) != 1 else ''}:\n"]
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] Source: {r.source_title} ({r.source_url})\nTrust: {r.trust_score}\nContent: {r.content}\n"
            )

        return "\n".join(parts)

    def _fts_only_search(self, query: str, top_k: int) -> str:
        """Fallback keyword-only search when embedding is unavailable."""
        fts_query = escape_fts_query(query)
        if not fts_query:
            return "Error: Embedding model unavailable and no keyword query provided."

        rows = self._conn.execute(
            """SELECT c.id, c.content, s.title, s.url, s.trust_score
               FROM chunks_fts f
               JOIN chunks c ON c.id = f.rowid
               JOIN sources s ON s.id = c.source_id AND s.is_stale = 0
               WHERE chunks_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (fts_query, top_k),
        ).fetchall()

        if not rows:
            return "No relevant documents found (keyword search only — embedding model unavailable)."

        parts = [f"Found {len(rows)} result{'s' if len(rows) != 1 else ''} (keyword search only):\n"]
        for i, r in enumerate(rows, 1):
            parts.append(
                f"[{i}] Source: {r['title']} ({r['url']})\nTrust: {r['trust_score']}\nContent: {r['content']}\n"
            )

        return "\n".join(parts)
