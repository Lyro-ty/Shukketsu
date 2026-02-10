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
                f"[{i}] Source: {r.source_title} ({r.source_url})\nTrust: {r.trust_score}\nContent: {r.content}\n"
            )

        return "\n".join(parts)
