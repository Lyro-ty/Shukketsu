"""RAG search tool for vector similarity search over the knowledge base."""

import logging
import sqlite3
import struct
from collections.abc import Callable, Coroutine
from typing import Any

from code.shukketsu import config
from code.shukketsu.tools.schemas import Tool

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Coroutine[Any, Any, list[float]]]


class RagSearchTool(Tool):
    """Search the knowledge base using vector similarity.

    Queries chunks_vec for cosine similarity, joins back to chunks
    and sources for content and metadata.
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
        """Execute a vector similarity search."""
        query = tool_input.get("query", "")
        top_k = tool_input.get("top_k", config.RAG_SEARCH_TOP_K)

        if not query:
            return "Error: 'query' parameter is required."

        embedding = await self._embed_fn(query)
        query_blob = struct.pack(f"{len(embedding)}f", *embedding)

        rows = self._conn.execute(
            """
            SELECT c.id, c.content, c.chunk_index, s.url, s.title, s.trust_score,
                   v.distance
            FROM (
                SELECT rowid, distance
                FROM chunks_vec
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            ) v
            JOIN chunks c ON c.id = v.rowid
            JOIN sources s ON s.id = c.source_id
            """,
            (query_blob, top_k),
        ).fetchall()

        if not rows:
            return "No relevant documents found for this query."

        parts = [f"Found {len(rows)} result{'s' if len(rows) != 1 else ''}:\n"]
        for i, row in enumerate(rows, 1):
            parts.append(
                f"[{i}] Source: {row['title']} ({row['url']})\nTrust: {row['trust_score']}\nContent: {row['content']}\n"
            )

        return "\n".join(parts)
