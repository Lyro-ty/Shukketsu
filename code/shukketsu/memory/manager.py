"""MemoryManager: session memory recall and strategy learning.

Stores facts extracted from conversations and retrieval strategies.
All writes are fire-and-forget -- errors are logged, never raised.
"""

import asyncio
import json
import logging
import math
import sqlite3
import struct
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from code.shukketsu import config
from code.shukketsu.memory.models import MemoryExtraction, SessionMemory

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Awaitable[list[float]]]


def _composite_score(similarity: float, recency: float, quality: float) -> float:
    """Compute composite memory score: 0.6*sim + 0.2*recency + 0.2*quality."""
    return (
        config.MEMORY_COMPOSITE_SIM_WEIGHT * similarity
        + config.MEMORY_COMPOSITE_RECENCY_WEIGHT * recency
        + config.MEMORY_COMPOSITE_QUALITY_WEIGHT * quality
    )


def _recency_score(created_at: str, half_life_days: float = config.MEMORY_RECENCY_HALF_LIFE_DAYS) -> float:
    """Exponential decay recency score: exp(-days / half_life_days)."""
    try:
        created = datetime.fromisoformat(created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        age_days = max(0.0, (datetime.now(UTC) - created).total_seconds() / 86400.0)
        return math.exp(-age_days / half_life_days)
    except (ValueError, TypeError):
        logger.debug("Invalid created_at timestamp %r, using default recency", created_at)
        return 0.5


class MemoryManager:
    """Manages session memories and retrieval strategy learning.

    All public methods are fire-and-forget on write paths -- errors are
    logged but never propagated to callers.
    """

    def __init__(self, conn: sqlite3.Connection, embed_fn: EmbedFn) -> None:
        self._conn = conn
        self._embed_fn = embed_fn
        self._write_lock = asyncio.Lock()

    async def extract_session_memory(
        self,
        query: str,
        answer: str,
        trajectory: list[dict],
        session_id: str | None = None,
    ) -> None:
        """Extract key facts from a conversation turn and store in DB.

        Uses Qwen 4B via get_structured_output to extract a MemoryExtraction,
        then stores the result + embedding. Fire-and-forget: errors are logged.
        """
        try:
            from code.shukketsu.llm.structured import ModelBackend, get_structured_output

            messages = [
                {
                    "role": "system",
                    "content": (
                        "Extract the key facts from this conversation. "
                        "Summarize the answer concisely. List specific facts "
                        "and any WoW entities mentioned."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nAnswer: {answer}",
                },
            ]

            extraction: MemoryExtraction = await get_structured_output(
                response_model=MemoryExtraction,
                messages=messages,
                backend=ModelBackend.ROUTER,
                max_tokens=512,
            )

            # Embed before acquiring lock (network I/O, not DB-bound)
            embedding = await self._embed_fn(query)
            blob = struct.pack(f"{len(embedding)}f", *embedding)

            # Hold lock for all DB writes to prevent interleaved transactions
            async with self._write_lock:
                cursor = self._conn.execute(
                    """INSERT INTO session_memories
                       (query, answer_summary, key_facts_json, entities_mentioned,
                        retrieval_quality, session_id)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        query,
                        extraction.summary,
                        json.dumps(extraction.key_facts),
                        json.dumps(extraction.entities_mentioned),
                        0.5,  # Default quality; updated on feedback
                        session_id,
                    ),
                )
                row_id = cursor.lastrowid
                if row_id is None or row_id == 0:
                    raise RuntimeError("INSERT into session_memories returned no rowid")

                self._conn.execute(
                    "INSERT INTO session_memories_vec (rowid, embedding) VALUES (?, ?)",
                    (row_id, blob),
                )
                self._conn.commit()

        except Exception:
            self._conn.rollback()
            logger.warning("Memory extraction failed", exc_info=True)

    async def recall_relevant(
        self,
        query: str,
        top_k: int = config.MEMORY_RECALL_TOP_K,
    ) -> list[SessionMemory]:
        """Recall the most relevant session memories for a query.

        Embeds the query, performs KNN search on session_memories_vec,
        then scores with composite: 0.6*cosine_sim + 0.2*recency + 0.2*quality.

        Returns up to top_k SessionMemory objects sorted by composite score.
        """
        try:
            # Check if any memories exist
            count = self._conn.execute("SELECT COUNT(*) FROM session_memories").fetchone()[0]
            if count == 0:
                return []

            embedding = await self._embed_fn(query)
            blob = struct.pack(f"{len(embedding)}f", *embedding)

            # KNN search -- fetch more than top_k so composite scoring can reorder
            fetch_k = min(count, top_k * 3)
            rows = self._conn.execute(
                """SELECT v.rowid, v.distance, m.query, m.answer_summary,
                          m.key_facts_json, m.entities_mentioned,
                          m.retrieval_quality, m.created_at
                   FROM (
                       SELECT rowid, distance
                       FROM session_memories_vec
                       WHERE embedding MATCH ? AND k = ?
                   ) v
                   JOIN session_memories m ON m.id = v.rowid""",
                (blob, fetch_k),
            ).fetchall()

            memories: list[SessionMemory] = []
            for row in rows:
                # cosine distance -> similarity: 1 - distance (sqlite-vec cosine returns distance)
                similarity = max(0.0, 1.0 - row["distance"])
                recency = _recency_score(row["created_at"])
                quality = row["retrieval_quality"]
                score = _composite_score(similarity, recency, quality)

                memories.append(
                    SessionMemory(
                        id=row["rowid"],
                        query=row["query"],
                        answer_summary=row["answer_summary"],
                        key_facts=json.loads(row["key_facts_json"]),
                        entities_mentioned=json.loads(row["entities_mentioned"]),
                        retrieval_quality=quality,
                        created_at=row["created_at"],
                        score=score,
                    )
                )

            memories.sort(key=lambda m: m.score, reverse=True)
            return memories[:top_k]

        except Exception:
            logger.warning("Memory recall failed", exc_info=True)
            return []

    async def record_strategy(
        self,
        query: str,
        tools_used: list[str],
        quality: float,
        strategy_type: str = "routing",
    ) -> None:
        """Record or update a retrieval strategy. Fire-and-forget.

        If a strategy with the same query_pattern exists, increments
        times_reinforced and updates the running average quality.
        Otherwise inserts a new row.
        """
        try:
            async with self._write_lock:
                existing = self._conn.execute(
                    "SELECT id, times_reinforced, avg_quality FROM strategy_memories WHERE query_pattern = ?",
                    (query,),
                ).fetchone()

                if existing:
                    new_reinforced = existing["times_reinforced"] + 1
                    new_avg = (existing["avg_quality"] * existing["times_reinforced"] + quality) / new_reinforced
                    self._conn.execute(
                        """UPDATE strategy_memories
                           SET times_reinforced = ?,
                               avg_quality = ?,
                               last_used = datetime('now'),
                               successful_tools = ?
                           WHERE id = ?""",
                        (
                            new_reinforced,
                            new_avg,
                            json.dumps(tools_used),
                            existing["id"],
                        ),
                    )
                else:
                    self._conn.execute(
                        """INSERT INTO strategy_memories
                           (query_pattern, strategy_type, successful_tools, avg_quality)
                           VALUES (?, ?, ?, ?)""",
                        (query, strategy_type, json.dumps(tools_used), quality),
                    )
                self._conn.commit()

        except Exception:
            logger.warning("Strategy recording failed", exc_info=True)

    async def recall_strategies(
        self,
        top_k: int = config.MEMORY_STRATEGY_TOP_K,
    ) -> list[dict]:
        """Recall the highest-quality strategies.

        Returns up to top_k strategy dicts sorted by avg_quality descending.
        """
        try:
            rows = self._conn.execute(
                """SELECT query_pattern, strategy_type, successful_tools,
                          failed_tools, best_sources, times_reinforced,
                          avg_quality, last_used
                   FROM strategy_memories
                   ORDER BY avg_quality DESC
                   LIMIT ?""",
                (top_k,),
            ).fetchall()

            return [
                {
                    "query_pattern": row["query_pattern"],
                    "strategy_type": row["strategy_type"],
                    "successful_tools": json.loads(row["successful_tools"]),
                    "failed_tools": json.loads(row["failed_tools"]),
                    "best_sources": json.loads(row["best_sources"]),
                    "times_reinforced": row["times_reinforced"],
                    "avg_quality": row["avg_quality"],
                    "last_used": row["last_used"],
                }
                for row in rows
            ]

        except Exception:
            logger.warning("Strategy recall failed", exc_info=True)
            return []
