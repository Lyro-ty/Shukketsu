"""Ingest pipeline: chunk text, embed it, extract entities, store in SQLite."""

import hashlib
import logging
import sqlite3
import struct
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from code.shukketsu.ingest.chunker import chunk_text
from code.shukketsu.ingest.embedder import Embedder
from code.shukketsu.rag.entities import ChunkExtraction
from code.shukketsu.rag.graph import GraphStore

logger = logging.getLogger(__name__)

# Type alias for the extraction function
ExtractFn = Callable[[str], Awaitable[ChunkExtraction]]


@dataclass(frozen=True)
class IngestResult:
    """Result of an ingest operation."""

    source_id: int
    chunk_count: int
    already_existed: bool
    entity_count: int = 0
    relationship_count: int = 0


class IngestPipeline:
    """Orchestrates chunking, embedding, entity extraction, and storage.

    Handles deduplication via content hashing and stores chunks, vectors,
    entities, and source metadata in a single SQLite transaction.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedder: Embedder,
        *,
        graph_store: GraphStore | None = None,
        extract_fn: ExtractFn | None = None,
    ) -> None:
        self._conn = conn
        self._embedder = embedder
        self._graph_store = graph_store
        self._extract_fn = extract_fn

    async def ingest(
        self,
        text: str,
        url: str,
        title: str,
        source_type: str = "guide",
    ) -> IngestResult:
        """Ingest text into the knowledge base.

        Args:
            text: The full text content to ingest.
            url: Source URL (used as unique key for dedup).
            title: Human-readable title for the source.
            source_type: Category of source (guide, forum, wiki, etc.).

        Returns:
            IngestResult with source_id, chunk_count, entity stats, and dedup status.
        """
        content_hash = hashlib.sha256(text.encode()).hexdigest() if text.strip() else ""

        # Check for existing source with the same URL
        existing = self._conn.execute("SELECT id, content_hash FROM sources WHERE url = ?", (url,)).fetchone()

        if existing:
            if existing["content_hash"] == content_hash:
                chunk_count = self._conn.execute(
                    "SELECT chunk_count FROM sources WHERE id = ?", (existing["id"],)
                ).fetchone()["chunk_count"]
                return IngestResult(
                    source_id=existing["id"],
                    chunk_count=chunk_count,
                    already_existed=True,
                )

        # Embed BEFORE touching the database (this is the most likely failure point)
        if text.strip():
            chunks = chunk_text(text)
            embeddings = await self._embedder.embed_texts([c.content for c in chunks])
        else:
            chunks = []
            embeddings = []

        # Now write everything in a single transaction
        entity_count = 0
        relationship_count = 0
        try:
            if existing:
                source_id = existing["id"]
                self._delete_chunks_and_vectors(source_id)
                now_iso = datetime.now(UTC).isoformat()
                self._conn.execute(
                    "UPDATE sources SET title = ?, source_type = ?, "
                    "content_hash_previous = content_hash, content_hash = ?, "
                    "change_count = change_count + 1, is_stale = 0, "
                    "fetched_at = ?, last_checked = ?, chunk_count = 0 WHERE id = ?",
                    (title, source_type, content_hash, now_iso, now_iso, source_id),
                )
            else:
                from code.shukketsu.config import get_check_interval

                now_iso = datetime.now(UTC).isoformat()
                check_interval = get_check_interval(url)
                cursor = self._conn.execute(
                    "INSERT INTO sources (url, title, source_type, content_hash, fetched_at, "
                    "check_interval_hours, last_checked) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (url, title, source_type, content_hash, now_iso, check_interval, now_iso),
                )
                source_id = cursor.lastrowid

            chunk_ids: list[int] = []
            for chunk, embedding in zip(chunks, embeddings):
                cursor = self._conn.execute(
                    "INSERT INTO chunks (source_id, content, chunk_index, metadata_json) VALUES (?, ?, ?, NULL)",
                    (source_id, chunk.content, chunk.chunk_index),
                )
                chunk_id = int(cursor.lastrowid)  # type: ignore[arg-type]
                chunk_ids.append(chunk_id)
                embedding_blob = struct.pack(f"{len(embedding)}f", *embedding)
                self._conn.execute(
                    "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                    (chunk_id, embedding_blob),
                )

            self._conn.execute(
                "UPDATE sources SET chunk_count = ? WHERE id = ?",
                (len(chunks), source_id),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        # Entity extraction (after commit — non-critical, best-effort)
        if self._graph_store and self._extract_fn and chunks:
            e_count, r_count = await self._extract_and_store(chunks, chunk_ids)
            entity_count = e_count
            relationship_count = r_count

        logger.info(
            "Ingested %d chunks, %d entities from %s (%s)",
            len(chunks),
            entity_count,
            url,
            title,
        )

        return IngestResult(
            source_id=source_id,
            chunk_count=len(chunks),
            already_existed=False,
            entity_count=entity_count,
            relationship_count=relationship_count,
        )

    async def _extract_and_store(
        self,
        chunks: list,
        chunk_ids: list[int],
    ) -> tuple[int, int]:
        """Extract entities from chunks and store in graph. Best-effort."""
        total_entities = 0
        total_relationships = 0

        for chunk, chunk_id in zip(chunks, chunk_ids):
            try:
                extraction = await self._extract_fn(chunk.content)  # type: ignore[misc]
            except Exception as exc:
                logger.warning("Entity extraction failed for chunk %d: %s", chunk_id, exc)
                continue

            entity_id_map: dict[str, int] = {}
            for entity in extraction.entities:
                eid = self._graph_store.upsert_entity(  # type: ignore[union-attr]
                    entity.name,
                    entity.entity_type,
                    properties=entity.properties or None,
                    source_chunk_id=chunk_id,
                )
                entity_id_map[entity.name] = eid
                total_entities += 1

            for rel in extraction.relationships:
                src_id = entity_id_map.get(rel.source)
                tgt_id = entity_id_map.get(rel.target)
                if src_id is None or tgt_id is None:
                    logger.debug(
                        "Skipping relationship %s->%s: entity not in this chunk's extraction",
                        rel.source,
                        rel.target,
                    )
                    continue
                self._graph_store.upsert_relationship(  # type: ignore[union-attr]
                    src_id,
                    tgt_id,
                    rel.relation_type,
                    properties=rel.properties or None,
                    source_chunk_id=chunk_id,
                )
                total_relationships += 1

        self._conn.commit()
        return total_entities, total_relationships

    def _delete_chunks_and_vectors(self, source_id: int) -> None:
        """Delete all chunks and their vectors for a source.

        Must delete from chunks_vec first since it is a virtual table
        not covered by ON DELETE CASCADE.
        """
        self._conn.execute(
            "DELETE FROM chunks_vec WHERE rowid IN (SELECT id FROM chunks WHERE source_id = ?)",
            (source_id,),
        )
        self._conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
