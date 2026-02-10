"""Ingest pipeline: chunk text, embed it, store in SQLite."""

import hashlib
import logging
import sqlite3
import struct
from dataclasses import dataclass
from datetime import UTC, datetime

from code.shukketsu.ingest.chunker import chunk_text
from code.shukketsu.ingest.embedder import Embedder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    """Result of an ingest operation."""

    source_id: int
    chunk_count: int
    already_existed: bool


class IngestPipeline:
    """Orchestrates chunking, embedding, and storage of documents.

    Handles deduplication via content hashing and stores chunks, vectors,
    and source metadata in a single SQLite transaction.
    """

    def __init__(self, conn: sqlite3.Connection, embedder: Embedder) -> None:
        self._conn = conn
        self._embedder = embedder

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
            IngestResult with source_id, chunk_count, and dedup status.
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
        try:
            if existing:
                source_id = existing["id"]
                self._delete_chunks_and_vectors(source_id)
                self._conn.execute(
                    "UPDATE sources SET title = ?, source_type = ?, content_hash = ?, "
                    "fetched_at = ?, chunk_count = 0 WHERE id = ?",
                    (title, source_type, content_hash, datetime.now(UTC).isoformat(), source_id),
                )
            else:
                cursor = self._conn.execute(
                    "INSERT INTO sources (url, title, source_type, content_hash, fetched_at) VALUES (?, ?, ?, ?, ?)",
                    (url, title, source_type, content_hash, datetime.now(UTC).isoformat()),
                )
                source_id = cursor.lastrowid

            for chunk, embedding in zip(chunks, embeddings):
                cursor = self._conn.execute(
                    "INSERT INTO chunks (source_id, content, chunk_index, metadata_json) VALUES (?, ?, ?, NULL)",
                    (source_id, chunk.content, chunk.chunk_index),
                )
                chunk_id = cursor.lastrowid
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

        logger.info("Ingested %d chunks from %s (%s)", len(chunks), url, title)

        return IngestResult(
            source_id=source_id,
            chunk_count=len(chunks),
            already_existed=False,
        )

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
