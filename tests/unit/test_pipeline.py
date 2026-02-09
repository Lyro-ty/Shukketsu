"""Tests for the ingest pipeline."""

import sqlite3
from unittest.mock import AsyncMock

from code.shukketsu.ingest.pipeline import IngestPipeline, IngestResult

EMBEDDING_DIM = 768


def _mock_embedder() -> AsyncMock:
    """Create a mock embedder that returns deterministic vectors."""
    embedder = AsyncMock()

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[0.1] * EMBEDDING_DIM for _ in texts]

    embedder.embed_texts = AsyncMock(side_effect=fake_embed_texts)
    return embedder


class TestIngestPipeline:
    """Tests for IngestPipeline.ingest()."""

    async def test_ingest_stores_source(self, test_db: sqlite3.Connection) -> None:
        pipeline = IngestPipeline(conn=test_db, embedder=_mock_embedder())
        result = await pipeline.ingest(
            text="The hit cap for combat rogues is 9%.",
            url="https://example.com/guide",
            title="Rogue Guide",
        )
        row = test_db.execute(
            "SELECT url, title, source_type FROM sources WHERE id = ?", (result.source_id,)
        ).fetchone()
        assert row["url"] == "https://example.com/guide"
        assert row["title"] == "Rogue Guide"
        assert row["source_type"] == "guide"

    async def test_ingest_stores_chunks(self, test_db: sqlite3.Connection) -> None:
        pipeline = IngestPipeline(conn=test_db, embedder=_mock_embedder())
        text = "The hit cap for combat rogues is 9%. " * 50  # enough to split
        result = await pipeline.ingest(
            text=text,
            url="https://example.com/guide",
            title="Guide",
        )
        assert result.chunk_count > 0
        rows = test_db.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_id = ?", (result.source_id,)
        ).fetchone()
        assert rows[0] == result.chunk_count

    async def test_ingest_stores_vectors(self, test_db: sqlite3.Connection) -> None:
        pipeline = IngestPipeline(conn=test_db, embedder=_mock_embedder())
        result = await pipeline.ingest(
            text="The hit cap for combat rogues is 9%.",
            url="https://example.com/guide",
            title="Guide",
        )
        # chunks_vec should have the same number of rows as chunks
        chunk_ids = test_db.execute(
            "SELECT id FROM chunks WHERE source_id = ?", (result.source_id,)
        ).fetchall()
        for row in chunk_ids:
            vec_row = test_db.execute(
                "SELECT rowid FROM chunks_vec WHERE rowid = ?", (row["id"],)
            ).fetchone()
            assert vec_row is not None

    async def test_ingest_updates_chunk_count(self, test_db: sqlite3.Connection) -> None:
        pipeline = IngestPipeline(conn=test_db, embedder=_mock_embedder())
        result = await pipeline.ingest(
            text="The hit cap for combat rogues is 9%.",
            url="https://example.com/guide",
            title="Guide",
        )
        row = test_db.execute("SELECT chunk_count FROM sources WHERE id = ?", (result.source_id,)).fetchone()
        assert row["chunk_count"] == result.chunk_count

    async def test_ingest_returns_correct_result(self, test_db: sqlite3.Connection) -> None:
        pipeline = IngestPipeline(conn=test_db, embedder=_mock_embedder())
        result = await pipeline.ingest(
            text="Short text.",
            url="https://example.com/guide",
            title="Guide",
        )
        assert isinstance(result, IngestResult)
        assert result.source_id > 0
        assert result.chunk_count >= 1
        assert result.already_existed is False

    async def test_dedup_skips_identical_content(self, test_db: sqlite3.Connection) -> None:
        pipeline = IngestPipeline(conn=test_db, embedder=_mock_embedder())
        text = "Exact same content."
        result1 = await pipeline.ingest(text=text, url="https://example.com/a", title="A")
        result2 = await pipeline.ingest(text=text, url="https://example.com/a", title="A")
        assert result2.already_existed is True
        assert result2.source_id == result1.source_id

    async def test_changed_content_reingests(self, test_db: sqlite3.Connection) -> None:
        pipeline = IngestPipeline(conn=test_db, embedder=_mock_embedder())
        result1 = await pipeline.ingest(text="Version one.", url="https://example.com/a", title="A")
        result2 = await pipeline.ingest(text="Version two.", url="https://example.com/a", title="A")
        assert result2.already_existed is False
        assert result2.source_id == result1.source_id
        # Old chunks should be replaced
        rows = test_db.execute("SELECT content FROM chunks WHERE source_id = ?", (result2.source_id,)).fetchall()
        contents = [r["content"] for r in rows]
        assert any("Version two" in c for c in contents)
        assert not any("Version one" in c for c in contents)

    async def test_empty_text_returns_zero_chunks(self, test_db: sqlite3.Connection) -> None:
        pipeline = IngestPipeline(conn=test_db, embedder=_mock_embedder())
        result = await pipeline.ingest(text="", url="https://example.com/empty", title="Empty")
        assert result.chunk_count == 0
