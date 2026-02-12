"""Tests for ingest pipeline freshness integration."""

import sqlite3
from unittest.mock import AsyncMock

from code.shukketsu.ingest.pipeline import IngestPipeline

EMBEDDING_DIM = 768


def _mock_embedder() -> AsyncMock:
    embedder = AsyncMock()

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[0.1] * EMBEDDING_DIM for _ in texts]

    embedder.embed_texts = AsyncMock(side_effect=fake_embed_texts)
    return embedder


class TestIngestFreshness:
    async def test_new_source_gets_domain_check_interval(self, test_db: sqlite3.Connection) -> None:
        """wowhead.com source gets 720h, unknown domain gets 168h."""
        pipeline = IngestPipeline(conn=test_db, embedder=_mock_embedder())

        await pipeline.ingest("Some guide content here.", "https://wowhead.com/tbc/guide", "WoWHead Guide")
        row = test_db.execute(
            "SELECT check_interval_hours FROM sources WHERE url = ?", ("https://wowhead.com/tbc/guide",)
        ).fetchone()
        assert row["check_interval_hours"] == 720

        await pipeline.ingest("Another guide.", "https://unknown-site.com/guide", "Unknown Guide")
        row = test_db.execute(
            "SELECT check_interval_hours FROM sources WHERE url = ?", ("https://unknown-site.com/guide",)
        ).fetchone()
        assert row["check_interval_hours"] == 168

    async def test_reingest_changed_content_updates_tracking(self, test_db: sqlite3.Connection) -> None:
        """On re-ingest with different content: hash_previous set, change_count incremented, is_stale cleared."""
        pipeline = IngestPipeline(conn=test_db, embedder=_mock_embedder())

        # First ingest
        await pipeline.ingest("Original content.", "https://example.com/guide", "Guide")
        row = test_db.execute(
            "SELECT id, content_hash, change_count, is_stale FROM sources WHERE url = ?",
            ("https://example.com/guide",),
        ).fetchone()
        original_hash = row["content_hash"]
        assert row["change_count"] == 0

        # Mark as stale (simulating freshness checker)
        test_db.execute("UPDATE sources SET is_stale = 1 WHERE id = ?", (row["id"],))
        test_db.commit()

        # Re-ingest with different content
        await pipeline.ingest("Updated content.", "https://example.com/guide", "Guide")
        row = test_db.execute(
            "SELECT content_hash, content_hash_previous, change_count, is_stale FROM sources WHERE url = ?",
            ("https://example.com/guide",),
        ).fetchone()
        assert row["content_hash_previous"] == original_hash
        assert row["content_hash"] != original_hash
        assert row["change_count"] == 1
        assert row["is_stale"] == 0

    async def test_reingest_same_content_no_tracking_change(self, test_db: sqlite3.Connection) -> None:
        """Unchanged content doesn't modify tracking columns."""
        pipeline = IngestPipeline(conn=test_db, embedder=_mock_embedder())

        await pipeline.ingest("Same content.", "https://example.com/guide", "Guide")
        result = await pipeline.ingest("Same content.", "https://example.com/guide", "Guide")
        assert result.already_existed

        row = test_db.execute(
            "SELECT change_count, is_stale FROM sources WHERE url = ?", ("https://example.com/guide",)
        ).fetchone()
        assert row["change_count"] == 0
        assert row["is_stale"] == 0
