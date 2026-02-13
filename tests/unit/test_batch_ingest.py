"""Tests for batch ingest engine."""

import sqlite3
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code.shukketsu.ingest.batch import (
    BatchIngest,
    BatchReport,
    SourceResult,
    extract_entities_pass,
    print_db_stats,
)
from code.shukketsu.ingest.manifest import SourceEntry
from code.shukketsu.ingest.pipeline import IngestResult
from code.shukketsu.rag.entities import EntityType
from code.shukketsu.scraping.fetcher import FetchResult


def _entry(url: str = "https://example.com", title: str = "Test", **kwargs) -> SourceEntry:  # type: ignore[no-untyped-def]
    return SourceEntry(url=url, title=title, **kwargs)


class TestBatchReport:
    """Tests for BatchReport aggregation."""

    def test_empty_report(self) -> None:
        report = BatchReport()
        assert report.succeeded == 0
        assert report.failed == 0
        assert report.skipped == 0
        assert report.total_chunks == 0

    def test_mixed_results(self) -> None:
        report = BatchReport(
            results=[
                SourceResult(url="a", title="A", success=True, chunks=10),
                SourceResult(url="b", title="B", success=True, chunks=5, already_existed=True),
                SourceResult(url="c", title="C", success=False, error="boom"),
            ]
        )
        assert report.succeeded == 1
        assert report.skipped == 1
        assert report.failed == 1
        assert report.total_chunks == 10  # Only new chunks counted


class TestBatchIngest:
    """Tests for BatchIngest."""

    @pytest.fixture
    def fetcher(self) -> AsyncMock:
        mock = AsyncMock()
        mock.fetch.return_value = FetchResult(
            html="<html><body><h1>Test</h1><p>Content here.</p></body></html>",
            status_code=200,
            final_url="https://example.com",
        )
        return mock

    @pytest.fixture
    def pipeline(self) -> AsyncMock:
        mock = AsyncMock()
        mock.ingest.return_value = IngestResult(source_id=1, chunk_count=5, already_existed=False)
        return mock

    @pytest.fixture
    def batch(self, fetcher: AsyncMock, pipeline: AsyncMock) -> BatchIngest:
        return BatchIngest(fetcher, pipeline, concurrency=2)

    async def test_single_success(self, batch: BatchIngest) -> None:
        report = await batch.ingest_all([_entry()])
        assert report.succeeded == 1
        assert report.total_chunks == 5

    async def test_fetch_failure(self, batch: BatchIngest, fetcher: AsyncMock) -> None:
        fetcher.fetch.side_effect = Exception("Connection refused")
        report = await batch.ingest_all([_entry()])
        assert report.failed == 1
        assert "Connection refused" in report.results[0].error  # type: ignore[operator]

    @patch("code.shukketsu.ingest.batch.trafilatura")
    async def test_empty_extraction(self, mock_traf: MagicMock, batch: BatchIngest) -> None:
        mock_traf.extract.return_value = ""
        report = await batch.ingest_all([_entry()])
        assert report.failed == 1
        assert "Empty extraction" in report.results[0].error  # type: ignore[operator]

    async def test_dedup_skipped(self, batch: BatchIngest, pipeline: AsyncMock) -> None:
        pipeline.ingest.return_value = IngestResult(source_id=1, chunk_count=5, already_existed=True)
        report = await batch.ingest_all([_entry()])
        assert report.skipped == 1
        assert report.succeeded == 0

    async def test_batch_aggregation(self, batch: BatchIngest) -> None:
        sources = [
            _entry(url="https://a.com", title="A"),
            _entry(url="https://b.com", title="B"),
            _entry(url="https://c.com", title="C"),
        ]
        report = await batch.ingest_all(sources)
        assert len(report.results) == 3
        assert report.succeeded == 3

    async def test_disabled_sources_filtered_before_batch(self) -> None:
        """Disabled sources are filtered by Manifest.filter(), not by BatchIngest."""
        from code.shukketsu.ingest.manifest import Manifest

        m = Manifest(
            sources=[
                _entry(url="https://a.com", title="A"),
                _entry(url="https://b.com", title="B", enabled=False),
            ]
        )
        filtered = m.filter()
        assert len(filtered) == 1
        assert filtered[0].title == "A"

    async def test_concurrency_limiting(self, fetcher: AsyncMock, pipeline: AsyncMock) -> None:
        """Concurrency semaphore limits parallel fetches."""
        call_count = 0
        max_concurrent = 0

        original_fetch = fetcher.fetch

        async def tracking_fetch(url: str) -> FetchResult:
            nonlocal call_count, max_concurrent
            call_count += 1
            current = call_count
            if current > max_concurrent:
                max_concurrent = current
            result = await original_fetch(url)
            call_count -= 1
            return result

        fetcher.fetch = tracking_fetch
        batch = BatchIngest(fetcher, pipeline, concurrency=1)
        sources = [_entry(url=f"https://{i}.com", title=f"S{i}") for i in range(3)]
        report = await batch.ingest_all(sources)
        assert report.succeeded == 3
        # With concurrency=1, max concurrent should be 1
        assert max_concurrent <= 1

    async def test_dry_run_not_in_batch(self) -> None:
        """Dry run is handled at CLI level, not in BatchIngest."""
        # Just ensure BatchIngest doesn't have a dry_run param
        assert "dry_run" not in BatchIngest.__init__.__code__.co_varnames


class TestExtractEntitiesPass:
    """Tests for the standalone entity extraction pass."""

    async def test_processes_unextracted_chunks(self, test_db: sqlite3.Connection) -> None:
        """Extract entities from chunks not yet in the entities table."""
        # Insert a source and chunk manually
        test_db.execute(
            "INSERT INTO sources (url, title, source_type, content_hash, fetched_at) "
            "VALUES ('https://a.com', 'A', 'guide', 'abc', '2026-01-01')"
        )
        test_db.execute(
            "INSERT INTO chunks (source_id, content, chunk_index) VALUES (1, 'Sinister Strike is a combat ability', 0)"
        )
        test_db.commit()

        @dataclass
        class FakeEntity:
            name: str = "Sinister Strike"
            entity_type: EntityType = EntityType.SPELL
            properties: dict | None = None

        @dataclass
        class FakeExtraction:
            entities: list = None  # type: ignore[assignment]
            relationships: list = None  # type: ignore[assignment]

            def __post_init__(self) -> None:
                if self.entities is None:
                    self.entities = [FakeEntity()]
                if self.relationships is None:
                    self.relationships = []

        extract_fn = AsyncMock(return_value=FakeExtraction())

        from code.shukketsu.rag.graph import GraphStore

        graph_store = GraphStore(test_db)
        graph_store.seed_entity_types()

        processed, entities, rels = await extract_entities_pass(test_db, graph_store, extract_fn)
        assert processed == 1
        assert entities == 1
        extract_fn.assert_called_once()

    async def test_skips_already_extracted(self, test_db: sqlite3.Connection) -> None:
        """Chunks already associated with entities are skipped."""
        from code.shukketsu.rag.graph import GraphStore

        graph_store = GraphStore(test_db)
        graph_store.seed_entity_types()

        # Insert source, chunk, and entity (already extracted)
        test_db.execute(
            "INSERT INTO sources (url, title, source_type, content_hash, fetched_at) "
            "VALUES ('https://a.com', 'A', 'guide', 'abc', '2026-01-01')"
        )
        test_db.execute("INSERT INTO chunks (source_id, content, chunk_index) VALUES (1, 'Test content', 0)")
        test_db.commit()

        # Create an entity linked to this chunk
        graph_store.upsert_entity("Test Entity", EntityType.SPELL, source_chunk_id=1)
        test_db.commit()

        extract_fn = AsyncMock()
        processed, _, _ = await extract_entities_pass(test_db, graph_store, extract_fn)
        assert processed == 0
        extract_fn.assert_not_called()

    async def test_extraction_failure_continues(self, test_db: sqlite3.Connection) -> None:
        """Failed extraction for one chunk doesn't stop the batch."""
        test_db.execute(
            "INSERT INTO sources (url, title, source_type, content_hash, fetched_at) "
            "VALUES ('https://a.com', 'A', 'guide', 'abc', '2026-01-01')"
        )
        test_db.execute("INSERT INTO chunks (source_id, content, chunk_index) VALUES (1, 'Chunk 1', 0)")
        test_db.execute("INSERT INTO chunks (source_id, content, chunk_index) VALUES (1, 'Chunk 2', 1)")
        test_db.commit()

        from code.shukketsu.rag.graph import GraphStore

        graph_store = GraphStore(test_db)
        graph_store.seed_entity_types()

        extract_fn = AsyncMock(side_effect=Exception("LLM timeout"))
        processed, entities, rels = await extract_entities_pass(test_db, graph_store, extract_fn)
        assert processed == 0  # Both failed
        assert entities == 0
        assert extract_fn.call_count == 2  # Tried both


class TestPrintDbStats:
    """Tests for database statistics output."""

    def test_prints_stats(self, test_db: sqlite3.Connection, capsys: pytest.CaptureFixture[str]) -> None:
        print_db_stats(test_db)
        output = capsys.readouterr().out
        assert "Sources:" in output
        assert "Chunks:" in output
        assert "Entities:" in output
