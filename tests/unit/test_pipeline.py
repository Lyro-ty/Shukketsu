"""Tests for the ingest pipeline."""

import sqlite3
from unittest.mock import AsyncMock

import pytest

from code.shukketsu.ingest.pipeline import IngestPipeline, IngestResult
from code.shukketsu.rag.entities import (
    ChunkExtraction,
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
    RelationType,
)
from code.shukketsu.rag.graph import GraphStore
from code.shukketsu.resilience.errors import EmbeddingError

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
        rows = test_db.execute("SELECT COUNT(*) FROM chunks WHERE source_id = ?", (result.source_id,)).fetchone()
        assert rows[0] == result.chunk_count

    async def test_ingest_stores_vectors(self, test_db: sqlite3.Connection) -> None:
        pipeline = IngestPipeline(conn=test_db, embedder=_mock_embedder())
        result = await pipeline.ingest(
            text="The hit cap for combat rogues is 9%.",
            url="https://example.com/guide",
            title="Guide",
        )
        # chunks_vec should have the same number of rows as chunks
        chunk_ids = test_db.execute("SELECT id FROM chunks WHERE source_id = ?", (result.source_id,)).fetchall()
        for row in chunk_ids:
            vec_row = test_db.execute("SELECT rowid FROM chunks_vec WHERE rowid = ?", (row["id"],)).fetchone()
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

    async def test_embedding_failure_rolls_back_source(self, test_db: sqlite3.Connection) -> None:
        """If embedding fails, the source row should be rolled back so re-ingest works."""
        embedder = _mock_embedder()
        embedder.embed_texts = AsyncMock(side_effect=EmbeddingError("GPU OOM"))
        pipeline = IngestPipeline(conn=test_db, embedder=embedder)

        with pytest.raises(EmbeddingError):
            await pipeline.ingest(text="Some real content.", url="https://example.com/fail", title="Fail")

        # Source should NOT exist — it was rolled back
        row = test_db.execute("SELECT id FROM sources WHERE url = ?", ("https://example.com/fail",)).fetchone()
        assert row is None

        # Re-ingest with a working embedder should succeed
        pipeline2 = IngestPipeline(conn=test_db, embedder=_mock_embedder())
        result = await pipeline2.ingest(text="Some real content.", url="https://example.com/fail", title="Fail")
        assert result.already_existed is False
        assert result.chunk_count >= 1


def _mock_extractor(extraction: ChunkExtraction | None = None) -> AsyncMock:
    """Create a mock entity extractor."""
    if extraction is None:
        extraction = ChunkExtraction(
            entities=[
                ExtractedEntity(name="Sinister Strike", entity_type=EntityType.SPELL),
                ExtractedEntity(name="Hit Rating", entity_type=EntityType.STAT),
            ],
            relationships=[
                ExtractedRelationship(
                    source="Sinister Strike",
                    target="Hit Rating",
                    relation_type=RelationType.AFFECTED_BY,
                ),
            ],
        )

    async def fake_extract(chunk_text: str) -> ChunkExtraction:
        return extraction

    return AsyncMock(side_effect=fake_extract)


class TestIngestPipelineWithExtraction:
    """Tests for entity extraction during ingest."""

    async def test_extraction_stores_entities(self, test_db: sqlite3.Connection) -> None:
        graph = GraphStore(test_db)
        graph.seed_entity_types()
        pipeline = IngestPipeline(
            conn=test_db,
            embedder=_mock_embedder(),
            graph_store=graph,
            extract_fn=_mock_extractor(),
        )
        await pipeline.ingest(
            text="The hit cap for Sinister Strike is 9%.",
            url="https://example.com/guide",
            title="Guide",
        )
        entity = graph.get_entity_by_name("Sinister Strike")
        assert entity is not None

    async def test_extraction_stores_relationships(self, test_db: sqlite3.Connection) -> None:
        graph = GraphStore(test_db)
        graph.seed_entity_types()
        pipeline = IngestPipeline(
            conn=test_db,
            embedder=_mock_embedder(),
            graph_store=graph,
            extract_fn=_mock_extractor(),
        )
        await pipeline.ingest(
            text="Sinister Strike is affected by Hit Rating.",
            url="https://example.com/guide",
            title="Guide",
        )
        entity = graph.get_entity_by_name("Sinister Strike")
        rels = graph.get_relationships(entity["id"])
        assert len(rels) >= 1

    async def test_no_extraction_without_graph_store(self, test_db: sqlite3.Connection) -> None:
        """Pipeline without graph_store should not attempt extraction."""
        extractor = _mock_extractor()
        pipeline = IngestPipeline(
            conn=test_db,
            embedder=_mock_embedder(),
            extract_fn=extractor,
        )
        await pipeline.ingest(
            text="Some content.",
            url="https://example.com/guide",
            title="Guide",
        )
        extractor.assert_not_called()

    async def test_extraction_failure_does_not_abort_ingest(
        self, test_db: sqlite3.Connection
    ) -> None:
        """If extraction fails for a chunk, ingest should still succeed."""
        from code.shukketsu.resilience.errors import EntityExtractionError

        graph = GraphStore(test_db)
        graph.seed_entity_types()

        failing_extractor = AsyncMock(
            side_effect=EntityExtractionError("LLM down")
        )
        pipeline = IngestPipeline(
            conn=test_db,
            embedder=_mock_embedder(),
            graph_store=graph,
            extract_fn=failing_extractor,
        )
        result = await pipeline.ingest(
            text="The hit cap for combat rogues is 9%.",
            url="https://example.com/guide",
            title="Guide",
        )
        assert result.chunk_count >= 1  # ingest succeeded despite extraction failure

    async def test_extraction_called_per_chunk(self, test_db: sqlite3.Connection) -> None:
        """Extractor should be called once per chunk."""
        graph = GraphStore(test_db)
        graph.seed_entity_types()
        extractor = _mock_extractor(ChunkExtraction(entities=[], relationships=[]))
        pipeline = IngestPipeline(
            conn=test_db,
            embedder=_mock_embedder(),
            graph_store=graph,
            extract_fn=extractor,
        )
        text = "The hit cap for combat rogues is 9%. " * 50  # enough to split into multiple chunks
        result = await pipeline.ingest(text=text, url="https://example.com/guide", title="Guide")
        assert extractor.call_count == result.chunk_count

    async def test_ingest_result_includes_entity_count(self, test_db: sqlite3.Connection) -> None:
        """IngestResult should report how many entities were extracted."""
        graph = GraphStore(test_db)
        graph.seed_entity_types()
        pipeline = IngestPipeline(
            conn=test_db,
            embedder=_mock_embedder(),
            graph_store=graph,
            extract_fn=_mock_extractor(),
        )
        result = await pipeline.ingest(
            text="Sinister Strike and Hit Rating.",
            url="https://example.com/guide",
            title="Guide",
        )
        assert result.entity_count >= 0  # new field

    async def test_existing_pipeline_still_works(self, test_db: sqlite3.Connection) -> None:
        """Pipeline without new params should work exactly as before."""
        pipeline = IngestPipeline(conn=test_db, embedder=_mock_embedder())
        result = await pipeline.ingest(
            text="Basic content.",
            url="https://example.com/guide",
            title="Guide",
        )
        assert result.chunk_count >= 1
        assert result.already_existed is False
