"""Integration test fixtures."""

import sqlite3
from pathlib import Path

import pytest

from code.shukketsu.db.connection import get_connection, init_db


@pytest.fixture
def seed_content_dir() -> Path:
    """Path to the eval seed content directory."""
    return Path(__file__).parent.parent.parent / "code" / "shukketsu" / "evals" / "datasets" / "seed_content"


@pytest.fixture
def integration_db(tmp_path: Path) -> sqlite3.Connection:
    """Fresh database for integration tests."""
    conn = get_connection(tmp_path / "integration.db")
    init_db(conn)
    yield conn  # type: ignore[misc]
    conn.close()


@pytest.fixture
async def seeded_db(integration_db: sqlite3.Connection, seed_content_dir: Path) -> sqlite3.Connection:
    """Ingest seed content files into a test database.

    Requires Ollama running (for nomic-embed-text embeddings).
    """
    from code.shukketsu.ingest.embedder import get_embedder
    from code.shukketsu.ingest.pipeline import IngestPipeline

    embedder = get_embedder()
    pipeline = IngestPipeline(conn=integration_db, embedder=embedder)

    for md_file in sorted(seed_content_dir.glob("*.md")):
        content = md_file.read_text()
        await pipeline.ingest(content, f"file://{md_file.name}", md_file.stem)

    return integration_db
