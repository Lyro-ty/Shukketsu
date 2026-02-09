"""Shared test fixtures for Shukketsu test suite."""

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def test_db(tmp_path):
    """Create a fresh test database with schema."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    schema_path = Path(__file__).parent.parent / "code" / "shukketsu" / "db" / "schema.sql"
    if schema_path.exists():
        conn.executescript(schema_path.read_text())

    yield conn
    conn.close()
