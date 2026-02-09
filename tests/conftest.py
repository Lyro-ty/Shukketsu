"""Shared test fixtures for Shukketsu test suite."""

import sqlite3

import pytest


@pytest.fixture
def test_db(tmp_path) -> sqlite3.Connection:
    """Create a fresh test database with full schema.

    Uses the same connection factory as production, ensuring WAL mode,
    sqlite-vec, and foreign keys are configured identically.
    """
    from code.shukketsu.db.connection import get_connection, init_db

    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    yield conn
    conn.close()
