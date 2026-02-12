"""Shared test fixtures for Shukketsu test suite."""

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_breakers() -> Generator[None]:
    """Reset all circuit breakers before each test to prevent state leakage."""
    from code.shukketsu.resilience.circuit_breaker import reset_all_breakers

    reset_all_breakers()
    yield
    reset_all_breakers()


@pytest.fixture
def test_db(tmp_path: Path) -> Generator[sqlite3.Connection]:
    """Create a fresh test database with full schema.

    Uses the same connection factory as production, ensuring WAL mode,
    sqlite-vec, and foreign keys are configured identically.
    """
    from code.shukketsu.db.connection import get_connection, init_db

    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    yield conn
    conn.close()
