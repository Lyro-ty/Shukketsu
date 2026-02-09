"""Tests for database connection and schema initialization."""

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def db_conn(tmp_path: Path) -> sqlite3.Connection:
    """Create a configured connection without schema."""
    from code.shukketsu.db.connection import get_connection

    conn = get_connection(tmp_path / "test.db")
    yield conn
    conn.close()


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    """Create a fully initialized test database."""
    from code.shukketsu.db.connection import get_connection, init_db

    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    yield conn
    conn.close()


# --- Connection tests ---


class TestGetConnection:
    """Tests for get_connection factory."""

    def test_returns_connection(self, db_conn: sqlite3.Connection) -> None:
        """get_connection should return a sqlite3.Connection."""
        assert isinstance(db_conn, sqlite3.Connection)

    def test_sets_wal_mode(self, db_conn: sqlite3.Connection) -> None:
        """Connection should use WAL journal mode."""
        mode = db_conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_sets_foreign_keys(self, db_conn: sqlite3.Connection) -> None:
        """Foreign key enforcement should be enabled."""
        fk = db_conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

    def test_sets_row_factory(self, db_conn: sqlite3.Connection) -> None:
        """Connection should use sqlite3.Row for dict-like access."""
        assert db_conn.row_factory is sqlite3.Row

    def test_loads_sqlite_vec(self, db_conn: sqlite3.Connection) -> None:
        """sqlite-vec extension should be loaded and functional."""
        version = db_conn.execute("SELECT vec_version()").fetchone()[0]
        assert version  # non-empty version string


# --- Schema tests ---


class TestInitDb:
    """Tests for init_db schema initialization."""

    def test_creates_core_tables(self, db: sqlite3.Connection) -> None:
        """init_db should create sources, chunks, articles, schema_version."""
        tables = _get_tables(db)
        for name in ("sources", "chunks", "articles", "schema_version"):
            assert name in tables, f"Missing table: {name}"

    def test_creates_fts_virtual_table(self, db: sqlite3.Connection) -> None:
        """init_db should create the FTS5 virtual table for keyword search."""
        tables = _get_tables(db)
        assert "chunks_fts" in tables

    def test_creates_vec_virtual_table(self, db: sqlite3.Connection) -> None:
        """init_db should create the vec0 virtual table for vector search."""
        tables = _get_tables(db)
        assert "chunks_vec" in tables

    def test_sets_schema_version(self, db: sqlite3.Connection) -> None:
        """Schema version should be 1 after initialization."""
        version = db.execute("SELECT version FROM schema_version").fetchone()[0]
        assert version == 1

    def test_is_idempotent(self, db: sqlite3.Connection) -> None:
        """Calling init_db twice should not raise or duplicate data."""
        from code.shukketsu.db.connection import init_db

        init_db(db)  # second call — should be a no-op
        count = db.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        assert count == 1

    def test_foreign_key_enforcement(self, db: sqlite3.Connection) -> None:
        """Inserting a chunk with invalid source_id should raise IntegrityError."""
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO chunks (source_id, content, chunk_index) VALUES (999, 'test', 0)")

    def test_fts_sync_on_insert(self, db: sqlite3.Connection) -> None:
        """Inserting into chunks should auto-populate FTS index via trigger."""
        db.execute("INSERT INTO sources (url, title) VALUES ('https://example.com', 'Test')")
        db.execute(
            "INSERT INTO chunks (source_id, content, chunk_index) VALUES (1, 'combat rogue hit cap guide for TBC', 0)"
        )
        db.commit()

        rows = db.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'combat rogue'").fetchall()
        assert len(rows) == 1

    def test_fts_sync_on_delete(self, db: sqlite3.Connection) -> None:
        """Deleting a chunk should remove it from FTS index via trigger."""
        db.execute("INSERT INTO sources (url, title) VALUES ('https://example.com', 'Test')")
        db.execute("INSERT INTO chunks (source_id, content, chunk_index) VALUES (1, 'unique search term xyzzy', 0)")
        db.commit()

        db.execute("DELETE FROM chunks WHERE id = 1")
        db.commit()

        rows = db.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'xyzzy'").fetchall()
        assert len(rows) == 0

    def test_cascade_delete_removes_chunks(self, db: sqlite3.Connection) -> None:
        """Deleting a source should cascade-delete its chunks."""
        db.execute("INSERT INTO sources (url, title) VALUES ('https://example.com', 'Test')")
        db.execute("INSERT INTO chunks (source_id, content, chunk_index) VALUES (1, 'test content', 0)")
        db.commit()

        db.execute("DELETE FROM sources WHERE id = 1")
        db.commit()

        count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count == 0

    def test_source_url_unique(self, db: sqlite3.Connection) -> None:
        """Inserting duplicate source URLs should raise IntegrityError."""
        db.execute("INSERT INTO sources (url, title) VALUES ('https://example.com', 'First')")
        db.commit()

        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO sources (url, title) VALUES ('https://example.com', 'Duplicate')")


# --- Helpers ---


def _get_tables(conn: sqlite3.Connection) -> set[str]:
    """Get all table names from the database."""
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}
