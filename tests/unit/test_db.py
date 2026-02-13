"""Tests for database connection and schema initialization."""

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def db_conn(tmp_path: Path) -> Generator[sqlite3.Connection]:
    """Create a configured connection without schema."""
    from code.shukketsu.db.connection import get_connection

    conn = get_connection(tmp_path / "test.db")
    yield conn
    conn.close()


@pytest.fixture
def db(tmp_path: Path) -> Generator[sqlite3.Connection]:
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
        """Schema version should be 4 after initialization."""
        version = db.execute("SELECT version FROM schema_version").fetchone()[0]
        assert version == 4

    def test_is_idempotent(self, db: sqlite3.Connection) -> None:
        """Calling init_db twice should not raise or duplicate data."""
        from code.shukketsu.db.connection import init_db

        count_before = db.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        init_db(db)  # second call — should be a no-op
        count_after = db.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        assert count_after == count_before

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

    def test_creates_graph_tables(self, db: sqlite3.Connection) -> None:
        """init_db should create entity_types, entities, relationships tables."""
        tables = _get_tables(db)
        for name in ("entity_types", "entities", "relationships"):
            assert name in tables, f"Missing table: {name}"

    def test_schema_version_is_7(self, db: sqlite3.Connection) -> None:
        """Schema version should be 7 after fresh initialization."""
        version = db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert version == 7

    def test_entity_type_unique_name(self, db: sqlite3.Connection) -> None:
        """entity_types.name should enforce uniqueness."""
        db.execute("INSERT INTO entity_types (name, display_name) VALUES ('item', 'Item')")
        db.commit()
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO entity_types (name, display_name) VALUES ('item', 'Item Duplicate')")

    def test_entity_unique_canonical_per_type(self, db: sqlite3.Connection) -> None:
        """entities should enforce UNIQUE(canonical_name, entity_type_id)."""
        db.execute("INSERT INTO entity_types (name, display_name) VALUES ('item', 'Item')")
        type_id = db.execute("SELECT id FROM entity_types WHERE name = 'item'").fetchone()["id"]
        db.execute(
            "INSERT INTO entities (name, entity_type_id, canonical_name) VALUES (?, ?, ?)",
            ("Dragonspine Trophy", type_id, "dragonspine trophy"),
        )
        db.commit()
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO entities (name, entity_type_id, canonical_name) VALUES (?, ?, ?)",
                ("DST", type_id, "dragonspine trophy"),
            )

    def test_relationship_unique_triple(self, db: sqlite3.Connection) -> None:
        """relationships should enforce UNIQUE(source_entity_id, target_entity_id, relation_type)."""
        db.execute("INSERT INTO entity_types (name, display_name) VALUES ('item', 'Item')")
        db.execute("INSERT INTO entity_types (name, display_name) VALUES ('boss', 'Boss')")
        type_item = db.execute("SELECT id FROM entity_types WHERE name = 'item'").fetchone()["id"]
        type_boss = db.execute("SELECT id FROM entity_types WHERE name = 'boss'").fetchone()["id"]
        db.execute(
            "INSERT INTO entities (name, entity_type_id, canonical_name) VALUES (?, ?, ?)",
            ("DST", type_item, "dragonspine trophy"),
        )
        db.execute(
            "INSERT INTO entities (name, entity_type_id, canonical_name) VALUES (?, ?, ?)",
            ("Gruul", type_boss, "gruul"),
        )
        db.commit()
        item_id = db.execute("SELECT id FROM entities WHERE canonical_name = 'dragonspine trophy'").fetchone()["id"]
        boss_id = db.execute("SELECT id FROM entities WHERE canonical_name = 'gruul'").fetchone()["id"]
        db.execute(
            "INSERT INTO relationships (source_entity_id, target_entity_id, relation_type) VALUES (?, ?, ?)",
            (item_id, boss_id, "drops_from"),
        )
        db.commit()
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO relationships (source_entity_id, target_entity_id, relation_type) VALUES (?, ?, ?)",
                (item_id, boss_id, "drops_from"),
            )

    def test_entity_fk_to_entity_types(self, db: sqlite3.Connection) -> None:
        """entities.entity_type_id should reference entity_types(id)."""
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO entities (name, entity_type_id, canonical_name) VALUES (?, ?, ?)",
                ("Ghost Item", 999, "ghost item"),
            )

    def test_relationship_fk_cascade_on_entity_delete(self, db: sqlite3.Connection) -> None:
        """Deleting an entity should cascade-delete its relationships."""
        db.execute("INSERT INTO entity_types (name, display_name) VALUES ('item', 'Item')")
        db.execute("INSERT INTO entity_types (name, display_name) VALUES ('boss', 'Boss')")
        type_item = db.execute("SELECT id FROM entity_types WHERE name = 'item'").fetchone()["id"]
        type_boss = db.execute("SELECT id FROM entity_types WHERE name = 'boss'").fetchone()["id"]
        db.execute(
            "INSERT INTO entities (name, entity_type_id, canonical_name) VALUES (?, ?, ?)",
            ("DST", type_item, "dragonspine trophy"),
        )
        db.execute(
            "INSERT INTO entities (name, entity_type_id, canonical_name) VALUES (?, ?, ?)",
            ("Gruul", type_boss, "gruul"),
        )
        db.commit()
        item_id = db.execute("SELECT id FROM entities WHERE canonical_name = 'dragonspine trophy'").fetchone()["id"]
        boss_id = db.execute("SELECT id FROM entities WHERE canonical_name = 'gruul'").fetchone()["id"]
        db.execute(
            "INSERT INTO relationships (source_entity_id, target_entity_id, relation_type) VALUES (?, ?, ?)",
            (item_id, boss_id, "drops_from"),
        )
        db.commit()
        db.execute("DELETE FROM entities WHERE id = ?", (item_id,))
        db.commit()
        count = db.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        assert count == 0

    def test_articles_table_v3_columns(self, db: sqlite3.Connection) -> None:
        """articles table should have v3 columns: status, spec, category, created_at."""
        columns = {row[1] for row in db.execute("PRAGMA table_info(articles)").fetchall()}
        for col in ("status", "spec", "category", "created_at", "last_updated"):
            assert col in columns, f"Missing column: {col}"
        assert "needs_review" not in columns, "needs_review should be removed in v3"

    def test_articles_status_index_exists(self, db: sqlite3.Connection) -> None:
        """articles table should have indexes on status and spec."""
        indexes = {row[1] for row in db.execute("PRAGMA index_list(articles)").fetchall()}
        assert "idx_articles_status" in indexes
        assert "idx_articles_spec" in indexes

    def test_creates_memory_tables(self, db: sqlite3.Connection) -> None:
        """init_db should create session_memories, strategy_memories, trust_events."""
        tables = _get_tables(db)
        for name in ("session_memories", "strategy_memories", "trust_events"):
            assert name in tables, f"Missing table: {name}"

    def test_creates_session_memories_vec(self, db: sqlite3.Connection) -> None:
        """init_db should create the session_memories_vec virtual table."""
        tables = _get_tables(db)
        assert "session_memories_vec" in tables

    def test_trust_events_fk_to_sources(self, db: sqlite3.Connection) -> None:
        """trust_events.source_id should reference sources(id)."""
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO trust_events (source_id, event_type, delta) VALUES (999, 'test', 0.1)")

    def test_trust_events_cascade_on_source_delete(self, db: sqlite3.Connection) -> None:
        """Deleting a source should cascade-delete its trust_events."""
        db.execute("INSERT INTO sources (url, title) VALUES ('https://example.com', 'Test')")
        db.commit()
        source_id = db.execute("SELECT id FROM sources WHERE url = 'https://example.com'").fetchone()["id"]
        db.execute(
            "INSERT INTO trust_events (source_id, event_type, delta) VALUES (?, 'contradiction', -0.1)",
            (source_id,),
        )
        db.commit()
        db.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        db.commit()
        count = db.execute("SELECT COUNT(*) FROM trust_events").fetchone()[0]
        assert count == 0


def test_schema_v7_validation_runs_table(test_db: sqlite3.Connection) -> None:
    """Schema v7 adds validation_runs table."""
    tables = test_db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='validation_runs'").fetchone()
    assert tables is not None

    cols = test_db.execute("PRAGMA table_info(validation_runs)").fetchall()
    col_names = [c[1] for c in cols]
    assert "character_name" in col_names
    assert "run_type" in col_names
    assert "overall_dps_drift_pct" in col_names
    assert "report_json" in col_names


# --- Helpers ---


def _get_tables(conn: sqlite3.Connection) -> set[str]:
    """Get all table names from the database."""
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}
