"""Tests for WCL database schema v5 migration.

Verifies that all WCL tables, columns, constraints, and indexes are created
correctly after init_db().
"""

import sqlite3
from pathlib import Path

import pytest

from code.shukketsu.db.connection import get_connection, init_db

_WCL_TABLES = [
    "wcl_tracked_characters",
    "wcl_rankings",
    "wcl_reports",
    "wcl_fights",
    "wcl_combatants",
    "wcl_damage",
    "wcl_buffs",
    "wcl_casts",
    "wcl_fight_rankings",
    "wcl_character_log",
]


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    """Create a fresh database with full schema including v5 migration."""
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    return conn


def _get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Return column names for a table via PRAGMA table_info."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row[1] for row in rows]


class TestWCLTablesExist:
    """All 10 WCL tables should be created by the v5 migration."""

    def test_wcl_tables_exist(self, db: sqlite3.Connection) -> None:
        rows = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'wcl_%'").fetchall()
        table_names = sorted(row[0] for row in rows)
        assert table_names == sorted(_WCL_TABLES)


class TestWCLRankingsColumns:
    """Verify wcl_rankings has the expected columns."""

    def test_wcl_rankings_columns(self, db: sqlite3.Connection) -> None:
        cols = _get_table_columns(db, "wcl_rankings")
        expected = [
            "id",
            "zone_id",
            "encounter_id",
            "encounter_name",
            "player_name",
            "spec",
            "dps",
            "duration_ms",
            "report_code",
            "fight_id",
            "guild_name",
            "server_name",
            "server_region",
            "faction",
            "raid_size",
            "bracket_data",
            "fetched_at",
        ]
        assert cols == expected


class TestWCLCombatantsColumns:
    """Verify wcl_combatants has stat columns and JSON blob columns."""

    def test_wcl_combatants_columns(self, db: sqlite3.Connection) -> None:
        cols = _get_table_columns(db, "wcl_combatants")
        # Check stat columns are present
        stat_cols = [
            "strength",
            "agility",
            "stamina",
            "intellect",
            "spirit",
            "crit_melee",
            "crit_ranged",
            "crit_spell",
            "haste_melee",
            "haste_ranged",
            "haste_spell",
            "hit_melee",
            "hit_ranged",
            "hit_spell",
            "expertise",
            "dodge",
            "parry",
            "block",
            "armor",
        ]
        for col in stat_cols:
            assert col in cols, f"Missing stat column: {col}"
        # Check JSON blob columns
        for json_col in ["gear_json", "talents_json", "auras_json"]:
            assert json_col in cols, f"Missing JSON column: {json_col}"


class TestWCLRankingsUniqueConstraint:
    """Inserting a duplicate ranking row should raise IntegrityError."""

    def test_wcl_rankings_unique_constraint(self, db: sqlite3.Connection) -> None:
        row = {
            "zone_id": 1012,
            "encounter_id": 725,
            "encounter_name": "Brutallus",
            "player_name": "Lyroo",
            "spec": "Combat",
            "dps": 3150.8,
            "duration_ms": 165000,
            "report_code": "SWP2024x",
            "fight_id": 7,
            "server_name": "Whitemane",
            "server_region": "US",
        }
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        sql = f"INSERT INTO wcl_rankings ({cols}) VALUES ({placeholders})"
        db.execute(sql, list(row.values()))
        db.commit()
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(sql, list(row.values()))


class TestWCLReportsPrimaryKey:
    """wcl_reports uses code as the primary key."""

    def test_wcl_reports_primary_key(self, db: sqlite3.Connection) -> None:
        db.execute("INSERT INTO wcl_reports (code) VALUES ('ABC123')")
        db.commit()
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO wcl_reports (code) VALUES ('ABC123')")


class TestWCLFightsForeignKey:
    """wcl_fights has a foreign key to wcl_reports(code)."""

    def test_wcl_fights_foreign_key(self, db: sqlite3.Connection) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO wcl_fights "
                "(report_code, fight_id, encounter_id, encounter_name, kill, duration_ms) "
                "VALUES ('NONEXISTENT', 1, 725, 'Brutallus', 1, 165000)"
            )


class TestWCLIndexesExist:
    """All 5 WCL indexes should be created."""

    def test_wcl_indexes_exist(self, db: sqlite3.Connection) -> None:
        rows = db.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_wcl_%'").fetchall()
        index_names = sorted(row[0] for row in rows)
        expected = sorted(
            [
                "idx_wcl_rankings_encounter",
                "idx_wcl_rankings_player",
                "idx_wcl_combatants_report",
                "idx_wcl_damage_report",
                "idx_wcl_character_log_char",
            ]
        )
        assert index_names == expected


class TestSchemaVersionIs7:
    """After init_db, schema_version should contain version 7."""

    def test_schema_version_is_7(self, db: sqlite3.Connection) -> None:
        version = db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert version == 7
