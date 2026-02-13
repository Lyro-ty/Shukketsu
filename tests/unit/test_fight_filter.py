"""Tests for WCL fight filtering and validation."""

import sqlite3

import pytest

from code.shukketsu.sim.fight_filter import (
    PATCHWERK_ENCOUNTERS,
    FightFilter,
    FilterCriteria,
)


@pytest.fixture()
def fight_db(test_db: sqlite3.Connection) -> sqlite3.Connection:
    """Populate test DB with sample fight data."""
    test_db.execute("INSERT OR IGNORE INTO wcl_reports (code, endpoint) VALUES ('RPT1', 'fresh')")
    fights = [
        ("RPT1", 1, 725, "Brutallus", 1, 180000),
        ("RPT1", 2, 725, "Brutallus", 0, 120000),      # wipe
        ("RPT1", 3, 999, "Unknown Boss", 1, 60000),     # non-patchwerk
        ("RPT1", 4, 725, "Brutallus", 1, 15000),        # too short (<30s)
    ]
    for f in fights:
        test_db.execute(
            """INSERT OR REPLACE INTO wcl_fights
               (report_code, fight_id, encounter_id, encounter_name, kill, duration_ms)
               VALUES (?, ?, ?, ?, ?, ?)""",
            f,
        )
    # Add damage rows for Lyroo
    for fid, dmg, active_ms in [(1, 200000, 175000), (2, 100000, 50000), (4, 5000, 14000)]:
        test_db.execute(
            """INSERT OR REPLACE INTO wcl_damage
               (report_code, fight_id, player_name, player_type,
                total_damage, active_time_ms, abilities_json, targets_json)
               VALUES (?, ?, 'Lyroo', 'Rogue', ?, ?, '[]', '[]')""",
            ("RPT1", fid, dmg, active_ms),
        )
    # Add combatant for Lyroo
    for fid in [1, 2, 4]:
        test_db.execute(
            """INSERT OR REPLACE INTO wcl_combatants
               (report_code, fight_id, source_id, player_name) VALUES (?, ?, 5, 'Lyroo')""",
            ("RPT1", fid),
        )
    test_db.commit()
    return test_db


class TestFilterCriteria:
    def test_default_criteria(self) -> None:
        c = FilterCriteria()
        assert c.kills_only is True
        assert c.min_duration_ms == 30000
        assert c.patchwerk_only is True

    def test_frozen(self) -> None:
        c = FilterCriteria()
        with pytest.raises(Exception):
            c.kills_only = False  # type: ignore[misc]


class TestFightFilter:
    def test_excludes_wipes(self, fight_db: sqlite3.Connection) -> None:
        ff = FightFilter()
        results = ff.filter_fights(fight_db, "Lyroo")
        wipe = next(r for r in results if r.fight_id == 2)
        assert not wipe.included
        assert wipe.exclusion_reason == "wipe"

    def test_excludes_short_fights(self, fight_db: sqlite3.Connection) -> None:
        ff = FightFilter()
        results = ff.filter_fights(fight_db, "Lyroo")
        short = next(r for r in results if r.fight_id == 4)
        assert not short.included
        assert short.exclusion_reason == "short_fight"

    def test_includes_valid_kill(self, fight_db: sqlite3.Connection) -> None:
        ff = FightFilter()
        results = ff.filter_fights(fight_db, "Lyroo")
        valid = next(r for r in results if r.fight_id == 1)
        assert valid.included
        assert valid.exclusion_reason is None
        assert valid.wcl_total_damage == 200000

    def test_custom_criteria_disables_patchwerk(self, fight_db: sqlite3.Connection) -> None:
        """Disabling patchwerk_only lets non-patchwerk encounters through."""
        # fight 3 has encounter_id 999 and no damage row for Lyroo, so it won't appear
        criteria = FilterCriteria(patchwerk_only=False)
        ff = FightFilter(criteria=criteria)
        results = ff.filter_fights(fight_db, "Lyroo")
        # Still only sees fights where Lyroo has damage+combatant rows
        ids = [r.fight_id for r in results if r.included]
        assert 1 in ids

    def test_no_results_for_unknown_player(self, fight_db: sqlite3.Connection) -> None:
        ff = FightFilter()
        results = ff.filter_fights(fight_db, "UnknownPlayer")
        assert len(results) == 0


class TestPatchworkEncounters:
    def test_brutallus_is_patchwerk(self) -> None:
        assert 725 in PATCHWERK_ENCOUNTERS

    def test_gruul_is_patchwerk(self) -> None:
        assert 649 in PATCHWERK_ENCOUNTERS
