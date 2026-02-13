"""Tests for WCL ingest orchestrators.

Verifies RankingsIngestor, ReportDiver, and CharacterSyncer correctly
fetch data via WCLClient (mocked) and store it in SQLite tables.
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from code.shukketsu.apis.wcl.auth import WCLAuth
from code.shukketsu.apis.wcl.client import WCLClient
from code.shukketsu.apis.wcl.ingest import CharacterSyncer, RankingsIngestor, ReportDiver
from code.shukketsu.db.connection import get_connection, init_db
from code.shukketsu.resilience.errors import WCLQueryError


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    """Create a fresh database with full schema including v5 migration."""
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    return conn


@pytest.fixture()
def mock_client() -> WCLClient:
    """Provide a WCLClient with a mocked async query method."""
    auth = WCLAuth(client_id="test", client_secret="test")
    client = WCLClient(auth)
    client.query = AsyncMock()  # type: ignore[method-assign]
    return client


# ---------------------------------------------------------------------------
# Mock response factories
# ---------------------------------------------------------------------------


def _rankings_response(count: int = 1, report_code: str = "ABC123") -> dict:
    """Build a mock rankings API response."""
    rankings = []
    for i in range(count):
        rankings.append(
            {
                "name": f"Player{i}",
                "class": "Rogue",
                "spec": "Combat",
                "amount": 1500.0 + i * 10,
                "duration": 180000,
                "report": {"code": report_code, "fightID": 1, "startTime": 1000},
                "server": {"name": "Nightslayer", "region": "US"},
                "guild": {"name": "TestGuild"},
                "faction": 1,
                "size": 25,
                "bracketData": 0,
            }
        )
    return {
        "worldData": {
            "encounter": {
                "characterRankings": {
                    "rankings": rankings,
                },
            },
        },
    }


def _fights_response(code: str = "ABC123") -> dict:
    """Build a mock report fights response with one boss fight."""
    return {
        "reportData": {
            "report": {
                "fights": [
                    {
                        "id": 5,
                        "encounterID": 652,
                        "name": "Gruul the Dragonkiller",
                        "kill": True,
                        "duration": 200000,
                        "bossPercentage": 0,
                        "averageItemLevel": 125.0,
                        "size": 25,
                        "difficulty": 0,
                    },
                    {
                        "id": 99,
                        "encounterID": 0,
                        "name": "Trash",
                        "kill": None,
                        "duration": 60000,
                    },
                ],
                "masterData": {"actors": [{"id": 1, "name": "Lyroo", "type": "Player", "subType": "Rogue"}]},
            },
        },
    }


def _combatant_info_response() -> dict:
    """Build a mock CombatantInfo events response."""
    return {
        "reportData": {
            "report": {
                "events": {
                    "data": [
                        {
                            "sourceID": 1,
                            "specID": 260,
                            "faction": 1,
                            "fight": 5,
                            "agility": 800,
                            "hitMelee": 142,
                            "critMelee": 300,
                            "gear": [{"id": 28830, "slot": 0, "itemLevel": 125}],
                            "talents": [{"guid": 12345, "type": 1}],
                            "auras": [],
                        },
                    ],
                    "nextPageTimestamp": None,
                },
            },
        },
    }


def _damage_table_response() -> dict:
    """Build a mock damage table response."""
    return {
        "reportData": {
            "report": {
                "table": {
                    "data": {
                        "entries": [
                            {
                                "name": "Lyroo",
                                "type": "Rogue",
                                "total": 450000,
                                "activeTime": 195000,
                                "abilities": [{"name": "Sinister Strike", "total": 200000, "type": 1}],
                                "targets": [{"name": "Gruul", "total": 450000}],
                            },
                        ],
                    },
                },
            },
        },
    }


def _buff_table_response() -> dict:
    """Build a mock buff table response."""
    return {
        "reportData": {
            "report": {
                "table": {
                    "data": {
                        "auras": [
                            {
                                "name": "Slice and Dice",
                                "guid": 6774,
                                "totalUptime": 185000,
                                "totalUses": 8,
                                "bands": [{"startTime": 1000, "endTime": 186000}],
                            },
                        ],
                    },
                },
            },
        },
    }


def _cast_table_response() -> dict:
    """Build a mock cast table response."""
    return {
        "reportData": {
            "report": {
                "table": {
                    "data": {
                        "entries": [
                            {
                                "name": "Lyroo",
                                "id": 1,
                                "type": "Rogue",
                                "total": 120,
                                "abilities": [
                                    {"name": "Sinister Strike", "total": 80},
                                    {"name": "Eviscerate", "total": 40},
                                ],
                            },
                        ],
                    },
                },
            },
        },
    }


def _report_rankings_response() -> dict:
    """Build a mock per-fight rankings response."""
    return {
        "reportData": {
            "report": {
                "rankings": {
                    "data": [
                        {
                            "fightID": 5,
                            "encounterID": 652,
                            "roles": {
                                "dps": {
                                    "characters": [
                                        {
                                            "name": "Lyroo",
                                            "server": {"name": "Nightslayer"},
                                            "class": "Rogue",
                                            "spec": "Combat",
                                            "amount": 1500.0,
                                            "rankPercent": 95,
                                            "totalParses": 5000,
                                        },
                                    ],
                                },
                            },
                        },
                    ],
                },
            },
        },
    }


def _character_response(report_codes: list[str] | None = None) -> dict:
    """Build a mock character profile with recent reports."""
    codes = report_codes or ["RPT001", "RPT002"]
    reports = [{"code": c, "title": f"Report {c}", "startTime": 1000, "endTime": 2000} for c in codes]
    return {
        "characterData": {
            "character": {
                "id": 42,
                "name": "Lyroo",
                "classID": 4,
                "server": {"id": 1, "name": "Nightslayer", "slug": "nightslayer", "region": {"slug": "us"}},
                "recentReports": {"data": reports},
            },
        },
    }


# ---------------------------------------------------------------------------
# Rankings tests
# ---------------------------------------------------------------------------


class TestRankingsIngestor:
    """Tests for RankingsIngestor."""

    async def test_ingest_rankings_stores_to_db(self, db: sqlite3.Connection, mock_client: WCLClient) -> None:
        """Rankings data is stored in the wcl_rankings table."""
        mock_client.query.return_value = _rankings_response(count=2)  # type: ignore[union-attr]

        ingestor = RankingsIngestor(mock_client, db)
        await ingestor.ingest_encounter(652, "Gruul the Dragonkiller", 1001)

        rows = db.execute("SELECT player_name, dps, encounter_id FROM wcl_rankings ORDER BY player_name").fetchall()
        assert len(rows) == 2
        assert rows[0]["player_name"] == "Player0"
        assert rows[0]["dps"] == 1500.0
        assert rows[0]["encounter_id"] == 652
        assert rows[1]["player_name"] == "Player1"
        assert rows[1]["dps"] == 1510.0

    async def test_ingest_rankings_collects_report_codes(self, db: sqlite3.Connection, mock_client: WCLClient) -> None:
        """ingest_encounter returns the set of unique report codes."""
        mock_client.query.return_value = _rankings_response(count=3, report_code="XYZ789")  # type: ignore[union-attr]

        ingestor = RankingsIngestor(mock_client, db)
        codes = await ingestor.ingest_encounter(652, "Gruul", 1001)

        assert codes == {"XYZ789"}

    async def test_ingest_rankings_skips_duplicates(self, db: sqlite3.Connection, mock_client: WCLClient) -> None:
        """Running ingest twice with same data does not crash (INSERT OR REPLACE)."""
        mock_client.query.return_value = _rankings_response(count=1)  # type: ignore[union-attr]

        ingestor = RankingsIngestor(mock_client, db)
        await ingestor.ingest_encounter(652, "Gruul", 1001)
        await ingestor.ingest_encounter(652, "Gruul", 1001)

        rows = db.execute("SELECT COUNT(*) FROM wcl_rankings").fetchone()
        assert rows[0] == 1  # Replaced, not duplicated


# ---------------------------------------------------------------------------
# Report deep-dive tests
# ---------------------------------------------------------------------------


class TestReportDiver:
    """Tests for ReportDiver."""

    async def test_deep_dive_stores_combatants(self, db: sqlite3.Connection, mock_client: WCLClient) -> None:
        """CombatantInfo events are stored in wcl_combatants."""
        mock_client.query.side_effect = [  # type: ignore[union-attr]
            _fights_response(),
            _combatant_info_response(),
            _damage_table_response(),
            _buff_table_response(),
            _cast_table_response(),
            _report_rankings_response(),
        ]

        diver = ReportDiver(mock_client, db)
        await diver.dive("ABC123")

        rows = db.execute("SELECT source_id, agility, hit_melee, gear_json FROM wcl_combatants").fetchall()
        assert len(rows) == 1
        assert rows[0]["source_id"] == 1
        assert rows[0]["agility"] == 800
        assert rows[0]["hit_melee"] == 142
        gear = json.loads(rows[0]["gear_json"])
        assert gear[0]["id"] == 28830

    async def test_deep_dive_stores_damage(self, db: sqlite3.Connection, mock_client: WCLClient) -> None:
        """Damage table entries are stored in wcl_damage."""
        mock_client.query.side_effect = [  # type: ignore[union-attr]
            _fights_response(),
            _combatant_info_response(),
            _damage_table_response(),
            _buff_table_response(),
            _cast_table_response(),
            _report_rankings_response(),
        ]

        diver = ReportDiver(mock_client, db)
        await diver.dive("ABC123")

        rows = db.execute("SELECT player_name, total_damage, abilities_json FROM wcl_damage").fetchall()
        assert len(rows) == 1
        assert rows[0]["player_name"] == "Lyroo"
        assert rows[0]["total_damage"] == 450000
        abilities = json.loads(rows[0]["abilities_json"])
        assert abilities[0]["name"] == "Sinister Strike"

    async def test_deep_dive_stores_buffs(self, db: sqlite3.Connection, mock_client: WCLClient) -> None:
        """Buff table auras are stored in wcl_buffs."""
        mock_client.query.side_effect = [  # type: ignore[union-attr]
            _fights_response(),
            _combatant_info_response(),
            _damage_table_response(),
            _buff_table_response(),
            _cast_table_response(),
            _report_rankings_response(),
        ]

        diver = ReportDiver(mock_client, db)
        await diver.dive("ABC123")

        rows = db.execute("SELECT buff_name, buff_guid, total_uptime_ms FROM wcl_buffs").fetchall()
        assert len(rows) == 1
        assert rows[0]["buff_name"] == "Slice and Dice"
        assert rows[0]["buff_guid"] == 6774
        assert rows[0]["total_uptime_ms"] == 185000

    async def test_deep_dive_stores_casts(self, db: sqlite3.Connection, mock_client: WCLClient) -> None:
        """Cast table entries are stored in wcl_casts (one row per ability)."""
        mock_client.query.side_effect = [  # type: ignore[union-attr]
            _fights_response(),
            _combatant_info_response(),
            _damage_table_response(),
            _buff_table_response(),
            _cast_table_response(),
            _report_rankings_response(),
        ]

        diver = ReportDiver(mock_client, db)
        await diver.dive("ABC123")

        rows = db.execute(
            "SELECT player_name, ability_name, cast_count FROM wcl_casts ORDER BY ability_name"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["ability_name"] == "Eviscerate"
        assert rows[0]["cast_count"] == 40
        assert rows[1]["ability_name"] == "Sinister Strike"
        assert rows[1]["cast_count"] == 80

    async def test_deep_dive_stores_rankings(self, db: sqlite3.Connection, mock_client: WCLClient) -> None:
        """Per-fight rankings are stored in wcl_fight_rankings."""
        mock_client.query.side_effect = [  # type: ignore[union-attr]
            _fights_response(),
            _combatant_info_response(),
            _damage_table_response(),
            _buff_table_response(),
            _cast_table_response(),
            _report_rankings_response(),
        ]

        diver = ReportDiver(mock_client, db)
        await diver.dive("ABC123")

        rows = db.execute("SELECT player_name, dps, rank_percent, server_name FROM wcl_fight_rankings").fetchall()
        assert len(rows) == 1
        assert rows[0]["player_name"] == "Lyroo"
        assert rows[0]["dps"] == 1500.0
        assert rows[0]["rank_percent"] == 95
        assert rows[0]["server_name"] == "Nightslayer"

    async def test_deep_dive_skips_archived(self, db: sqlite3.Connection, mock_client: WCLClient) -> None:
        """Archived reports are marked in DB and the dive is skipped."""
        mock_client.query.side_effect = WCLQueryError(  # type: ignore[union-attr]
            "Report is archived: This report has been archived"
        )

        diver = ReportDiver(mock_client, db)
        await diver.dive("OLD_REPORT")

        row = db.execute("SELECT is_archived FROM wcl_reports WHERE code = ?", ("OLD_REPORT",)).fetchone()
        assert row is not None
        assert row["is_archived"] == 1


# ---------------------------------------------------------------------------
# Character sync tests
# ---------------------------------------------------------------------------


class TestCharacterSyncer:
    """Tests for CharacterSyncer."""

    async def test_sync_character_finds_new_reports(self, db: sqlite3.Connection, mock_client: WCLClient) -> None:
        """sync returns new report codes not already in the log."""
        mock_client.query.return_value = _character_response(["RPT001", "RPT002", "RPT003"])  # type: ignore[union-attr]

        syncer = CharacterSyncer(mock_client, db)
        new_codes = await syncer.sync({"wcl_id": 42, "name": "Lyroo"})

        assert set(new_codes) == {"RPT001", "RPT002", "RPT003"}

    async def test_sync_character_stores_log(self, db: sqlite3.Connection, mock_client: WCLClient) -> None:
        """sync stores entries in wcl_character_log."""
        mock_client.query.return_value = _character_response(["RPT001", "RPT002"])  # type: ignore[union-attr]

        syncer = CharacterSyncer(mock_client, db)
        await syncer.sync({"wcl_id": 42, "name": "Lyroo"})

        rows = db.execute(
            "SELECT report_code FROM wcl_character_log WHERE character_wcl_id = 42 ORDER BY report_code"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["report_code"] == "RPT001"
        assert rows[1]["report_code"] == "RPT002"

        # Second sync with same data returns no new codes
        mock_client.query.return_value = _character_response(["RPT001", "RPT002"])  # type: ignore[union-attr]
        new_codes = await syncer.sync({"wcl_id": 42, "name": "Lyroo"})
        assert new_codes == []

    async def test_detect_gear_changes(self, db: sqlite3.Connection, mock_client: WCLClient) -> None:
        """detect_gear_changes identifies differences between consecutive combatant rows."""
        # Set up: insert report, character log entries, and two combatant rows with different gear
        db.execute("INSERT INTO wcl_reports (code, endpoint) VALUES ('RPT_A', 'fresh')")
        db.execute("INSERT INTO wcl_reports (code, endpoint) VALUES ('RPT_B', 'fresh')")
        db.execute(
            "INSERT INTO wcl_character_log (character_wcl_id, report_code, fight_id, encounter_id)"
            " VALUES (42, 'RPT_A', 1, 652)"
        )
        db.execute(
            "INSERT INTO wcl_character_log (character_wcl_id, report_code, fight_id, encounter_id)"
            " VALUES (42, 'RPT_B', 1, 652)"
        )

        gear_old = json.dumps([{"id": 28830, "slot": 0, "itemLevel": 125}])
        gear_new = json.dumps([{"id": 29383, "slot": 0, "itemLevel": 141}])

        db.execute(
            """INSERT INTO wcl_combatants (report_code, fight_id, source_id, gear_json)
               VALUES ('RPT_A', 1, 1, ?)""",
            (gear_old,),
        )
        db.execute(
            """INSERT INTO wcl_combatants (report_code, fight_id, source_id, gear_json)
               VALUES ('RPT_B', 1, 1, ?)""",
            (gear_new,),
        )
        db.commit()

        syncer = CharacterSyncer(mock_client, db)
        changes = syncer.detect_gear_changes(42)

        assert len(changes) == 1
        assert changes[0]["from_report"] == "RPT_A"
        assert changes[0]["to_report"] == "RPT_B"
        assert changes[0]["old_gear"][0]["id"] == 28830
        assert changes[0]["new_gear"][0]["id"] == 29383
