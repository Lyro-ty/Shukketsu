"""Ingest orchestrators for WCL API data.

Three ingest modes:
- **RankingsIngestor** — fetches top Rogue rankings per encounter.
- **ReportDiver** — deep-dives a report (combatants, damage, buffs, casts, rankings).
- **CharacterSyncer** — syncs tracked characters, detects gear changes.
"""

import json
import logging
import sqlite3

from code.shukketsu.apis.wcl.client import WCLClient
from code.shukketsu.apis.wcl.queries import (
    build_buff_table_query,
    build_cast_table_query,
    build_character_query,
    build_combatant_info_query,
    build_damage_table_query,
    build_rankings_query,
    build_report_fights_query,
    build_report_rankings_query,
)
from code.shukketsu.resilience.errors import WCLQueryError

logger = logging.getLogger(__name__)


class RankingsIngestor:
    """Fetches top Rogue rankings for TBC encounters and stores in DB."""

    def __init__(self, client: WCLClient, conn: sqlite3.Connection) -> None:
        self._client = client
        self._conn = conn

    async def ingest_encounter(
        self,
        encounter_id: int,
        encounter_name: str,
        zone_id: int,
        endpoint: str = "fresh",
    ) -> set[str]:
        """Fetch top 100 Rogue rankings for an encounter.

        Returns set of unique report codes found.
        """
        query, variables = build_rankings_query(encounter_id, page=1)
        data = await self._client.query(query, variables, endpoint=endpoint)

        # Response structure: worldData -> encounter -> characterRankings
        rankings_data = data.get("worldData", {}).get("encounter", {}).get("characterRankings", {})
        rankings = rankings_data.get("rankings", [])
        report_codes: set[str] = set()

        for r in rankings:
            report = r.get("report", {})
            report_code = report.get("code", "")
            if report_code:
                report_codes.add(report_code)

            server = r.get("server", {})
            guild = r.get("guild", {})

            self._conn.execute(
                """INSERT OR REPLACE INTO wcl_rankings
                   (zone_id, encounter_id, encounter_name, player_name, spec, dps, duration_ms,
                    report_code, fight_id, guild_name, server_name, server_region,
                    faction, raid_size, bracket_data)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    zone_id,
                    encounter_id,
                    encounter_name,
                    r.get("name", ""),
                    r.get("spec", ""),
                    r.get("amount", 0),
                    r.get("duration", 0),
                    report_code,
                    report.get("fightID", 0),
                    guild.get("name"),
                    server.get("name", ""),
                    server.get("region", ""),
                    r.get("faction"),
                    r.get("size"),
                    r.get("bracketData"),
                ),
            )

        self._conn.commit()
        logger.info("Stored %d rankings for encounter %s (%d)", len(rankings), encounter_name, encounter_id)
        return report_codes


class ReportDiver:
    """Deep-dives a report: CombatantInfo, damage, buffs, casts, rankings."""

    def __init__(self, client: WCLClient, conn: sqlite3.Connection) -> None:
        self._client = client
        self._conn = conn

    async def dive(
        self,
        report_code: str,
        fight_ids: list[int] | None = None,
        endpoint: str = "fresh",
    ) -> None:
        """Deep-dive a report, storing all data."""
        # Ensure report metadata row exists before inserting child rows (FK constraint)
        self._store_report(report_code, endpoint)

        if fight_ids is None:
            fight_ids = await self._fetch_fights(report_code, endpoint)

        if not fight_ids:
            logger.warning("No fights found in report %s", report_code)
            return

        # Fetch all data types
        await self._fetch_combatant_info(report_code, fight_ids, endpoint)
        await self._fetch_damage(report_code, fight_ids, endpoint)
        await self._fetch_buffs(report_code, fight_ids, endpoint)
        await self._fetch_casts(report_code, fight_ids, endpoint)
        await self._fetch_rankings(report_code, endpoint)

        self._conn.commit()
        logger.info("Deep-dive complete for report %s (%d fights)", report_code, len(fight_ids))

    async def _fetch_fights(self, code: str, endpoint: str) -> list[int]:
        """Fetch fight list from report, store fights, return boss fight IDs."""
        query, variables = build_report_fights_query(code)
        try:
            data = await self._client.query(query, variables, endpoint=endpoint)
        except WCLQueryError as exc:
            if "archived" in str(exc).lower():
                logger.warning("Report %s is archived, skipping", code)
                self._conn.execute(
                    "INSERT OR REPLACE INTO wcl_reports (code, is_archived, endpoint) VALUES (?, 1, ?)",
                    (code, endpoint),
                )
                self._conn.commit()
                return []
            raise

        report = data.get("reportData", {}).get("report", {})
        fights = report.get("fights", [])
        fight_ids: list[int] = []

        for f in fights:
            eid = f.get("encounterID", 0)
            if eid == 0:  # Skip trash
                continue
            fight_ids.append(f["id"])
            self._conn.execute(
                """INSERT OR REPLACE INTO wcl_fights
                   (report_code, fight_id, encounter_id, encounter_name, kill, duration_ms,
                    boss_percentage, avg_item_level, raid_size, difficulty)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    code,
                    f["id"],
                    eid,
                    f.get("name", ""),
                    1 if f.get("kill") else 0,
                    f.get("duration", 0) if f.get("duration") else 0,
                    f.get("bossPercentage"),
                    f.get("averageItemLevel"),
                    f.get("size"),
                    f.get("difficulty"),
                ),
            )

        return fight_ids

    def _store_report(self, code: str, endpoint: str) -> None:
        """Ensure report metadata row exists."""
        self._conn.execute(
            "INSERT OR IGNORE INTO wcl_reports (code, endpoint) VALUES (?, ?)",
            (code, endpoint),
        )

    async def _fetch_combatant_info(self, code: str, fight_ids: list[int], endpoint: str) -> None:
        """Fetch and store CombatantInfo events."""
        query, variables = build_combatant_info_query(code, fight_ids)
        data = await self._client.query(query, variables, endpoint=endpoint)
        events = data.get("reportData", {}).get("report", {}).get("events", {}).get("data", [])

        for evt in events:
            gear_json = json.dumps(evt.get("gear", []))
            talents_json = json.dumps(evt.get("talents", []))
            auras_json = json.dumps(evt.get("auras", []))

            # Events may include a fight field; fall back to first fight_id
            fight_id = evt.get("fight", fight_ids[0] if fight_ids else 0)

            self._conn.execute(
                """INSERT OR REPLACE INTO wcl_combatants
                   (report_code, fight_id, source_id, spec_id, faction,
                    strength, agility, stamina, intellect, spirit,
                    crit_melee, crit_ranged, crit_spell,
                    haste_melee, haste_ranged, haste_spell,
                    hit_melee, hit_ranged, hit_spell,
                    expertise, dodge, parry, block, armor,
                    gear_json, talents_json, auras_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    code,
                    fight_id,
                    evt.get("sourceID", 0),
                    evt.get("specID"),
                    evt.get("faction"),
                    evt.get("strength"),
                    evt.get("agility"),
                    evt.get("stamina"),
                    evt.get("intellect"),
                    evt.get("spirit"),
                    evt.get("critMelee"),
                    evt.get("critRanged"),
                    evt.get("critSpell"),
                    evt.get("hasteMelee"),
                    evt.get("hasteRanged"),
                    evt.get("hasteSpell"),
                    evt.get("hitMelee"),
                    evt.get("hitRanged"),
                    evt.get("hitSpell"),
                    evt.get("expertise"),
                    evt.get("dodge"),
                    evt.get("parry"),
                    evt.get("block"),
                    evt.get("armor"),
                    gear_json,
                    talents_json,
                    auras_json,
                ),
            )

    async def _fetch_damage(self, code: str, fight_ids: list[int], endpoint: str) -> None:
        """Fetch and store damage table."""
        query, variables = build_damage_table_query(code, fight_ids)
        data = await self._client.query(query, variables, endpoint=endpoint)
        table = data.get("reportData", {}).get("report", {}).get("table", {})
        entries = table.get("data", {}).get("entries", [])

        for entry in entries:
            self._conn.execute(
                """INSERT OR REPLACE INTO wcl_damage
                   (report_code, fight_id, player_name, player_type, total_damage,
                    active_time_ms, abilities_json, targets_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    code,
                    fight_ids[0] if fight_ids else 0,
                    entry.get("name", ""),
                    entry.get("type"),
                    entry.get("total", 0),
                    entry.get("activeTime"),
                    json.dumps(entry.get("abilities", [])),
                    json.dumps(entry.get("targets", [])),
                ),
            )

    async def _fetch_buffs(self, code: str, fight_ids: list[int], endpoint: str) -> None:
        """Fetch and store buff table."""
        query, variables = build_buff_table_query(code, fight_ids)
        data = await self._client.query(query, variables, endpoint=endpoint)
        table = data.get("reportData", {}).get("report", {}).get("table", {})
        auras = table.get("data", {}).get("auras", [])

        for aura in auras:
            self._conn.execute(
                """INSERT OR REPLACE INTO wcl_buffs
                   (report_code, fight_id, buff_name, buff_guid, total_uptime_ms,
                    total_uses, bands_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    code,
                    fight_ids[0] if fight_ids else 0,
                    aura.get("name", ""),
                    aura.get("guid", 0),
                    aura.get("totalUptime", 0),
                    aura.get("totalUses", 0),
                    json.dumps(aura.get("bands", [])),
                ),
            )

    async def _fetch_casts(self, code: str, fight_ids: list[int], endpoint: str) -> None:
        """Fetch and store cast table."""
        query, variables = build_cast_table_query(code, fight_ids)
        data = await self._client.query(query, variables, endpoint=endpoint)
        table = data.get("reportData", {}).get("report", {}).get("table", {})
        entries = table.get("data", {}).get("entries", [])

        for entry in entries:
            for ability in entry.get("abilities", []):
                self._conn.execute(
                    """INSERT OR REPLACE INTO wcl_casts
                       (report_code, fight_id, player_name, ability_name, cast_count)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        code,
                        fight_ids[0] if fight_ids else 0,
                        entry.get("name", ""),
                        ability.get("name", ""),
                        ability.get("total", 0),
                    ),
                )

    async def _fetch_rankings(self, code: str, endpoint: str) -> None:
        """Fetch and store per-fight rankings."""
        query, variables = build_report_rankings_query(code)
        data = await self._client.query(query, variables, endpoint=endpoint)
        rankings = data.get("reportData", {}).get("report", {}).get("rankings", {})

        if not rankings or not isinstance(rankings, dict):
            return

        for fight_data in rankings.get("data", []):
            fight_id = fight_data.get("fightID", 0)
            encounter_id = fight_data.get("encounterID", 0)
            for role in fight_data.get("roles", {}).values():
                for char in role.get("characters", []):
                    server_val = char.get("server")
                    server_name = server_val.get("name") if isinstance(server_val, dict) else server_val

                    self._conn.execute(
                        """INSERT OR REPLACE INTO wcl_fight_rankings
                           (report_code, fight_id, encounter_id, player_name, server_name,
                            class, spec, dps, rank_percent, total_parses)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            code,
                            fight_id,
                            encounter_id,
                            char.get("name", ""),
                            server_name,
                            char.get("class", ""),
                            char.get("spec"),
                            char.get("amount", 0),
                            char.get("rankPercent"),
                            char.get("totalParses"),
                        ),
                    )


class CharacterSyncer:
    """Syncs tracked characters, stores log entries, detects gear changes."""

    def __init__(self, client: WCLClient, conn: sqlite3.Connection) -> None:
        self._client = client
        self._conn = conn

    async def sync(self, character_config: dict) -> list[str]:
        """Sync a tracked character's recent reports.

        Returns list of new report codes found.
        """
        wcl_id: int = character_config["wcl_id"]
        endpoint = character_config.get("endpoint", "fresh")

        query, variables = build_character_query(wcl_id, report_limit=10)
        data = await self._client.query(query, variables, endpoint=endpoint)

        char_data = data.get("characterData", {}).get("character", {})
        reports = char_data.get("recentReports", {}).get("data", [])

        # Find reports not yet in character_log
        existing: set[str] = set()
        rows = self._conn.execute(
            "SELECT report_code FROM wcl_character_log WHERE character_wcl_id = ?",
            (wcl_id,),
        ).fetchall()
        for row in rows:
            existing.add(row[0])

        new_codes: list[str] = []
        for report in reports:
            code = report.get("code", "")
            if code and code not in existing:
                new_codes.append(code)
                # Store a basic log entry (will be enriched by deep-dive)
                self._conn.execute(
                    """INSERT OR IGNORE INTO wcl_character_log
                       (character_wcl_id, report_code, fight_id, encounter_id)
                       VALUES (?, ?, 0, 0)""",
                    (wcl_id, code),
                )

        self._conn.commit()
        logger.info("Character %s: %d new reports found", character_config.get("name", wcl_id), len(new_codes))
        return new_codes

    def detect_gear_changes(self, wcl_id: int) -> list[dict]:
        """Compare gear JSON between consecutive syncs.

        Returns list of change dicts with old/new gear.
        """
        rows = self._conn.execute(
            """SELECT c.fight_id, c.gear_json, c.report_code
               FROM wcl_combatants c
               JOIN wcl_character_log cl ON cl.report_code = c.report_code
               WHERE cl.character_wcl_id = ?
               ORDER BY c.report_code, c.fight_id""",
            (wcl_id,),
        ).fetchall()

        changes: list[dict] = []
        prev_gear: str | None = None
        prev_report: str | None = None

        for row in rows:
            gear_json = row[1] or "[]"
            report_code = row[2]
            if prev_gear is not None and gear_json != prev_gear:
                changes.append(
                    {
                        "from_report": prev_report,
                        "to_report": report_code,
                        "old_gear": json.loads(prev_gear),
                        "new_gear": json.loads(gear_json),
                    }
                )
            prev_gear = gear_json
            prev_report = report_code

        return changes
