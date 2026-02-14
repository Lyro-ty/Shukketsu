"""Ingest orchestrators for WCL API data.

Three ingest modes:
- **RankingsIngestor** — fetches top Rogue rankings per encounter.
- **ReportDiver** — deep-dives a report (combatants, damage, buffs, casts, rankings).
- **CharacterSyncer** — syncs tracked characters, detects gear changes.

Also provides a CLI entry point (``python3 -m code.shukketsu.apis.wcl.ingest``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3

from code.shukketsu import config
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
    build_zone_metadata_query,
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

        try:
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
        except Exception:
            self._conn.rollback()
            raise
        logger.info("Stored %d rankings for encounter %s (%d)", len(rankings), encounter_name, encounter_id)
        return report_codes


class ReportDiver:
    """Deep-dives a report: CombatantInfo, damage, buffs, casts, rankings."""

    def __init__(self, client: WCLClient, conn: sqlite3.Connection) -> None:
        self._client = client
        self._conn = conn
        self._actor_map: dict[int, str] = {}

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
        try:
            await self._fetch_combatant_info(report_code, fight_ids, endpoint)
            await self._fetch_damage(report_code, fight_ids, endpoint)
            await self._fetch_buffs(report_code, fight_ids, endpoint)
            await self._fetch_casts(report_code, fight_ids, endpoint)
            await self._fetch_rankings(report_code, endpoint)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
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
                    (f.get("endTime", 0) or 0) - (f.get("startTime", 0) or 0),
                    f.get("bossPercentage"),
                    f.get("averageItemLevel"),
                    f.get("size"),
                    f.get("difficulty"),
                ),
            )

        # Update report metadata now that we have fight timing data
        if fights:
            start_times = [f.get("startTime", 0) for f in fights if f.get("startTime")]
            end_times = [f.get("endTime", 0) for f in fights if f.get("endTime")]
            if start_times and end_times:
                self._conn.execute(
                    "UPDATE wcl_reports SET start_time = ?, end_time = ? WHERE code = ?",
                    (min(start_times), max(end_times), code),
                )
            title = report.get("title", "")
            zone_id = report.get("zone", {}).get("id") if report.get("zone") else None
            zone_name = report.get("zone", {}).get("name", "") if report.get("zone") else ""
            if title or zone_id:
                self._conn.execute(
                    "UPDATE wcl_reports SET title = COALESCE(?, title), "
                    "zone_id = COALESCE(?, zone_id), zone_name = COALESCE(?, zone_name) WHERE code = ?",
                    (title or None, zone_id, zone_name or None, code),
                )

        # Build actor map for player_name resolution in _fetch_combatant_info
        actors = report.get("masterData", {}).get("actors", [])
        self._actor_map = {a["id"]: a["name"] for a in actors if a.get("id") and a.get("name")}

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
            fight_id = evt.get("fight")
            if fight_id is None:
                fight_id = fight_ids[0] if fight_ids else 0
                logger.warning("CombatantInfo event missing fight field, defaulting to fight %d", fight_id)
            player_name = self._actor_map.get(evt.get("sourceID", 0), "")

            self._conn.execute(
                """INSERT OR REPLACE INTO wcl_combatants
                   (report_code, fight_id, source_id, player_name, spec_id, faction,
                    strength, agility, stamina, intellect, spirit,
                    crit_melee, crit_ranged, crit_spell,
                    haste_melee, haste_ranged, haste_spell,
                    hit_melee, hit_ranged, hit_spell,
                    expertise, dodge, parry, block, armor,
                    gear_json, talents_json, auras_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    code,
                    fight_id,
                    evt.get("sourceID", 0),
                    player_name,
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
        """Fetch and store damage table per fight."""
        for fid in fight_ids:
            query, variables = build_damage_table_query(code, [fid])
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
                        fid,
                        entry.get("name", ""),
                        entry.get("type"),
                        entry.get("total", 0),
                        entry.get("activeTime"),
                        json.dumps(entry.get("abilities", [])),
                        json.dumps(entry.get("targets", [])),
                    ),
                )

    async def _fetch_buffs(self, code: str, fight_ids: list[int], endpoint: str) -> None:
        """Fetch and store buff table per fight."""
        for fid in fight_ids:
            query, variables = build_buff_table_query(code, [fid])
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
                        fid,
                        aura.get("name", ""),
                        aura.get("guid", 0),
                        aura.get("totalUptime", 0),
                        aura.get("totalUses", 0),
                        json.dumps(aura.get("bands", [])),
                    ),
                )

    async def _fetch_casts(self, code: str, fight_ids: list[int], endpoint: str) -> None:
        """Fetch and store cast table per fight."""
        for fid in fight_ids:
            query, variables = build_cast_table_query(code, [fid])
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
                            fid,
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
        try:
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
        except Exception:
            self._conn.rollback()
            raise
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
               JOIN wcl_reports r ON r.code = c.report_code
               WHERE cl.character_wcl_id = ?
               ORDER BY COALESCE(r.start_time, 0), c.fight_id""",
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Classic Anniversary TBC zones
ACTIVE_TBC_ZONES: dict[int, str] = {
    1047: "Karazhan Anniversary",
    1048: "Gruul / Magtheridon Anniversary",
    1052: "SSC / TK Anniversary",
}

# Classic Fresh vanilla zones (current progression)
ACTIVE_FRESH_ZONES: dict[int, str] = {
    1049: "Molten Core",
    1034: "Blackwing Lair",
    1035: "Temple of Ahn'Qiraj",
    1036: "Naxxramas",
}


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for WCL ingest."""
    parser = argparse.ArgumentParser(
        description="Ingest data from Warcraft Logs API",
        prog="python3 -m code.shukketsu.apis.wcl",
    )
    parser.add_argument("--sync-characters", action="store_true", help="Sync tracked characters (pull new reports)")
    parser.add_argument("--rankings", action="store_true", help="Ingest top 100 rankings")
    parser.add_argument("--zone", type=int, default=None, help="Zone ID for rankings (required with --rankings)")
    parser.add_argument("--encounter", type=int, default=None, help="Specific encounter ID (optional with --rankings)")
    parser.add_argument("--report", type=str, default=None, help="Deep-dive a specific report code")
    parser.add_argument("--full", action="store_true", help="Full ingest: all active TBC zones")
    parser.add_argument("--rate-limit", action="store_true", help="Check and display rate limit status")
    parser.add_argument("--stats", action="store_true", help="Show WCL database statistics")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    parser.add_argument(
        "--endpoint",
        choices=["fresh", "classic"],
        default="fresh",
        help="API endpoint (default: %(default)s)",
    )
    return parser


def _print_stats(conn: sqlite3.Connection) -> None:
    """Print WCL database statistics."""
    tables = [
        ("wcl_rankings", "Rankings"),
        ("wcl_reports", "Reports"),
        ("wcl_fights", "Fights"),
        ("wcl_combatants", "Combatants"),
        ("wcl_damage", "Damage entries"),
        ("wcl_buffs", "Buff entries"),
        ("wcl_casts", "Cast entries"),
        ("wcl_fight_rankings", "Fight rankings"),
        ("wcl_character_log", "Character log"),
    ]
    print("\nWCL Database Statistics:")
    print("-" * 40)
    for table, label in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
        print(f"  {label}: {count}")


async def _async_main(args: argparse.Namespace) -> None:
    """Async entry point for WCL ingest CLI."""
    from code.shukketsu.apis.wcl.auth import WCLAuth
    from code.shukketsu.db.connection import get_connection, init_db

    conn = get_connection(config.DB_PATH)
    init_db(conn)
    client: WCLClient | None = None

    try:
        auth = WCLAuth()
        client = WCLClient(auth)
        endpoint: str = args.endpoint

        if args.rate_limit:
            rate_data = await client.check_rate_limit(endpoint)
            print(f"Points spent: {rate_data.get('pointsSpentThisHour', 0)}/{rate_data.get('limitPerHour', 0)}")
            print(f"Reset in: {rate_data.get('pointsResetIn', 0)}s")
            return

        if args.stats:
            _print_stats(conn)
            return

        if args.sync_characters:
            syncer = CharacterSyncer(client, conn)
            diver = ReportDiver(client, conn)
            for char_config in config.WCL_TRACKED_CHARACTERS:
                new_codes = await syncer.sync(char_config)
                char_endpoint = str(char_config.get("endpoint", endpoint))
                for code in new_codes:
                    if not args.dry_run:
                        await diver.dive(code, endpoint=char_endpoint)
                    else:
                        print(f"  Would deep-dive report: {code}")
            return

        if args.report:
            if args.dry_run:
                print(f"Would deep-dive report: {args.report}")
            else:
                diver = ReportDiver(client, conn)
                await diver.dive(args.report, endpoint=endpoint)
            return

        if args.rankings:
            if not args.zone:
                print("Error: --zone is required with --rankings")
                return
            ingestor = RankingsIngestor(client, conn)
            query, variables = build_zone_metadata_query(args.zone)
            data = await client.query(query, variables, endpoint=endpoint)
            zone = data.get("worldData", {}).get("zone", {})
            encounters = zone.get("encounters", [])
            if args.encounter:
                encounters = [e for e in encounters if e.get("id") == args.encounter]
            for enc in encounters:
                if args.dry_run:
                    print(f"Would ingest rankings for: {enc.get('name')} ({enc.get('id')})")
                else:
                    await ingestor.ingest_encounter(enc["id"], enc.get("name", ""), args.zone, endpoint)
            return

        if args.full:
            ingestor = RankingsIngestor(client, conn)
            diver = ReportDiver(client, conn)
            zones = ACTIVE_TBC_ZONES if endpoint == "classic" else ACTIVE_FRESH_ZONES
            for zone_id, zone_name in zones.items():
                print(f"\nIngesting zone: {zone_name} ({zone_id})")
                query, variables = build_zone_metadata_query(zone_id)
                data = await client.query(query, variables, endpoint=endpoint)
                zone = data.get("worldData", {}).get("zone", {})
                encounters = zone.get("encounters", [])
                all_codes: set[str] = set()
                for enc in encounters:
                    if args.dry_run:
                        print(f"  Would ingest: {enc.get('name')}")
                    else:
                        codes = await ingestor.ingest_encounter(enc["id"], enc.get("name", ""), zone_id, endpoint)
                        all_codes.update(codes)
                if not args.dry_run:
                    for code in all_codes:
                        await diver.dive(code, endpoint=endpoint)
            return

        print("No action specified. Use --help for options.")
    finally:
        if client is not None:
            await client.close()
        conn.close()


def main() -> None:
    """CLI entry point for WCL ingest."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
