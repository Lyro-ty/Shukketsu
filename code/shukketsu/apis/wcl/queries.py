"""Pre-built GraphQL query strings and builder functions for the WCL v2 API.

Each constant is a complete GraphQL query string.  Builder functions return
``(query, variables)`` tuples ready for submission to the GraphQL endpoint.
"""

# ---------------------------------------------------------------------------
# Zone & Encounter Metadata
# ---------------------------------------------------------------------------

ZONE_METADATA = """
query($zoneID: Int!) {
  worldData {
    zone(id: $zoneID) {
      id
      name
      encounters { id name }
    }
  }
}
""".strip()

# ---------------------------------------------------------------------------
# Encounter Rankings
# ---------------------------------------------------------------------------

ENCOUNTER_RANKINGS = """
query($encounterID: Int!, $page: Int) {
  worldData {
    encounter(id: $encounterID) {
      characterRankings(className: "Rogue", metric: dps, page: $page)
    }
  }
}
""".strip()

# ---------------------------------------------------------------------------
# Report Queries
# ---------------------------------------------------------------------------

REPORT_FIGHTS = """
query($code: String!) {
  reportData {
    report(code: $code) {
      fights {
        id encounterID name kill startTime endTime
        bossPercentage averageItemLevel size difficulty
      }
      masterData {
        actors(type: "Player") { id name type subType }
      }
    }
  }
}
""".strip()

REPORT_COMBATANT_INFO = """
query($code: String!, $fightIDs: [Int], $limit: Int) {
  reportData {
    report(code: $code) {
      events(dataType: CombatantInfo, fightIDs: $fightIDs, limit: $limit) {
        data
        nextPageTimestamp
      }
    }
  }
}
""".strip()

REPORT_DAMAGE_TABLE = """
query($code: String!, $fightIDs: [Int]) {
  reportData {
    report(code: $code) {
      table(dataType: DamageDone, fightIDs: $fightIDs, sourceClass: "Rogue")
    }
  }
}
""".strip()

REPORT_BUFF_TABLE = """
query($code: String!, $fightIDs: [Int]) {
  reportData {
    report(code: $code) {
      table(dataType: Buffs, fightIDs: $fightIDs, sourceClass: "Rogue")
    }
  }
}
""".strip()

REPORT_CAST_TABLE = """
query($code: String!, $fightIDs: [Int]) {
  reportData {
    report(code: $code) {
      table(dataType: Casts, fightIDs: $fightIDs, sourceClass: "Rogue")
    }
  }
}
""".strip()

REPORT_RANKINGS = """
query($code: String!) {
  reportData {
    report(code: $code) {
      rankings
    }
  }
}
""".strip()

REPORT_DAMAGE_EVENTS = """
query($code: String!, $fightIDs: [Int], $startTime: Float, $endTime: Float, $limit: Int) {
  reportData {
    report(code: $code) {
      events(dataType: DamageDone, fightIDs: $fightIDs, sourceClass: "Rogue",
             startTime: $startTime, endTime: $endTime, limit: $limit) {
        data
        nextPageTimestamp
      }
    }
  }
}
""".strip()

# ---------------------------------------------------------------------------
# Character Queries
# ---------------------------------------------------------------------------

CHARACTER_BY_ID = """
query($id: Int!, $limit: Int) {
  characterData {
    character(id: $id) {
      id name classID
      server { id name slug region { slug name } }
      recentReports(limit: $limit) {
        data { code title startTime endTime zone { id name } }
      }
    }
  }
}
""".strip()

CHARACTER_ZONE_RANKINGS = """
query($id: Int!, $zoneID: Int!) {
  characterData {
    character(id: $id) {
      zoneRankings(zoneID: $zoneID, metric: dps)
    }
  }
}
""".strip()

# ---------------------------------------------------------------------------
# Rate Limit
# ---------------------------------------------------------------------------

RATE_LIMIT = """
query {
  rateLimitData {
    pointsSpentThisHour
    limitPerHour
    pointsResetIn
  }
}
""".strip()


# ---------------------------------------------------------------------------
# Builder functions — return (query_string, variables_dict) tuples
# ---------------------------------------------------------------------------


def build_zone_metadata_query(zone_id: int) -> tuple[str, dict]:
    """Build a zone metadata query for the given zone ID."""
    return ZONE_METADATA, {"zoneID": zone_id}


def build_rankings_query(encounter_id: int, page: int = 1) -> tuple[str, dict]:
    """Build an encounter rankings query for top Rogues."""
    return ENCOUNTER_RANKINGS, {"encounterID": encounter_id, "page": page}


def build_report_fights_query(code: str) -> tuple[str, dict]:
    """Build a report fights + actors query."""
    return REPORT_FIGHTS, {"code": code}


def build_combatant_info_query(code: str, fight_ids: list[int], limit: int = 100) -> tuple[str, dict]:
    """Build a CombatantInfo events query for the given fights."""
    return REPORT_COMBATANT_INFO, {"code": code, "fightIDs": fight_ids, "limit": limit}


def build_damage_table_query(code: str, fight_ids: list[int]) -> tuple[str, dict]:
    """Build a damage table query filtered to Rogues."""
    return REPORT_DAMAGE_TABLE, {"code": code, "fightIDs": fight_ids}


def build_buff_table_query(code: str, fight_ids: list[int]) -> tuple[str, dict]:
    """Build a buff table query filtered to Rogues."""
    return REPORT_BUFF_TABLE, {"code": code, "fightIDs": fight_ids}


def build_cast_table_query(code: str, fight_ids: list[int]) -> tuple[str, dict]:
    """Build a cast table query filtered to Rogues."""
    return REPORT_CAST_TABLE, {"code": code, "fightIDs": fight_ids}


def build_report_rankings_query(code: str) -> tuple[str, dict]:
    """Build a report rankings query."""
    return REPORT_RANKINGS, {"code": code}


def build_damage_events_query(
    code: str,
    fight_ids: list[int],
    start_time: float | None = None,
    end_time: float | None = None,
    limit: int = 10000,
) -> tuple[str, dict]:
    """Build a raw damage events query with optional time window."""
    variables: dict = {
        "code": code,
        "fightIDs": fight_ids,
        "limit": limit,
    }
    if start_time is not None:
        variables["startTime"] = start_time
    if end_time is not None:
        variables["endTime"] = end_time
    return REPORT_DAMAGE_EVENTS, variables


def build_character_query(character_id: int, report_limit: int = 10) -> tuple[str, dict]:
    """Build a character profile query with recent reports."""
    return CHARACTER_BY_ID, {"id": character_id, "limit": report_limit}


def build_character_zone_rankings_query(character_id: int, zone_id: int) -> tuple[str, dict]:
    """Build a character zone rankings query."""
    return CHARACTER_ZONE_RANKINGS, {"id": character_id, "zoneID": zone_id}
