"""Tests for WCL GraphQL query constants and builder functions.

Verifies query strings contain expected GraphQL keywords and that builder
functions produce correct (query, variables) tuples.
"""

import pytest

from code.shukketsu.apis.wcl.queries import (
    CHARACTER_BY_ID,
    CHARACTER_ZONE_RANKINGS,
    ENCOUNTER_RANKINGS,
    RATE_LIMIT,
    REPORT_BUFF_TABLE,
    REPORT_CAST_TABLE,
    REPORT_COMBATANT_INFO,
    REPORT_DAMAGE_EVENTS,
    REPORT_DAMAGE_TABLE,
    REPORT_FIGHTS,
    REPORT_RANKINGS,
    ZONE_METADATA,
    build_buff_table_query,
    build_cast_table_query,
    build_character_query,
    build_character_zone_rankings_query,
    build_combatant_info_query,
    build_damage_events_query,
    build_damage_table_query,
    build_rankings_query,
    build_report_fights_query,
    build_report_rankings_query,
    build_zone_metadata_query,
)

# ---------------------------------------------------------------------------
# Query string constants
# ---------------------------------------------------------------------------

ALL_QUERIES = [
    ZONE_METADATA,
    ENCOUNTER_RANKINGS,
    REPORT_FIGHTS,
    REPORT_COMBATANT_INFO,
    REPORT_DAMAGE_TABLE,
    REPORT_BUFF_TABLE,
    REPORT_CAST_TABLE,
    REPORT_RANKINGS,
    REPORT_DAMAGE_EVENTS,
    CHARACTER_BY_ID,
    CHARACTER_ZONE_RANKINGS,
    RATE_LIMIT,
]


class TestQueryConstants:
    """Tests for GraphQL query string constants."""

    @pytest.mark.parametrize("query", ALL_QUERIES)
    def test_non_empty_string(self, query: str) -> None:
        """Every query constant is a non-empty string."""
        assert isinstance(query, str)
        assert len(query) > 0

    def test_expected_root_fields(self) -> None:
        """Each query references the correct top-level GraphQL field."""
        assert "worldData" in ZONE_METADATA
        assert "worldData" in ENCOUNTER_RANKINGS
        assert "reportData" in REPORT_FIGHTS
        assert "reportData" in REPORT_COMBATANT_INFO
        assert "reportData" in REPORT_DAMAGE_TABLE
        assert "reportData" in REPORT_BUFF_TABLE
        assert "reportData" in REPORT_CAST_TABLE
        assert "reportData" in REPORT_RANKINGS
        assert "reportData" in REPORT_DAMAGE_EVENTS
        assert "characterData" in CHARACTER_BY_ID
        assert "characterData" in CHARACTER_ZONE_RANKINGS
        assert "rateLimitData" in RATE_LIMIT

    def test_encounter_rankings_rogue_filter(self) -> None:
        """Encounter rankings query hard-codes Rogue class filter."""
        assert 'className: "Rogue"' in ENCOUNTER_RANKINGS
        assert "characterRankings" in ENCOUNTER_RANKINGS

    def test_damage_table_rogue_filter(self) -> None:
        """Table queries filter by sourceClass Rogue."""
        assert 'sourceClass: "Rogue"' in REPORT_DAMAGE_TABLE
        assert 'sourceClass: "Rogue"' in REPORT_DAMAGE_EVENTS


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


class TestBuildZoneMetadataQuery:
    """Tests for build_zone_metadata_query."""

    def test_returns_tuple(self) -> None:
        query, variables = build_zone_metadata_query(zone_id=1012)
        assert isinstance(query, str)
        assert isinstance(variables, dict)

    def test_variables(self) -> None:
        _, variables = build_zone_metadata_query(zone_id=1012)
        assert variables == {"zoneID": 1012}


class TestBuildRankingsQuery:
    """Tests for build_rankings_query."""

    def test_default_page(self) -> None:
        _, variables = build_rankings_query(encounter_id=725)
        assert variables["encounterID"] == 725
        assert variables["page"] == 1

    def test_custom_page(self) -> None:
        _, variables = build_rankings_query(encounter_id=725, page=3)
        assert variables["page"] == 3


class TestBuildReportQueries:
    """Tests for report-level builder functions."""

    def test_report_fights(self) -> None:
        query, variables = build_report_fights_query(code="SWP2024x")
        assert query is REPORT_FIGHTS
        assert variables == {"code": "SWP2024x"}

    def test_combatant_info_defaults(self) -> None:
        _, variables = build_combatant_info_query(code="abc123", fight_ids=[7, 8])
        assert variables["code"] == "abc123"
        assert variables["fightIDs"] == [7, 8]
        assert variables["limit"] == 100

    def test_combatant_info_custom_limit(self) -> None:
        _, variables = build_combatant_info_query(code="abc123", fight_ids=[7], limit=50)
        assert variables["limit"] == 50

    def test_damage_table(self) -> None:
        query, variables = build_damage_table_query(code="xyz", fight_ids=[1, 2])
        assert query is REPORT_DAMAGE_TABLE
        assert variables == {"code": "xyz", "fightIDs": [1, 2]}

    def test_buff_table(self) -> None:
        query, variables = build_buff_table_query(code="xyz", fight_ids=[3])
        assert query is REPORT_BUFF_TABLE
        assert variables == {"code": "xyz", "fightIDs": [3]}

    def test_cast_table(self) -> None:
        query, variables = build_cast_table_query(code="xyz", fight_ids=[4])
        assert query is REPORT_CAST_TABLE
        assert variables == {"code": "xyz", "fightIDs": [4]}

    def test_report_rankings(self) -> None:
        query, variables = build_report_rankings_query(code="SWP2024x")
        assert query is REPORT_RANKINGS
        assert variables == {"code": "SWP2024x"}


class TestBuildDamageEventsQuery:
    """Tests for build_damage_events_query."""

    def test_defaults(self) -> None:
        _, variables = build_damage_events_query(code="abc", fight_ids=[7])
        assert variables["code"] == "abc"
        assert variables["fightIDs"] == [7]
        assert variables["limit"] == 10000
        assert "startTime" not in variables
        assert "endTime" not in variables

    def test_with_time_window(self) -> None:
        _, variables = build_damage_events_query(code="abc", fight_ids=[7], start_time=1000.0, end_time=180000.0)
        assert variables["startTime"] == 1000.0
        assert variables["endTime"] == 180000.0

    def test_partial_time_window(self) -> None:
        """Only start_time provided, end_time omitted."""
        _, variables = build_damage_events_query(code="abc", fight_ids=[7], start_time=5000.0)
        assert variables["startTime"] == 5000.0
        assert "endTime" not in variables


class TestBuildCharacterQueries:
    """Tests for character builder functions."""

    def test_character_default_limit(self) -> None:
        _, variables = build_character_query(character_id=104956434)
        assert variables["id"] == 104956434
        assert variables["limit"] == 10

    def test_character_custom_limit(self) -> None:
        _, variables = build_character_query(character_id=104956434, report_limit=5)
        assert variables["limit"] == 5

    def test_character_zone_rankings(self) -> None:
        query, variables = build_character_zone_rankings_query(character_id=104956434, zone_id=1012)
        assert query is CHARACTER_ZONE_RANKINGS
        assert variables == {"id": 104956434, "zoneID": 1012}
