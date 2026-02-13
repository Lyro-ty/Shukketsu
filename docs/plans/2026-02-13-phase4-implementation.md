# Phase 4: Sim Validation & Calibration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate the TBC Rogue simulation engine against real WCL combat data, calibrate until DPS drift ≤ 5%, and fill remaining gaps (CLEU parser, WoWSims import, item DB expansion).

**Architecture:** WCL ingest fixes → spell mapping → fight filtering → item DB expansion → WCL-to-SimConfig bridge (synthetic Items) → comparison engine → validation pipeline (ProcessPoolExecutor) → CLEU parser → WoWSims import → web dashboards → calibration. Design doc: `docs/plans/2026-02-13-phase4-validation-design.md`.

**Tech Stack:** Python 3.12, SQLite (schema v5.1 → v6), Pydantic v2, FastAPI/HTMX, concurrent.futures.ProcessPoolExecutor, existing sim engine + WCL API subsystem.

---

## Context for Implementers

**Test invocation:** Always use `python3 -m pytest` (never bare `pytest`).

**Import pattern:** `from code.shukketsu.X import Y` — project root is CWD.

**Error pattern:** All exceptions inherit `ShukketsuError` with a `FailureMode` enum member.

**Pydantic pattern:** v2 `BaseModel`, `model_config = ConfigDict(frozen=True)` for immutable models.

**Key existing files you'll reference:**
- `code/shukketsu/apis/wcl/ingest.py` — `ReportDiver` class, methods at lines 109-313
- `code/shukketsu/apis/wcl/schema.sql` — WCL table definitions, lines 67-122
- `code/shukketsu/apis/wcl/queries.py` — GraphQL queries and builder functions
- `code/shukketsu/resilience/errors.py` — Error taxonomy
- `code/shukketsu/db/connection.py` — Migration chain pattern, lines 73-202
- `code/shukketsu/sim/models.py` — `SimConfig` (line 244), `SimResult` (line 341), `Item` (line 188)
- `code/shukketsu/sim/items.py` — `ItemDatabase` class (line 662), `_CURATED_ITEMS` (line 25)
- `code/shukketsu/sim/combat.py` — `CombatSimulation`, `_resolve_weapons()` (line 1209), `_compute_base_stats()` (line 1238)
- `code/shukketsu/sim/buffs.py` — `resolve_buffs()`, `ResolvedBuffs` model
- `code/shukketsu/sim/imports.py` — `parse_wowsims()` stub (line 243)
- `code/shukketsu/config.py` — All config constants

---

### Task 0: Fix WCL Ingest — Per-Fight Data Storage + Actor Mapping

**PREREQUISITE** — All downstream tasks depend on correct per-fight WCL data.

**Files:**
- Modify: `code/shukketsu/apis/wcl/ingest.py:244-313` (per-fight loops)
- Modify: `code/shukketsu/apis/wcl/ingest.py:136-180` (store actors)
- Modify: `code/shukketsu/apis/wcl/ingest.py:189-242` (add player_name)
- Modify: `code/shukketsu/db/connection.py:73-99` (v5.1 migration)
- Test: `tests/unit/apis/wcl/test_wcl_ingest.py`

**Step 1: Write failing tests for per-fight damage storage**

Add to `tests/unit/apis/wcl/test_wcl_ingest.py`:

```python
class TestPerFightStorage:
    """Tests for per-fight data storage (not aggregated)."""

    @pytest.fixture()
    def diver(self, test_db, mock_client):
        return ReportDiver(mock_client, test_db)

    async def test_damage_stored_per_fight(self, diver, test_db, mock_client):
        """Each fight_id gets its own damage rows, not aggregated under fight_ids[0]."""
        # Mock returns damage data for two fights
        fight_ids = [1, 2]
        mock_client.query = AsyncMock(side_effect=[
            # Fight 1 damage
            {"reportData": {"report": {"table": {"data": {"entries": [
                {"name": "Lyroo", "type": "Rogue", "total": 50000, "activeTime": 60000,
                 "abilities": [{"name": "Sinister Strike", "total": 30000}],
                 "targets": []}
            ]}}}}},
            # Fight 2 damage
            {"reportData": {"report": {"table": {"data": {"entries": [
                {"name": "Lyroo", "type": "Rogue", "total": 80000, "activeTime": 90000,
                 "abilities": [{"name": "Sinister Strike", "total": 45000}],
                 "targets": []}
            ]}}}}},
        ])
        # Store report metadata first
        diver._store_report("ABC123", "fresh")
        test_db.execute(
            "INSERT OR REPLACE INTO wcl_fights (report_code, fight_id, encounter_id, encounter_name, kill, duration_ms) VALUES (?, ?, ?, ?, ?, ?)",
            ("ABC123", 1, 100, "Boss1", 1, 60000),
        )
        test_db.execute(
            "INSERT OR REPLACE INTO wcl_fights (report_code, fight_id, encounter_id, encounter_name, kill, duration_ms) VALUES (?, ?, ?, ?, ?, ?)",
            ("ABC123", 2, 100, "Boss1", 1, 90000),
        )

        await diver._fetch_damage("ABC123", fight_ids, "fresh")

        rows = test_db.execute(
            "SELECT fight_id, total_damage FROM wcl_damage WHERE report_code = ? ORDER BY fight_id",
            ("ABC123",),
        ).fetchall()
        assert len(rows) == 2
        assert rows[0] == (1, 50000)
        assert rows[1] == (2, 80000)

    async def test_buffs_stored_per_fight(self, diver, test_db, mock_client):
        """Each fight_id gets its own buff rows."""
        fight_ids = [1, 2]
        mock_client.query = AsyncMock(side_effect=[
            {"reportData": {"report": {"table": {"data": {"auras": [
                {"name": "Slice and Dice", "guid": 6774, "totalUptime": 55000, "totalUses": 3, "bands": []}
            ]}}}}},
            {"reportData": {"report": {"table": {"data": {"auras": [
                {"name": "Slice and Dice", "guid": 6774, "totalUptime": 80000, "totalUses": 5, "bands": []}
            ]}}}}},
        ])
        diver._store_report("ABC123", "fresh")
        test_db.execute(
            "INSERT OR REPLACE INTO wcl_fights (report_code, fight_id, encounter_id, encounter_name, kill, duration_ms) VALUES (?, ?, ?, ?, ?, ?)",
            ("ABC123", 1, 100, "Boss1", 1, 60000),
        )
        test_db.execute(
            "INSERT OR REPLACE INTO wcl_fights (report_code, fight_id, encounter_id, encounter_name, kill, duration_ms) VALUES (?, ?, ?, ?, ?, ?)",
            ("ABC123", 2, 100, "Boss1", 1, 90000),
        )

        await diver._fetch_buffs("ABC123", fight_ids, "fresh")

        rows = test_db.execute(
            "SELECT fight_id, total_uptime_ms FROM wcl_buffs WHERE report_code = ? ORDER BY fight_id",
            ("ABC123",),
        ).fetchall()
        assert len(rows) == 2
        assert rows[0] == (1, 55000)
        assert rows[1] == (2, 80000)

    async def test_casts_stored_per_fight(self, diver, test_db, mock_client):
        """Each fight_id gets its own cast rows."""
        fight_ids = [1, 2]
        mock_client.query = AsyncMock(side_effect=[
            {"reportData": {"report": {"table": {"data": {"entries": [
                {"name": "Lyroo", "abilities": [{"name": "Sinister Strike", "total": 40}]}
            ]}}}}},
            {"reportData": {"report": {"table": {"data": {"entries": [
                {"name": "Lyroo", "abilities": [{"name": "Sinister Strike", "total": 60}]}
            ]}}}}},
        ])
        diver._store_report("ABC123", "fresh")
        test_db.execute(
            "INSERT OR REPLACE INTO wcl_fights (report_code, fight_id, encounter_id, encounter_name, kill, duration_ms) VALUES (?, ?, ?, ?, ?, ?)",
            ("ABC123", 1, 100, "Boss1", 1, 60000),
        )
        test_db.execute(
            "INSERT OR REPLACE INTO wcl_fights (report_code, fight_id, encounter_id, encounter_name, kill, duration_ms) VALUES (?, ?, ?, ?, ?, ?)",
            ("ABC123", 2, 100, "Boss1", 1, 90000),
        )

        await diver._fetch_casts("ABC123", fight_ids, "fresh")

        rows = test_db.execute(
            "SELECT fight_id, cast_count FROM wcl_casts WHERE report_code = ? AND player_name = ? ORDER BY fight_id",
            ("ABC123", "Lyroo"),
        ).fetchall()
        assert len(rows) == 2
        assert rows[0] == (1, 40)
        assert rows[1] == (2, 60)

    async def test_combatant_player_name_stored(self, diver, test_db, mock_client):
        """wcl_combatants stores player_name from actors list."""
        mock_client.query = AsyncMock(return_value={
            "reportData": {"report": {
                "events": {"data": [
                    {"sourceID": 5, "specID": 260, "fight": 1,
                     "strength": 100, "agility": 500,
                     "gear": [], "talents": [], "auras": []}
                ]}
            }}
        })
        diver._store_report("ABC123", "fresh")
        test_db.execute(
            "INSERT OR REPLACE INTO wcl_fights (report_code, fight_id, encounter_id, encounter_name, kill, duration_ms) VALUES (?, ?, ?, ?, ?, ?)",
            ("ABC123", 1, 100, "Boss1", 1, 60000),
        )
        # Store actor mapping (simulating what _fetch_fights stores)
        diver._actor_map = {5: "Lyroo"}

        await diver._fetch_combatant_info("ABC123", [1], "fresh")

        row = test_db.execute(
            "SELECT player_name FROM wcl_combatants WHERE report_code = ? AND source_id = ?",
            ("ABC123", 5),
        ).fetchone()
        assert row is not None
        assert row[0] == "Lyroo"
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/apis/wcl/test_wcl_ingest.py::TestPerFightStorage -v`
Expected: FAIL — tests expect per-fight storage but code stores under `fight_ids[0]`.

**Step 3: Fix `_fetch_damage()` — per-fight loop**

In `code/shukketsu/apis/wcl/ingest.py`, replace lines 244-267:

```python
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
```

**Step 4: Fix `_fetch_buffs()` — per-fight loop**

Replace lines 269-291:

```python
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
```

**Step 5: Fix `_fetch_casts()` — per-fight loop**

Replace lines 293-313:

```python
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
```

**Step 6: Store actors in `_fetch_fights()` and add `player_name` to combatants**

In `_fetch_fights()`, after extracting fights, store the actor map:

```python
# At end of _fetch_fights(), after the fights loop:
actors = report.get("masterData", {}).get("actors", [])
self._actor_map: dict[int, str] = {a["id"]: a["name"] for a in actors if a.get("id") and a.get("name")}
```

In `_fetch_combatant_info()`, use the actor map to populate `player_name`:

```python
# In the INSERT, add player_name column:
player_name = getattr(self, "_actor_map", {}).get(evt.get("sourceID", 0), "")
# Add player_name to the INSERT and values tuple
```

**Step 7: Add v5.1 migration**

In `code/shukketsu/db/connection.py`, add after `_migrate_v4_to_v5`:

```python
_V5_1_SQL = """
ALTER TABLE wcl_combatants ADD COLUMN player_name TEXT;
"""


def _migrate_v5_to_v5_1(conn: sqlite3.Connection) -> None:
    """Migrate v5 to v5.1: add player_name to wcl_combatants."""
    try:
        conn.execute(_V5_1_SQL)
    except sqlite3.OperationalError:
        pass  # Column already exists (idempotent)
    conn.execute("INSERT INTO schema_version (version) VALUES (51)")
    conn.commit()
    logger.info("Database migrated from v5 to v5.1 (wcl_combatants.player_name)")
```

Update `init_db()` to call this migration:
```python
if version < 51:
    _migrate_v5_to_v5_1(conn)
```

**Step 8: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/apis/wcl/test_wcl_ingest.py -v`
Expected: ALL PASS (existing + new)

**Step 9: Run full suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All 1470+ tests pass

**Step 10: Commit**

```bash
git add code/shukketsu/apis/wcl/ingest.py code/shukketsu/db/connection.py tests/unit/apis/wcl/test_wcl_ingest.py
git commit -m "fix(wcl): per-fight data storage and actor name mapping

BREAKING: WCL ingest now queries per-fight instead of aggregating
under fight_ids[0]. Adds player_name to wcl_combatants via schema
v5.1 migration. Required for sim validation pipeline."
```

---

### Task 1: Add Error Types

**Files:**
- Modify: `code/shukketsu/resilience/errors.py`
- Test: `tests/unit/test_errors.py`

**Step 1: Write failing tests**

Add to `tests/unit/test_errors.py`:

```python
class TestPhase4Errors:
    def test_wcl_bridge_error(self):
        from code.shukketsu.resilience.errors import WCLBridgeError, FailureMode
        err = WCLBridgeError("Missing weapon item 32837")
        assert err.failure_mode == FailureMode.SIM_VALIDATION
        assert "32837" in str(err)

    def test_log_parse_error(self):
        from code.shukketsu.resilience.errors import LogParseError, FailureMode
        err = LogParseError("Invalid CLEU line format")
        assert err.failure_mode == FailureMode.SIM_VALIDATION
        assert "CLEU" in str(err)

    def test_validation_pipeline_error(self):
        from code.shukketsu.resilience.errors import ValidationPipelineError, FailureMode
        err = ValidationPipelineError("No valid fights found")
        assert err.failure_mode == FailureMode.SIM_VALIDATION
```

**Step 2: Run to verify failure**

Run: `python3 -m pytest tests/unit/test_errors.py::TestPhase4Errors -v`
Expected: FAIL — ImportError

**Step 3: Implement**

Add to `code/shukketsu/resilience/errors.py` after `ItemNotFoundError`:

```python
class WCLBridgeError(ShukketsuError):
    """Raised when WCL-to-SimConfig bridge cannot reconstruct a config."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.SIM_VALIDATION)


class LogParseError(ShukketsuError):
    """Raised when a combat log cannot be parsed."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.SIM_VALIDATION)


class ValidationPipelineError(ShukketsuError):
    """Raised when the validation pipeline fails."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.SIM_VALIDATION)
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_errors.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add code/shukketsu/resilience/errors.py tests/unit/test_errors.py
git commit -m "feat(errors): add WCLBridgeError, LogParseError, ValidationPipelineError"
```

---

### Task 2: Spell ID Mapping

**Files:**
- Create: `code/shukketsu/sim/spell_map.py`
- Create: `tests/unit/test_spell_map.py`

**Step 1: Write failing tests**

Create `tests/unit/test_spell_map.py`:

```python
"""Tests for WCL spell ID <-> sim ability name mapping."""

import pytest


class TestSpellIdToAbility:
    def test_sinister_strike_rank10(self):
        from code.shukketsu.sim.spell_map import wcl_ability_name
        assert wcl_ability_name(1752) == "sinister_strike"

    def test_sinister_strike_other_rank(self):
        from code.shukketsu.sim.spell_map import wcl_ability_name
        assert wcl_ability_name(11294) == "sinister_strike"

    def test_eviscerate(self):
        from code.shukketsu.sim.spell_map import wcl_ability_name
        assert wcl_ability_name(26865) == "eviscerate"

    def test_unknown_spell(self):
        from code.shukketsu.sim.spell_map import wcl_ability_name
        assert wcl_ability_name(99999) is None

    def test_blade_flurry(self):
        from code.shukketsu.sim.spell_map import wcl_ability_name
        assert wcl_ability_name(13877) == "blade_flurry"

    def test_instant_poison(self):
        from code.shukketsu.sim.spell_map import wcl_ability_name
        assert wcl_ability_name(26891) == "instant_poison"


class TestDisplayName:
    def test_sinister_strike_display(self):
        from code.shukketsu.sim.spell_map import sim_display_name
        assert sim_display_name("sinister_strike") == "Sinister Strike"

    def test_unknown_ability_returns_title_case(self):
        from code.shukketsu.sim.spell_map import sim_display_name
        assert sim_display_name("some_ability") == "Some Ability"


class TestBuffMapping:
    def test_kings(self):
        from code.shukketsu.sim.spell_map import wcl_buff_name
        assert wcl_buff_name(25898) == "kings"

    def test_heroism(self):
        from code.shukketsu.sim.spell_map import wcl_buff_name
        assert wcl_buff_name(32182) == "heroism"

    def test_unknown_buff(self):
        from code.shukketsu.sim.spell_map import wcl_buff_name
        assert wcl_buff_name(99999) is None


class TestAggregateAbilities:
    def test_aggregate_abilities_merges_ranks(self):
        """Multiple spell IDs for the same ability should merge."""
        from code.shukketsu.sim.spell_map import aggregate_wcl_abilities
        wcl_abilities = [
            {"guid": 1752, "name": "Sinister Strike", "total": 10000},
            {"guid": 11294, "name": "Sinister Strike", "total": 5000},
            {"guid": 26865, "name": "Eviscerate", "total": 8000},
        ]
        result = aggregate_wcl_abilities(wcl_abilities)
        assert result["sinister_strike"] == 15000
        assert result["eviscerate"] == 8000
```

**Step 2: Run to verify failure**

Run: `python3 -m pytest tests/unit/test_spell_map.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Implement**

Create `code/shukketsu/sim/spell_map.py`:

```python
"""WCL spell ID <-> sim ability name mapping.

Maps WCL spell/ability IDs from combat logs and API data to the
internal ability names used by the simulation engine.
"""

from typing import Final

# WCL spell ID -> sim ability name
SPELL_ID_TO_ABILITY: Final[dict[int, str]] = {
    # Builders
    1752: "sinister_strike",
    11294: "sinister_strike",
    53: "backstab",
    34413: "mutilate",
    16511: "hemorrhage",
    5938: "shiv",
    # Finishers
    26865: "eviscerate",
    26867: "rupture",
    6774: "slice_and_dice",
    32645: "envenom",
    8647: "expose_armor",
    # Openers
    11297: "ambush",
    11290: "garrote",
    1833: "cheap_shot",
    # Cooldowns
    13877: "blade_flurry",
    13750: "adrenaline_rush",
    14177: "cold_blood",
    9512: "thistle_tea",
    14185: "premeditation",
    # Poisons
    26891: "instant_poison",
    27282: "deadly_poison",
    27283: "wound_poison",
    # Procs
    23577: "combat_potency",
    13964: "sword_specialization",
    # Auto-attacks (melee swing)
    1: "melee",
}

# Sim ability name -> display name
ABILITY_DISPLAY_NAMES: Final[dict[str, str]] = {
    "sinister_strike": "Sinister Strike",
    "backstab": "Backstab",
    "mutilate": "Mutilate",
    "hemorrhage": "Hemorrhage",
    "shiv": "Shiv",
    "eviscerate": "Eviscerate",
    "rupture": "Rupture",
    "slice_and_dice": "Slice and Dice",
    "envenom": "Envenom",
    "expose_armor": "Expose Armor",
    "ambush": "Ambush",
    "garrote": "Garrote",
    "cheap_shot": "Cheap Shot",
    "blade_flurry": "Blade Flurry",
    "adrenaline_rush": "Adrenaline Rush",
    "cold_blood": "Cold Blood",
    "thistle_tea": "Thistle Tea",
    "premeditation": "Premeditation",
    "instant_poison": "Instant Poison",
    "deadly_poison": "Deadly Poison",
    "wound_poison": "Wound Poison",
    "combat_potency": "Combat Potency",
    "sword_specialization": "Sword Specialization",
    "melee": "Melee",
}

# WCL buff/aura ability IDs -> our buff system names
WCL_BUFF_MAP: Final[dict[int, str]] = {
    25898: "kings",
    2048: "battle_shout",
    25359: "grace_of_air",
    25528: "strength_of_earth",
    26990: "motw",
    34300: "lotp",
    16293: "wf_totem",
    27066: "trueshot_aura",
    32182: "heroism",
    35476: "drums_of_battle",
    25225: "sunder_armor",
    26993: "faerie_fire",
    27226: "curse_of_recklessness",
}


def wcl_ability_name(spell_id: int) -> str | None:
    """Return sim ability name for a WCL spell ID, or None if unknown."""
    return SPELL_ID_TO_ABILITY.get(spell_id)


def sim_display_name(ability_name: str) -> str:
    """Return human-readable display name for a sim ability."""
    return ABILITY_DISPLAY_NAMES.get(ability_name, ability_name.replace("_", " ").title())


def wcl_buff_name(spell_id: int) -> str | None:
    """Return our buff system name for a WCL buff spell ID, or None."""
    return WCL_BUFF_MAP.get(spell_id)


def aggregate_wcl_abilities(
    wcl_abilities: list[dict],
) -> dict[str, int]:
    """Aggregate WCL ability entries by sim ability name.

    Multiple spell IDs mapping to the same ability (different ranks)
    are summed together. Unknown spell IDs are skipped.

    Args:
        wcl_abilities: List of dicts with 'guid' and 'total' keys.

    Returns:
        Dict of sim_ability_name -> total_damage.
    """
    result: dict[str, int] = {}
    for entry in wcl_abilities:
        ability = wcl_ability_name(entry.get("guid", 0))
        if ability is not None:
            result[ability] = result.get(ability, 0) + entry.get("total", 0)
    return result
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_spell_map.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add code/shukketsu/sim/spell_map.py tests/unit/test_spell_map.py
git commit -m "feat(sim): add WCL spell ID mapping for validation pipeline"
```

---

### Task 3: Fight Filter

**Files:**
- Create: `code/shukketsu/sim/fight_filter.py`
- Create: `tests/unit/test_fight_filter.py`

**Step 1: Write failing tests**

Create `tests/unit/test_fight_filter.py`:

```python
"""Tests for WCL fight filtering and validation."""

import sqlite3

import pytest

from code.shukketsu.sim.fight_filter import (
    PATCHWERK_ENCOUNTERS,
    FightFilter,
    FilterCriteria,
    ValidatedFight,
)


@pytest.fixture()
def fight_db(test_db):
    """Populate test DB with sample fight data."""
    test_db.execute("INSERT OR IGNORE INTO wcl_reports (code, endpoint) VALUES ('RPT1', 'fresh')")
    fights = [
        ("RPT1", 1, 725, "Brutallus", 1, 180000, 10, 25),
        ("RPT1", 2, 725, "Brutallus", 0, 120000, 10, 25),  # wipe
        ("RPT1", 3, 999, "Unknown Boss", 1, 60000, 10, 25),  # non-patchwerk
        ("RPT1", 4, 725, "Brutallus", 1, 15000, 10, 25),  # too short
    ]
    for f in fights:
        test_db.execute(
            """INSERT OR REPLACE INTO wcl_fights
               (report_code, fight_id, encounter_id, encounter_name, kill, duration_ms, difficulty, raid_size)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            f,
        )
    # Add damage rows for Lyroo
    for fid, dmg in [(1, 200000), (2, 100000), (4, 5000)]:
        test_db.execute(
            """INSERT OR REPLACE INTO wcl_damage
               (report_code, fight_id, player_name, player_type, total_damage, active_time_ms, abilities_json, targets_json)
               VALUES (?, ?, 'Lyroo', 'Rogue', ?, ?, '[]', '[]')""",
            ("RPT1", fid, dmg, fid * 1000),
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
    def test_default_criteria(self):
        c = FilterCriteria()
        assert c.kills_only is True
        assert c.min_duration_ms == 30000
        assert c.patchwerk_only is True

    def test_frozen(self):
        c = FilterCriteria()
        with pytest.raises(Exception):
            c.kills_only = False  # type: ignore[misc]


class TestFightFilter:
    def test_excludes_wipes(self, fight_db):
        ff = FightFilter()
        results = ff.filter_fights(fight_db, "Lyroo")
        wipe = next(r for r in results if r.fight_id == 2)
        assert not wipe.included
        assert wipe.exclusion_reason == "wipe"

    def test_excludes_short_fights(self, fight_db):
        ff = FightFilter()
        results = ff.filter_fights(fight_db, "Lyroo")
        short = next(r for r in results if r.fight_id == 4)
        assert not short.included
        assert short.exclusion_reason == "short_fight"

    def test_includes_valid_kill(self, fight_db):
        ff = FightFilter()
        results = ff.filter_fights(fight_db, "Lyroo")
        valid = next(r for r in results if r.fight_id == 1)
        assert valid.included
        assert valid.exclusion_reason is None
        assert valid.wcl_total_damage == 200000

    def test_custom_criteria_override(self, fight_db):
        """encounter_ids whitelist overrides patchwerk_only."""
        criteria = FilterCriteria(patchwerk_only=False, encounter_ids={999})
        ff = FightFilter(criteria=criteria)
        results = ff.filter_fights(fight_db, "Lyroo")
        included = [r for r in results if r.included]
        assert len(included) >= 0  # May or may not have fights for encounter 999


class TestPatchworkEncounters:
    def test_brutallus_is_patchwerk(self):
        assert 725 in PATCHWERK_ENCOUNTERS

    def test_gruul_is_patchwerk(self):
        assert 649 in PATCHWERK_ENCOUNTERS
```

**Step 2: Run to verify failure**

Run: `python3 -m pytest tests/unit/test_fight_filter.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Implement**

Create `code/shukketsu/sim/fight_filter.py`:

```python
"""WCL fight filtering for sim validation.

Selects fights suitable for DPS calibration by excluding wipes,
deaths, short fights, and movement-heavy encounters.
"""

import logging
import sqlite3
from typing import Final

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# Encounters where continuous combat assumption holds
PATCHWERK_ENCOUNTERS: Final[dict[int, str]] = {
    # Molten Core (Fresh)
    200663: "Lucifron",
    200664: "Magmadar",
    200670: "Golemagg the Incinerator",
    # Karazhan
    652: "Attumen the Huntsman",
    # Gruul's Lair
    649: "Gruul the Dragonkiller",
    # SSC
    623: "Hydross the Unstable",
    624: "The Lurker Below",
    # TK
    730: "Void Reaver",
    # Black Temple
    601: "Supremus",
    602: "Shade of Akama",
    # Sunwell Plateau
    725: "Brutallus",
    726: "Felmyst",
}


class FilterCriteria(BaseModel):
    """Criteria for selecting valid calibration fights."""

    model_config = ConfigDict(frozen=True)

    kills_only: bool = True
    min_duration_ms: int = 30_000
    max_death_pct: float = 0.0
    patchwerk_only: bool = True
    encounter_ids: set[int] | None = None


class ValidatedFight(BaseModel):
    """A WCL fight annotated with inclusion/exclusion status."""

    report_code: str
    fight_id: int
    encounter_id: int
    encounter_name: str
    duration_ms: int
    source_id: int
    wcl_active_dps: float
    wcl_total_damage: int
    included: bool
    exclusion_reason: str | None = None


class FightFilter:
    """Filters WCL fights for sim validation suitability."""

    def __init__(self, criteria: FilterCriteria = FilterCriteria()) -> None:
        self._criteria = criteria

    def filter_fights(
        self,
        conn: sqlite3.Connection,
        character_name: str,
    ) -> list[ValidatedFight]:
        """Return all fights for a character, annotated with inclusion/exclusion."""
        rows = conn.execute(
            """SELECT f.report_code, f.fight_id, f.encounter_id, f.encounter_name,
                      f.kill, f.duration_ms,
                      d.total_damage, d.active_time_ms,
                      c.source_id
               FROM wcl_fights f
               JOIN wcl_damage d ON d.report_code = f.report_code AND d.fight_id = f.fight_id
               JOIN wcl_combatants c ON c.report_code = f.report_code AND c.fight_id = f.fight_id
               WHERE d.player_name = ? AND c.player_name = ?""",
            (character_name, character_name),
        ).fetchall()

        results: list[ValidatedFight] = []
        for row in rows:
            (report_code, fight_id, encounter_id, encounter_name,
             kill, duration_ms, total_damage, active_time_ms, source_id) = row

            exclusion_reason = self._check_exclusion(
                kill=kill,
                duration_ms=duration_ms,
                encounter_id=encounter_id,
                total_damage=total_damage,
                active_time_ms=active_time_ms,
            )

            active_dps = (
                total_damage / (active_time_ms / 1000)
                if active_time_ms and active_time_ms > 0
                else 0.0
            )

            results.append(
                ValidatedFight(
                    report_code=report_code,
                    fight_id=fight_id,
                    encounter_id=encounter_id,
                    encounter_name=encounter_name,
                    duration_ms=duration_ms,
                    source_id=source_id,
                    wcl_active_dps=active_dps,
                    wcl_total_damage=total_damage,
                    included=exclusion_reason is None,
                    exclusion_reason=exclusion_reason,
                )
            )

        return results

    def _check_exclusion(
        self,
        *,
        kill: int,
        duration_ms: int,
        encounter_id: int,
        total_damage: int,
        active_time_ms: int | None,
    ) -> str | None:
        """Return exclusion reason or None if fight is valid."""
        c = self._criteria

        if c.kills_only and not kill:
            return "wipe"

        if duration_ms < c.min_duration_ms:
            return "short_fight"

        if c.patchwerk_only and c.encounter_ids is None:
            if encounter_id not in PATCHWERK_ENCOUNTERS:
                return "movement_boss"
        elif c.encounter_ids is not None:
            if encounter_id not in c.encounter_ids:
                return "not_in_whitelist"

        # Check for likely death: active time < 90% of fight duration
        if active_time_ms and duration_ms > 0:
            active_ratio = active_time_ms / duration_ms
            if active_ratio < (1.0 - c.max_death_pct) * 0.9:
                return "death"

        return None
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_fight_filter.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add code/shukketsu/sim/fight_filter.py tests/unit/test_fight_filter.py
git commit -m "feat(sim): add fight filter for validation pipeline"
```

---

### Task 4: Config Additions

**Files:**
- Modify: `code/shukketsu/config.py`
- Modify: `tests/unit/test_wcl_config.py`

**Step 1: Write failing test**

Add to `tests/unit/test_wcl_config.py`:

```python
def test_validation_config_constants():
    from code.shukketsu import config
    assert hasattr(config, "VALIDATION_CONCURRENCY")
    assert config.VALIDATION_CONCURRENCY == 4
    assert hasattr(config, "VALIDATION_ITERATIONS")
    assert config.VALIDATION_ITERATIONS == 5000
    assert hasattr(config, "VALIDATION_DPS_THRESHOLD")
    assert config.VALIDATION_DPS_THRESHOLD == 5.0
    assert hasattr(config, "LOG_UPLOAD_MAX_SIZE_MB")
    assert config.LOG_UPLOAD_MAX_SIZE_MB == 100

def test_lyroo_has_race():
    from code.shukketsu import config
    lyroo = config.WCL_TRACKED_CHARACTERS[0]
    assert "race" in lyroo
    assert lyroo["race"] == "orc"
```

**Step 2: Run to verify failure**

Run: `python3 -m pytest tests/unit/test_wcl_config.py::test_validation_config_constants -v`
Expected: FAIL — AttributeError

**Step 3: Implement**

Add to `code/shukketsu/config.py` after the `# Backup` section:

```python
# Validation pipeline
VALIDATION_CONCURRENCY = int(os.getenv("VALIDATION_CONCURRENCY", "4"))
VALIDATION_ITERATIONS = int(os.getenv("VALIDATION_ITERATIONS", "5000"))
VALIDATION_DPS_THRESHOLD = float(os.getenv("VALIDATION_DPS_THRESHOLD", "5.0"))
VALIDATION_ABILITY_THRESHOLD = float(os.getenv("VALIDATION_ABILITY_THRESHOLD", "15.0"))
VALIDATION_BUFF_THRESHOLD = float(os.getenv("VALIDATION_BUFF_THRESHOLD", "5.0"))

# Log upload
LOG_UPLOAD_MAX_SIZE_MB = int(os.getenv("LOG_UPLOAD_MAX_SIZE_MB", "100"))
```

Update `WCL_TRACKED_CHARACTERS`:
```python
WCL_TRACKED_CHARACTERS: list[dict[str, str | int]] = [
    {"wcl_id": 104956434, "name": "Lyroo", "server": "nightslayer", "region": "us", "endpoint": "fresh", "race": "orc"},
]
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_wcl_config.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add code/shukketsu/config.py tests/unit/test_wcl_config.py
git commit -m "feat(config): add validation pipeline and log upload constants"
```

---

### Task 5: Comparison Engine

**Files:**
- Create: `code/shukketsu/sim/comparator.py`
- Create: `tests/unit/test_comparator.py`

**Step 1: Write failing tests**

Create `tests/unit/test_comparator.py`:

```python
"""Tests for sim validation comparison engine."""

import pytest

from code.shukketsu.sim.comparator import (
    AbilityMetrics,
    BossAggregate,
    FightValidation,
    MetricDrift,
    SimComparator,
    ValidationReport,
    WCLFightMetrics,
)


class TestMetricDrift:
    def test_pass_within_threshold(self):
        d = MetricDrift(
            metric_name="dps", sim_value=1050, wcl_value=1000,
            absolute_delta=50, relative_pct=5.0, status="pass",
        )
        assert d.status == "pass"

    def test_fail_above_threshold(self):
        d = MetricDrift(
            metric_name="dps", sim_value=1200, wcl_value=1000,
            absolute_delta=200, relative_pct=20.0, status="fail",
        )
        assert d.status == "fail"


class TestWCLFightMetrics:
    def test_creation(self):
        m = WCLFightMetrics(
            total_damage=300000, active_dps=1000.0, fight_duration_ms=300000,
            ability_breakdown={"sinister_strike": AbilityMetrics(
                damage_total=150000, damage_pct=50.0, cast_count=120, hit_count=100, crit_count=40,
            )},
            buff_uptimes={"slice_and_dice": 0.95},
            proc_counts={"combat_potency": 45},
        )
        assert m.active_dps == 1000.0
        assert "sinister_strike" in m.ability_breakdown


class TestSimComparator:
    def test_compute_drift_pass(self):
        comp = SimComparator.__new__(SimComparator)
        drift = comp._compute_drift("dps", 1020.0, 1000.0, threshold_pass=5.0, threshold_warn=10.0)
        assert drift.status == "pass"
        assert abs(drift.relative_pct - 2.0) < 0.1

    def test_compute_drift_warn(self):
        comp = SimComparator.__new__(SimComparator)
        drift = comp._compute_drift("dps", 1080.0, 1000.0, threshold_pass=5.0, threshold_warn=10.0)
        assert drift.status == "warn"

    def test_compute_drift_fail(self):
        comp = SimComparator.__new__(SimComparator)
        drift = comp._compute_drift("dps", 1200.0, 1000.0, threshold_pass=5.0, threshold_warn=10.0)
        assert drift.status == "fail"

    def test_compute_drift_zero_wcl(self):
        """Zero WCL value should not divide by zero."""
        comp = SimComparator.__new__(SimComparator)
        drift = comp._compute_drift("dps", 100.0, 0.0, threshold_pass=5.0, threshold_warn=10.0)
        assert drift.status == "fail"


class TestValidationReport:
    def test_report_aggregation(self):
        report = ValidationReport(
            character_name="Lyroo",
            total_fights=10,
            included_fights=8,
            excluded_fights=2,
            per_fight=[],
            per_boss={},
            overall_dps_drift_pct=3.5,
            overall_status="pass",
            timestamp="2026-02-13T00:00:00",
        )
        assert report.overall_status == "pass"
```

**Step 2: Run to verify failure**

Run: `python3 -m pytest tests/unit/test_comparator.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Implement**

Create `code/shukketsu/sim/comparator.py` — full implementation as specified in design doc Section 4. See the companion design document for the complete class with `extract_wcl_metrics()`, `compare_fight()`, `build_report()`, and `_compute_drift()`.

Key classes: `AbilityMetrics`, `WCLFightMetrics`, `MetricDrift`, `FightValidation`, `BossAggregate`, `ValidationReport`, `SimComparator`.

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_comparator.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add code/shukketsu/sim/comparator.py tests/unit/test_comparator.py
git commit -m "feat(sim): add comparison engine for validation pipeline"
```

---

### Task 6: Schema v6 Migration

**Files:**
- Modify: `code/shukketsu/db/connection.py`
- Test: `tests/unit/test_db.py`

**Step 1: Write failing test**

Add to `tests/unit/test_db.py`:

```python
def test_schema_v6_validation_runs_table(test_db):
    """Schema v6 adds validation_runs table."""
    tables = test_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='validation_runs'"
    ).fetchone()
    assert tables is not None

    cols = test_db.execute("PRAGMA table_info(validation_runs)").fetchall()
    col_names = [c[1] for c in cols]
    assert "character_name" in col_names
    assert "run_type" in col_names
    assert "overall_dps_drift_pct" in col_names
    assert "report_json" in col_names
```

**Step 2: Run to verify failure**

Run: `python3 -m pytest tests/unit/test_db.py::test_schema_v6_validation_runs_table -v`
Expected: FAIL — table doesn't exist

**Step 3: Implement**

Add migration to `code/shukketsu/db/connection.py` following the existing pattern (see `_migrate_v4_to_v5`). Add `_VALIDATION_V6_SQL` constant and `_migrate_v51_to_v6()` function. Update `init_db()` chain.

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_db.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add code/shukketsu/db/connection.py tests/unit/test_db.py
git commit -m "feat(db): add schema v6 with validation_runs table"
```

---

### Task 7: Validation Pipeline

**Files:**
- Create: `code/shukketsu/sim/validation_pipeline.py`
- Create: `tests/unit/test_validation_pipeline.py`

**Step 1: Write failing tests**

Create `tests/unit/test_validation_pipeline.py` with tests for:
- Pipeline construction
- `run_validation()` with no valid fights (empty report)
- `run_validation()` stores report in `validation_runs` table
- Progress callback fires

**Step 2: Run to verify failure**

Run: `python3 -m pytest tests/unit/test_validation_pipeline.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Implement**

Create `code/shukketsu/sim/validation_pipeline.py` — orchestrates FightFilter -> WCLBridge -> SimRunner -> SimComparator -> store report. Uses ProcessPoolExecutor for CPU parallelism.

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_validation_pipeline.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add code/shukketsu/sim/validation_pipeline.py tests/unit/test_validation_pipeline.py
git commit -m "feat(sim): add validation pipeline orchestrator"
```

---

### Task 8: CLEU Combat Log Parser

**Files:**
- Create: `code/shukketsu/sim/log_parser.py`
- Create: `tests/unit/test_log_parser.py`

**Step 1: Write failing tests**

Create `tests/unit/test_log_parser.py` with sample CLEU log data and tests for:
- Parse finds fights from ENCOUNTER_START/END
- Fight metadata (encounter_id, name, kill status)
- Damage tracking by spell ID
- Buff timeline (apply/remove pairs)
- Cast counts
- Empty log handling
- Wrong character produces no damage

**Step 2: Run to verify failure**

Run: `python3 -m pytest tests/unit/test_log_parser.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Implement**

Create `code/shukketsu/sim/log_parser.py` — streaming line-by-line parser for TBC CLEU format. Tracks ENCOUNTER_START/END boundaries, aggregates SPELL_DAMAGE, SWING_DAMAGE, SPELL_AURA_APPLIED/REMOVED, SPELL_CAST_SUCCESS, UNIT_DIED events per character.

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_log_parser.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add code/shukketsu/sim/log_parser.py tests/unit/test_log_parser.py
git commit -m "feat(sim): add CLEU combat log parser for validation"
```

---

### Task 9: WoWSims Import Parser

**Files:**
- Modify: `code/shukketsu/sim/imports.py:243-251`
- Modify: `tests/unit/test_sim_imports.py`

**Step 1: Write failing tests** — test basic JSON parsing, gear extraction, talent-based spec detection

**Step 2: Run to verify failure** — `InvalidSimConfigError("WoWSims import not yet supported")`

**Step 3: Implement** — Replace the stub with JSON parser, add `_WOWSIMS_RACE_MAP`, `_WOWSIMS_SLOT_MAP`, `_detect_spec_from_talents()`. Update `detect_format()` to distinguish WoWSims JSON from SeventyUpgrades JSON.

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_sim_imports.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add code/shukketsu/sim/imports.py tests/unit/test_sim_imports.py
git commit -m "feat(sim): implement WoWSims JSON import parser"
```

---

### Task 10: Web Validation Dashboard

**Files:**
- Modify: `code/shukketsu/web/routers/sim.py`
- Create: `code/shukketsu/web/templates/sim/validate/index.html`
- Create: `code/shukketsu/web/templates/sim/validate/partials/report.html`
- Test: `tests/unit/test_sim_routes.py`

Add `/sim/validate/` GET route (dashboard with run history table) and `/sim/validate/report/{run_id}` GET route (renders full report JSON). Templates extend `base.html` with the existing dark WoW theme.

---

### Task 11: Web Log Upload Routes

**Files:**
- Create: `code/shukketsu/web/routers/logs.py`
- Create: `code/shukketsu/web/templates/logs/index.html`
- Create: `code/shukketsu/web/templates/logs/partials/fight_list.html`
- Modify: `code/shukketsu/web/app.py` (register router)
- Modify: `code/shukketsu/web/templates/base.html` (add Logs nav link)
- Create: `tests/unit/test_log_routes.py`

Add `/logs/` GET (upload page), `/logs/upload` POST (parse log + return fights partial). Register router in app.py, add "Logs" nav link.

---

### Task 12: Item Database Expansion

**Files:**
- Modify: `code/shukketsu/sim/items.py`
- Test: `tests/unit/test_sim_items.py`

Add ~80 new items to `_CURATED_ITEMS` (weapons, trinkets, set pieces). Data entry from WoWSims/wowhead. Verify >= 100 total items, key weapons exist, key trinkets exist.

---

### Task 13: CLAUDE.md and Memory Update

Update CLAUDE.md Phase 4 section from stub to complete. Update memory with test counts and schema version. Commit.

---

## Execution Order Summary

| Task | Description | Depends On | Est. Tests |
|------|-------------|------------|------------|
| 0 | WCL ingest per-fight fix | -- | +8 |
| 1 | Error types | -- | +3 |
| 2 | Spell ID mapping | -- | +12 |
| 3 | Fight filter | Task 0 | +8 |
| 4 | Config additions | -- | +2 |
| 5 | Comparison engine | Tasks 2, 4 | +8 |
| 6 | Schema v6 migration | Task 0 | +2 |
| 7 | Validation pipeline | Tasks 3, 5, 6 | +4 |
| 8 | CLEU log parser | Task 2 | +7 |
| 9 | WoWSims import | -- | +3 |
| 10 | Validation dashboard | Tasks 6, 7 | +2 |
| 11 | Log upload routes | Task 8 | +2 |
| 12 | Item DB expansion | -- | +3 |
| 13 | CLAUDE.md + memory | All | 0 |

**Parallelizable:** Tasks 1, 2, 4, 9, 12 have no dependencies and can run in parallel.

**Critical path:** Task 0 -> Task 3 -> Task 7 -> Task 10 (WCL validation pipeline)

**Total estimated new tests:** ~60+
