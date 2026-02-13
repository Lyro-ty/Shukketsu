# Phase 4: Sim Validation & Calibration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate the TBC Rogue simulation engine against real WCL combat data, calibrate until DPS drift ≤ 5%, and fill remaining gaps (CLEU parser, WoWSims import, item DB expansion).

**Architecture:** WCL ingest fixes → spell mapping → item DB expansion → fight filtering → WCL-to-SimConfig bridge (synthetic Items) → comparison engine → schema v7 → validation pipeline (ProcessPoolExecutor) → CLEU parser → WoWSims import → web dashboards → calibration. Design doc: `docs/plans/2026-02-13-phase4-validation-design.md`.

**Tech Stack:** Python 3.12, SQLite (schema v5 → v7), Pydantic v2, FastAPI/HTMX, concurrent.futures.ProcessPoolExecutor, existing sim engine + WCL API subsystem.

---

## Context for Implementers

**Test invocation:** Always use `python3 -m pytest` (never bare `pytest`).

**Import pattern:** `from code.shukketsu.X import Y` — project root is CWD.

**Error pattern:** All exceptions inherit `ShukketsuError` with a `FailureMode` enum member.

**Pydantic pattern:** v2 `BaseModel`, `model_config = ConfigDict(frozen=True)` for immutable models.

**Key existing files you'll reference:**
- `code/shukketsu/apis/wcl/ingest.py` — `ReportDiver` class (line 102), `_fetch_damage` (line 244), `_fetch_buffs` (line 269), `_fetch_casts` (line 293)
- `code/shukketsu/apis/wcl/schema.sql` — WCL table definitions (UNIQUE constraints matter)
- `code/shukketsu/apis/wcl/queries.py` — `build_damage_table_query(code, fight_ids)` etc.
- `code/shukketsu/resilience/errors.py` — Error taxonomy, `FailureMode.SIM_VALIDATION` already exists
- `code/shukketsu/db/connection.py` — Migration chain (versions 1-5), `init_db()` at line 73
- `code/shukketsu/sim/models.py` — `SimConfig` (line 244), `SimResult` (line 341), `Item` (line 188), `Race` (line 54), `RogueSpec` (line 23), `GearSlot` (line 32)
- `code/shukketsu/sim/items.py` — `ItemDatabase` class (line 662), `_CURATED_ITEMS` (line 25)
- `code/shukketsu/sim/combat.py` — `_resolve_weapons()` (line 1209), `_compute_base_stats()` (line 1238)
- `code/shukketsu/sim/buffs.py` — `resolve_buffs()`, `ResolvedBuffs` model, `RAID_BUFFS` registry
- `code/shukketsu/sim/imports.py` — `CharacterImport` (line 42), `build_config()` (line 271), `parse_wowsims()` stub (line 243)
- `code/shukketsu/sim/runner.py` — `SimRunner` (line 131), `sim_run()` (line 146)
- `code/shukketsu/sim/validation.py` — Existing static validation profiles (separate from WCL validation)
- `code/shukketsu/config.py` — `WCL_TRACKED_CHARACTERS` (line 161), env var pattern
- `tests/conftest.py` — `test_db(tmp_path)` fixture, `_reset_breakers()` autouse
- `tests/unit/apis/wcl/test_wcl_ingest.py` — `db(tmp_path)` fixture, `mock_client()` fixture, mock response factories

---

### Task 0: Fix WCL Ingest — Per-Fight Data Storage + Actor Mapping

**PREREQUISITE** — All downstream tasks depend on correct per-fight WCL data.

**Problem:** `_fetch_damage`, `_fetch_buffs`, `_fetch_casts` all store every row under `fight_ids[0]` (lines 259, 284, 308 of `ingest.py`). The GraphQL query aggregates across all fights, so per-fight data is lost. Also, `wcl_combatants` has no `player_name` column — can't join combatant gear to damage rows.

**Files:**
- Modify: `code/shukketsu/apis/wcl/ingest.py`
- Modify: `code/shukketsu/db/connection.py`
- Test: `tests/unit/apis/wcl/test_wcl_ingest.py`

**Step 1: Write failing tests for per-fight damage storage**

Add to `tests/unit/apis/wcl/test_wcl_ingest.py`:

```python
class TestPerFightStorage:
    """Tests for per-fight data storage (not aggregated)."""

    async def test_damage_stored_per_fight(self, db: sqlite3.Connection, mock_client: WCLClient) -> None:
        """Each fight_id gets its own damage rows, not aggregated under fight_ids[0]."""
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
        diver = ReportDiver(mock_client, db)
        diver._store_report("ABC123", "fresh")
        for fid, dur in [(1, 60000), (2, 90000)]:
            db.execute(
                """INSERT OR REPLACE INTO wcl_fights
                   (report_code, fight_id, encounter_id, encounter_name, kill, duration_ms)
                   VALUES (?, ?, 100, 'Boss1', 1, ?)""",
                ("ABC123", fid, dur),
            )

        await diver._fetch_damage("ABC123", [1, 2], "fresh")

        rows = db.execute(
            "SELECT fight_id, total_damage FROM wcl_damage WHERE report_code = ? ORDER BY fight_id",
            ("ABC123",),
        ).fetchall()
        assert len(rows) == 2
        assert dict(rows[0]) == {"fight_id": 1, "total_damage": 50000}
        assert dict(rows[1]) == {"fight_id": 2, "total_damage": 80000}

    async def test_buffs_stored_per_fight(self, db: sqlite3.Connection, mock_client: WCLClient) -> None:
        """Each fight_id gets its own buff rows."""
        mock_client.query = AsyncMock(side_effect=[
            {"reportData": {"report": {"table": {"data": {"auras": [
                {"name": "Slice and Dice", "guid": 6774, "totalUptime": 55000, "totalUses": 3, "bands": []}
            ]}}}}},
            {"reportData": {"report": {"table": {"data": {"auras": [
                {"name": "Slice and Dice", "guid": 6774, "totalUptime": 80000, "totalUses": 5, "bands": []}
            ]}}}}},
        ])
        diver = ReportDiver(mock_client, db)
        diver._store_report("ABC123", "fresh")
        for fid, dur in [(1, 60000), (2, 90000)]:
            db.execute(
                """INSERT OR REPLACE INTO wcl_fights
                   (report_code, fight_id, encounter_id, encounter_name, kill, duration_ms)
                   VALUES (?, ?, 100, 'Boss1', 1, ?)""",
                ("ABC123", fid, dur),
            )

        await diver._fetch_buffs("ABC123", [1, 2], "fresh")

        rows = db.execute(
            "SELECT fight_id, total_uptime_ms FROM wcl_buffs WHERE report_code = ? ORDER BY fight_id",
            ("ABC123",),
        ).fetchall()
        assert len(rows) == 2
        assert dict(rows[0]) == {"fight_id": 1, "total_uptime_ms": 55000}
        assert dict(rows[1]) == {"fight_id": 2, "total_uptime_ms": 80000}

    async def test_casts_stored_per_fight(self, db: sqlite3.Connection, mock_client: WCLClient) -> None:
        """Each fight_id gets its own cast rows."""
        mock_client.query = AsyncMock(side_effect=[
            {"reportData": {"report": {"table": {"data": {"entries": [
                {"name": "Lyroo", "abilities": [{"name": "Sinister Strike", "total": 40}]}
            ]}}}}},
            {"reportData": {"report": {"table": {"data": {"entries": [
                {"name": "Lyroo", "abilities": [{"name": "Sinister Strike", "total": 60}]}
            ]}}}}},
        ])
        diver = ReportDiver(mock_client, db)
        diver._store_report("ABC123", "fresh")
        for fid, dur in [(1, 60000), (2, 90000)]:
            db.execute(
                """INSERT OR REPLACE INTO wcl_fights
                   (report_code, fight_id, encounter_id, encounter_name, kill, duration_ms)
                   VALUES (?, ?, 100, 'Boss1', 1, ?)""",
                ("ABC123", fid, dur),
            )

        await diver._fetch_casts("ABC123", [1, 2], "fresh")

        rows = db.execute(
            "SELECT fight_id, cast_count FROM wcl_casts WHERE report_code = ? AND player_name = ? ORDER BY fight_id",
            ("ABC123", "Lyroo"),
        ).fetchall()
        assert len(rows) == 2
        assert dict(rows[0]) == {"fight_id": 1, "cast_count": 40}
        assert dict(rows[1]) == {"fight_id": 2, "cast_count": 60}

    async def test_actor_map_stored_from_fights(self, db: sqlite3.Connection, mock_client: WCLClient) -> None:
        """_fetch_fights populates _actor_map from masterData.actors."""
        mock_client.query = AsyncMock(return_value={
            "reportData": {"report": {
                "fights": [
                    {"id": 5, "encounterID": 652, "name": "Gruul", "kill": True, "duration": 200000},
                ],
                "masterData": {"actors": [
                    {"id": 1, "name": "Lyroo", "type": "Player"},
                    {"id": 2, "name": "Gruul", "type": "NPC"},
                ]},
            }},
        })
        diver = ReportDiver(mock_client, db)
        diver._store_report("ABC123", "fresh")

        await diver._fetch_fights("ABC123", "fresh")

        assert diver._actor_map == {1: "Lyroo", 2: "Gruul"}

    async def test_combatant_player_name_stored(self, db: sqlite3.Connection, mock_client: WCLClient) -> None:
        """wcl_combatants stores player_name from _actor_map."""
        mock_client.query = AsyncMock(return_value={
            "reportData": {"report": {
                "events": {"data": [
                    {"sourceID": 5, "specID": 260, "fight": 1,
                     "strength": 100, "agility": 500,
                     "gear": [], "talents": [], "auras": []}
                ]},
            }},
        })
        diver = ReportDiver(mock_client, db)
        diver._actor_map = {5: "Lyroo"}
        diver._store_report("ABC123", "fresh")
        db.execute(
            """INSERT OR REPLACE INTO wcl_fights
               (report_code, fight_id, encounter_id, encounter_name, kill, duration_ms)
               VALUES ('ABC123', 1, 100, 'Boss1', 1, 60000)""",
        )

        await diver._fetch_combatant_info("ABC123", [1], "fresh")

        row = db.execute(
            "SELECT player_name FROM wcl_combatants WHERE report_code = ? AND source_id = ?",
            ("ABC123", 5),
        ).fetchone()
        assert row is not None
        assert row["player_name"] == "Lyroo"
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/apis/wcl/test_wcl_ingest.py::TestPerFightStorage -v`
Expected: FAIL — per-fight tests fail because code stores under `fight_ids[0]`, actor_map tests fail because attribute doesn't exist.

**Step 3: Initialize `_actor_map` in `__init__`**

In `code/shukketsu/apis/wcl/ingest.py`, modify `ReportDiver.__init__` (line 105-107):

```python
def __init__(self, client: WCLClient, conn: sqlite3.Connection) -> None:
    self._client = client
    self._conn = conn
    self._actor_map: dict[int, str] = {}
```

**Step 4: Store actors in `_fetch_fights()`**

In `_fetch_fights()`, after the fights loop (line 178, before `return fight_ids`), add:

```python
        # Build actor map for player_name resolution in _fetch_combatant_info
        actors = report.get("masterData", {}).get("actors", [])
        self._actor_map = {a["id"]: a["name"] for a in actors if a.get("id") and a.get("name")}

        return fight_ids
```

**Step 5: Fix `_fetch_damage()` — per-fight loop**

Replace `_fetch_damage` (lines 244-267):

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

**Step 6: Fix `_fetch_buffs()` — per-fight loop**

Replace `_fetch_buffs` (lines 269-291):

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

**Step 7: Fix `_fetch_casts()` — per-fight loop**

Replace `_fetch_casts` (lines 293-313):

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

**Step 8: Add `player_name` to `_fetch_combatant_info()`**

In `_fetch_combatant_info()` (line 189), add `player_name` column. Replace the INSERT (lines 203-242):

```python
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
```

**Step 9: Add v6 migration (player_name column)**

In `code/shukketsu/db/connection.py`, add after `_migrate_v4_to_v5` (line 202):

```python
def _migrate_v5_to_v6(conn: sqlite3.Connection) -> None:
    """Migrate v5 to v6: add player_name to wcl_combatants."""
    try:
        conn.execute("ALTER TABLE wcl_combatants ADD COLUMN player_name TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists (idempotent)
    conn.execute("INSERT INTO schema_version (version) VALUES (6)")
    conn.commit()
    logger.info("Database migrated from v5 to v6 (wcl_combatants.player_name)")
```

Update `init_db()` — add after `if version < 5:` block (line 88):

```python
            if version < 6:
                _migrate_v5_to_v6(conn)
```

And after the base schema init (line 98):

```python
    if version < 6:
        _migrate_v5_to_v6(conn)
```

**Step 10: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/apis/wcl/test_wcl_ingest.py -v`
Expected: ALL PASS (existing + new)

**Step 11: Run full suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All 1461+ tests pass

**Step 12: Commit**

```bash
git add code/shukketsu/apis/wcl/ingest.py code/shukketsu/db/connection.py tests/unit/apis/wcl/test_wcl_ingest.py
git commit -m "fix(wcl): per-fight data storage and actor name mapping

BREAKING: WCL ingest now queries per-fight instead of aggregating
under fight_ids[0]. Adds player_name to wcl_combatants via schema
v6 migration. Required for sim validation pipeline."
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
    def test_wcl_bridge_error(self) -> None:
        from code.shukketsu.resilience.errors import WCLBridgeError, FailureMode
        err = WCLBridgeError("Missing weapon item 32837")
        assert err.failure_mode == FailureMode.SIM_VALIDATION
        assert "32837" in str(err)

    def test_log_parse_error(self) -> None:
        from code.shukketsu.resilience.errors import LogParseError, FailureMode
        err = LogParseError("Invalid CLEU line format")
        assert err.failure_mode == FailureMode.SIM_VALIDATION
        assert "CLEU" in str(err)

    def test_validation_pipeline_error(self) -> None:
        from code.shukketsu.resilience.errors import ValidationPipelineError, FailureMode
        err = ValidationPipelineError("No valid fights found")
        assert err.failure_mode == FailureMode.SIM_VALIDATION
```

**Step 2: Run to verify failure**

Run: `python3 -m pytest tests/unit/test_errors.py::TestPhase4Errors -v`
Expected: FAIL — ImportError

**Step 3: Implement**

Add to `code/shukketsu/resilience/errors.py` after `ItemNotFoundError` (line 196):

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
    def test_sinister_strike_rank10(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_ability_name
        assert wcl_ability_name(1752) == "sinister_strike"

    def test_sinister_strike_other_rank(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_ability_name
        assert wcl_ability_name(11294) == "sinister_strike"

    def test_eviscerate(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_ability_name
        assert wcl_ability_name(26865) == "eviscerate"

    def test_unknown_spell(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_ability_name
        assert wcl_ability_name(99999) is None

    def test_blade_flurry(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_ability_name
        assert wcl_ability_name(13877) == "blade_flurry"

    def test_instant_poison(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_ability_name
        assert wcl_ability_name(26891) == "instant_poison"


class TestDisplayName:
    def test_sinister_strike_display(self) -> None:
        from code.shukketsu.sim.spell_map import sim_display_name
        assert sim_display_name("sinister_strike") == "Sinister Strike"

    def test_unknown_ability_returns_title_case(self) -> None:
        from code.shukketsu.sim.spell_map import sim_display_name
        assert sim_display_name("some_ability") == "Some Ability"


class TestBuffMapping:
    def test_kings(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_buff_name
        assert wcl_buff_name(25898) == "kings"

    def test_heroism(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_buff_name
        assert wcl_buff_name(32182) == "heroism"

    def test_unknown_buff(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_buff_name
        assert wcl_buff_name(99999) is None


class TestAggregateAbilities:
    def test_aggregate_merges_ranks(self) -> None:
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

    def test_aggregate_logs_unmapped_spells(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unknown spell IDs are logged as warnings."""
        import logging
        from code.shukketsu.sim.spell_map import aggregate_wcl_abilities
        with caplog.at_level(logging.WARNING, logger="code.shukketsu.sim.spell_map"):
            result = aggregate_wcl_abilities([{"guid": 99999, "name": "Mystery", "total": 42000}])
        assert len(result) == 0
        assert "99999" in caplog.text
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

import logging
from typing import Final

logger = logging.getLogger(__name__)

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

# WCL buff/aura ability IDs -> our buff system names (keys match buffs.py RAID_BUFFS keys)
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


def aggregate_wcl_abilities(wcl_abilities: list[dict]) -> dict[str, int]:
    """Aggregate WCL ability entries by sim ability name.

    Multiple spell IDs mapping to the same ability (different ranks)
    are summed together. Unknown spell IDs are logged and skipped.

    Args:
        wcl_abilities: List of dicts with 'guid' and 'total' keys.

    Returns:
        Dict of sim_ability_name -> total_damage.
    """
    result: dict[str, int] = {}
    for entry in wcl_abilities:
        spell_id = entry.get("guid", 0)
        ability = wcl_ability_name(spell_id)
        if ability is not None:
            result[ability] = result.get(ability, 0) + entry.get("total", 0)
        else:
            total = entry.get("total", 0)
            name = entry.get("name", "unknown")
            logger.warning("Unmapped WCL spell ID %d (%s) with damage %d", spell_id, name, total)
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

### Task 3: Config Additions

**Files:**
- Modify: `code/shukketsu/config.py`
- Modify: `tests/unit/test_wcl_config.py`

**Step 1: Write failing tests**

Add to `tests/unit/test_wcl_config.py`:

```python
def test_validation_config_constants() -> None:
    from code.shukketsu import config
    assert hasattr(config, "VALIDATION_CONCURRENCY")
    assert config.VALIDATION_CONCURRENCY == 4
    assert hasattr(config, "VALIDATION_ITERATIONS")
    assert config.VALIDATION_ITERATIONS == 5000
    assert hasattr(config, "VALIDATION_DPS_THRESHOLD")
    assert config.VALIDATION_DPS_THRESHOLD == 5.0
    assert hasattr(config, "LOG_UPLOAD_MAX_SIZE_MB")
    assert config.LOG_UPLOAD_MAX_SIZE_MB == 100


def test_lyroo_has_race() -> None:
    from code.shukketsu import config
    lyroo = config.WCL_TRACKED_CHARACTERS[0]
    assert "race" in lyroo
    assert lyroo["race"] == "orc"
```

**Step 2: Run to verify failure**

Run: `python3 -m pytest tests/unit/test_wcl_config.py::test_validation_config_constants -v`
Expected: FAIL — AttributeError

**Step 3: Implement**

Add to `code/shukketsu/config.py` after the `# Backup` section (after line 168):

```python
# Validation pipeline
VALIDATION_CONCURRENCY = int(os.getenv("VALIDATION_CONCURRENCY", "4"))
VALIDATION_ITERATIONS = int(os.getenv("VALIDATION_ITERATIONS", "5000"))
VALIDATION_DPS_THRESHOLD = float(os.getenv("VALIDATION_DPS_THRESHOLD", "5.0"))
VALIDATION_DPS_WARN_THRESHOLD = float(os.getenv("VALIDATION_DPS_WARN_THRESHOLD", "10.0"))
VALIDATION_ABILITY_THRESHOLD = float(os.getenv("VALIDATION_ABILITY_THRESHOLD", "15.0"))
VALIDATION_BUFF_THRESHOLD = float(os.getenv("VALIDATION_BUFF_THRESHOLD", "5.0"))
VALIDATION_PROC_THRESHOLD = float(os.getenv("VALIDATION_PROC_THRESHOLD", "20.0"))

# Log upload
LOG_UPLOAD_MAX_SIZE_MB = int(os.getenv("LOG_UPLOAD_MAX_SIZE_MB", "100"))
```

Update `WCL_TRACKED_CHARACTERS` (line 161-163) to include `race`:

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

### Task 4: Item Database Expansion

**Must come BEFORE the WCL bridge (Task 6)** — the bridge resolves weapon item IDs from `gear_json`, so weapons must exist in the database.

**Files:**
- Modify: `code/shukketsu/sim/items.py`
- Test: `tests/unit/test_sim_items.py`

**Step 1: Write failing tests**

Add to `tests/unit/test_sim_items.py`:

```python
class TestItemDBExpansion:
    """Tests for expanded item database (100+ items)."""

    def test_item_count_minimum(self) -> None:
        from code.shukketsu.sim.items import ItemDatabase
        db = ItemDatabase()
        assert len(db._items) >= 100

    def test_p1_sword_latros(self) -> None:
        from code.shukketsu.sim.items import ItemDatabase
        db = ItemDatabase()
        item = db.get_item(28189)  # Latro's Shifting Sword
        assert item is not None
        assert item.weapon is not None
        assert item.weapon.weapon_type.value == "sword"

    def test_p5_warglaive_mh(self) -> None:
        from code.shukketsu.sim.items import ItemDatabase
        db = ItemDatabase()
        item = db.get_item(32837)  # Warglaive of Azzinoth MH
        assert item is not None

    def test_dst_trinket(self) -> None:
        from code.shukketsu.sim.items import ItemDatabase
        db = ItemDatabase()
        item = db.get_item(28830)  # Dragonspine Trophy
        assert item is not None
        assert item.proc is not None

    def test_t6_slayer_set_exists(self) -> None:
        from code.shukketsu.sim.items import ItemDatabase
        db = ItemDatabase()
        slayer_items = [i for i in db._items.values() if i.set_id == "slayer"]
        assert len(slayer_items) >= 4

    def test_all_raid_weapon_types_present(self) -> None:
        """At least one weapon of each combat-relevant type."""
        from code.shukketsu.sim.items import ItemDatabase
        from code.shukketsu.sim.models import GearSlot
        db = ItemDatabase()
        mh_weapons = db.items_for_slot(GearSlot.MAIN_HAND)
        weapon_types = {w.weapon.weapon_type for w in mh_weapons if w.weapon}
        assert "sword" in weapon_types
        assert "dagger" in weapon_types
        assert "fist" in weapon_types
        assert "mace" in weapon_types
```

**Step 2: Run to verify failure**

Run: `python3 -m pytest tests/unit/test_sim_items.py::TestItemDBExpansion -v`
Expected: FAIL — `len(db._items)` is ~50, below 100 threshold.

**Step 3: Implement**

Add ~60 new items to `_CURATED_ITEMS` in `code/shukketsu/sim/items.py`. Focus on:
- **Weapons (~25):** All P1-P5 raid swords, daggers, fists, maces for both MH and OH slots. Source item IDs/stats from WoWSims or wowhead TBC.
- **Trinkets (~10):** All proc trinkets with verified rates (DST already exists; add Bloodlust Brooch 29383, Tsunami Talisman 30627, Madness of the Betrayer 32505, etc.).
- **Set pieces (~15):** Complete Netherblade (4pc) and Slayer's Armor (5pc) with `set_id`.
- **Armor (~10):** Key non-set pieces (Skulker's Greaves, Belt of 100 Deaths, etc.).

Data entry task — use WoWSims and wowhead TBC Classic as source of truth for stats and item IDs.

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_sim_items.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add code/shukketsu/sim/items.py tests/unit/test_sim_items.py
git commit -m "feat(sim): expand item database to 100+ curated TBC Rogue items"
```

---

### Task 5: Fight Filter

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
               (report_code, fight_id, player_name, player_type, total_damage, active_time_ms, abilities_json, targets_json)
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

# Encounters where continuous combat assumption holds (encounter_id -> name)
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

# If active_time / fight_duration < this, the player likely died
_DEATH_ACTIVE_RATIO: Final[float] = 0.85


class FilterCriteria(BaseModel):
    """Criteria for selecting valid calibration fights."""

    model_config = ConfigDict(frozen=True)

    kills_only: bool = True
    min_duration_ms: int = 30_000
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

    def __init__(self, criteria: FilterCriteria | None = None) -> None:
        self._criteria = criteria or FilterCriteria()

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
            exclusion_reason = self._check_exclusion(
                kill=row["kill"],
                duration_ms=row["duration_ms"],
                encounter_id=row["encounter_id"],
                active_time_ms=row["active_time_ms"],
            )

            active_time_ms = row["active_time_ms"] or 0
            active_dps = (
                row["total_damage"] / (active_time_ms / 1000)
                if active_time_ms > 0
                else 0.0
            )

            results.append(
                ValidatedFight(
                    report_code=row["report_code"],
                    fight_id=row["fight_id"],
                    encounter_id=row["encounter_id"],
                    encounter_name=row["encounter_name"],
                    duration_ms=row["duration_ms"],
                    source_id=row["source_id"],
                    wcl_active_dps=active_dps,
                    wcl_total_damage=row["total_damage"],
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

        # Detect death: active time < 85% of fight duration
        if active_time_ms and duration_ms > 0:
            if active_time_ms / duration_ms < _DEATH_ACTIVE_RATIO:
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

### Task 6: WCL-to-SimConfig Bridge

**This is the most complex task.** Bridges WCL combatant data into `SimConfig` objects the sim engine can run.

**Key challenge:** WCL stats are **post-buff totals** (include Kings, Grace of Air, etc.). If passed directly, buffs would be double-counted. Solution: create synthetic `Item` objects carrying unbuffed stats, using negative IDs to avoid collision with real items.

**Files:**
- Create: `code/shukketsu/sim/wcl_bridge.py`
- Create: `tests/unit/test_wcl_bridge.py`

**Step 1: Write failing tests**

Create `tests/unit/test_wcl_bridge.py`:

```python
"""Tests for WCL-to-SimConfig bridge."""

import json
import sqlite3

import pytest

from code.shukketsu.sim.wcl_bridge import WCLBridge


@pytest.fixture()
def bridge_db(test_db: sqlite3.Connection) -> sqlite3.Connection:
    """Populate DB with a minimal WCL fight for bridge testing."""
    test_db.execute("INSERT OR IGNORE INTO wcl_reports (code, endpoint) VALUES ('RPT1', 'fresh')")
    test_db.execute(
        """INSERT OR REPLACE INTO wcl_fights
           (report_code, fight_id, encounter_id, encounter_name, kill, duration_ms)
           VALUES ('RPT1', 1, 725, 'Brutallus', 1, 180000)""",
    )
    gear = json.dumps([
        {"id": 28189, "slot": 15, "itemLevel": 115},  # MH: Latro's (slot 15 = main_hand)
        {"id": 28189, "slot": 16, "itemLevel": 115},  # OH: same for simplicity (slot 16 = off_hand)
    ])
    auras = json.dumps([
        {"source": 0, "ability": 25898, "stacks": 0, "icon": "spell_magic.jpg"},  # Kings
    ])
    test_db.execute(
        """INSERT OR REPLACE INTO wcl_combatants
           (report_code, fight_id, source_id, player_name, spec_id,
            strength, agility, stamina, hit_melee, crit_melee, expertise, haste_melee,
            gear_json, talents_json, auras_json)
           VALUES ('RPT1', 1, 5, 'Lyroo', 260,
                   200, 800, 300, 142, 300, 60, 120,
                   ?, '[]', ?)""",
        (gear, auras),
    )
    test_db.commit()
    return test_db


class TestWCLBridge:
    def test_build_config_returns_sim_config(self, bridge_db: sqlite3.Connection) -> None:
        bridge = WCLBridge(bridge_db)
        config = bridge.build_config("RPT1", 1, 5, race="orc")
        from code.shukketsu.sim.models import SimConfig
        assert isinstance(config, SimConfig)

    def test_build_config_has_race(self, bridge_db: sqlite3.Connection) -> None:
        bridge = WCLBridge(bridge_db)
        config = bridge.build_config("RPT1", 1, 5, race="orc")
        assert config.race.value == "orc"

    def test_build_config_has_weapons(self, bridge_db: sqlite3.Connection) -> None:
        """Config must include MH and OH weapon item IDs."""
        bridge = WCLBridge(bridge_db)
        config = bridge.build_config("RPT1", 1, 5, race="orc")
        from code.shukketsu.sim.models import GearSlot
        assert GearSlot.MAIN_HAND in config.gear
        assert GearSlot.OFF_HAND in config.gear

    def test_build_config_has_talents(self, bridge_db: sqlite3.Connection) -> None:
        bridge = WCLBridge(bridge_db)
        config = bridge.build_config("RPT1", 1, 5, race="orc")
        assert len(config.talents) > 0

    def test_build_config_has_buffs_from_auras(self, bridge_db: sqlite3.Connection) -> None:
        """Kings aura in wcl_combatants should map to 'kings' buff."""
        bridge = WCLBridge(bridge_db)
        config = bridge.build_config("RPT1", 1, 5, race="orc")
        assert "kings" in config.buffs

    def test_build_config_synthetic_stat_item(self, bridge_db: sqlite3.Connection) -> None:
        """A synthetic item with unbuffed stats should be registered."""
        bridge = WCLBridge(bridge_db)
        config = bridge.build_config("RPT1", 1, 5, race="orc")
        from code.shukketsu.sim.models import GearSlot
        # Chest slot holds synthetic stat body
        assert GearSlot.CHEST in config.gear
        # Synthetic items use negative IDs
        assert config.gear[GearSlot.CHEST] < 0

    def test_build_config_missing_combatant_raises(self, bridge_db: sqlite3.Connection) -> None:
        from code.shukketsu.resilience.errors import WCLBridgeError
        bridge = WCLBridge(bridge_db)
        with pytest.raises(WCLBridgeError, match="No combatant"):
            bridge.build_config("RPT1", 1, 999, race="orc")

    def test_spec_detection_combat(self, bridge_db: sqlite3.Connection) -> None:
        """spec_id 260 (Combat) maps to combat_swords."""
        bridge = WCLBridge(bridge_db)
        config = bridge.build_config("RPT1", 1, 5, race="orc")
        assert "combat" in config.spec.value

    def test_item_db_has_synthetic_items(self, bridge_db: sqlite3.Connection) -> None:
        """After build_config, bridge's item_db includes synthetic items."""
        bridge = WCLBridge(bridge_db)
        config = bridge.build_config("RPT1", 1, 5, race="orc")
        from code.shukketsu.sim.models import GearSlot
        chest_id = config.gear[GearSlot.CHEST]
        item = bridge.item_db.get_item(chest_id)
        assert item is not None
        assert item.stats.get("agility", 0) > 0
```

**Step 2: Run to verify failure**

Run: `python3 -m pytest tests/unit/test_wcl_bridge.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Implement**

Create `code/shukketsu/sim/wcl_bridge.py`. Key design:

1. Read `wcl_combatants` row for (report_code, fight_id, source_id)
2. Parse `gear_json` → resolve MH/OH weapons from `ItemDatabase` (or create minimal synthetic weapon if not in DB)
3. Parse `auras_json` → map WCL aura IDs via `spell_map.wcl_buff_name()` → build `buffs` list
4. Compute what buff contributions those buffs would add → subtract from WCL post-buff stats to get unbuffed gear stats
5. Create synthetic `Item` for chest slot with unbuffed stats (negative ID like -1)
6. Map `spec_id` to `RogueSpec` and canonical talent string
7. Register all synthetic items into the bridge's `ItemDatabase`
8. Return `SimConfig`

The bridge holds its own `ItemDatabase` instance (copy of global + synthetic items).

```python
"""WCL-to-SimConfig bridge.

Reconstructs SimConfig from WCL combatant data using synthetic Item
objects. WCL stats are post-buff totals, so buff contributions are
subtracted to produce unbuffed gear stats.
"""

import json
import logging
import sqlite3

from code.shukketsu.resilience.errors import WCLBridgeError
from code.shukketsu.sim.items import ItemDatabase
from code.shukketsu.sim.models import (
    BossConfig,
    GearSlot,
    Item,
    PoisonConfig,
    Race,
    RogueSpec,
    SimConfig,
    WeaponStats,
    WeaponType,
)
from code.shukketsu.sim.spell_map import wcl_buff_name

logger = logging.getLogger(__name__)

# WCL spec_id -> (RogueSpec, canonical talent string)
_SPEC_MAP: dict[int, tuple[RogueSpec, str]] = {
    259: (RogueSpec.ASSASSINATION_MUTILATE, "41/20/0"),
    260: (RogueSpec.COMBAT_SWORDS, "20/41/0"),
    261: (RogueSpec.COMBAT_SWORDS, "20/41/0"),  # Subtlety → treat as combat for sim
}

# WCL gear slot index -> GearSlot (only weapon slots needed for resolution)
_WCL_WEAPON_SLOTS: dict[int, GearSlot] = {
    15: GearSlot.MAIN_HAND,
    16: GearSlot.OFF_HAND,
}

# Counter for synthetic item IDs (negative to avoid collision)
_SYNTHETIC_ID_COUNTER = -1000


def _next_synthetic_id() -> int:
    global _SYNTHETIC_ID_COUNTER
    _SYNTHETIC_ID_COUNTER -= 1
    return _SYNTHETIC_ID_COUNTER


class WCLBridge:
    """Bridges WCL combatant data into SimConfig via synthetic Items."""

    def __init__(self, conn: sqlite3.Connection, item_db: ItemDatabase | None = None) -> None:
        self._conn = conn
        self._item_db = item_db or ItemDatabase()

    @property
    def item_db(self) -> ItemDatabase:
        return self._item_db

    def build_config(
        self,
        report_code: str,
        fight_id: int,
        source_id: int,
        *,
        race: str = "human",
        fight_length: int | None = None,
        iterations: int | None = None,
    ) -> SimConfig:
        """Build a SimConfig from WCL combatant data.

        Args:
            report_code: WCL report code.
            fight_id: Fight ID within the report.
            source_id: Player's source ID in the report.
            race: Player race string (lowercase).
            fight_length: Override fight length (default: use fight duration).
            iterations: Override iteration count.

        Returns:
            SimConfig ready for simulation.

        Raises:
            WCLBridgeError: If combatant data is missing or insufficient.
        """
        row = self._conn.execute(
            """SELECT * FROM wcl_combatants
               WHERE report_code = ? AND fight_id = ? AND source_id = ?""",
            (report_code, fight_id, source_id),
        ).fetchone()

        if row is None:
            raise WCLBridgeError(
                f"No combatant found for report={report_code} fight={fight_id} source={source_id}"
            )

        # Resolve spec and talents
        spec_id = row["spec_id"] or 260
        spec, talents = _SPEC_MAP.get(spec_id, (RogueSpec.COMBAT_SWORDS, "20/41/0"))

        # Parse gear JSON for weapons
        gear_json = json.loads(row["gear_json"] or "[]")
        gear: dict[GearSlot, int] = {}
        self._resolve_weapons(gear_json, gear)

        # Parse auras JSON for buff detection
        auras_json = json.loads(row["auras_json"] or "[]")
        buffs = self._detect_buffs(auras_json)

        # Build synthetic stat body item (unbuffed gear stats)
        stat_item = self._build_stat_item(row, buffs)
        self._item_db._items[stat_item.id] = stat_item
        gear[GearSlot.CHEST] = stat_item.id

        # Get fight duration for fight_length
        fight_row = self._conn.execute(
            "SELECT duration_ms FROM wcl_fights WHERE report_code = ? AND fight_id = ?",
            (report_code, fight_id),
        ).fetchone()
        duration_s = (fight_row["duration_ms"] // 1000) if fight_row else 300

        # Build race
        race_enum = Race(race) if race in [r.value for r in Race] else Race.HUMAN

        from code.shukketsu import config as cfg
        return SimConfig(
            spec=spec,
            race=race_enum,
            talents=talents,
            gear=gear,
            buffs=buffs,
            consumables=[],
            boss=BossConfig(),
            poisons=PoisonConfig(),
            fight_length=fight_length or duration_s,
            iterations=iterations or cfg.VALIDATION_ITERATIONS,
        )

    def _resolve_weapons(self, gear_json: list[dict], gear: dict[GearSlot, int]) -> None:
        """Resolve MH/OH weapons from gear_json, creating synthetic weapons if needed."""
        for entry in gear_json:
            slot_idx = entry.get("slot", -1)
            if slot_idx not in _WCL_WEAPON_SLOTS:
                continue
            gear_slot = _WCL_WEAPON_SLOTS[slot_idx]
            item_id = entry.get("id", 0)
            existing = self._item_db.get_item(item_id)
            if existing is not None and existing.weapon is not None:
                gear[gear_slot] = item_id
            else:
                # Create minimal synthetic weapon
                synth = Item(
                    id=_next_synthetic_id(),
                    name=f"WCL Weapon {item_id}",
                    slot=gear_slot,
                    item_level=entry.get("itemLevel", 100),
                    phase=1,
                    weapon=WeaponStats(
                        min_damage=100, max_damage=200, speed=2.6,
                        weapon_type=WeaponType.SWORD, dps=57.7,
                    ),
                )
                self._item_db._items[synth.id] = synth
                gear[gear_slot] = synth.id
                logger.warning("Created synthetic weapon for unknown item ID %d", item_id)

    def _detect_buffs(self, auras_json: list[dict]) -> list[str]:
        """Map WCL auras to sim buff IDs."""
        buffs: list[str] = []
        for aura in auras_json:
            ability_id = aura.get("ability", 0)
            buff = wcl_buff_name(ability_id)
            if buff is not None and buff not in buffs:
                buffs.append(buff)
        return buffs

    def _build_stat_item(self, row: sqlite3.Row, buffs: list[str]) -> Item:
        """Build a synthetic chest item carrying unbuffed gear stats.

        WCL stats are post-buff. We subtract known buff contributions
        to approximate the gear-only stats.
        """
        # Raw WCL stats (post-buff)
        agi = float(row["agility"] or 0)
        strength = float(row["strength"] or 0)
        stamina = float(row["stamina"] or 0)
        hit_rating = float(row["hit_melee"] or 0)
        crit_rating = float(row["crit_melee"] or 0)
        haste_rating = float(row["haste_melee"] or 0)
        expertise_rating = float(row["expertise"] or 0)

        # Subtract Kings 10% multiplicative buff if present
        if "kings" in buffs:
            agi /= 1.10
            strength /= 1.10
            stamina /= 1.10

        # Subtract base Rogue stats (level 70)
        agi -= 180.0
        strength -= 110.0
        stamina -= 100.0

        # Clamp to 0 (WCL data can be noisy)
        agi = max(0.0, agi)
        strength = max(0.0, strength)
        stamina = max(0.0, stamina)

        return Item(
            id=_next_synthetic_id(),
            name="WCL Synthetic Stats",
            slot=GearSlot.CHEST,
            item_level=0,
            phase=1,
            stats={
                "agility": agi,
                "strength": strength,
                "stamina": stamina,
                "hit_rating": hit_rating,
                "crit_rating": crit_rating,
                "haste_rating": haste_rating,
                "expertise_rating": expertise_rating,
            },
        )
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_wcl_bridge.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add code/shukketsu/sim/wcl_bridge.py tests/unit/test_wcl_bridge.py
git commit -m "feat(sim): add WCL-to-SimConfig bridge with synthetic items"
```

---

### Task 7: Comparison Engine

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
    compute_drift,
)


class TestMetricDrift:
    def test_pass_within_threshold(self) -> None:
        d = MetricDrift(
            metric_name="dps", sim_value=1050, wcl_value=1000,
            absolute_delta=50, relative_pct=5.0, status="pass",
        )
        assert d.status == "pass"

    def test_fail_above_threshold(self) -> None:
        d = MetricDrift(
            metric_name="dps", sim_value=1200, wcl_value=1000,
            absolute_delta=200, relative_pct=20.0, status="fail",
        )
        assert d.status == "fail"


class TestWCLFightMetrics:
    def test_creation(self) -> None:
        m = WCLFightMetrics(
            total_damage=300000, active_dps=1000.0, fight_duration_ms=300000,
            ability_breakdown={"sinister_strike": AbilityMetrics(
                damage_total=150000, damage_pct=50.0, cast_count=120,
            )},
            buff_uptimes={"slice_and_dice": 0.95},
            proc_counts={"combat_potency": 45},
        )
        assert m.active_dps == 1000.0
        assert "sinister_strike" in m.ability_breakdown


class TestComputeDrift:
    """Tests for the compute_drift static function."""

    def test_pass(self) -> None:
        drift = compute_drift("dps", 1020.0, 1000.0, threshold_pass=5.0, threshold_warn=10.0)
        assert drift.status == "pass"
        assert abs(drift.relative_pct - 2.0) < 0.1

    def test_warn(self) -> None:
        drift = compute_drift("dps", 1080.0, 1000.0, threshold_pass=5.0, threshold_warn=10.0)
        assert drift.status == "warn"

    def test_fail(self) -> None:
        drift = compute_drift("dps", 1200.0, 1000.0, threshold_pass=5.0, threshold_warn=10.0)
        assert drift.status == "fail"

    def test_zero_wcl_value(self) -> None:
        """Zero WCL value should not divide by zero."""
        drift = compute_drift("dps", 100.0, 0.0, threshold_pass=5.0, threshold_warn=10.0)
        assert drift.status == "fail"

    def test_negative_drift(self) -> None:
        """Sim below WCL is still measured by absolute relative pct."""
        drift = compute_drift("dps", 800.0, 1000.0, threshold_pass=5.0, threshold_warn=10.0)
        assert drift.status == "fail"
        assert drift.relative_pct == pytest.approx(-20.0)


class TestValidationReport:
    def test_report_aggregation(self) -> None:
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

Create `code/shukketsu/sim/comparator.py`. Key design:
- `compute_drift()` is a **module-level function** (not a method) — easy to test without constructing a class.
- `SimComparator` holds a DB connection, extracts `WCLFightMetrics` from DB tables, compares against `SimResult`.
- Models: `AbilityMetrics`, `WCLFightMetrics`, `MetricDrift`, `FightValidation`, `BossAggregate`, `ValidationReport`.

```python
"""Comparison engine for sim validation.

Measures drift between SimResult and WCL fight data across DPS,
ability breakdown, buff uptimes, and proc rates.
"""

import logging
import sqlite3
from datetime import datetime, timezone

from pydantic import BaseModel

from code.shukketsu import config
from code.shukketsu.sim.models import SimResult
from code.shukketsu.sim.spell_map import aggregate_wcl_abilities, wcl_buff_name

logger = logging.getLogger(__name__)


class AbilityMetrics(BaseModel):
    """Per-ability metrics from WCL."""

    damage_total: int
    damage_pct: float
    cast_count: int = 0


class WCLFightMetrics(BaseModel):
    """Extracted WCL fight metrics for comparison."""

    total_damage: int
    active_dps: float
    fight_duration_ms: int
    ability_breakdown: dict[str, AbilityMetrics]
    buff_uptimes: dict[str, float]  # 0.0-1.0
    proc_counts: dict[str, int]


class MetricDrift(BaseModel):
    """Drift measurement for a single metric."""

    metric_name: str
    sim_value: float
    wcl_value: float
    absolute_delta: float
    relative_pct: float  # (sim - wcl) / wcl * 100
    status: str  # "pass" | "warn" | "fail"


class FightValidation(BaseModel):
    """Validation result for a single fight."""

    report_code: str
    fight_id: int
    encounter_name: str
    fight_duration_ms: int
    dps_drift: MetricDrift
    ability_drifts: list[MetricDrift]
    buff_drifts: list[MetricDrift]
    overall_status: str


class BossAggregate(BaseModel):
    """Aggregated validation across fights for one boss."""

    encounter_name: str
    fight_count: int
    avg_sim_dps: float
    avg_wcl_dps: float
    avg_drift_pct: float
    status: str


class ValidationReport(BaseModel):
    """Full validation report."""

    character_name: str
    total_fights: int
    included_fights: int
    excluded_fights: int
    per_fight: list[FightValidation]
    per_boss: dict[str, BossAggregate]
    overall_dps_drift_pct: float
    overall_status: str
    timestamp: str


def compute_drift(
    metric_name: str,
    sim_value: float,
    wcl_value: float,
    *,
    threshold_pass: float,
    threshold_warn: float,
) -> MetricDrift:
    """Compute drift between sim and WCL values.

    Args:
        metric_name: Human-readable name for the metric.
        sim_value: Value from simulation.
        wcl_value: Value from WCL data.
        threshold_pass: Absolute relative % below which status is "pass".
        threshold_warn: Absolute relative % below which status is "warn".

    Returns:
        MetricDrift with computed status.
    """
    absolute_delta = sim_value - wcl_value
    if wcl_value != 0:
        relative_pct = (sim_value - wcl_value) / wcl_value * 100
    else:
        relative_pct = 100.0 if sim_value > 0 else 0.0

    abs_pct = abs(relative_pct)
    if abs_pct <= threshold_pass:
        status = "pass"
    elif abs_pct <= threshold_warn:
        status = "warn"
    else:
        status = "fail"

    return MetricDrift(
        metric_name=metric_name,
        sim_value=sim_value,
        wcl_value=wcl_value,
        absolute_delta=absolute_delta,
        relative_pct=relative_pct,
        status=status,
    )


class SimComparator:
    """Compares SimResult against WCL fight data."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def extract_wcl_metrics(
        self,
        report_code: str,
        fight_id: int,
        player_name: str,
        duration_ms: int,
    ) -> WCLFightMetrics:
        """Extract WCL metrics from database for one fight."""
        import json

        # Damage
        dmg_row = self._conn.execute(
            "SELECT total_damage, active_time_ms, abilities_json FROM wcl_damage WHERE report_code = ? AND fight_id = ? AND player_name = ?",
            (report_code, fight_id, player_name),
        ).fetchone()

        total_damage = dmg_row["total_damage"] if dmg_row else 0
        active_time_ms = (dmg_row["active_time_ms"] or 0) if dmg_row else 0
        active_dps = total_damage / (active_time_ms / 1000) if active_time_ms > 0 else 0.0

        # Ability breakdown
        abilities_raw = json.loads(dmg_row["abilities_json"]) if dmg_row else []
        ability_breakdown = {}
        for ab in aggregate_wcl_abilities(abilities_raw).items():
            ability_breakdown[ab[0]] = AbilityMetrics(
                damage_total=ab[1],
                damage_pct=(ab[1] / total_damage * 100) if total_damage > 0 else 0.0,
            )

        # Buff uptimes
        buff_rows = self._conn.execute(
            "SELECT buff_guid, total_uptime_ms FROM wcl_buffs WHERE report_code = ? AND fight_id = ?",
            (report_code, fight_id),
        ).fetchall()
        buff_uptimes: dict[str, float] = {}
        for br in buff_rows:
            buff = wcl_buff_name(br["buff_guid"])
            if buff is not None:
                buff_uptimes[buff] = br["total_uptime_ms"] / duration_ms if duration_ms > 0 else 0.0

        return WCLFightMetrics(
            total_damage=total_damage,
            active_dps=active_dps,
            fight_duration_ms=duration_ms,
            ability_breakdown=ability_breakdown,
            buff_uptimes=buff_uptimes,
            proc_counts={},
        )

    def compare_fight(
        self,
        sim_result: SimResult,
        wcl_metrics: WCLFightMetrics,
        report_code: str,
        fight_id: int,
        encounter_name: str,
    ) -> FightValidation:
        """Compare a single sim result against WCL metrics."""
        dps_drift = compute_drift(
            "dps", sim_result.dps_mean, wcl_metrics.active_dps,
            threshold_pass=config.VALIDATION_DPS_THRESHOLD,
            threshold_warn=config.VALIDATION_DPS_WARN_THRESHOLD,
        )

        ability_drifts: list[MetricDrift] = []
        for name, wcl_ab in wcl_metrics.ability_breakdown.items():
            sim_ab = next((a for a in sim_result.ability_breakdown if a.name == name), None)
            sim_pct = sim_ab.damage_pct if sim_ab else 0.0
            ability_drifts.append(compute_drift(
                name, sim_pct, wcl_ab.damage_pct,
                threshold_pass=config.VALIDATION_ABILITY_THRESHOLD,
                threshold_warn=config.VALIDATION_ABILITY_THRESHOLD * 2,
            ))

        buff_drifts: list[MetricDrift] = []
        for buff_name, wcl_uptime in wcl_metrics.buff_uptimes.items():
            sim_uptime = sim_result.buff_uptimes.get(buff_name, 0.0)
            buff_drifts.append(compute_drift(
                buff_name, sim_uptime * 100, wcl_uptime * 100,
                threshold_pass=config.VALIDATION_BUFF_THRESHOLD,
                threshold_warn=config.VALIDATION_BUFF_THRESHOLD * 2,
            ))

        statuses = [dps_drift.status] + [d.status for d in ability_drifts] + [d.status for d in buff_drifts]
        if "fail" in statuses:
            overall = "fail"
        elif "warn" in statuses:
            overall = "warn"
        else:
            overall = "pass"

        return FightValidation(
            report_code=report_code,
            fight_id=fight_id,
            encounter_name=encounter_name,
            fight_duration_ms=wcl_metrics.fight_duration_ms,
            dps_drift=dps_drift,
            ability_drifts=ability_drifts,
            buff_drifts=buff_drifts,
            overall_status=overall,
        )

    def build_report(
        self,
        validations: list[FightValidation],
        character_name: str,
        total_fights: int,
        excluded_fights: int,
    ) -> ValidationReport:
        """Build aggregated validation report."""
        per_boss: dict[str, list[FightValidation]] = {}
        for v in validations:
            per_boss.setdefault(v.encounter_name, []).append(v)

        boss_aggs: dict[str, BossAggregate] = {}
        for boss_name, fights in per_boss.items():
            avg_sim = sum(f.dps_drift.sim_value for f in fights) / len(fights)
            avg_wcl = sum(f.dps_drift.wcl_value for f in fights) / len(fights)
            avg_drift = sum(f.dps_drift.relative_pct for f in fights) / len(fights)
            status = "pass" if abs(avg_drift) <= config.VALIDATION_DPS_THRESHOLD else "fail"
            boss_aggs[boss_name] = BossAggregate(
                encounter_name=boss_name,
                fight_count=len(fights),
                avg_sim_dps=avg_sim,
                avg_wcl_dps=avg_wcl,
                avg_drift_pct=avg_drift,
                status=status,
            )

        all_drifts = [v.dps_drift.relative_pct for v in validations]
        overall_drift = sum(all_drifts) / len(all_drifts) if all_drifts else 0.0
        overall_status = "pass" if abs(overall_drift) <= config.VALIDATION_DPS_THRESHOLD else "fail"

        return ValidationReport(
            character_name=character_name,
            total_fights=total_fights,
            included_fights=len(validations),
            excluded_fights=excluded_fights,
            per_fight=validations,
            per_boss=boss_aggs,
            overall_dps_drift_pct=overall_drift,
            overall_status=overall_status,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_comparator.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add code/shukketsu/sim/comparator.py tests/unit/test_comparator.py
git commit -m "feat(sim): add comparison engine for validation pipeline"
```

---

### Task 8: Schema v7 Migration

**Files:**
- Modify: `code/shukketsu/db/connection.py`
- Test: `tests/unit/test_db.py`

**Step 1: Write failing test**

Add to `tests/unit/test_db.py`:

```python
def test_schema_v7_validation_runs_table(test_db: sqlite3.Connection) -> None:
    """Schema v7 adds validation_runs table."""
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

Run: `python3 -m pytest tests/unit/test_db.py::test_schema_v7_validation_runs_table -v`
Expected: FAIL — table doesn't exist

**Step 3: Implement**

Add to `code/shukketsu/db/connection.py`:

```python
_VALIDATION_V7_SQL = """
CREATE TABLE IF NOT EXISTS validation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_name TEXT NOT NULL,
    run_type TEXT NOT NULL,
    total_fights INTEGER NOT NULL,
    included_fights INTEGER NOT NULL,
    overall_dps_drift_pct REAL NOT NULL,
    overall_status TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _migrate_v6_to_v7(conn: sqlite3.Connection) -> None:
    """Migrate v6 to v7: add validation_runs table."""
    conn.executescript(_VALIDATION_V7_SQL)
    conn.execute("INSERT INTO schema_version (version) VALUES (7)")
    conn.commit()
    logger.info("Database migrated from v6 to v7 (validation_runs table)")
```

Update `init_db()` to include both v6 and v7:

```python
            if version < 6:
                _migrate_v5_to_v6(conn)
            if version < 7:
                _migrate_v6_to_v7(conn)
```

And after base schema init:

```python
    if version < 6:
        _migrate_v5_to_v6(conn)
    if version < 7:
        _migrate_v6_to_v7(conn)
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_db.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add code/shukketsu/db/connection.py tests/unit/test_db.py
git commit -m "feat(db): add schema v7 with validation_runs table"
```

---

### Task 9: Validation Pipeline

**Files:**
- Create: `code/shukketsu/sim/validation_pipeline.py`
- Create: `tests/unit/test_validation_pipeline.py`

**Step 1: Write failing tests**

Create `tests/unit/test_validation_pipeline.py`:

```python
"""Tests for validation pipeline orchestrator."""

import sqlite3
from unittest.mock import AsyncMock, patch

import pytest

from code.shukketsu.sim.validation_pipeline import ValidationPipeline


class TestValidationPipeline:
    def test_construction(self, test_db: sqlite3.Connection) -> None:
        pipeline = ValidationPipeline(test_db)
        assert pipeline is not None

    async def test_run_no_fights_returns_empty_report(self, test_db: sqlite3.Connection) -> None:
        """No valid fights → report with 0 included fights."""
        pipeline = ValidationPipeline(test_db)
        report = await pipeline.run_validation("NonexistentPlayer", race="orc")
        assert report.included_fights == 0
        assert report.overall_status == "pass"

    async def test_run_stores_report_in_db(self, test_db: sqlite3.Connection) -> None:
        """Validation report is stored in validation_runs table."""
        pipeline = ValidationPipeline(test_db)
        report = await pipeline.run_validation("Lyroo", race="orc")
        row = test_db.execute("SELECT * FROM validation_runs ORDER BY id DESC LIMIT 1").fetchone()
        assert row is not None
        assert row["character_name"] == "Lyroo"

    async def test_progress_callback_fires(self, test_db: sqlite3.Connection) -> None:
        calls: list[str] = []
        pipeline = ValidationPipeline(test_db)
        await pipeline.run_validation("Lyroo", race="orc", on_progress=lambda msg: calls.append(msg))
        assert len(calls) >= 1  # At least "filtering fights" and "complete"
```

**Step 2: Run to verify failure**

Run: `python3 -m pytest tests/unit/test_validation_pipeline.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Implement**

Create `code/shukketsu/sim/validation_pipeline.py`:

```python
"""Validation pipeline orchestrator.

Orchestrates: FightFilter → WCLBridge → SimRunner → SimComparator → store report.
Uses ProcessPoolExecutor for CPU-bound sim parallelism.
"""

import asyncio
import json
import logging
import sqlite3
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

from code.shukketsu import config
from code.shukketsu.sim.comparator import SimComparator, ValidationReport
from code.shukketsu.sim.fight_filter import FightFilter, ValidatedFight
from code.shukketsu.sim.models import SimConfig, SimResult
from code.shukketsu.sim.wcl_bridge import WCLBridge

logger = logging.getLogger(__name__)


def _run_single_sim(config_dict: dict) -> dict:
    """Run a single sim in a subprocess. Must be pickle-safe (module-level function).

    Creates its own ItemDatabase and sim components per call.
    Accepts/returns dicts for pickle safety.
    """
    from code.shukketsu.sim.combat import CombatSimulation
    from code.shukketsu.sim.items import ItemDatabase
    from code.shukketsu.sim.models import SimConfig
    from code.shukketsu.sim.rotation import RotationEngine
    from code.shukketsu.sim.talents import compute_modifiers, parse_talents
    from code.shukketsu.sim.buffs import resolve_buffs

    sim_config = SimConfig(**config_dict)
    item_db = ItemDatabase()
    talent_tree = parse_talents(sim_config.talents)
    modifiers = compute_modifiers(talent_tree)
    buffs = resolve_buffs(sim_config.buffs, sim_config.consumables, sim_config.boss.debuffs)
    rotation = RotationEngine(sim_config.spec, modifiers)
    sim = CombatSimulation(sim_config, item_db, modifiers, buffs, rotation)
    result = sim.run(sim_config.iterations)
    return result.model_dump()


class ValidationPipeline:
    """Orchestrates WCL validation: filter → bridge → sim → compare → store."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    async def run_validation(
        self,
        character_name: str,
        *,
        race: str = "human",
        on_progress: Callable[[str], None] | None = None,
    ) -> ValidationReport:
        """Run full validation for a character.

        Args:
            character_name: WCL player name to validate.
            race: Player race (lowercase).
            on_progress: Optional callback for progress messages.

        Returns:
            ValidationReport with per-fight and per-boss metrics.
        """
        def _progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)
            logger.info(msg)

        _progress(f"Filtering fights for {character_name}...")
        fight_filter = FightFilter()
        all_fights = fight_filter.filter_fights(self._conn, character_name)
        included = [f for f in all_fights if f.included]
        excluded_count = len(all_fights) - len(included)

        if not included:
            _progress("No valid fights found.")
            comparator = SimComparator(self._conn)
            report = comparator.build_report([], character_name, len(all_fights), excluded_count)
            self._store_report(report)
            return report

        _progress(f"Building configs for {len(included)} fights...")
        bridge = WCLBridge(self._conn)
        configs: list[tuple[ValidatedFight, SimConfig]] = []
        for fight in included:
            try:
                sim_config = bridge.build_config(
                    fight.report_code, fight.fight_id, fight.source_id, race=race,
                )
                configs.append((fight, sim_config))
            except Exception:
                logger.warning("Failed to build config for fight %d in %s", fight.fight_id, fight.report_code, exc_info=True)

        if not configs:
            _progress("No configs could be built.")
            comparator = SimComparator(self._conn)
            report = comparator.build_report([], character_name, len(all_fights), excluded_count)
            self._store_report(report)
            return report

        _progress(f"Running {len(configs)} simulations...")
        # For now use asyncio.to_thread; ProcessPoolExecutor needs synthetic items
        # which can't be pickled across processes (they're registered in the bridge's ItemDatabase).
        # TODO: Switch to ProcessPoolExecutor when synthetic items are serializable.
        sim_results = await self._run_sims_threaded(configs, bridge)

        _progress("Comparing results...")
        comparator = SimComparator(self._conn)
        validations = []
        for (fight, _), sim_result in zip(configs, sim_results):
            wcl_metrics = comparator.extract_wcl_metrics(
                fight.report_code, fight.fight_id, character_name, fight.duration_ms,
            )
            validation = comparator.compare_fight(
                sim_result, wcl_metrics, fight.report_code, fight.fight_id, fight.encounter_name,
            )
            validations.append(validation)

        report = comparator.build_report(validations, character_name, len(all_fights), excluded_count)
        self._store_report(report)
        _progress(f"Validation complete: {report.overall_status} ({report.overall_dps_drift_pct:.1f}% drift)")
        return report

    async def _run_sims_threaded(
        self,
        configs: list[tuple[ValidatedFight, SimConfig]],
        bridge: WCLBridge,
    ) -> list[SimResult]:
        """Run sims using asyncio.to_thread (GIL-limited but safe with synthetic items)."""
        from code.shukketsu.sim.combat import CombatSimulation
        from code.shukketsu.sim.rotation import RotationEngine
        from code.shukketsu.sim.talents import compute_modifiers, parse_talents
        from code.shukketsu.sim.buffs import resolve_buffs
        from code.shukketsu.sim.models import SimResult

        async def run_one(sim_config: SimConfig) -> SimResult:
            talent_tree = parse_talents(sim_config.talents)
            modifiers = compute_modifiers(talent_tree)
            buffs = resolve_buffs(sim_config.buffs, sim_config.consumables, sim_config.boss.debuffs)
            rotation = RotationEngine(sim_config.spec, modifiers)
            sim = CombatSimulation(sim_config, bridge.item_db, modifiers, buffs, rotation)
            return await asyncio.to_thread(sim.run, sim_config.iterations)

        tasks = [run_one(cfg) for _, cfg in configs]
        return await asyncio.gather(*tasks)

    def _store_report(self, report: ValidationReport) -> None:
        """Store report in validation_runs table."""
        self._conn.execute(
            """INSERT INTO validation_runs
               (character_name, run_type, total_fights, included_fights,
                overall_dps_drift_pct, overall_status, report_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                report.character_name,
                "wcl",
                report.total_fights,
                report.included_fights,
                report.overall_dps_drift_pct,
                report.overall_status,
                report.model_dump_json(),
            ),
        )
        self._conn.commit()
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_validation_pipeline.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add code/shukketsu/sim/validation_pipeline.py tests/unit/test_validation_pipeline.py
git commit -m "feat(sim): add validation pipeline orchestrator"
```

---

### Task 10: CLEU Combat Log Parser

**Files:**
- Create: `code/shukketsu/sim/log_parser.py`
- Create: `tests/unit/test_log_parser.py`

Streaming line-by-line parser for TBC CLEU format. Tracks ENCOUNTER_START/END boundaries, aggregates SPELL_DAMAGE, SWING_DAMAGE, SPELL_AURA_APPLIED/REMOVED, SPELL_CAST_SUCCESS events per character. Returns `WCLFightMetrics` (same type as comparator) so both sources flow through the same comparison pipeline.

Test with sample CLEU log lines embedded as string fixtures. Tests cover: encounter boundary detection, damage aggregation by spell ID via spell_map, buff uptime calculation, cast counting, empty log handling, wrong character filtering.

---

### Task 11: WoWSims Import Parser

**Files:**
- Modify: `code/shukketsu/sim/imports.py:243-251`
- Modify: `tests/unit/test_sim_imports.py`

Replace the `parse_wowsims()` stub with a JSON parser. WoWSims exports as base64-encoded protobuf, but the JSON export (`individual_sim_ui.ts` → "Export JSON") is more practical. Add `_WOWSIMS_RACE_MAP`, `_WOWSIMS_SLOT_MAP`, `_detect_spec_from_talents()`. Update `detect_format()` to distinguish WoWSims JSON from SeventyUpgrades JSON (WoWSims has `"player"` key, SeventyUpgrades has `"race"` key).

---

### Task 12: Web Validation Dashboard

**Files:**
- Modify: `code/shukketsu/web/routers/sim.py`
- Create: `code/shukketsu/web/templates/sim/validate/index.html`
- Create: `code/shukketsu/web/templates/sim/validate/partials/report.html`
- Test: `tests/unit/test_sim_routes.py`

Add `/sim/validate/` GET (dashboard with run history table), `/sim/validate/run` POST (HTMX trigger), `/sim/validate/report/{run_id}` GET (renders full report JSON). Templates extend `base.html` with the existing dark WoW theme.

---

### Task 13: Web Log Upload Routes

**Files:**
- Create: `code/shukketsu/web/routers/logs.py`
- Create: `code/shukketsu/web/templates/logs/index.html`
- Create: `code/shukketsu/web/templates/logs/partials/fight_list.html`
- Modify: `code/shukketsu/web/app.py` (register router)
- Modify: `code/shukketsu/web/templates/base.html` (add Logs nav link)
- Create: `tests/unit/test_log_routes.py`

Add `/logs/` GET (upload page), `/logs/upload` POST (parse log + return fights partial). Validate file size against `LOG_UPLOAD_MAX_SIZE_MB`. Register router in `app.py`, add "Logs" nav link.

---

### Task 14: CLAUDE.md and Memory Update

Update CLAUDE.md Phase 4 section from stub to complete. Update memory with test counts and schema version. Commit.

---

## Execution Order Summary

| Task | Description | Depends On | Est. Tests |
|------|-------------|------------|------------|
| 0 | WCL ingest per-fight fix + v6 migration | -- | +5 |
| 1 | Error types | -- | +3 |
| 2 | Spell ID mapping | -- | +12 |
| 3 | Config additions | -- | +2 |
| 4 | Item DB expansion | -- | +6 |
| 5 | Fight filter | Task 0 | +8 |
| 6 | WCL-to-SimConfig bridge | Tasks 0, 2, 4 | +8 |
| 7 | Comparison engine | Tasks 2, 3 | +8 |
| 8 | Schema v7 migration | Task 0 | +1 |
| 9 | Validation pipeline | Tasks 5, 6, 7, 8 | +4 |
| 10 | CLEU log parser | Task 2 | +7 |
| 11 | WoWSims import | -- | +3 |
| 12 | Validation dashboard | Tasks 8, 9 | +2 |
| 13 | Log upload routes | Task 10 | +2 |
| 14 | CLAUDE.md + memory | All | 0 |

**Parallelizable:** Tasks 1, 2, 3, 4, 11 have no dependencies and can run in parallel.

**Critical path:** Task 0 → Task 5 → Task 6 → Task 9 → Task 12

**Total estimated new tests:** ~70+
