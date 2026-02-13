# Phase 4: Sim Validation, Calibration & Gap Fill — Design Document

**Goal:** Validate the TBC Rogue simulation engine against real WCL combat data and uploaded combat logs, calibrate mechanics until DPS drift is within 5%, and fill remaining gaps (CLEU parser, WoWSims import, item database expansion).

**Architecture:** A WCL-to-SimConfig bridge reconstructs simulation inputs from real fight data. A fight filter selects valid calibration fights (kills, no deaths, Patchwerk-like bosses). A comparison engine measures drift across DPS, ability breakdown, buff uptime, and proc rates. A CLEU combat log parser provides a second input source through the same comparison pipeline. Results surface in a web validation dashboard.

**Tech Stack:** Python 3.12, SQLite (schema v6), Pydantic v2, FastAPI/HTMX, httpx, existing sim engine + WCL API subsystem.

---

## 1. WCL-to-SimConfig Bridge (`sim/wcl_bridge.py`)

Reconstructs a `SimConfig` from WCL combatant data for a specific fight.

### Input

A row from `wcl_combatants` (joined with `wcl_fights` for fight metadata):

```python
# Key columns from wcl_combatants:
source_id: int
spec_id: int | None
strength: int | None
agility: int | None
hit_melee: int | None
crit_melee: int | None
haste_melee: int | None
expertise: int | None
armor: int | None
gear_json: str    # JSON array of WCLGearItem dicts
talents_json: str # JSON array of WCLTalentEntry dicts
auras_json: str   # JSON array of WCLAuraEntry dicts

# Key columns from wcl_fights:
duration_ms: int
encounter_id: int
encounter_name: str
kill: int  # 1 = kill
raid_size: int | None
```

### Output

A `SimConfig` ready to simulate.

### Reconstruction Logic

**Stats (direct mapping):**

WCL combatant info includes total stat ratings. These bypass the need to reconstruct gear piece-by-piece for most stats. However, `SimConfig` currently takes `gear: dict[GearSlot, int]` (item IDs), not raw stat totals. We have two options:

- **Option A (recommended):** Add an alternate constructor `SimConfig.from_stats()` that accepts raw stat totals + weapon info and builds the config without individual gear pieces. The combat simulation reads stats from the config — it doesn't care whether they came from items or raw totals.
- **Option B:** Reverse-engineer every gear piece from item IDs. Fragile and unnecessary for validation.

We go with Option A. The `from_stats()` factory method on `SimConfig`:

```python
@classmethod
def from_stats(
    cls,
    *,
    spec: RogueSpec,
    race: Race,
    talents: str,
    # Raw stat totals from WCL
    strength: int = 0,
    agility: int = 0,
    hit_rating: int = 0,
    crit_rating: int = 0,
    haste_rating: int = 0,
    expertise_rating: int = 0,
    attack_power: int = 0,
    armor_penetration: int = 0,
    # Weapon info (required)
    mh_weapon: WeaponStats,
    oh_weapon: WeaponStats,
    # Proc items (resolved from gear_json)
    proc_items: list[ProcEffect] = [],
    set_bonuses: list[SetBonus] = [],
    # Fight context
    buffs: list[str] = [],
    consumables: list[str] = [],
    boss: BossConfig = BossConfig(),
    fight_length: int = 300,
    target_count: int = 1,
    **kwargs,
) -> SimConfig:
```

This requires the combat simulation to support reading stats from a `raw_stats` dict when `gear` is empty. The simulation's `_calc_stats()` method (or equivalent) needs a branch: if `raw_stats` is populated, use those directly instead of summing item stats.

**Weapon resolution:**

Parse `gear_json` for slots 15 (main hand) and 16 (off hand). Look up item IDs in `ItemDatabase`. If found, use the `WeaponStats` from the item. If not found, raise `WCLBridgeError` with the missing item ID — the item database needs expansion.

**Proc item detection:**

Parse `gear_json` for slots 12 and 13 (trinkets) and all other slots. Check each item ID against `ItemDatabase`. If the item has a `proc` or `on_use` field, include it in the config. Maintain a `KNOWN_PROC_ITEMS: dict[int, ProcEffect]` fallback for proc items not in the curated database.

**Set bonus detection:**

Parse `gear_json`, group items by `set_id` (from `WCLGearItem.set_id`). For each set with >= 2 pieces, check `ItemDatabase.get_set_bonuses()`. Apply matching bonuses.

**Talent parsing:**

WCL `talents_json` contains `WCLTalentEntry` objects with `guid` and `name`. Map these to our "X/Y/Z" talent string format. The mapping is: count talents by tree (assassination tree GUIDs, combat tree GUIDs, subtlety tree GUIDs). Maintain a `WCL_TALENT_TREE_MAP: dict[int, str]` mapping talent GUIDs to tree names.

**Race resolution:**

WCL doesn't expose race. Add a `race` field to `config.WCL_TRACKED_CHARACTERS`:

```python
WCL_TRACKED_CHARACTERS: list[dict[str, str | int]] = [
    {"wcl_id": 104956434, "name": "Lyroo", "server": "nightslayer",
     "region": "us", "endpoint": "fresh", "race": "orc"},
]
```

For unknown characters (from rankings), default to `Race.HUMAN`.

**Buff reconstruction:**

Parse `auras_json` to detect active raid buffs. Map WCL aura ability IDs to our buff system names (e.g., aura 25898 → `"kings"`, aura 2048 → `"battle_shout"`). Maintain a `WCL_BUFF_MAP: dict[int, str]`.

**Boss context:**

Map `encounter_id` to a `BossConfig` with appropriate armor value and target count. Maintain an `ENCOUNTER_BOSS_CONFIG: dict[int, BossConfig]` table. Default boss armor is 7700 (level 73 boss). Some encounters have multiple targets (e.g., Illidari Council = 4 targets for Blade Flurry value).

### Public API

```python
class WCLBridge:
    def __init__(self, item_db: ItemDatabase, conn: sqlite3.Connection) -> None: ...

    def build_config(
        self,
        report_code: str,
        fight_id: int,
        source_id: int,
        *,
        race: Race = Race.HUMAN,
    ) -> SimConfig:
        """Reconstruct SimConfig from WCL combatant data for a specific fight."""

    def build_configs_for_character(
        self,
        character_name: str,
        *,
        race: Race = Race.HUMAN,
        fight_filter: FightFilter | None = None,
    ) -> list[tuple[ValidatedFight, SimConfig]]:
        """Build SimConfigs for all valid fights of a tracked character."""
```

---

## 2. Fight Filter (`sim/fight_filter.py`)

Selects WCL fights suitable for calibration.

### Filter Criteria

```python
@dataclass(frozen=True)
class FilterCriteria:
    kills_only: bool = True
    min_duration_ms: int = 30_000       # 30 seconds
    max_death_pct: float = 0.0          # 0% = character must survive entire fight
    patchwerk_only: bool = True         # Only Patchwerk-like encounters
    encounter_ids: set[int] | None = None  # Explicit whitelist (overrides patchwerk_only)
```

### Patchwerk-Like Encounters

Encounters where the sim's continuous-combat assumption holds. Curated per zone:

```python
PATCHWERK_ENCOUNTERS: dict[int, str] = {
    # Molten Core (Fresh)
    200663: "Lucifron",
    200664: "Magmadar",
    200670: "Golemagg the Incinerator",
    # Karazhan
    652: "Attumen the Huntsman",
    # Gruul's Lair
    649: "Gruul the Dragonkiller",
    # SSC
    623: "Hydross the Unstable",  # burn phases
    624: "The Lurker Below",
    # TK
    730: "Void Reaver",
    # Black Temple
    601: "Supremus",              # phases, but melee-friendly
    602: "Shade of Akama",
    # Sunwell Plateau
    725: "Brutallus",             # gold standard Patchwerk
    726: "Felmyst",               # ground phase only
}
```

This list grows as we validate — start conservative, add encounters that show good sim correlation.

### Death Detection

Check if the tracked character's `active_time` in WCL damage tables is significantly less than fight duration. Compute:

```
active_ratio = character_active_time / fight_duration_ms
```

If `active_ratio < 0.9`, the character likely died or had excessive downtime. Exclude the fight.

Active time can be inferred from the WCL damage table: if we have per-second damage data, the last second with nonzero damage gives us approximate active time. Alternatively, query WCL events API for death events (more API points but precise).

For MVP, use a simpler heuristic: check if the character appears in the fight's damage table at all. If their total damage is zero or suspiciously low (< 10% of fight duration × expected DPS), mark as excluded.

### Output

```python
class ValidatedFight(BaseModel):
    report_code: str
    fight_id: int
    encounter_id: int
    encounter_name: str
    duration_ms: int
    source_id: int               # Character's source ID in this fight
    wcl_active_dps: float        # From WCL damage table
    wcl_total_damage: int
    included: bool
    exclusion_reason: str | None = None  # "wipe", "death", "short_fight", "movement_boss"

class FightFilter:
    def __init__(self, criteria: FilterCriteria = FilterCriteria()) -> None: ...

    def filter_fights(
        self,
        conn: sqlite3.Connection,
        character_name: str,
    ) -> list[ValidatedFight]:
        """Return all fights for a character, annotated with inclusion/exclusion."""
```

---

## 3. Spell ID Mapping (`sim/spell_map.py`)

Maps WCL spell IDs to sim ability names and back.

### Mapping Table

```python
# WCL spell ID → sim ability name
SPELL_ID_TO_ABILITY: dict[int, str] = {
    # Builders
    1752: "sinister_strike",    # Sinister Strike (Rank 10)
    11294: "sinister_strike",   # Sinister Strike (other ranks)
    53: "backstab",             # Backstab (Rank 10)
    34413: "mutilate",          # Mutilate
    16511: "hemorrhage",        # Hemorrhage
    5938: "shiv",               # Shiv
    # Finishers
    26865: "eviscerate",        # Eviscerate (Rank 10)
    26867: "rupture",           # Rupture (Rank 7)
    6774: "slice_and_dice",     # Slice and Dice (Rank 2)
    32645: "envenom",           # Envenom (Rank 2)
    8647: "expose_armor",       # Expose Armor (Rank 5)
    # Openers
    11297: "ambush",            # Ambush (Rank 7)
    11290: "garrote",           # Garrote (Rank 7)
    1833: "cheap_shot",         # Cheap Shot
    # Cooldowns
    13877: "blade_flurry",      # Blade Flurry
    13750: "adrenaline_rush",   # Adrenaline Rush
    14177: "cold_blood",        # Cold Blood
    9512: "thistle_tea",        # Thistle Tea
    14185: "premeditation",     # Preparation (for Premeditation)
    # Poisons
    26891: "instant_poison",    # Instant Poison VII
    27282: "deadly_poison",     # Deadly Poison VII
    27283: "wound_poison",      # Wound Poison V
    # Procs
    23577: "combat_potency",    # Combat Potency (energize)
    13964: "sword_specialization",  # Sword Specialization extra attack
}

# Sim ability name → display name
ABILITY_DISPLAY_NAMES: dict[str, str] = {
    "sinister_strike": "Sinister Strike",
    "backstab": "Backstab",
    "mutilate": "Mutilate",
    "eviscerate": "Eviscerate",
    "rupture": "Rupture",
    "slice_and_dice": "Slice and Dice",
    "envenom": "Envenom",
    "blade_flurry": "Blade Flurry",
    "adrenaline_rush": "Adrenaline Rush",
    "instant_poison": "Instant Poison",
    "deadly_poison": "Deadly Poison",
    # ... etc
}

# WCL buff/aura ability IDs → our buff system names
WCL_BUFF_MAP: dict[int, str] = {
    25898: "kings",              # Greater Blessing of Kings
    2048: "battle_shout",        # Battle Shout
    25359: "grace_of_air",       # Grace of Air Totem
    25528: "strength_of_earth",  # Strength of Earth Totem
    26990: "motw",               # Mark of the Wild
    34300: "lotp",               # Leader of the Pack
    16293: "wf_totem",           # Windfury Totem
    27066: "trueshot_aura",      # Trueshot Aura
    32182: "heroism",            # Heroism / Bloodlust
    35476: "drums_of_battle",    # Drums of Battle
    # Boss debuffs
    25225: "sunder_armor",       # Sunder Armor (5 stacks)
    26993: "faerie_fire",        # Faerie Fire
    27226: "curse_of_recklessness", # Curse of Recklessness
}
```

**Note:** Multiple spell IDs can map to the same ability (different ranks). The mapping covers the most common TBC rank for each ability. We can expand as we encounter additional rank IDs in WCL data.

### Public API

```python
def wcl_ability_name(spell_id: int) -> str | None:
    """Return sim ability name for a WCL spell ID, or None if unknown."""

def sim_display_name(ability_name: str) -> str:
    """Return human-readable display name for a sim ability."""

def wcl_buff_name(spell_id: int) -> str | None:
    """Return our buff system name for a WCL buff spell ID, or None if unknown."""
```

---

## 4. Comparison Engine (`sim/comparator.py`)

Compares `SimResult` against WCL fight data and produces a validation report.

### WCL Data Extraction

For a given fight, query the database for:

- `wcl_damage` rows where `report_code` and `fight_id` match and `source_id` matches the tracked character → ability-level damage breakdown
- `wcl_buffs` rows → buff uptime bands (start_time, end_time per buff)
- `wcl_casts` rows → cast counts per ability

Normalize into a common comparison format:

```python
class WCLFightMetrics(BaseModel):
    """Extracted metrics from a single WCL fight for comparison."""
    total_damage: int
    active_dps: float
    fight_duration_ms: int
    ability_breakdown: dict[str, AbilityMetrics]  # keyed by sim ability name
    buff_uptimes: dict[str, float]                 # buff name → uptime 0.0-1.0
    proc_counts: dict[str, int]                    # proc name → count

class AbilityMetrics(BaseModel):
    damage_total: int
    damage_pct: float
    cast_count: int
    hit_count: int
    crit_count: int
```

### Comparison Output

```python
class MetricDrift(BaseModel):
    """Drift for a single metric."""
    metric_name: str
    sim_value: float
    wcl_value: float
    absolute_delta: float
    relative_pct: float      # (sim - wcl) / wcl * 100
    status: str              # "pass" | "warn" | "fail"

class FightValidation(BaseModel):
    """Validation result for a single fight."""
    report_code: str
    fight_id: int
    encounter_name: str
    fight_duration_ms: int
    dps_drift: MetricDrift
    ability_drifts: list[MetricDrift]
    buff_drifts: list[MetricDrift]
    proc_drifts: list[MetricDrift]
    overall_status: str      # "pass" | "warn" | "fail"

class ValidationReport(BaseModel):
    """Aggregated validation across multiple fights."""
    character_name: str
    total_fights: int
    included_fights: int
    excluded_fights: int
    per_fight: list[FightValidation]
    per_boss: dict[str, BossAggregate]
    overall_dps_drift_pct: float
    overall_status: str      # "pass" | "warn" | "fail"
    timestamp: str

class BossAggregate(BaseModel):
    encounter_name: str
    fight_count: int
    avg_sim_dps: float
    avg_wcl_dps: float
    avg_drift_pct: float
    status: str
```

### Drift Thresholds

```python
DPS_PASS_THRESHOLD = 5.0       # within 5% = pass
DPS_WARN_THRESHOLD = 10.0      # 5-10% = warn, >10% = fail
ABILITY_PASS_THRESHOLD = 15.0  # ability % of total: 15% relative drift
BUFF_PASS_THRESHOLD = 5.0      # buff uptime: 5% absolute
PROC_PASS_THRESHOLD = 20.0     # proc counts: 20% relative (RNG-heavy)
```

### Public API

```python
class SimComparator:
    def __init__(self, conn: sqlite3.Connection) -> None: ...

    def compare_fight(
        self,
        sim_result: SimResult,
        fight: ValidatedFight,
    ) -> FightValidation:
        """Compare sim output against WCL data for a single fight."""

    def build_report(
        self,
        validations: list[FightValidation],
        character_name: str,
        excluded: list[ValidatedFight],
    ) -> ValidationReport:
        """Aggregate fight validations into a full report."""
```

---

## 5. Validation Pipeline (`sim/validation_pipeline.py`)

Orchestrates the end-to-end validation flow.

### Pipeline Steps

1. **Filter fights** — `FightFilter.filter_fights()` returns all fights annotated with inclusion/exclusion
2. **Build configs** — `WCLBridge.build_config()` for each included fight
3. **Run sims** — `SimRunner.sim_run()` for each config (parallelized with `asyncio.gather`, limited to 4 concurrent)
4. **Compare results** — `SimComparator.compare_fight()` for each sim result vs WCL data
5. **Aggregate** — `SimComparator.build_report()` produces the final `ValidationReport`
6. **Store** — Save the report to `validation_runs` table for dashboard caching

### Concurrency

Each sim run is CPU-bound (10,000 iterations × 300s fight). On the GB10 with 12 CPU cores, we can parallelize 4 sims at once via `asyncio.to_thread` (which the `SimRunner` already uses). 56 fights ÷ 4 concurrent = ~14 batches.

### Public API

```python
class ValidationPipeline:
    def __init__(
        self,
        conn: sqlite3.Connection,
        runner: SimRunner,
        item_db: ItemDatabase,
    ) -> None: ...

    async def run_validation(
        self,
        character_name: str,
        *,
        race: Race = Race.HUMAN,
        filter_criteria: FilterCriteria = FilterCriteria(),
        concurrency: int = 4,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ValidationReport:
        """Run full validation pipeline for a tracked character."""

    async def run_single_fight(
        self,
        report_code: str,
        fight_id: int,
        source_id: int,
        *,
        race: Race = Race.HUMAN,
    ) -> FightValidation:
        """Validate a single fight (for debugging calibration issues)."""
```

### Progress Reporting

The `progress_callback(completed, total)` fires after each sim finishes. The web UI uses this for a progress bar via SSE or polling.

---

## 6. CLEU Combat Log Parser (`sim/log_parser.py`)

Parses raw WoW combat log text files into the same comparison format.

### CLEU Event Format (TBC Classic)

```
M/D HH:MM:SS.mmm  SUBEVENT,sourceGUID,sourceName,sourceFlags,sourceRaidFlags,destGUID,destName,destFlags,destRaidFlags,...
```

### Relevant Sub-Events

| Sub-Event | Purpose |
|-----------|---------|
| `ENCOUNTER_START` | Fight boundary (encounter ID, name, difficulty, size) |
| `ENCOUNTER_END` | Fight boundary (kill flag) |
| `SWING_DAMAGE` | Auto-attack hits |
| `SWING_MISSED` | Auto-attack misses/dodges |
| `SPELL_DAMAGE` | Ability/poison/proc damage |
| `SPELL_MISSED` | Ability misses/dodges/resists |
| `SPELL_PERIODIC_DAMAGE` | DOT ticks (Rupture, Garrote, Deadly Poison) |
| `SPELL_AURA_APPLIED` | Buff/debuff applied |
| `SPELL_AURA_REMOVED` | Buff/debuff removed |
| `SPELL_CAST_SUCCESS` | Ability cast (for cast counts) |
| `SPELL_ENERGIZE` | Energy gains (Combat Potency, Thistle Tea) |
| `UNIT_DIED` | Death detection |

### Parser Architecture

Stream the file line by line (no full-file load — logs can be 100MB+):

```python
class LogFight(BaseModel):
    """A single fight extracted from a combat log."""
    encounter_id: int
    encounter_name: str
    kill: bool
    duration_ms: int
    start_time: datetime
    end_time: datetime
    damage_by_ability: dict[int, AbilityMetrics]  # spell_id → metrics
    buff_timeline: dict[int, list[tuple[int, int]]]  # spell_id → [(start_ms, end_ms)]
    cast_counts: dict[int, int]                    # spell_id → count
    deaths: list[str]                              # character names that died

class ParsedCombatLog(BaseModel):
    """Full parse result from a combat log file."""
    fights: list[LogFight]
    character_name: str
    parse_errors: int  # lines that couldn't be parsed

class CLEUParser:
    def __init__(self, character_name: str) -> None: ...

    def parse_file(self, path: Path) -> ParsedCombatLog:
        """Parse a combat log file, extracting fights for the given character."""

    def parse_stream(self, stream: IO[str]) -> ParsedCombatLog:
        """Parse from a stream (for web upload)."""
```

### Normalization

After parsing, `LogFight` data is normalized into the same `WCLFightMetrics` format used by the comparison engine. The `spell_map.py` mappings convert spell IDs to sim ability names. This means the `SimComparator` works identically regardless of whether the data came from WCL API or a raw log file.

```python
def log_fight_to_metrics(fight: LogFight, character_name: str) -> WCLFightMetrics:
    """Convert parsed log fight into comparison-ready metrics."""
```

---

## 7. WoWSims Import Parser

Fill the `parse_wowsims()` stub in `sim/imports.py`.

### WoWSims Export Format

WoWSims exports character data as a Protocol Buffer (protobuf) JSON serialization. The relevant fields for a Rogue:

```json
{
  "race": 2,
  "class": 4,
  "equipment": {
    "items": [
      {"id": 32837, "enchant": 2673, "gems": [32220, 32196]},
      ...
    ]
  },
  "talentsString": "005323105500010-0252051000035015223-03",
  "rotation": { ... },
  "buffs": { ... },
  "debuffs": { ... },
  "consumes": { ... }
}
```

### Implementation

```python
def parse_wowsims(raw: str) -> CharacterImport:
    """Parse WoWSims JSON export into CharacterImport."""
    data = json.loads(raw)

    # Map WoWSims race enum to our Race enum
    race = _WOWSIMS_RACE_MAP.get(data.get("race", 0), Race.HUMAN)

    # Extract talents string (already in X/Y/Z-compatible format)
    talents = data.get("talentsString", "")

    # Extract gear: items array → dict[GearSlot, int]
    # WoWSims items array is ordered by slot index
    gear = {}
    enchants = {}
    gems = {}
    for item_data in data.get("equipment", {}).get("items", []):
        slot = _WOWSIMS_SLOT_MAP.get(idx, None)
        if slot and item_data.get("id"):
            gear[slot] = item_data["id"]
            if item_data.get("enchant"):
                enchants[slot] = item_data["enchant"]
            if item_data.get("gems"):
                gems[slot] = item_data["gems"]

    # Detect spec from talents
    spec = _detect_spec_from_talents(talents)

    return CharacterImport(
        spec=spec, race=race, talents=talents,
        gear=gear, enchants=enchants, gems=gems,
    )
```

### Mapping Tables

- `_WOWSIMS_RACE_MAP: dict[int, Race]` — WoWSims race enum → our `Race`
- `_WOWSIMS_SLOT_MAP: dict[int, GearSlot]` — WoWSims slot index → our `GearSlot`
- `_detect_spec_from_talents(talents: str) -> RogueSpec` — count points per tree to determine spec

---

## 8. Item Database Expansion

Expand `sim/items.py` from ~50 to ~130 curated items.

### Priority Items to Add

**Weapons (~40 items):**
- All P1 raid weapons: Blinkstrike, Spiteblade, Malchazeen, Latro's Shifting Sword, Decapitator
- P2 weapons: Talon of Azshara, Fang of Vashj, Twinblade of the Phoenix
- P3 weapons: Cursed Vision of Sargeras (thrown), MH/OH from Hyjal/BT
- P4/P5 weapons: Mounting Vengeance, Hand of the Deceiver, Boundless Agility

**Trinkets (~15 items):**
- Bloodlust Brooch, Icon of Unyielding Courage, Shard of Contempt
- Romulo's Poison Vial, Hourglass of the Unraveller
- Badge of Tenacity, Empty Mug of Direbrew

**Set pieces (~20 items):**
- Complete Netherblade set (5 pieces) — we may only have partial
- Complete Slayer's set (5 pieces)
- Deathmantle set (Tier 4)

**Proc items verification:**
- Ensure every proc trinket has a `ProcEffect` with correct trigger, rate, ICD, duration, and effect
- Cross-reference with community proc rate data (Simonize spreadsheet, WoWSims source)

### Source of Truth

Item stats sourced from:
- WoWSims item database (Go source files on GitHub, most authoritative)
- seventyupgrades.com item data
- wowhead TBC Classic database

---

## 9. Database Schema v6 (`validation_runs` table)

Add a single new table to cache validation results.

### Migration: v5 → v6

```sql
CREATE TABLE IF NOT EXISTS validation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_name TEXT NOT NULL,
    run_type TEXT NOT NULL,         -- 'wcl' or 'log'
    total_fights INTEGER NOT NULL,
    included_fights INTEGER NOT NULL,
    overall_dps_drift_pct REAL NOT NULL,
    overall_status TEXT NOT NULL,   -- 'pass', 'warn', 'fail'
    report_json TEXT NOT NULL,      -- Full ValidationReport serialized
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_validation_runs_character
    ON validation_runs(character_name, created_at DESC);
```

The `report_json` column stores the full `ValidationReport.model_dump_json()`. This avoids a complex normalized schema for per-fight and per-boss data — we only need to query by character and timestamp, and render the full report in the UI.

### Migration Code

Following the established pattern in `connection.py`:

```python
_VALIDATION_V6_SQL = """
CREATE TABLE IF NOT EXISTS validation_runs (
    ...
);
CREATE INDEX IF NOT EXISTS idx_validation_runs_character
    ON validation_runs(character_name, created_at DESC);
"""

def _migrate_v5_to_v6(conn: sqlite3.Connection) -> None:
    conn.executescript(_VALIDATION_V6_SQL)
    conn.execute("INSERT INTO schema_version (version) VALUES (6)")
    conn.commit()
    logger.info("Database migrated from v5 to v6 (validation_runs table)")
```

---

## 10. Web UI: Log Upload & Validation Dashboard

### New Routes

**Log upload (`/logs/`):**

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/logs/` | `logs_page` | Upload form with file input + character name field |
| POST | `/logs/upload` | `logs_upload` | Parse uploaded log, return fight list partial |
| POST | `/logs/analyze` | `logs_analyze` | Run comparison for selected fights, return results partial |

**Validation dashboard (`/sim/validate/`):**

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/sim/validate/` | `validate_page` | Dashboard showing latest validation runs per character |
| POST | `/sim/validate/run` | `validate_run` | Trigger full WCL validation pipeline, return progress partial |
| GET | `/sim/validate/status/{run_id}` | `validate_status` | Poll progress (SSE or JSON) |
| GET | `/sim/validate/report/{run_id}` | `validate_report` | Render full report partial |

### Templates

- `templates/logs/index.html` — Upload page (file drop zone, character name input)
- `templates/logs/partials/fight_list.html` — Parsed fights table with checkboxes
- `templates/logs/partials/analysis_results.html` — Comparison results
- `templates/sim/validate/index.html` — Dashboard with run history and latest results
- `templates/sim/validate/partials/progress.html` — Progress bar
- `templates/sim/validate/partials/report.html` — Full validation report with per-boss tables
- `templates/sim/validate/partials/fight_row.html` — Expandable fight detail row

### Navigation

Add "Logs" link to `base.html` nav bar (alongside Chat, Wiki, Sim).

### File Upload Handling

- Max file size: 100MB (FastAPI `UploadFile`)
- Stream to temp file, parse line-by-line with `CLEUParser`
- Delete temp file after parsing
- Return parsed fights for user selection before running comparison

### Validation Dashboard

- Show a table of tracked characters
- Each character row shows: last validation date, fight count, overall drift %, status badge (green/yellow/red)
- Click to expand → per-boss breakdown table
- "Run Validation" button per character → triggers pipeline, shows progress bar
- Historical runs listed below with timestamps

---

## 11. Config Additions

New config constants in `config.py`:

```python
# Validation pipeline
VALIDATION_CONCURRENCY = int(os.getenv("VALIDATION_CONCURRENCY", "4"))
VALIDATION_ITERATIONS = int(os.getenv("VALIDATION_ITERATIONS", "5000"))  # fewer than default for speed
VALIDATION_DPS_THRESHOLD = float(os.getenv("VALIDATION_DPS_THRESHOLD", "5.0"))
VALIDATION_ABILITY_THRESHOLD = float(os.getenv("VALIDATION_ABILITY_THRESHOLD", "15.0"))
VALIDATION_BUFF_THRESHOLD = float(os.getenv("VALIDATION_BUFF_THRESHOLD", "5.0"))

# Log upload
LOG_UPLOAD_MAX_SIZE_MB = int(os.getenv("LOG_UPLOAD_MAX_SIZE_MB", "100"))
```

---

## 12. Phase Gate Criteria

### Primary (blocking)

- [ ] Sim DPS for Lyroo's Combat Swords fights on Patchwerk-like bosses is within **5%** of WCL active DPS, averaged per boss
- [ ] Validation pipeline runs end-to-end: WCL data → SimConfig → sim → compare → report
- [ ] At least 10 valid fights pass the filter and are compared

### Secondary (informational, not blocking)

- [ ] Ability breakdown proportions within 15% relative per major ability
- [ ] SND uptime within 5% absolute
- [ ] Assassination (Mutilate) spec also validates within 5% if Lyroo has Mutilate logs
- [ ] CLEU parser correctly extracts fights from an uploaded combat log
- [ ] WoWSims import parser handles standard exports

### Deliverables

1. `sim/wcl_bridge.py` — WCL-to-SimConfig reconstruction
2. `sim/fight_filter.py` — Fight selection and validation
3. `sim/spell_map.py` — Spell ID ↔ ability name mapping
4. `sim/comparator.py` — Comparison engine with drift metrics
5. `sim/validation_pipeline.py` — End-to-end orchestration
6. `sim/log_parser.py` — CLEU combat log parser
7. `sim/imports.py` — WoWSims import parser (fill stub)
8. `sim/items.py` — Expanded item database (~130 items)
9. `sim/models.py` — `SimConfig.from_stats()` alternate constructor + `raw_stats` support
10. `db/connection.py` — Schema v6 migration
11. `web/routers/logs.py` — Log upload routes
12. `web/routers/sim.py` — Validation dashboard routes
13. `web/templates/logs/` — Upload and analysis templates
14. `web/templates/sim/validate/` — Dashboard templates
15. `config.py` — Validation and upload config constants
16. Tests for all new modules

---

## 13. Implementation Order

Recommended task sequence (each task is independently testable):

1. **Spell map** — Pure data, no dependencies. Foundation for everything else.
2. **Fight filter** — Queries DB, returns annotated fights. Tests with fixture data.
3. **WCL bridge** — Depends on spell map. Reconstructs SimConfig from DB rows. Requires `SimConfig.from_stats()`.
4. **SimConfig.from_stats()** — Modify `models.py` and combat engine to support raw stat input.
5. **Comparison engine** — Depends on spell map. Compares SimResult vs WCL metrics.
6. **Validation pipeline** — Wires filter + bridge + runner + comparator. Integration-level.
7. **Item database expansion** — Independent. Add ~80 items, verify proc definitions.
8. **CLEU log parser** — Independent of WCL pipeline. Uses spell map for normalization.
9. **WoWSims import** — Independent. Fill the stub in imports.py.
10. **Schema v6 migration** — Add `validation_runs` table.
11. **Web: validation dashboard** — Routes + templates for WCL validation.
12. **Web: log upload** — Routes + templates for combat log upload + analysis.
13. **Calibration** — Run the pipeline, identify drift sources, tune mechanics until phase gate passes.
14. **CLAUDE.md + memory update** — Document completion.
