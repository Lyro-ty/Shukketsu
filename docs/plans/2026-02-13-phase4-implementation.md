# Phase 4: TBC Rogue DPS Simulation Engine — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a discrete-event DPS simulation engine for TBC 2.4.3 Rogues with AI Analyst agent integration and web UI.

**Architecture:** Pure Python sim engine using `heapq` priority queue for event scheduling, millisecond-resolution combat simulation. Static data (abilities, talents, items, buffs) as Pydantic models loaded from JSON. AI Analyst agent wraps sim as tools for natural-language quantitative analysis. HTMX web UI for character setup and results.

**Tech Stack:** Python 3.12, Pydantic v2, heapq, asyncio, FastAPI, HTMX, Chart.js, Jinja2

**Design reference:** `docs/plans/2026-02-13-phase4-sim-engine-design.md` (full specifications for all models, formulas, abilities, talents, items, buffs, rotation, combat loop, runner API)

---

## Step 1: Foundation — Data Models, Config, Errors

**Files:**
- Create: `code/shukketsu/sim/models.py`
- Modify: `code/shukketsu/resilience/errors.py`
- Modify: `code/shukketsu/config.py`
- Modify: `code/shukketsu/sim/__init__.py`
- Create: `tests/unit/sim/__init__.py`
- Create: `tests/unit/sim/test_models.py`

**What to build:**

All enums and Pydantic data models from Section 1 of the design doc:

**Enums** (all `StrEnum`):
- `WeaponType` (sword, dagger, fist, mace)
- `RogueSpec` (combat_swords, combat_fists, combat_daggers, assassination_mutilate)
- `GearSlot` (17 slots: head through ranged)
- `Race` (human, orc, night_elf, blood_elf, undead, dwarf, gnome, troll)
- `PoisonType` (instant, deadly, wound, anesthetic, none)
- `OpenerAbility` (garrote, ambush, cheap_shot, none)
- `ProcTrigger` (on_hit, ppm, on_crit, on_use)
- `FightType` (patchwerk, cleave, movement)
- `GemSlot` (red, yellow, blue, meta)
- `HitOutcome` (hit, crit, miss, dodge, glancing, block)
- `AbilityFlag` (builder, finisher, normalized, cannot_be_dodged, applies_lethality, physical, nature, ignores_armor, main_hand, off_hand, snapshot, off_gcd)
- `BuffCategory` (attack_power, stats_pct, agility_flat, strength_flat, melee_crit, ap_pct, flask, battle_elixir, guardian_elixir, food, uncategorized)

**Item/Gear models:**
- `WeaponStats` (min_damage, max_damage, speed, weapon_type, dps)
- `ProcEffect` (trigger, rate, icd, duration, effect dict, stacks)
- `SetBonus` (set_name, pieces_required, effect dict)
- `Item` (id, name, slot, item_level, phase, stats dict, sockets, socket_bonus, set_id, weapon, proc, on_use)
- `Enchant` (id, name, slot, stats, proc)
- `Gem` (id, name, color, stats, meta_condition)

**Config models:**
- `PoisonConfig` (main_hand, off_hand defaults)
- `BossConfig` (name, level=73, armor=7700, debuffs list)
- `SimConfig` (spec, race, talents, gear dict, enchants, gems, poisons, opener, expose_armor, use_premeditation, buffs, consumables, boss, fight_type, target_count, fight_length, iterations, raid_preset, latency_ms)

**Result models:**
- `AbilityBreakdown` (name, damage_total, damage_pct, casts, hit_pct, crit_pct, miss_pct, dodge_pct, glancing_pct)
- `ProcUptime` (name, source, uptime_pct, avg_procs_per_fight, avg_stacks)
- `PoisonStats` (poison_type, hand, procs_per_fight, damage_total, damage_pct, avg_deadly_stacks)
- `CooldownUsage` (name, casts_per_fight, avg_uptime_pct)
- `ResourceStats` (energy_per_second, avg_energy_waste, combo_points_per_second, combo_point_overcap_pct, gcd_utilization_pct, energy_starved_pct)
- `StatWeight` (stat, ep_value, dps_per_point, is_capped)
- `DpsTimeline` (bucket_seconds=5, buckets list)
- `SimResult` (dps_mean, dps_std, dps_median, dps_min, dps_max, iterations, fight_length, ability_breakdown, stat_weights, proc_uptimes, poison_stats, cooldown_usage, buff_uptimes, resource_stats, dps_timeline, dps_distribution, config)

**Errors** (add to `resilience/errors.py`):
- Add `SIM_ERROR`, `SIM_VALIDATION`, `SIM_TIMEOUT` to `FailureMode` enum
- `SimError(ShukketsuError)` with `FailureMode.SIM_ERROR`
- `InvalidSimConfigError(ShukketsuError)` with `FailureMode.SIM_VALIDATION`
- `SimTimeoutError(ShukketsuError)` with `FailureMode.SIM_TIMEOUT`
- `ItemNotFoundError(ShukketsuError)` with `FailureMode.NOT_FOUND`

**Config** (add to `config.py`):
- `SIM_DEFAULT_ITERATIONS = int(os.getenv("SIM_DEFAULT_ITERATIONS", "10000"))`
- `SIM_DEFAULT_FIGHT_LENGTH = int(os.getenv("SIM_DEFAULT_FIGHT_LENGTH", "300"))`
- `SIM_STAT_WEIGHT_DELTA = int(os.getenv("SIM_STAT_WEIGHT_DELTA", "80"))`
- `SIM_CACHE_ENABLED = os.getenv("SIM_CACHE_ENABLED", "true").lower() == "true"`
- `ANALYST_SYSTEM_PROMPT` (import placeholder, fill in Step 11)
- `ANALYST_MAX_ITERATIONS = int(os.getenv("ANALYST_MAX_ITERATIONS", "5"))`

**Tests (~35):**
- Every enum has correct values and is a StrEnum
- `SimConfig` validates with full defaults
- `SimConfig` rejects invalid spec/race/talent combinations
- `SimResult` can be constructed from sample data
- `Item` with and without weapon/proc/sockets
- `ProcEffect` validates trigger types
- `BossConfig` defaults to level 73 / 7700 armor
- Error classes have correct FailureMode
- Config values are correct defaults

**Run:** `python3 -m pytest tests/unit/sim/test_models.py -v`

---

## Step 2: Combat Mechanics

**Files:**
- Create: `code/shukketsu/sim/mechanics.py`
- Create: `tests/unit/sim/test_mechanics.py`

**What to build:**

All TBC 2.4.3 combat formula functions from Section 2 of the design doc. Every function is **pure** (no state, no side effects). All constants defined at module level.

**Constants** (module-level):
```python
# Rating conversions (Level 70)
HIT_RATING_PER_PCT = 15.77
CRIT_RATING_PER_PCT = 22.08
HASTE_RATING_PER_PCT = 15.77
EXPERTISE_RATING_PER_POINT = 3.9423
AP_PER_DPS = 14.0
AGI_PER_CRIT_PCT = 40.0
AGI_PER_AP = 1.0
STR_PER_AP = 1.0

# Boss (Level 73)
BASE_MISS_CHANCE = 0.08
HIT_SUPPRESSION = 0.01
DW_MISS_PENALTY = 0.19
BASE_DODGE_CHANCE = 0.065
BASE_PARRY_CHANCE = 0.14
BASE_GLANCING_CHANCE = 0.24
GLANCING_MULTIPLIER = 0.75
CRIT_SUPPRESSION = 0.048

# Armor
ARMOR_CONSTANT = 10557.5
MAX_ARMOR_REDUCTION = 0.75

# Weapon normalization speeds
NORM_SPEED_DAGGER = 1.7
NORM_SPEED_ONE_HAND = 2.4

# Energy
ENERGY_TICK_MS = 2020
ENERGY_PER_TICK = 20.2
BASE_MAX_ENERGY = 100
VIGOR_BONUS_ENERGY = 10

# Crit
MELEE_CRIT_MULTIPLIER = 2.0
SPELL_CRIT_MULTIPLIER = 1.5
ROGUE_BASE_CRIT_ADJUSTMENT = -0.003

# Rogue
ROGUE_THREAT_MULTIPLIER = 0.71
BUILDER_MISS_REFUND = 0.80

# Boss armor presets
BOSS_ARMOR_STANDARD = 7700
BOSS_ARMOR_CASTER = 6200

# Armor debuffs
SUNDER_ARMOR_PER_STACK = 520
EXPOSE_ARMOR_BASE = 2050
FAERIE_FIRE_ARMOR = 610
CURSE_OF_RECKLESSNESS_ARMOR = 800
```

**Functions:**
1. `resolve_white_hit(miss_chance, dodge_chance, glancing_chance, crit_chance, roll) -> HitOutcome` — Single-roll table: miss → dodge → glance → crit → hit. Crit CAN be pushed off.
2. `resolve_yellow_hit(miss_chance, dodge_chance, crit_chance, hit_roll, crit_roll, *, can_be_dodged=True) -> HitOutcome` — Two-roll: roll1 for miss/dodge, roll2 for crit. Crit CANNOT be pushed off.
3. `calc_miss_chance(hit_rating, *, is_dual_wield, is_yellow, precision_ranks=0) -> float` — Base miss + DW penalty (yellow: no DW penalty), minus hit%. Clamped to [0, 1].
4. `calc_dodge_chance(expertise_rating, *, weapon_expertise_ranks=0) -> float` — 6.5% base minus expertise. Clamped to [0, 1].
5. `calc_crit_chance(crit_rating, agility, *, base_crit=0.0, talent_crit=0.0, bonus_crit=0.0, crit_suppression=True) -> float` — Sum all sources, apply rogue base adjustment (-0.3%), subtract crit suppression vs boss.
6. `calc_crit_multiplier(base_mult, *, primary_mod=1.0, secondary_mod=0.0, has_meta_gem=False) -> float` — WoWSims formula: `1.0 + (base * primary - 1.0) * (1.0 + secondary)`. Meta gem: `primary *= 1.03`.
7. `calc_armor_reduction(armor, *, arpen=0, sunder_stacks=0, expose_armor_ranks=0, faerie_fire=False, curse_of_recklessness=False) -> float` — Debuffs reduce armor first, then ArP, then formula. Clamped to [0.25, 1.0] (max 75% reduction).
8. `calc_weapon_damage(min_dmg, max_dmg, speed, attack_power, *, normalized=False, norm_speed=2.4, roll=0.5) -> float` — `base + AP/14 * (norm_speed if normalized else speed)`. Roll interpolates min/max.
9. `calc_haste_multiplier(haste_rating, *multiplicative_buffs) -> float` — `(1 + rating/1577) * prod(buffs)`. All haste stacks multiplicatively.
10. `calc_effective_speed(base_speed, haste_multiplier) -> float` — `base_speed / haste_multiplier`.
11. `calc_poison_proc_chance(base_chance, imp_poisons_ranks=0) -> float` — `base + 0.02 * ranks`.
12. `calc_ppm_proc_chance(ppm, weapon_speed) -> float` — `ppm * weapon_speed / 60`.
13. `calc_glancing_reduction() -> float` — Returns `GLANCING_MULTIPLIER` (0.75). Simplified from range.
14. `calc_normalized_speed(weapon_type) -> float` — Dagger: 1.7, all others: 2.4.

**Tests (~25):**
- `resolve_white_hit`: miss at 0.0, dodge at miss boundary, glancing at dodge boundary, crit at glancing boundary, hit at crit boundary, crit pushed off when miss+dodge+glancing fill table
- `resolve_yellow_hit`: miss on roll1, dodge on roll1, crit on roll2 (independent), hit when both rolls pass, cannot_be_dodged flag
- `calc_miss_chance`: zero hit rating DW (27%), zero hit rating 2H (9%), yellow (8% base + 1% suppression), with Precision talent, at hit cap
- `calc_dodge_chance`: zero expertise (6.5%), with expertise rating, with Weapon Expertise talent, at dodge cap (0%)
- `calc_crit_chance`: base case, with agility, with crit suppression, rogue adjustment
- `calc_crit_multiplier`: base (2.0), with Lethality (secondary), with meta gem (primary), combined
- `calc_armor_reduction`: standard boss (7700), with full debuffs, armor cap (75%), zero armor
- `calc_weapon_damage`: normal, normalized dagger, normalized sword, with AP scaling
- `calc_haste_multiplier`: zero rating, with SND, multiplicative stacking
- `calc_poison_proc_chance`, `calc_ppm_proc_chance`: known values

**Run:** `python3 -m pytest tests/unit/sim/test_mechanics.py -v`

---

## Step 3: Abilities Database

**Files:**
- Create: `code/shukketsu/sim/abilities.py`
- Create: `tests/unit/sim/test_abilities.py`

**What to build:**

Static ability definitions from Section 3 of the design doc. No simulation logic — just data.

**Models:**
- `AbilityDef(BaseModel)` — name, spell_id, energy_cost, flat_damage, weapon_multiplier, normalized, norm_speed, combo_points_generated, combo_points_consumed (bool for finishers), cooldown_ms, duration_ms, flags (set of AbilityFlag), miss_refund_pct, ap_coefficient, bonus_per_combo_point, weapon_type_required (optional)
- `PoisonDef(BaseModel)` — name, spell_id, proc_chance_base, damage_per_proc, damage_per_stack_tick (for Deadly), max_stacks, tick_interval_ms, duration_ms, flags

**Static registries (module-level dicts):**
- `ABILITIES: dict[str, AbilityDef]` — All builders: sinister_strike, backstab, mutilate, hemorrhage, shiv, ambush, garrote, cheap_shot. All finishers: eviscerate, envenom, rupture, slice_and_dice, expose_armor. All cooldowns: blade_flurry, adrenaline_rush, cold_blood, thistle_tea, premeditation.
- `POISONS: dict[str, PoisonDef]` — instant_poison, deadly_poison, wound_poison

**Lookup functions:**
- `get_ability(name: str) -> AbilityDef` — Raises `KeyError` if not found.
- `get_poison(poison_type: PoisonType) -> PoisonDef` — Converts enum to lookup.
- `builders_for_spec(spec: RogueSpec) -> list[str]` — Returns priority-ordered builder names for spec.
- `finisher_for_spec(spec: RogueSpec, *, has_rupture: bool, fight_remaining: float) -> str` — Returns preferred damage finisher.

**Key data points to get right** (from design doc triple-check):
- SS energy: 45/42/40 (talent ranks), flat +98, normalized 2.4
- BS energy: 60, flat +170 (NOT +255), wpn mult 1.5, normalized 1.7, Lethality applies
- Mutilate energy: 60, flat +101 per sub-hit, 1.5x if DP active, generates 2 CP, Lethality applies
- Shiv energy: `20 + 10 * oh_speed` (dynamic), miss_refund 0.80, cannot_be_dodged
- Ambush energy: 60, flat +290, wpn mult 2.5, normalized 1.7
- Garrote: 672 total over 18s (6 ticks of 112)
- Thistle Tea: 100 energy (game-accurate, NOT WoWSims 40)
- AR, Cold Blood, Thistle Tea, Premeditation: OFF_GCD flag
- Envenom: nature damage, ignores armor, consumes DP stacks (1 per CP)

**Tests (~15):**
- Every ability exists in registry with correct energy cost
- Ability flags are correct (builder/finisher/normalized/off_gcd)
- `builders_for_spec` returns correct order per spec
- `finisher_for_spec` returns envenom for mutilate, evis for combat
- Poison proc chances match design doc values
- Shiv energy cost scales with OH speed
- All cooldowns have correct duration and cooldown values
- `get_ability` raises KeyError for unknown ability

**Run:** `python3 -m pytest tests/unit/sim/test_abilities.py -v`

---

## Step 4: Talent System

**Files:**
- Create: `code/shukketsu/sim/talents.py`
- Create: `tests/unit/sim/test_talents.py`

**What to build:**

Talent definitions and modifier computation from Section 4 of the design doc.

**Models:**
- `TalentDef(BaseModel)` — name, tree (assassination/combat/subtlety), tier, column, max_ranks, effect_per_rank (dict mapping modifier name to value per rank)
- `TalentAllocation(BaseModel)` — assassination (int), combat (int), subtlety (int), points (dict mapping talent_name to ranks). Parsed from "20/41/0" string + point distribution.
- `TalentModifiers(BaseModel, frozen=True)` — All computed modifier fields:
  - `bonus_crit_pct`, `bonus_hit_pct`, `bonus_expertise`
  - `ss_energy_reduction`, `ss_damage_bonus_pct`
  - `bs_crit_bonus_pct`, `bs_damage_bonus_pct`
  - `mutilate_crit_bonus_pct`, `mutilate_damage_bonus_pct`
  - `evis_damage_bonus_pct`, `rupture_damage_bonus_pct`
  - `snd_duration_mult`, `snd_haste_bonus` (for T6 2pc, applied separately)
  - `lethality_secondary_mod`
  - `crit_damage_primary_mod` (Mace Spec)
  - `find_weakness_damage_pct`
  - `seal_fate_proc_chance`
  - `ruthlessness_proc_chance`
  - `relentless_strikes_per_cp`
  - `quick_recovery_refund_pct`
  - `combat_potency_proc_chance`, `combat_potency_energy`
  - `sword_spec_proc_chance` (per rank)
  - `dagger_spec_crit_bonus`, `fist_spec_crit_bonus`
  - `dw_spec_oh_bonus_pct`
  - `imp_poisons_ranks`, `vile_poisons_pct`, `master_poisoner_hit_pct`
  - `murder_damage_pct`
  - `aggression_damage_pct`
  - `opportunity_damage_pct`
  - `surprise_attacks_damage_pct`, `surprise_attacks_finisher_undodgeable` (bool)
  - `vitality_agi_mult`, `sinister_calling_agi_mult`, `deadliness_ap_mult`
  - `serrated_blades_arpen`, `serrated_blades_rupture_pct`
  - `vigor` (bool), `cold_blood` (bool), `blade_flurry` (bool), `adrenaline_rush` (bool)
  - `mutilate_talented` (bool), `hemorrhage_talented` (bool), `premeditation_talented` (bool)

**Static registries:**
- `TALENT_DEFS: dict[str, TalentDef]` — All talents from design doc (Assassination, Combat, Subtlety trees)

**Functions:**
- `parse_talents(talent_string: str, spec: RogueSpec) -> TalentAllocation` — Parses "20/41/0" and infers point distribution based on spec template. Validates total points <= 61.
- `compute_modifiers(allocation: TalentAllocation) -> TalentModifiers` — Pure function. Iterates allocation.points, accumulates each talent's effect_per_rank * ranks into modifier fields.
- `get_spec_template(spec: RogueSpec) -> dict[str, int]` — Returns canonical talent point distribution for a spec (e.g., Combat Swords 20/41/0 standard build).

**Tests (~15):**
- `parse_talents` with valid "20/41/0" string
- `parse_talents` rejects >61 total points
- `compute_modifiers` for Combat Swords (20/41/0): verify Precision 5/5, Sword Spec 5/5, Combat Potency 5/5, Lethality 5/5, etc.
- `compute_modifiers` for Mutilate (41/20/0): verify Seal Fate, Find Weakness, Mutilate talented, etc.
- Individual talent effects are correct magnitude (e.g., Lethality 5/5 = 0.30 secondary mod)
- Multiplicative stat stacking: Vitality * Sinister Calling correctly computed
- `get_spec_template` returns known builds
- TalentModifiers is frozen (immutable)

**Run:** `python3 -m pytest tests/unit/sim/test_talents.py -v`

---

## Step 5: Item Database + Data Conversion

**Files:**
- Create: `code/shukketsu/sim/items.py`
- Create: `code/shukketsu/sim/data/` directory
- Create: `code/shukketsu/sim/data/items.json` (converted from WoWSims)
- Create: `code/shukketsu/sim/data/enchants.json`
- Create: `code/shukketsu/sim/data/gems.json`
- Create: `code/shukketsu/sim/data/sets.json`
- Create: `scripts/convert_wowsims_items.py` (offline converter)
- Create: `tests/unit/sim/test_items.py`

**What to build:**

Item database and WoWSims data conversion from Section 5 of the design doc.

**Data conversion script** (`scripts/convert_wowsims_items.py`):
- Clones/reads WoWSims `sim/core/items/` Go files
- Parses Go struct literals with regex
- Emits `items.json`, `enchants.json`, `gems.json`, `sets.json`
- Filters for Rogue-relevant items (leather/mail, jewelry, weapons, trinkets)
- This is a one-time offline script, not runtime code
- For initial implementation, manually curate a minimal set (~50 key items covering all phases, all specs) to unblock testing. Full conversion comes later.

**ItemDatabase class:**
```python
class ItemDatabase:
    def __init__(self, data_dir: Path | None = None) -> None: ...
    def get_item(self, item_id: int) -> Item | None: ...
    def get_enchant(self, enchant_id: int) -> Enchant | None: ...
    def get_gem(self, gem_id: int) -> Gem | None: ...
    def get_set_bonuses(self, set_name: str) -> list[SetBonus]: ...
    def items_for_slot(self, slot: GearSlot, *, phase: int | None = None, min_ilvl: int = 0) -> list[Item]: ...
    def search(self, query: str, *, slot: GearSlot | None = None) -> list[Item]: ...
    def rogue_items_for_slot(self, slot: GearSlot, spec: RogueSpec, phase: int = 5) -> list[Item]: ...
```

**Set bonus data** (hardcoded in `items.py`):
- Netherblade (T4): 2pc SND +3s, 4pc 15% CP on finisher
- Deathmantle (T5): 2pc +40/CP on Evis/Envenom, 4pc 1.0 PPM free finisher
- Slayer's Armor (T6): 2pc SND +5% haste, 4pc +6% SS/BS/Mut/Hemo

**Notable proc items** (manually defined with correct ICD/PPM/effects):
- Dragonspine Trophy: 1.0 PPM, 20s ICD, 10s +325 haste rating
- Tsunami Talisman: 10% on crit, 45s ICD, 10s +340 AP
- Madness of the Betrayer: ~1 PPM, 10s ICD, 10s +300 ArP
- Mongoose enchant: 1.0 PPM/weapon, 15s +120 agi +2% haste
- Executioner enchant: 1.0 PPM, 15s +840 ArP

**Minimal curated item set** (for testing, ~50 items):
- P1 BiS Combat Swords gear (Blinkstrike, Latro's, DST, Brooch, Netherblade 4pc)
- P3 BiS additions (Warglaives, Deathmantle)
- P5 BiS additions (Slayer's 4pc, Madness)
- Key daggers for Mutilate (Emerald Ripper, etc.)
- Common enchants (Mongoose, Executioner, etc.)
- Common gems (Delicate Living Ruby, etc.)

**Tests (~20):**
- `ItemDatabase` loads from JSON files
- `get_item` returns correct item by ID
- `get_item` returns None for unknown ID
- `items_for_slot` filters by slot
- `items_for_slot` filters by phase
- `search` finds items by name (case-insensitive substring)
- `rogue_items_for_slot` excludes non-Rogue items
- Set bonuses load correctly
- Proc items have correct ProcEffect data
- `get_enchant`, `get_gem` work correctly

**Run:** `python3 -m pytest tests/unit/sim/test_items.py -v`

---

## Step 6: Buffs, Consumables & Character Import

**Files:**
- Create: `code/shukketsu/sim/buffs.py`
- Create: `code/shukketsu/sim/imports.py`
- Create: `tests/unit/sim/test_buffs.py`
- Create: `tests/unit/sim/test_imports.py`

**What to build:**

### Buffs (`buffs.py`)

Buff definitions, stacking rules, and preset profiles from Section 6 of the design doc.

**Models:**
- `BuffDef(BaseModel)` — name, buff_id (str), category (BuffCategory), stats (dict), proc (ProcEffect | None), blocks_poison (bool, for WF), description
- `ResolvedBuffs(BaseModel)` — flat_stats (dict), stat_multipliers (dict), active_procs (list[ProcEffect]), boss_armor_reduction (int), active_buff_ids (set[str])

**Static registries:**
- `RAID_BUFFS: dict[str, BuffDef]` — All raid buffs: kings, imp_battle_shout, imp_grace_of_air, imp_soe, imp_motw, lotp, wf_totem, trueshot_aura, heroism, drums_of_battle
- `BOSS_DEBUFFS: dict[str, BuffDef]` — sunder_armor, faerie_fire, curse_of_recklessness
- `CONSUMABLES: dict[str, BuffDef]` — flask_relentless_assault, elixir_major_agi, food_clefthoof, food_warp_burger, haste_potion, etc.
- `RAID_PRESETS: dict[str, list[str]]` — full_25man, karazhan_10man, solo, custom

**Functions:**
- `resolve_buffs(buff_ids: list[str], debuff_ids: list[str], consumable_ids: list[str]) -> ResolvedBuffs` — Applies BuffCategory stacking rules (same category = highest wins), aggregates flat stats and multipliers, computes boss armor reduction from debuffs.
- `get_preset(name: str) -> tuple[list[str], list[str], list[str]]` — Returns (buffs, debuffs, consumables) for a preset name.

### Character Import (`imports.py`)

Three import formats with auto-detection from Section 5 of the design doc.

**Models:**
- `ImportFormat(StrEnum)` — SIMC, SEVENTYUPGRADES, WOWSIMS
- `CharacterImport(BaseModel)` — spec, race, talents, gear (dict[GearSlot, int]), enchants, gems

**Functions:**
- `detect_format(raw: str) -> ImportFormat` — `=` in first line → SIMC, starts with `{` → SEVENTYUPGRADES, else → WOWSIMS
- `parse_simc(raw: str) -> CharacterImport` — Parse `/simc` addon output (key=value lines)
- `parse_seventyupgrades(raw: str) -> CharacterImport` — Parse 70u JSON export
- `parse_wowsims(raw: str) -> CharacterImport` — Parse WoWSims Base64 URL/export
- `parse_import(raw: str) -> CharacterImport` — Auto-detect and delegate
- `build_config(char_import: CharacterImport, *, preset: str = "full_25man", **overrides) -> SimConfig` — Builds SimConfig from import + defaults

**Tests (~25):**

Buffs (~12):
- `resolve_buffs` with full 25-man preset produces correct aggregate stats
- BuffCategory stacking: two buffs in same category → highest wins
- BuffCategory stacking: different categories stack
- WF totem blocks_poison flag
- Preset names return correct buff lists
- Empty buff list returns zero stats
- Boss debuff armor reduction sums correctly
- Consumable flask vs elixir stacking (mutually exclusive)

Imports (~13):
- `detect_format` identifies SimC format
- `detect_format` identifies 70u JSON format
- `detect_format` identifies WoWSims Base64 format
- `parse_simc` extracts gear, talents, race from sample input
- `parse_seventyupgrades` extracts from sample JSON
- `parse_wowsims` extracts from sample Base64
- `parse_import` auto-detects and delegates correctly
- `build_config` produces valid SimConfig with preset buffs
- Invalid import string raises `InvalidSimConfigError`
- Missing required fields raise `InvalidSimConfigError`

**Run:** `python3 -m pytest tests/unit/sim/test_buffs.py tests/unit/sim/test_imports.py -v`

---

## Step 7: Rotation Engine

**Files:**
- Create: `code/shukketsu/sim/rotation.py`
- Create: `tests/unit/sim/test_rotation.py`

**What to build:**

State-machine rotation engine from Section 7 of the design doc.

**Models:**
- `RotationState(StrEnum)` — OPENER, SLICE_ASAP, DISPATCH, BUILD_FOR_SND, BUILD_FOR_EA, FILL_BEFORE_SND, FILL_BEFORE_EA
- `RotationAction(BaseModel)` — ability_name (str), target (str, "boss" or "self"), wait_for_energy (bool)
- `RotationContext(BaseModel, frozen=True)` — Read-only snapshot: combo_points, energy, max_energy, snd_remaining_ms, rupture_remaining_ms, ea_remaining_ms, dp_remaining_ms, dp_stacks, ar_active, bf_active, bf_ready, ar_ready, cb_ready, tea_ready, premeditation_ready, fight_remaining_ms, target_count, is_stealthed, gcd_ready_at_ms, current_time_ms

**RotationEngine class:**
```python
class RotationEngine:
    def __init__(self, spec: RogueSpec, modifiers: TalentModifiers, *, expose_armor: bool = False) -> None: ...
    def decide(self, ctx: RotationContext) -> RotationAction: ...
    def _dispatch(self, ctx: RotationContext) -> RotationAction: ...
    def _should_use_cooldown(self, cd_name: str, ctx: RotationContext) -> bool: ...
    def _select_builder(self, ctx: RotationContext) -> str: ...
    def _select_finisher(self, ctx: RotationContext) -> str: ...
    def _should_pool_energy(self, ctx: RotationContext) -> bool: ...
```

**Key behaviors to implement:**
1. **State transitions**: OPENER → SLICE_ASAP → DISPATCH is the standard flow. DISPATCH is the central decision hub.
2. **DISPATCH priority**: (1) SND expired → SLICE_ASAP, (2) EA refresh needed → EA states, (3) SND refresh needed → BUILD_FOR_SND, (4) enough CPs → damage finisher (pool check), (5) Shiv for DP (Mut only), (6) build CPs
3. **Energy pooling**: Don't finisher below 50 energy (30 during AR) unless SND dropping
4. **SND refresh buffer**: `4.0 - combo_points * 0.8` seconds
5. **Builder selection**: Mutilate (if talented + daggers) > Backstab (dagger MH) > SS (fallback)
6. **Finisher selection**: Rupture if not active and fight long enough. Else Envenom (Mut) or Evis (Combat).
7. **Cooldown timing**: AR at energy <= 85, BF with SND active, Cold Blood paired with AR or on CD, Tea at energy <= max-100

**Tests (~20):**
- OPENER state returns opener ability
- SLICE_ASAP state returns SND at any CP count
- DISPATCH with SND expired → SLICE_ASAP
- DISPATCH with enough CPs and SND active → damage finisher
- DISPATCH with low CPs → builder
- Energy pooling: don't finisher at 40 energy
- Energy pooling: DO finisher at 40 energy during AR
- Builder selection: combat_swords → sinister_strike
- Builder selection: assassination_mutilate → mutilate
- Builder selection: combat_daggers → backstab
- Finisher selection: combat + rupture not active → rupture
- Finisher selection: mutilate → envenom
- Cooldown timing: AR offered when energy <= 85 and SND active
- Cooldown timing: BF offered when SND active
- Cooldown timing: Tea offered when energy <= max-100
- Shiv for DP: offered when DP < 2s remaining (Mutilate only)
- SND refresh buffer calculation
- State machine doesn't get stuck (fuzzy: random contexts, always returns an action)

**Run:** `python3 -m pytest tests/unit/sim/test_rotation.py -v`

---

## Step 8: Combat Event Loop

**Files:**
- Create: `code/shukketsu/sim/combat.py`
- Create: `tests/unit/sim/test_combat.py`

**What to build:**

The heart of the simulation: discrete-event combat loop from Section 8 of the design doc.

**Models/Enums:**
- `EventType(StrEnum)` — MH_AUTO, OH_AUTO, ENERGY_TICK, ABILITY_USE, DOT_TICK, BUFF_EXPIRE, PROC_TRIGGER, COOLDOWN_USE, POTION_USE
- `SimEvent(BaseModel)` — timestamp_ms (int), event_type (EventType), priority (int), data (dict). Implements `__lt__` for heapq ordering by (timestamp_ms, priority).
- `CombatState` (dataclass, mutable) — current_time_ms, energy, max_energy, combo_points, health_pct, gcd_ready_at_ms, mh_swing_at_ms, oh_swing_at_ms, next_energy_tick_ms, buff_timers (dict[str, int]), dot_timers (dict[str, DotState]), proc_icds (dict[str, int]), sword_spec_icd_ms (int), damage_by_ability (defaultdict), casts_by_ability (defaultdict), outcome_counts (nested defaultdict), proc_counts (defaultdict), proc_uptime_ms (defaultdict), resource_tracker (energy waste, CP waste, etc.)

**DotState(BaseModel):**
- remaining_ticks, tick_interval_ms, damage_per_tick, next_tick_ms, snapshot_ap (for Rupture)

**CombatSimulation class:**
```python
class CombatSimulation:
    def __init__(
        self,
        config: SimConfig,
        modifiers: TalentModifiers,
        resolved_buffs: ResolvedBuffs,
        item_db: ItemDatabase,
        rotation: RotationEngine,
    ) -> None: ...

    def run(self, iterations: int, *, seed: int = 42) -> SimResult: ...
    def _run_iteration(self, rng: Random) -> float: ...
    def _schedule(self, event: SimEvent) -> None: ...
    def _dispatch_event(self, event: SimEvent, state: CombatState, rng: Random) -> None: ...
    def _handle_mh_auto(self, state, rng) -> None: ...
    def _handle_oh_auto(self, state, rng) -> None: ...
    def _handle_energy_tick(self, state) -> None: ...
    def _handle_ability(self, ability_name, state, rng) -> None: ...
    def _handle_dot_tick(self, dot_name, state) -> None: ...
    def _handle_buff_expire(self, buff_name, state) -> None: ...
    def _handle_proc(self, proc_name, state, rng) -> None: ...
    def _apply_damage(self, ability, base_damage, outcome, state) -> float: ...
    def _check_procs(self, source, outcome, state, rng) -> None: ...
    def _apply_poison(self, hand, state, rng) -> None: ...
    def _adjust_swing_timers(self, old_haste, new_haste, state) -> None: ...
    def _aggregate_results(self, iteration_dps: list[float]) -> SimResult: ...
```

**Key mechanics to implement:**
1. **Event scheduling**: `heapq.heappush(queue, event)`, pop min timestamp
2. **Auto-attack startup**: One weapon gets random 0-50% delay
3. **Energy tick**: First at random offset [0, 2020ms], then every 2020ms
4. **GCD**: 1.0s for Rogues. Abilities can't be used until GCD ready.
5. **Proc-on-proc**: White → can trigger SwordSpec + WF + poisons. Yellow → SwordSpec + poisons (no WF). SwordSpec → WF + poisons (not self). WF → SwordSpec + poisons (not self). Sword Spec 500ms ICD.
6. **Haste swing adjustment**: When SND/BF activates/expires, adjust in-progress swing timers proportionally.
7. **Blade Flurry cleave**: Mirror damage to 2nd target (re-apply armor on cleave).
8. **Deterministic RNG**: `Random(seed=iteration_number)` per iteration.
9. **Result aggregation**: Mean, std, median, min, max DPS across iterations. Per-ability breakdown. Proc uptimes. Resource stats.

**Tests (~25):**
- Single iteration produces non-zero DPS
- Deterministic: same seed → same DPS
- Different seeds → different DPS
- Auto-attacks fire at correct intervals
- Energy ticks provide 20.2 energy
- GCD prevents double-casting
- Sinister Strike costs energy and generates 1 CP
- Eviscerate at 5 CP deals more than at 1 CP
- SND activates haste buff (swing timer shortens)
- Blade Flurry adds cleave damage when target_count >= 2
- Sword Spec procs generate extra MH hits
- Sword Spec 500ms ICD is respected
- Combat Potency procs on OH auto hits
- Poison procs on melee hits
- Deadly Poison stacks accumulate
- Adrenaline Rush doubles energy regen
- Rupture ticks deal bleed damage (ignores armor)
- Proc-on-proc: Sword Spec can trigger WF
- Proc-on-proc: WF cannot trigger another WF
- Proc-on-proc: Yellow hits don't trigger WF
- Result aggregation produces correct mean/std
- DPS timeline has correct bucket count
- Resource stats track energy waste

**Run:** `python3 -m pytest tests/unit/sim/test_combat.py -v`

---

## Step 9: Runner + Public API

**Files:**
- Create: `code/shukketsu/sim/runner.py`
- Create: `tests/unit/sim/test_runner.py`

**What to build:**

Public API from Section 9 of the design doc.

**SimRunner class:**
```python
class SimRunner:
    def __init__(self, item_db: ItemDatabase | None = None) -> None: ...

    async def sim_run(self, config: SimConfig) -> SimResult: ...
    async def sim_compare(self, config_a: SimConfig, config_b: SimConfig) -> CompareResult: ...
    async def sim_optimize(self, config: SimConfig, slot: GearSlot, *, top_n: int = 5, phase: int = 5) -> OptimizeResult: ...
    async def stat_weights(self, config: SimConfig, *, delta: int | None = None) -> list[StatWeight]: ...

    def build_config_from_import(self, raw: str, **overrides: Any) -> SimConfig: ...
    def swap_item(self, config: SimConfig, slot: GearSlot, item_query: str) -> SimConfig: ...

    def _build_simulation(self, config: SimConfig) -> CombatSimulation: ...
    def _pre_filter_candidates(self, config: SimConfig, slot: GearSlot, phase: int) -> list[Item]: ...
```

**Additional result models:**
- `StatDiff(BaseModel)` — stat_name, before, after, delta
- `AbilityDiff(BaseModel)` — ability_name, dps_before, dps_after, delta, delta_pct
- `CompareResult(BaseModel)` — dps_before, dps_after, dps_delta, dps_delta_pct, stat_changes (list[StatDiff]), ability_changes (list[AbilityDiff]), summary (str)
- `ItemRecommendation(BaseModel)` — item_name, item_id, dps, dps_delta, source, phase
- `OptimizeResult(BaseModel)` — current_item, current_dps, recommendations (list[ItemRecommendation])

**Key behaviors:**
1. `sim_run`: Builds CombatSimulation, runs via `asyncio.to_thread` (CPU-bound). Constructs TalentModifiers, resolves buffs, creates RotationEngine, creates CombatSimulation, calls `.run()`.
2. `sim_compare`: Runs two configs in parallel via `asyncio.gather(sim_run(a), sim_run(b))`. Diffs the results.
3. `sim_optimize`: Pre-filters candidates for slot using EP approximation, sims top ~20, returns ranked.
4. `stat_weights`: Delta-sim — run base config, then re-run with +delta of each stat (hit, crit, haste, AP, agi, str, ArP, expertise, weapon DPS). Normalize to AP = 1.0 EP. Flag capped stats.
5. `swap_item`: Fuzzy search item_db, return new config with item swapped in slot.
6. `_pre_filter_candidates`: Compute approximate EP for each candidate item vs current, take top 20.

**Tests (~15):**
- `sim_run` returns valid SimResult with non-zero DPS
- `sim_run` with different configs returns different DPS
- `sim_compare` returns CompareResult with correct delta sign
- `sim_compare` two identical configs → delta ~0
- `sim_optimize` returns ranked recommendations
- `stat_weights` returns EP values for all stats
- `stat_weights` flags hit cap when at cap
- `build_config_from_import` with SimC format produces valid config
- `swap_item` changes the correct slot
- `swap_item` with unknown item raises error
- `_pre_filter_candidates` returns <= 20 items
- Async execution doesn't block event loop

**Run:** `python3 -m pytest tests/unit/sim/test_runner.py -v`

---

## Step 10: Agent Tools + Analyst Agent

**Files:**
- Create: `code/shukketsu/tools/analysis/sim_run.py`
- Create: `code/shukketsu/tools/analysis/sim_compare.py`
- Create: `code/shukketsu/tools/analysis/sim_optimize.py`
- Create: `code/shukketsu/tools/analysis/__init__.py`
- Create: `code/shukketsu/agents/analyst.py`
- Create: `code/shukketsu/llm/prompts/analyst.py`
- Modify: `code/shukketsu/agents/tasks.py` (add AgentRole.ANALYST, AnalysisTask, AnalysisResult)
- Modify: `code/shukketsu/agents/factory.py` (register AnalystAgent)
- Modify: `code/shukketsu/config.py` (ANALYST_SYSTEM_PROMPT import)
- Create: `tests/unit/sim/test_tools.py`
- Create: `tests/unit/agents/test_analyst.py`

**What to build:**

### Tool schemas (following `tools/schemas.py` + `tools/registry.py` pattern):

**SimRunTool** (`tools/analysis/sim_run.py`):
- `name = "sim_run"`, description for LLM
- `parameters_schema`: spec, talents, gear_overrides, buff_preset, boss_armor, fight_length, iterations, compute_stat_weights
- `execute()`: Builds SimConfig, calls `SimRunner.sim_run()`, formats result as observation string

**SimCompareTool** (`tools/analysis/sim_compare.py`):
- `name = "sim_compare"`, description for LLM
- `parameters_schema`: swap_slot, swap_item, OR change_type + change_value
- `execute()`: Builds two configs, calls `SimRunner.sim_compare()`, formats as observation

**SimOptimizeTool** (`tools/analysis/sim_optimize.py`):
- `name = "sim_optimize"`, description for LLM
- `parameters_schema`: slot, phase, top_n
- `execute()`: Calls `SimRunner.sim_optimize()`, formats as observation

### Analyst Agent (`agents/analyst.py`):

Following `agents/researcher.py` pattern:

```python
class AnalystAgent(BaseAgent):
    """Specialist agent for quantitative DPS analysis using the sim engine."""

    async def execute(self, task: AnalysisTask, ...) -> AnalysisResult:
        outcome = await self._run_loop(...)
        # Post-process: extract key numbers from tool results
        return AnalysisResult(
            output=outcome.output,
            dps_mean=...,
            stat_weights=...,
            trajectory=[...],
        )
```

### System Prompt (`llm/prompts/analyst.py`):

The full analyst system prompt from the analyst-agent design doc (Section: System Prompt). Key principles: always sim before claiming, explain WHY not just WHAT, reference specific mechanics, show numbers, consider current gear, explain trade-offs, flag stat caps.

### Task Protocol additions (`agents/tasks.py`):

- Add `ANALYST = "analyst"` to `AgentRole` enum
- `AnalysisTask(AgentTask)` — import_string, comparison_mode, optimization_slot, etc.
- `AnalysisResult(AgentResult)` — dps_mean, stat_weights, comparison, recommendations

### Factory registration (`agents/factory.py`):

Add to `_ROLE_PROMPTS`, `_ROLE_MAX_ITERATIONS`, `_ROLE_CLASSES` dicts.

**Tests (~20):**

Tools (~10):
- SimRunTool schema validates correct input
- SimRunTool schema rejects invalid spec
- SimRunTool execute returns formatted observation string
- SimCompareTool parses swap_slot + swap_item correctly
- SimOptimizeTool formats ranked results correctly
- Tools register correctly in ToolRegistry

Agent (~10):
- AnalystAgent initializes with correct role
- AnalystAgent registered in factory
- AgentRole.ANALYST exists in enum
- AnalysisTask/AnalysisResult models validate
- Analyst system prompt is non-empty and contains key phrases
- Factory creates AnalystAgent with correct tools
- Orchestrator can dispatch to Analyst for ANALYSIS tasks

**Run:** `python3 -m pytest tests/unit/sim/test_tools.py tests/unit/agents/test_analyst.py -v`

---

## Step 11: Sim Web UI

**Files:**
- Create: `code/shukketsu/web/routers/sim.py`
- Create: `code/shukketsu/web/templates/sim/index.html`
- Create: `code/shukketsu/web/templates/sim/partials/gear_table.html`
- Create: `code/shukketsu/web/templates/sim/partials/results_panel.html`
- Create: `code/shukketsu/web/templates/sim/partials/import_form.html`
- Create: `code/shukketsu/web/templates/sim/partials/buff_toggles.html`
- Create: `code/shukketsu/web/templates/sim/partials/stat_weights.html`
- Create: `code/shukketsu/web/templates/sim/partials/ability_breakdown.html`
- Create: `code/shukketsu/web/static/js/sim-charts.js`
- Modify: `code/shukketsu/web/app.py` (register sim router)
- Modify: `code/shukketsu/web/templates/base.html` (add Sim nav link)
- Create: `tests/unit/web/test_sim_routes.py`

**What to build:**

From the sim-ui design doc (`2026-02-13-phase4-sim-ui.md`).

### Routes (`web/routers/sim.py`):

**Page routes (HTML):**
- `GET /sim/` — Render sim page with empty state
- `POST /sim/import` — Parse import string, return gear_table partial
- `POST /sim/run` — Run sim, return results_panel partial
- `POST /sim/swap/{slot}` — Swap item, return updated gear row

**API routes (JSON, for agent tools):**
- `POST /api/sim/run` — JSON API wrapping SimRunner.sim_run
- `POST /api/sim/compare` — JSON API wrapping SimRunner.sim_compare
- `POST /api/sim/optimize` — JSON API wrapping SimRunner.sim_optimize
- `GET /api/sim/items/{slot}` — Search items for dropdown
- `GET /api/sim/presets` — List buff presets

### Templates:

**`sim/index.html`** — Full page layout (extends base.html):
1. Import bar (textarea + submit button, HTMX post to /sim/import)
2. Character header (spec, race, talents — populated from import)
3. Gear table (17 rows, one per slot, with swap buttons)
4. Buff configuration (preset selector + individual toggles)
5. Run button + results area

**Partials** — HTMX fragments returned by POST endpoints:
- `gear_table.html`: Slot name, item name + ilvl, [Swap] button
- `results_panel.html`: DPS summary, Chart.js placeholders for charts
- `import_form.html`: Success/error feedback
- `buff_toggles.html`: Preset selector + individual checkbox toggles
- `stat_weights.html`: Stat weight table
- `ability_breakdown.html`: Ability pie chart + table

### Chart.js (`sim-charts.js`):
- `renderAbilityPie(canvasId, breakdownData)` — Pie chart for ability DPS breakdown
- `renderStatWeightBar(canvasId, weightData)` — Horizontal bar chart for stat weights
- `renderDpsHistogram(canvasId, distributionData)` — Histogram for DPS spread

### App registration:
- Add `from code.shukketsu.web.routers.sim import router as sim_router` to `app.py`
- `app.include_router(sim_router)`
- Add "Sim" link to base.html navigation

**Tests (~15):**
- `GET /sim/` returns 200 with sim page content
- `POST /sim/import` with valid SimC string returns gear table
- `POST /sim/import` with invalid string returns error
- `POST /sim/run` returns results panel with DPS data
- `POST /sim/swap/trinket_1` returns updated row
- `POST /api/sim/run` returns JSON SimResult
- `POST /api/sim/compare` returns JSON CompareResult
- `POST /api/sim/optimize` returns JSON OptimizeResult
- `GET /api/sim/items/trinket_1` returns item list
- `GET /api/sim/presets` returns preset names
- Sim page template renders without errors
- Chart.js data format is correct in response
- Navigation includes "Sim" link

**Run:** `python3 -m pytest tests/unit/web/test_sim_routes.py -v`

---

## Step 12: Validation + Integration Tests

**Files:**
- Create: `code/shukketsu/sim/validation.py`
- Create: `tests/unit/sim/test_validation.py`
- Create: `tests/integration/sim/test_sim_integration.py`
- Create: `tests/integration/sim/__init__.py`

**What to build:**

### Validation module (`sim/validation.py`):

**ValidationProfile** model:
```python
class ValidationProfile(BaseModel):
    name: str
    spec: RogueSpec
    phase: int
    config: SimConfig
    expected_dps_range: tuple[float, float]  # (min, max) from design doc
    tolerance_pct: float = 2.0  # 4.0 for Mutilate profiles
```

**VALIDATION_PROFILES** — 10 canonical profiles from the validation design doc:
1. P1 BiS Combat Swords (1200-1400 DPS)
2. P2 BiS Combat Swords (1500-1700 DPS)
3. P3 BiS Combat Swords (1800-2000 DPS)
4. P5 BiS Combat Swords (2200-2500 DPS)
5. P1 BiS Combat Daggers
6. P3 BiS Combat Fists
7. P3 BiS Mutilate (4% tolerance)
8. P5 BiS Mutilate (4% tolerance)
9. No WF group (Combat, P3)
10. Solo / no buffs (Combat, P3)

**Functions:**
- `get_validation_profiles() -> list[ValidationProfile]` — Returns all profiles
- `run_validation(runner: SimRunner, profile: ValidationProfile) -> ValidationResult` — Runs sim and checks against expected range
- `ValidationResult(BaseModel)` — profile_name, our_dps, expected_range, within_range (bool), drift_pct

### Sanity Tests (`test_validation.py`):
- Each validation profile has a valid SimConfig
- Profile configs can be loaded by SimRunner
- Expected DPS ranges are reasonable (> 500, < 5000)

### Integration Tests (`test_sim_integration.py`):
- Full sim_run end-to-end: P1 BiS Combat Swords produces DPS in expected range
- Full sim_compare: DST vs Brooch produces positive delta for DST
- Stat weights: Hit below cap has higher EP than hit above cap
- Character import → sim_run produces reasonable DPS
- Analyst agent tool → sim_run → formatted observation
- **All marked `@pytest.mark.integration`** (require item database)

**Tests (~15):**

Validation unit (~5):
- All 10 profiles load without error
- Profile configs pass SimConfig validation
- Expected ranges are reasonable
- Tolerance values are correct (2% for combat, 4% for mutilate)
- ValidationResult model validates

Integration (~10):
- P1 BiS Combat Swords DPS in range [1000, 1600]
- P3 BiS Combat Swords DPS in range [1600, 2200]
- sim_compare: DST > Brooch (dps_delta > 0)
- stat_weights: all EP values > 0
- stat_weights: AP EP ~1.0 (normalization check)
- SimC import → sim_run → reasonable DPS
- Analyst tool → observation string contains DPS number
- Rotation doesn't deadlock (10-second timeout on sim_run)
- Different iteration counts produce similar mean DPS (convergence)
- Blade Flurry on 2-target cleave fight > single target

**Run:**
```bash
python3 -m pytest tests/unit/sim/test_validation.py -v
python3 -m pytest tests/integration/sim/ -v -m integration
```

---

## Dependency Graph

```
Step 1 (Models/Config/Errors)
  ├── Step 2 (Mechanics)
  ├── Step 3 (Abilities)
  ├── Step 4 (Talents)
  ├── Step 5 (Items)
  └── Step 6 (Buffs + Imports)
        │
        └── Step 7 (Rotation) ← depends on 3, 4, 6
              │
              └── Step 8 (Combat Loop) ← depends on 2, 3, 4, 5, 6, 7
                    │
                    └── Step 9 (Runner API) ← depends on 8
                          │
                          ├── Step 10 (Agent + Tools)
                          ├── Step 11 (Web UI)
                          └── Step 12 (Validation + Integration)
```

**Parallelizable batches:**
- Batch 1: Step 1 (foundation)
- Batch 2: Steps 2, 3, 4, 5, 6 (all depend only on Step 1)
- Batch 3: Step 7 (rotation, depends on 3+4+6)
- Batch 4: Step 8 (combat loop, depends on everything)
- Batch 5: Step 9 (runner, depends on 8)
- Batch 6: Steps 10, 11, 12 (all depend only on 9)

---

## Estimated Test Growth

| Step | New Tests | Running Total |
|------|-----------|---------------|
| 1. Models + Config + Errors | ~35 | ~813 |
| 2. Combat Mechanics | ~25 | ~838 |
| 3. Abilities Database | ~15 | ~853 |
| 4. Talent System | ~15 | ~868 |
| 5. Item Database | ~20 | ~888 |
| 6. Buffs + Imports | ~25 | ~913 |
| 7. Rotation Engine | ~20 | ~933 |
| 8. Combat Event Loop | ~25 | ~958 |
| 9. Runner + API | ~15 | ~973 |
| 10. Agent + Tools | ~20 | ~993 |
| 11. Web UI | ~15 | ~1008 |
| 12. Validation + Integration | ~15 | ~1023 |

**Starting count:** 778 tests (Phase 3 complete)
**Estimated final:** ~1023 tests

---

## Phase Gate Checklist

After all 12 steps:

- [ ] P1 BiS Combat Swords sim produces DPS in range [1000, 1600]
- [ ] P5 BiS Combat Swords sim produces DPS in range [2000, 2800]
- [ ] P3 BiS Mutilate sim produces DPS in range [1400, 2000]
- [ ] Analyst agent can answer "what's my best trinket?" end-to-end
- [ ] Character import works for SimC format
- [ ] `/sim` page renders gear overview and sim results
- [ ] Stat weights computed and displayed
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] `ruff check` and `ruff format` clean
- [ ] `mypy` passes
