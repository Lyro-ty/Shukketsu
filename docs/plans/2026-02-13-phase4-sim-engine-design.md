# Phase 4: TBC Rogue DPS Simulation Engine — Design Document

## Overview

A discrete-event DPS simulation engine for TBC 2.4.3 Rogues, modeled after WoWSims TBC Rogue. The key differentiator is an AI Analyst agent that can autonomously run sims, swap gear, compare setups, and explain results in natural language.

## Scope Decisions

| Area | Decision |
|------|----------|
| Item data source | WoWSims GitHub database + addon import |
| Sim engine | Pure Python discrete-event, WoWSims WASM as validation oracle |
| Specs | Combat (Swords/Fists/Daggers) + Assassination Mutilate |
| Buffs | Full mechanics under the hood, preset raid profiles as UX |
| Agent tools | `sim_run`, `sim_compare`, `sim_optimize` |
| Import formats | SimC addon, Seventyupgrades JSON, WoWSims Base64 — auto-detect |
| Output | Full analysis package (DPS, breakdown, stat weights, timelines, uptimes) |
| UI | Hybrid: lightweight `/sim` page for setup + chat for iteration |
| Validation | Triple: automated 2% regression vs WoWSims + real WCL log comparison |

## Module Architecture

```
sim/
├── __init__.py
├── models.py       # All Pydantic data models (enums, configs, results)
├── mechanics.py    # Combat formulas: hit tables, crit, armor, glancing
├── abilities.py    # Ability database: damage, cost, cooldown, coefficients
├── talents.py      # Talent trees: point allocation → modifier calculation
├── items.py        # Item database: WoWSims import, query API, character import
├── buffs.py        # Raid buffs, consumables, debuffs, preset profiles
├── rotation.py     # Per-spec priority system (Combat, Mutilate)
├── combat.py       # Event loop: processes events in timestamp order
├── runner.py       # Public API: SimConfig → SimResult, agent tools
├── validation.py   # WoWSims WASM oracle + WCL log comparison
└── data/           # Converted WoWSims item/ability JSON files
    ├── items.json
    ├── enchants.json
    ├── gems.json
    └── sets.json
```

---

## 1. Data Models (`models.py`)

### Enums

```python
class WeaponType(StrEnum):
    SWORD = "sword"
    DAGGER = "dagger"
    FIST = "fist"
    MACE = "mace"

class RogueSpec(StrEnum):
    COMBAT_SWORDS = "combat_swords"
    COMBAT_FISTS = "combat_fists"
    COMBAT_DAGGERS = "combat_daggers"
    ASSASSINATION_MUTILATE = "assassination_mutilate"

class GearSlot(StrEnum):
    HEAD = "head"
    NECK = "neck"
    SHOULDER = "shoulder"
    BACK = "back"
    CHEST = "chest"
    WRIST = "wrist"
    HANDS = "hands"
    WAIST = "waist"
    LEGS = "legs"
    FEET = "feet"
    RING_1 = "ring_1"
    RING_2 = "ring_2"
    TRINKET_1 = "trinket_1"
    TRINKET_2 = "trinket_2"
    MAIN_HAND = "main_hand"
    OFF_HAND = "off_hand"
    RANGED = "ranged"

class Race(StrEnum):
    HUMAN = "human"
    ORC = "orc"
    NIGHT_ELF = "night_elf"
    BLOOD_ELF = "blood_elf"
    UNDEAD = "undead"
    DWARF = "dwarf"
    GNOME = "gnome"
    TROLL = "troll"

class PoisonType(StrEnum):
    INSTANT = "instant_poison"
    DEADLY = "deadly_poison"
    WOUND = "wound_poison"
    ANESTHETIC = "anesthetic_poison"
    NONE = "none"

class OpenerAbility(StrEnum):
    GARROTE = "garrote"
    AMBUSH = "ambush"
    CHEAP_SHOT = "cheap_shot"
    NONE = "none"

class ProcTrigger(StrEnum):
    ON_HIT = "on_hit"
    PPM = "ppm"
    ON_CRIT = "on_crit"
    ON_USE = "on_use"

class FightType(StrEnum):
    PATCHWERK = "patchwerk"
    CLEAVE = "cleave"
    MOVEMENT = "movement"

class GemSlot(StrEnum):
    RED = "red"
    YELLOW = "yellow"
    BLUE = "blue"
    META = "meta"
```

### Item/Gear Database Models

```python
class WeaponStats(BaseModel):
    min_damage: float
    max_damage: float
    speed: float
    weapon_type: WeaponType
    dps: float

class ProcEffect(BaseModel):
    trigger: ProcTrigger
    rate: float                 # % chance or PPM value
    icd: float = 0.0           # internal cooldown in seconds
    duration: float = 0.0
    effect: dict[str, float]   # {"attack_power": 325} etc.
    stacks: int = 1

class SetBonus(BaseModel):
    set_name: str
    pieces_required: int
    effect: dict[str, float]

class Item(BaseModel):
    id: int
    name: str
    slot: GearSlot
    item_level: int
    phase: int
    stats: dict[str, float]
    sockets: list[GemSlot] = []
    socket_bonus: dict[str, float] = {}
    set_id: str | None = None
    weapon: WeaponStats | None = None
    proc: ProcEffect | None = None
    on_use: ProcEffect | None = None

class Enchant(BaseModel):
    id: int
    name: str
    slot: GearSlot
    stats: dict[str, float] = {}
    proc: ProcEffect | None = None

class Gem(BaseModel):
    id: int
    name: str
    color: GemSlot
    stats: dict[str, float]
    meta_condition: str | None = None
```

### Sim Config (Input)

```python
class PoisonConfig(BaseModel):
    main_hand: PoisonType = PoisonType.INSTANT
    off_hand: PoisonType = PoisonType.DEADLY

class BossConfig(BaseModel):
    name: str = "Raid Boss"
    level: int = 73
    armor: int = 7700
    debuffs: list[str] = []

class SimConfig(BaseModel):
    spec: RogueSpec
    race: Race = Race.ORC
    talents: str                          # "20/41/0" format
    gear: dict[GearSlot, int]             # slot -> item_id
    enchants: dict[GearSlot, int] = {}
    gems: dict[GearSlot, list[int]] = {}
    poisons: PoisonConfig = PoisonConfig()
    opener: OpenerAbility = OpenerAbility.GARROTE
    expose_armor: bool = False
    use_premeditation: bool = False
    buffs: list[str] = []
    consumables: list[str] = []
    boss: BossConfig = BossConfig()
    fight_type: FightType = FightType.PATCHWERK
    target_count: int = 1
    fight_length: int = 300
    iterations: int = 10000
    raid_preset: str = "full_25man"
    latency_ms: int = 0
```

### Sim Result (Output)

```python
class AbilityBreakdown(BaseModel):
    name: str
    damage_total: float
    damage_pct: float
    casts: float
    hit_pct: float
    crit_pct: float
    miss_pct: float
    dodge_pct: float
    glancing_pct: float = 0.0

class ProcUptime(BaseModel):
    name: str
    source: str
    uptime_pct: float
    avg_procs_per_fight: float
    avg_stacks: float = 1.0

class PoisonStats(BaseModel):
    poison_type: PoisonType
    hand: str
    procs_per_fight: float
    damage_total: float
    damage_pct: float
    avg_deadly_stacks: float = 0.0

class CooldownUsage(BaseModel):
    name: str
    casts_per_fight: float
    avg_uptime_pct: float

class ResourceStats(BaseModel):
    energy_per_second: float
    avg_energy_waste: float
    combo_points_per_second: float
    combo_point_overcap_pct: float
    gcd_utilization_pct: float
    energy_starved_pct: float

class StatWeight(BaseModel):
    stat: str
    ep_value: float
    dps_per_point: float
    is_capped: bool = False

class DpsTimeline(BaseModel):
    bucket_seconds: int = 5
    buckets: list[float]

class SimResult(BaseModel):
    dps_mean: float
    dps_std: float
    dps_median: float
    dps_min: float
    dps_max: float
    iterations: int
    fight_length: int
    ability_breakdown: list[AbilityBreakdown]
    stat_weights: list[StatWeight]
    proc_uptimes: list[ProcUptime]
    poison_stats: list[PoisonStats]
    cooldown_usage: list[CooldownUsage]
    buff_uptimes: dict[str, float]
    resource_stats: ResourceStats
    dps_timeline: DpsTimeline
    dps_distribution: list[float]
    config: SimConfig
```

---

## 2. Combat Mechanics (`mechanics.py`)

All formulas sourced from WoWSims Go source code (`wowsims/tbc`) and verified against community research (Elitist Jerks, magey/tbc-warrior, ShadowPanther).

### Constants

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
BASE_PARRY_CHANCE = 0.14           # 0% from behind
BASE_GLANCING_CHANCE = 0.24
GLANCING_MULTIPLIER = 0.75         # simplified from [0.65, 0.85]
CRIT_SUPPRESSION = 0.048           # 3.0% skill + 1.8% aura

# Armor
ARMOR_CONSTANT = 10557.5           # 467.5 * 70 - 22167.5
MAX_ARMOR_REDUCTION = 0.75

# Weapon normalization
NORM_SPEED_DAGGER = 1.7
NORM_SPEED_ONE_HAND = 2.4
NORM_SPEED_TWO_HAND = 3.3
NORM_SPEED_RANGED = 2.8

# Energy
ENERGY_TICK_MS = 2020              # 2.02 seconds
ENERGY_PER_TICK = 20.2             # Blizzard quirk
BASE_MAX_ENERGY = 100
VIGOR_BONUS_ENERGY = 10

# Crit multipliers
MELEE_CRIT_MULTIPLIER = 2.0
SPELL_CRIT_MULTIPLIER = 1.5

# Rogue
ROGUE_THREAT_MULTIPLIER = 0.71
ROGUE_BASE_CRIT_ADJUSTMENT = -0.003
BUILDER_MISS_REFUND = 0.80

# Common boss armor
BOSS_ARMOR_STANDARD = 7700
BOSS_ARMOR_CASTER = 6200
BOSS_ARMOR_HEAVY = 8800

# Armor debuffs
SUNDER_ARMOR_PER_STACK = 520
EXPOSE_ARMOR_BASE = 2050
FAERIE_FIRE_ARMOR = 610
CURSE_OF_RECKLESSNESS_ARMOR = 800
```

### Hit Tables

**White hits (auto-attacks) = Single-roll system.** One random number, outcomes stacked: miss -> dodge -> glance -> crit -> hit. Crit CAN be pushed off the table.

**Yellow hits (specials) = Two-roll system.** Roll 1 for miss/dodge, roll 2 for crit. Crit CANNOT be pushed off.

### Key Functions

- `resolve_white_hit(miss, dodge, crit, roll) -> outcome`
- `resolve_yellow_hit(miss, dodge, crit, hit_roll, crit_roll, can_be_dodged) -> outcome`
- `calc_miss_chance(hit_rating, is_dw, is_yellow, precision) -> float`
- `calc_dodge_chance(expertise_rating, bonus) -> float`
- `calc_crit_chance(crit_rating, agi, base_crit, talent_crit, bonus_crit) -> float`
- `calc_crit_multiplier(base, primary_mod, secondary_mod, has_meta) -> float`
- `calc_armor_reduction(armor, arpen, sunder, ea, ff, cor) -> float`
- `calc_weapon_damage(min, max, speed, ap, normalized, norm_speed, roll) -> float`
- `calc_haste_multiplier(haste_rating, *multiplicative_buffs) -> float`
- `calc_poison_proc_chance(base, imp_poisons) -> float`
- `calc_ppm_proc_chance(ppm, weapon_speed) -> float`

### Critical formulas

```
# Armor reduction
damage_mult = max(0.25, 1 - effective_armor / (effective_armor + 10557.5))

# Crit multiplier (WoWSims formula)
crit_mult = 1.0 + (base_mult * primary_mod - 1.0) * (1.0 + secondary_mod)
# With Relentless Earthstorm Diamond: primary_mod *= 1.03

# Haste stacking (multiplicative)
total_haste = (1 + haste_rating/1577) * snd_mult * bf_mult * heroism_mult * ...
effective_speed = base_speed / total_haste

# Energy regen
energy_per_tick = 20.2 * energy_tick_multiplier  # 2x during Adrenaline Rush
tick_interval = 2.02 seconds
```

---

## 3. Abilities Database (`abilities.py`)

Static data only — no simulation logic. Each ability defined as a Pydantic model.

### Ability Flags

```python
class AbilityFlag(StrEnum):
    BUILDER = "builder"
    FINISHER = "finisher"
    NORMALIZED = "normalized"
    CANNOT_BE_DODGED = "cannot_be_dodged"
    APPLIES_LETHALITY = "applies_lethality"
    PHYSICAL = "physical"
    NATURE = "nature"
    IGNORES_ARMOR = "ignores_armor"
    MAIN_HAND = "main_hand"
    OFF_HAND = "off_hand"
    SNAPSHOT = "snapshot"
    OFF_GCD = "off_gcd"
```

### Builders

| Ability | SpellID | Energy | Flat Dmg | Wpn Mult | Normalized | CP | Notes |
|---------|---------|--------|----------|----------|------------|-----|-------|
| Sinister Strike | 26862 | 45/42/40 | +98 | 1.0 | Yes (2.4) | 1 | Lethality applies |
| Backstab | 26863 | 60 | +170 | 1.5+SC | Yes (1.7) | 1 | +10%/rank PW crit, Lethality |
| Mutilate | 34413 | 60 | +101/sub | 1.0 | Yes (1.7) | 2 | 1.5x if DP active, dual sub-hits, Lethality (WoWSims) |
| Hemorrhage | 26864 | 35 | 0 | 1.1+SC | Yes | 1 | +42 phys debuff (10 charges, 15s), Lethality |
| Shiv | 5938 | 20+10*OH_spd | 0 | 1.0 | No | 1 | Can't be dodged, guarantees poison |
| Ambush | 27611 | 60 | +290 | 2.5 | Yes (1.7) | 1 | Stealth + dagger MH required |
| Garrote | 26884 | 50 | DoT | — | — | 1 | Bleed: 672 over 18s (6 ticks) |
| Cheap Shot | 1833 | 60 | 0 | — | — | 2 | Stealth, stun 4s, no damage |

All builders have 80% energy refund on miss. Sinister Calling adds +0.01/rank to Backstab and Hemorrhage weapon multipliers.

### Finishers

| Ability | SpellID | Energy | Formula | Notes |
|---------|---------|--------|---------|-------|
| Eviscerate | 26865 | 35 | 60 + 185*CP + rand(0,120) + AP*0.03*CP + BonusWepDmg | Physical, +ImpEvis +Aggression |
| Envenom | 32684 | 35 | 60 + 180*CP + AP*0.03*CP | Nature, ignores armor, no Lethality, consumes DP stacks (1/CP) |
| Rupture | 26867 | 25 | tick: 70 + CP*11 + AP*coeff[CP] | Bleed, snapshot, ticks=CP+3, AP coeff=[0.01,0.02,0.03,0.03,0.03] |
| Slice and Dice | 6774 | 25 | — | 30% haste (35% T6 2pc), dur=[9,12,15,18,21]s * ImpSND mult |
| Expose Armor | 26866 | 25 | — | 2050*(1+0.25*ImpEA) armor reduction, 30s, exclusive with Sunder |

T4 4pc: Evis/Envenom cost -10 energy. Deathmantle 2pc: +40 per CP on Evis/Envenom.

### Cooldowns

| Ability | SpellID | Energy | CD | Duration | Effect | GCD |
|---------|---------|--------|----|----------|--------|-----|
| Blade Flurry | 13877 | 25 | 2m | 15s | +20% haste, cleave to 2nd target | Yes |
| Adrenaline Rush | 13750 | 0 | 5m | 15s | 2x energy regen, resets energy tick | No |
| Cold Blood | 14177 | 0 | 3m | — | +100% crit on next ability | No |
| Thistle Tea | 7676 | 0 | 5m | — | Restores 100 energy | No |
| Premeditation | 14183 | 0 | 20s | — | +2 CP from stealth | No |

### Poisons

| Poison | SpellID | Proc Chance | Damage | Notes |
|--------|---------|-------------|--------|-------|
| Instant VII | 26891 | 20% + 2%/ImpP | 146-194 nature | Can crit, +4%/VilePoisons, +5%/MasterPoisoner spell hit |
| Deadly VII | 27186 | 30% + 2%/ImpP | 45/stack/tick (4 ticks, 12s, max 5 stacks) | +4%/VP, +5%/MP spell hit |
| Wound V | 27189 | 50% | 65 nature | -10% healing/stack (PvP) |

WF Totem blocks MH poison application.

### Known Discrepancies

- **Envenom DP consumption**: WoWSims does NOT consume DP stacks. Real game consumes 1 stack per CP. We model game-accurate behavior and accept WoWSims divergence for Mutilate.
- **Thistle Tea**: WoWSims uses 40 energy. Game data says 100 energy. We use 100 (game-accurate).
- **Mutilate + Lethality**: WoWSims applies Lethality to Mutilate sub-hits despite the tooltip not listing it. We match WoWSims.

---

## 4. Talents (`talents.py`)

### Assassination Tree (relevant talents)

| Talent | Max Ranks | Effect per Rank |
|--------|-----------|-----------------|
| Improved Eviscerate | 3 | +5% Evis damage |
| Malice | 5 | +1% crit |
| Ruthlessness | 3 | 20% chance for 1 CP after finisher |
| Murder | 2 | +1% damage vs Humanoid/Beast/Giant/Dragonkin |
| Puncturing Wounds | 3 | +10% BS crit, +5% Mut crit |
| Relentless Strikes | 1 | 20% per CP chance for 25 energy on finisher |
| Lethality | 5 | +6% crit damage (secondary modifier) on SS/BS/Hemo/Mut |
| Vile Poisons | 5 | +4% poison and Envenom damage |
| Improved Poisons | 5 | +2% proc chance for IP and DP |
| Cold Blood | 1 | Enables Cold Blood |
| Quick Recovery | 2 | 40% finisher energy refund on miss |
| Seal Fate | 5 | 20% chance for extra CP when builder crits |
| Master Poisoner | 2 | +5% spell hit for poisons |
| Vigor | 1 | +10 max energy |
| Find Weakness | 5 | +2% damage on abilities for 10s after finisher |
| Mutilate | 1 | Enables Mutilate |

### Combat Tree (relevant talents)

| Talent | Max Ranks | Effect per Rank |
|--------|-----------|-----------------|
| Improved Sinister Strike | 2 | -3/-5 energy cost (45 -> 42 -> 40) |
| Improved Slice and Dice | 3 | +15% SND duration |
| Precision | 5 | +1% hit |
| Dual Wield Specialization | 5 | +10% OH auto-attack damage |
| Blade Flurry | 1 | Enables Blade Flurry |
| Dagger Specialization | 5 | +1% crit with daggers |
| Fist Weapon Specialization | 5 | +1% crit with fists |
| Sword Specialization | 5 | 1% chance for extra MH attack (500ms ICD) |
| Mace Specialization | 5 | +1% crit damage multiplier with maces |
| Weapon Expertise | 2 | +5 expertise |
| Aggression | 3 | +2% damage to SS/BS/Evis |
| Vitality | 2 | +1% agi (mult), +2% stam (mult) |
| Adrenaline Rush | 1 | Enables Adrenaline Rush |
| Combat Potency | 5 | 20% on OH hit -> +3 energy |
| Surprise Attacks | 1 | +10% SS/BS/Shiv, finishers can't be dodged |

### Subtlety Tree (relevant talents)

| Talent | Max Ranks | Effect per Rank |
|--------|-----------|-----------------|
| Opportunity | 5 | +4% BS and Mut damage |
| Serrated Blades | 3 | +186 ArP, +10% Rupture damage |
| Hemorrhage | 1 | Enables Hemorrhage |
| Premeditation | 1 | Enables Premeditation |
| Sinister Calling | 5 | +3% agi (mult), +1% BS/Hemo weapon mult |
| Deadliness | 5 | +2% AP (mult) |

### `compute_modifiers(allocation) -> TalentModifiers`

Pure function that takes talent point allocation and returns a frozen modifier struct. Called once at sim setup. Set bonuses applied separately after this function returns.

### Multiplicative Stat Stacking

Vitality (+1% agi/rank), Sinister Calling (+3% agi/rank), and Blessing of Kings (+10% all stats) all stack **multiplicatively**: `total_agi = base_agi * vitality_mult * sinister_calling_mult * kings_mult`.

---

## 5. Items Database (`items.py`)

### Data Source

Primary: WoWSims `wowsims/tbc` GitHub repo (`sim/core/items/`). Items stored as Go structs, converted to our JSON format via an offline build script. The converter parses Go struct literals with regex and emits `items.json`, `enchants.json`, `gems.json`, `sets.json`.

### ItemDatabase API

```python
class ItemDatabase:
    def get_item(item_id: int) -> Item | None
    def get_enchant(enchant_id: int) -> Enchant | None
    def get_gem(gem_id: int) -> Gem | None
    def get_set_bonuses(set_name: str) -> list[SetBonus]
    def items_for_slot(slot, phase=None, min_ilvl=0) -> list[Item]
    def search(query: str, slot=None) -> list[Item]  # fuzzy name search
    def rogue_items_for_slot(slot, spec, phase=5) -> list[Item]  # pre-filtered
```

### Rogue Set Bonuses (hardcoded)

| Set | Pieces | Effect |
|-----|--------|--------|
| Netherblade (T4) | 2pc | SND +3s (before ImpSND multiplier) |
| Netherblade (T4) | 4pc | 15% chance for extra CP on finisher |
| Deathmantle (T5) | 2pc | Evis/Envenom +40 damage per CP |
| Deathmantle (T5) | 4pc | 1.0 PPM free finisher proc (15s buff) |
| Slayer's Armor (T6) | 2pc | SND haste +5% (30% -> 35%) |
| Slayer's Armor (T6) | 4pc | SS/BS/Mut/Hemo +6% damage |

### Notable Proc Items (manually verified)

| Item | Proc | Rate | ICD | Duration | Effect |
|------|------|------|-----|----------|--------|
| Dragonspine Trophy | On hit | 1.0 PPM | 20s | 10s | +325 haste rating |
| Tsunami Talisman | On crit | 10% | 45s | 10s | +340 AP |
| Madness of the Betrayer | On hit | ~1 PPM | 10s | 10s | +300 ArP |
| Warp-Spring Coil | On hit | ~1 PPM | 30s | 15s | +225 haste rating |
| Mongoose (enchant) | PPM | 1.0/weapon | None | 15s | +120 agi, +2% haste |
| Executioner (enchant) | PPM | 1.0 | None | 15s | +840 ArP |

### Character Import

Three formats with auto-detection:

1. **SimC addon** (`/simc`): Plaintext `key=value` lines. Detected by `=` in first line.
2. **Seventyupgrades**: JSON export. Detected by leading `{`.
3. **WoWSims**: Base64-encoded protobuf URL. Detected by URL prefix or non-JSON/non-key-value.

`detect_format(raw) -> ImportFormat` then `parse_import(raw) -> CharacterImport`.

---

## 6. Buffs & Consumables (`buffs.py`)

### Stacking Rules

Buffs in the same `BuffCategory` are mutually exclusive (highest value wins). Buffs in different categories or `UNCATEGORIZED` always stack.

| Category | Example Buffs |
|----------|--------------|
| ATTACK_POWER | BoM vs Battle Shout |
| STATS_PCT | Blessing of Kings |
| AGILITY_FLAT | Grace of Air Totem |
| STRENGTH_FLAT | Strength of Earth Totem |
| MELEE_CRIT | Leader of the Pack |
| AP_PCT | Trueshot Aura |
| FLASK | Flask of Relentless Assault |
| BATTLE_ELIXIR | Elixir of Major Agility |
| GUARDIAN_ELIXIR | Elixir of Draenic Wisdom |
| FOOD | Roasted Clefthoof / Warp Burger / Grilled Mudfish |

### Raid Presets

**Full 25-Man**: Kings, Imp Battle Shout, Imp Grace of Air, Imp SoE, Imp MotW, LotP, WF Totem, Trueshot Aura, Heroism, Drums + Sunder 5, FF, CoR + Flask of Relentless Assault, Roasted Clefthoof, Haste Pot

**Karazhan 10-Man**: Kings, Imp BoM, Imp MotW, LotP, Trueshot Aura, Heroism + Sunder 5, FF + Elixir of Major Agi, Grilled Mudfish, Haste Pot

**Solo Target Dummy**: No buffs/debuffs/consumables.

**Custom**: Start empty, toggle individually.

### `resolve_buffs(buff_ids, debuff_ids, consumable_ids) -> ResolvedBuffs`

Applies stacking rules and returns aggregate flat stats, multipliers, active procs, and total boss armor reduction.

### Windfury Totem

20% on MH auto-attack for an extra attack with +445 AP. Blocks MH poison procs while active. Extra attacks can trigger Sword Spec (but not another WF). Modeled as `ProcEffect` with `blocks_poison=True`.

---

## 7. Rotation Logic (`rotation.py`)

### State Machine

Based on WoWSims' plan-based system with 7 states:

```
OPENER -> SLICE_ASAP -> DISPATCH -> {BUILD_FOR_SND, BUILD_FOR_EA, FILL_BEFORE_SND, FILL_BEFORE_EA}
```

**DISPATCH** is the central decision point. Priority order:
1. SND expired -> SLICE_ASAP
2. EA needs refresh and SND outlasts EA -> FILL_BEFORE_EA, else BUILD_FOR_EA
3. SND needs refresh -> BUILD_FOR_SND
4. Have enough CPs -> damage finisher (with energy pooling check)
5. Shiv to refresh DP (Mutilate only, < 2s remaining)
6. Default -> build combo points

### Key Behaviors

- **Energy pooling**: Don't finisher below 50 energy unless SND is about to drop or AR is active (threshold drops to 30 during AR).
- **SND refresh timing**: Buffer = `4.0 - combo_points * 0.8` seconds. More CPs = less buffer needed.
- **Damage finisher choice**: Rupture if not active + fight long enough + single target. Otherwise Evis (Combat) or Envenom (Mutilate).
- **Shiv for DP**: Mutilate only. If DP stacks > 0 and remaining < 2s and energy >= shiv cost, Shiv to prevent losing 1.5x bonus.
- **Builder auto-selection**: Mutilate (if talented + dual daggers) > Backstab (if dagger MH) > SS (fallback).

### Cooldown Timing

- **Adrenaline Rush**: Energy <= 85 (or <= 60 if tick imminent), SND active.
- **Blade Flurry**: SND active, ideally paired with AR.
- **Cold Blood**: Paired with AR or on cooldown.
- **Thistle Tea**: Energy <= max_energy - 100.
- **Haste Potion**: Configurable — "with_heroism", "with_bf", "on_pull".

### RotationContext

Read-only frozen snapshot passed to the engine each decision. Contains: combo_points, energy, all buff/debuff timers, cooldown readiness, fight_remaining, target_count, DP state.

---

## 8. Combat Event Loop (`combat.py`)

### Event Types

```python
MH_AUTO, OH_AUTO, ENERGY_TICK, ABILITY_USE, DOT_TICK,
BUFF_EXPIRE, PROC_TRIGGER, COOLDOWN_USE, POTION_USE,
HEROISM_START, HEROISM_END
```

### Architecture

- **Priority queue** (`heapq` min-heap) of `SimEvent` objects ordered by `(timestamp_ms, priority)`.
- **CombatState** dataclass: mutable state for one iteration (energy, CPs, buff timers, damage trackers).
- **Pre-computed tables**: Miss/dodge/crit/armor computed once in `__init__`, reused across all iterations.
- **Deterministic RNG**: `Random(seed=iteration_number)` per iteration for reproducibility.

### Main Loop

```
for each iteration:
    create fresh CombatState
    schedule initial events (energy tick, MH/OH auto, first GCD, heroism)
    while events remain and time < fight_duration:
        pop next event
        advance time
        dispatch by event type
        if GCD is free: ask rotation engine for next action
    record iteration DPS
aggregate results
```

### Auto-Attack Startup

One weapon gets a random 0-50% delay (matching WoWSims): `delay = random() * swing_duration / 2`. Which weapon is delayed is also random.

### Energy Tick

First tick at random offset [0, 2020ms]. Each tick: `energy = min(max_energy, energy + 20.2 * multiplier)`. Then check if ability is now affordable.

### Proc-on-Proc Rules (post-2.2)

| Source | Can trigger Sword Spec? | Can trigger WF? | Can trigger poisons/gear? |
|--------|------------------------|------------------|--------------------------|
| White hit | Yes | Yes | Yes |
| Yellow hit | Yes | No (MH auto only) | Yes |
| Sword Spec extra | No (self) | Yes | Yes |
| WF extra | Yes | No (self) | Yes |

Sword Spec has 500ms ICD. All extra attacks are MH white hits that can miss/dodge.

### Swing Timer Haste Adjustment

When a haste buff activates (SND, BF) or expires, in-progress swing timers are proportionally adjusted:
```
remaining = swing_at - current_time
new_remaining = remaining / haste_change_ratio
swing_at = current_time + new_remaining
```

### Blade Flurry Cleave

On each melee damage event (if BF active and target_count >= 2): reverse armor on source damage, reapply on cleave target. Modeled as immediate damage to "Blade Flurry" tracker.

---

## 9. Public API & Agent Tools (`runner.py`)

### SimRunner

```python
class SimRunner:
    async sim_run(config: SimConfig) -> SimResult
    async sim_compare(config_a, config_b) -> CompareResult
    async sim_optimize(config, slot=None, top_n=5, phase=5) -> OptimizeResult
    build_config_from_import(raw: str, overrides=None) -> SimConfig
    swap_item(config, slot, item_query: str) -> SimConfig
```

### sim_run

Primary entry point. Runs CombatSimulation in thread pool via `asyncio.to_thread` (CPU-bound). Results cached by JSON-serialized config key.

### sim_compare

Runs two configs in parallel (`asyncio.gather`), returns structured diff:
- **StatDiff**: What stats changed between configs
- **AbilityDiff**: Per-ability DPS change
- **Weight shift**: How stat weights changed
- **Buff uptime changes**: Proc uptimes
- **Natural language summary**: Pre-generated explanation of why DPS changed

### sim_optimize

Pre-filters candidates per slot using approximate EP weights (avoid simming 200+ items), then sims top ~20 candidates. Returns ranked list with DPS deltas.

Pre-filter EP weights are spec-specific rough approximations (e.g., ArP is worth less for Mutilate since Envenom ignores armor).

### Stat Weights

Delta-sim approach: re-run with +N of each stat, measure DPS gain, normalize to AP = 1.0 EP. Runs 9 additional sim runs (one per stat). Flags stats at cap (e.g., hit at yellow cap).

### Config Builders

- `build_config_from_import()`: Auto-detects format (SimC/70u/WoWSims), parses to `CharacterImport`, builds `SimConfig` with preset defaults.
- `swap_item()`: Fuzzy name search against ItemDatabase, returns modified config copy.

---

## 10. Validation (`validation.py`)

### WoWSims WASM Oracle

WoWSims compiles to WASM. We wrap it as a subprocess to get reference DPS values.

**Approach**: Convert our `SimConfig` to WoWSims protobuf format, invoke WASM binary, parse output, compare.

**Regression suite**: 10-20 canonical profiles (P1-P5 BiS for each spec). CI runs both engines, fails if drift > 2%.

### WCL Log Comparison

Parse real Warcraft Logs data for Patchwerk-style fights (Brutallus, Patchwerk). Compare sim predictions against actual parse distributions.

Not a hard gate — real logs have player skill variance, deaths, movement. Used as a realism check to flag when the model diverges significantly from reality.

### Validation Profiles

| Profile | Spec | Phase | Expected DPS Range |
|---------|------|-------|-------------------|
| P1 BiS Combat Swords | combat_swords | 1 | ~1200-1400 |
| P2 BiS Combat Swords | combat_swords | 2 | ~1500-1700 |
| P3 BiS Combat Swords | combat_swords | 3 | ~1800-2000 |
| P5 BiS Combat Swords | combat_swords | 5 | ~2200-2500 |
| P3 BiS Mutilate | assassination_mutilate | 3 | ~1600-1800 |
| P5 BiS Mutilate | assassination_mutilate | 5 | ~2000-2300 |

Phase gate: P1 BiS Combat Swords on Patchwerk-style fight within 2% of WoWSims.

---

## 11. Web UI (`/sim` page)

### Hybrid Approach

Lightweight sim page for setup + chat for smart iteration.

**Sim page (`/sim/`):**
- Character import (paste SimC/70u/WoWSims string)
- Gear overview table (all slots with item names, ilvl)
- Quick-swap dropdowns per slot (search + select)
- Talent string input with spec auto-detection
- Buff preset selector + individual toggles
- Boss config (armor, debuffs, fight length)
- "Run Sim" button -> results panel
- Results: DPS summary, ability breakdown pie chart, stat weights bar chart, DPS distribution histogram

**Chat integration:**
- "Take my current sim profile and swap in DST"
- "What's my best trinket upgrade?"
- "Sim me with full P5 BiS"
- "Compare combat swords vs mutilate with my current gear"

Results rendered inline in chat with charts and tables.

### Technology

HTMX + Jinja2 templates (matching existing web app). Chart.js for DPS distribution and ability breakdown visualizations. WebSocket for streaming sim progress during long runs.

---

## 12. Analyst Agent

### Agent Subclass

```python
class AnalystAgent(BaseAgent):
    """Specialist agent for quantitative analysis using the sim engine."""
    tools = [sim_run_tool, sim_compare_tool, sim_optimize_tool,
             rag_search_tool, graph_search_tool]
```

### Tools

| Tool | Input | Output | Use Case |
|------|-------|--------|----------|
| `sim_run` | SimRunInput | SimResult | "Sim my character" |
| `sim_compare` | SimCompareInput | CompareResult | "What if I swap X for Y?" |
| `sim_optimize` | SimOptimizeInput | OptimizeResult | "What's my best MH weapon?" |
| `rag_search` | query | SearchResults | Domain knowledge for interpretation |
| `graph_search` | entity | Relationships | Item/stat relationships |

### Integration with Orchestrator

The Orchestrator dispatches ANALYSIS category tasks to the Analyst. The Analyst can also be invoked directly through the router for sim-related queries.

Query flow: User asks "what trinket should I use?" -> Router classifies as ANALYSIS -> Orchestrator dispatches to Analyst -> Analyst runs `sim_optimize` for trinket slots -> interprets results with domain knowledge -> returns natural language answer with data.

### Interpretation Layer

The Analyst doesn't just return numbers — it explains WHY:
- "DST gains 47 DPS over Bloodlust Brooch because the +325 haste proc (23% uptime) increases your MH auto-attack frequency, which generates more Combat Potency procs (+2.3 energy/sec) and more Sword Spec procs."
- "You're 47 hit rating below yellow cap — gaining hit is worth 2.2 EP until capped, making hit your most valuable stat right now."

This leverages the knowledge graph (stat thresholds, item relationships) and the sim's structured output (ability breakdown, proc uptimes, stat weights).

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Single sim run (10k iterations, 5min fight) | < 5 seconds |
| Stat weight computation (9 extra sim runs) | < 45 seconds |
| sim_compare (2 parallel sims) | < 5 seconds |
| sim_optimize per slot (20 candidates) | < 60 seconds |
| WoWSims DPS parity | Within 2% for Combat specs |
| WCL realism check | Within 10% of median parse DPS |

---

## Phase Gate

- [ ] P1 BiS Combat Swords sim within 2% of WoWSims
- [ ] P5 BiS Combat Swords sim within 2% of WoWSims
- [ ] P3 BiS Mutilate sim within 2% of WoWSims (accepting Envenom DP divergence)
- [ ] Analyst agent can answer "what's my best trinket?" end-to-end
- [ ] Character import works for all 3 formats
- [ ] `/sim` page renders gear overview and sim results
- [ ] Stat weights computed and displayed
- [ ] Regression test suite passes in CI
