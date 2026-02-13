"""Data models for the TBC Rogue DPS simulation engine.

All enums, item/gear models, simulation config, and result types used
throughout the sim package.
"""

from enum import StrEnum

from pydantic import BaseModel

# --- Enums ---


class WeaponType(StrEnum):
    """TBC melee weapon types relevant to Rogues."""

    SWORD = "sword"
    DAGGER = "dagger"
    FIST = "fist"
    MACE = "mace"


class RogueSpec(StrEnum):
    """Supported TBC Rogue specializations."""

    COMBAT_SWORDS = "combat_swords"
    COMBAT_FISTS = "combat_fists"
    COMBAT_DAGGERS = "combat_daggers"
    ASSASSINATION_MUTILATE = "assassination_mutilate"


class GearSlot(StrEnum):
    """Equipment slots for a Rogue character."""

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
    """Playable races that can be Rogues in TBC."""

    HUMAN = "human"
    ORC = "orc"
    NIGHT_ELF = "night_elf"
    BLOOD_ELF = "blood_elf"
    UNDEAD = "undead"
    DWARF = "dwarf"
    GNOME = "gnome"
    TROLL = "troll"


class PoisonType(StrEnum):
    """Rogue poison types."""

    INSTANT = "instant_poison"
    DEADLY = "deadly_poison"
    WOUND = "wound_poison"
    ANESTHETIC = "anesthetic_poison"
    NONE = "none"


class OpenerAbility(StrEnum):
    """Opening abilities from stealth."""

    GARROTE = "garrote"
    AMBUSH = "ambush"
    CHEAP_SHOT = "cheap_shot"
    NONE = "none"


class ProcTrigger(StrEnum):
    """How a proc effect is triggered."""

    ON_HIT = "on_hit"
    PPM = "ppm"
    ON_CRIT = "on_crit"
    ON_USE = "on_use"


class FightType(StrEnum):
    """Encounter movement profiles."""

    PATCHWERK = "patchwerk"
    CLEAVE = "cleave"
    MOVEMENT = "movement"


class GemSlot(StrEnum):
    """Socket colors for gems."""

    RED = "red"
    YELLOW = "yellow"
    BLUE = "blue"
    META = "meta"


class HitOutcome(StrEnum):
    """Possible outcomes of a melee attack roll."""

    HIT = "hit"
    CRIT = "crit"
    MISS = "miss"
    DODGE = "dodge"
    GLANCING = "glancing"
    BLOCK = "block"


class AbilityFlag(StrEnum):
    """Flags describing ability properties for the combat engine."""

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


class BuffCategory(StrEnum):
    """Buff stacking categories. Buffs in the same category are mutually exclusive."""

    ATTACK_POWER = "attack_power"
    STATS_PCT = "stats_pct"
    AGILITY_FLAT = "agility_flat"
    STRENGTH_FLAT = "strength_flat"
    MELEE_CRIT = "melee_crit"
    AP_PCT = "ap_pct"
    FLASK = "flask"
    BATTLE_ELIXIR = "battle_elixir"
    GUARDIAN_ELIXIR = "guardian_elixir"
    FOOD = "food"
    UNCATEGORIZED = "uncategorized"


# --- Item / Gear Models ---


class WeaponStats(BaseModel):
    """Weapon-specific stats attached to weapon items."""

    min_damage: float
    max_damage: float
    speed: float
    weapon_type: WeaponType
    dps: float


class ProcEffect(BaseModel):
    """Proc or on-use effect for items and enchants."""

    trigger: ProcTrigger
    rate: float
    icd: float = 0.0
    duration: float = 0.0
    effect: dict[str, float] = {}
    stacks: int = 1


class SetBonus(BaseModel):
    """A set bonus activated at a piece threshold."""

    set_name: str
    pieces_required: int
    effect: dict[str, float]


class Item(BaseModel):
    """An equippable item with optional weapon stats, sockets, and procs."""

    id: int
    name: str
    slot: GearSlot
    item_level: int
    phase: int
    stats: dict[str, float] = {}
    sockets: list[GemSlot] = []
    socket_bonus: dict[str, float] = {}
    set_id: str | None = None
    weapon: WeaponStats | None = None
    proc: ProcEffect | None = None
    on_use: ProcEffect | None = None


class Enchant(BaseModel):
    """An enchant applicable to a gear slot."""

    id: int
    name: str
    slot: GearSlot
    stats: dict[str, float] = {}
    proc: ProcEffect | None = None


class Gem(BaseModel):
    """A socketable gem with color requirements."""

    id: int
    name: str
    color: GemSlot
    stats: dict[str, float]
    meta_condition: str | None = None


# --- Simulation Config (Input) ---


class PoisonConfig(BaseModel):
    """Which poisons are applied to each weapon."""

    main_hand: PoisonType = PoisonType.INSTANT
    off_hand: PoisonType = PoisonType.DEADLY


class BossConfig(BaseModel):
    """Target boss configuration."""

    name: str = "Raid Boss"
    level: int = 73
    armor: int = 7700
    debuffs: list[str] = []


class SimConfig(BaseModel):
    """Full simulation configuration input."""

    spec: RogueSpec
    race: Race = Race.ORC
    talents: str
    gear: dict[GearSlot, int]
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


# --- Simulation Result (Output) ---


class AbilityBreakdown(BaseModel):
    """Per-ability damage breakdown in sim results."""

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
    """Proc effect uptime statistics."""

    name: str
    source: str
    uptime_pct: float
    avg_procs_per_fight: float
    avg_stacks: float = 1.0


class PoisonStats(BaseModel):
    """Per-poison damage and proc statistics."""

    poison_type: PoisonType
    hand: str
    procs_per_fight: float
    damage_total: float
    damage_pct: float
    avg_deadly_stacks: float = 0.0


class CooldownUsage(BaseModel):
    """Cooldown usage statistics."""

    name: str
    casts_per_fight: float
    avg_uptime_pct: float


class ResourceStats(BaseModel):
    """Energy and combo point resource statistics."""

    energy_per_second: float
    avg_energy_waste: float
    combo_points_per_second: float
    combo_point_overcap_pct: float
    gcd_utilization_pct: float
    energy_starved_pct: float


class StatWeight(BaseModel):
    """Equivalence point value for a stat relative to AP."""

    stat: str
    ep_value: float
    dps_per_point: float
    is_capped: bool = False


class DpsTimeline(BaseModel):
    """DPS over time bucketed into intervals."""

    bucket_seconds: int = 5
    buckets: list[float] = []


class SimResult(BaseModel):
    """Complete simulation result output."""

    dps_mean: float
    dps_std: float
    dps_median: float
    dps_min: float
    dps_max: float
    iterations: int
    fight_length: int
    ability_breakdown: list[AbilityBreakdown]
    stat_weights: list[StatWeight] = []
    proc_uptimes: list[ProcUptime] = []
    poison_stats: list[PoisonStats] = []
    cooldown_usage: list[CooldownUsage] = []
    buff_uptimes: dict[str, float] = {}
    resource_stats: ResourceStats
    dps_timeline: DpsTimeline = DpsTimeline()
    dps_distribution: list[float] = []
    config: SimConfig
