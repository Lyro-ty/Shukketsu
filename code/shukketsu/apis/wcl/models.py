"""Pydantic v2 models for Warcraft Logs v2 GraphQL API response types.

All models use ``populate_by_name=True`` so they accept both camelCase
(as returned by the WCL API) and snake_case field names.
"""

from enum import IntEnum, StrEnum

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WCLEndpoint(StrEnum):
    """WCL API endpoint selection."""

    CLASSIC = "classic"
    FRESH = "fresh"


class WCLHitType(IntEnum):
    """WCL damage event hit types."""

    NORMAL_HIT = 1
    CRIT = 2
    ABSORB = 3
    DODGE = 7
    MISS = 8
    IMMUNE = 10
    PARTIAL_RESIST = 16


# ---------------------------------------------------------------------------
# Shared / Reference models
# ---------------------------------------------------------------------------


class WCLGuildInfo(BaseModel):
    """Guild reference from a ranking entry."""

    id: int
    name: str
    faction: int | None = None


class WCLServerInfo(BaseModel):
    """Server (realm) reference."""

    id: int
    name: str
    region: str | None = None


class WCLReportRef(BaseModel):
    """Reference to a report from a ranking entry."""

    code: str
    fight_id: int = Field(alias="fightID")
    start_time: int = Field(alias="startTime")
    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Rankings & Characters
# ---------------------------------------------------------------------------


class WCLRanking(BaseModel):
    """A single ranking entry from ``characterRankings``."""

    name: str
    class_name: str = Field(alias="class")  # "class" is reserved in Python
    spec: str
    amount: float  # DPS
    duration: int  # ms
    hard_mode_level: int | None = Field(default=None, alias="hardModeLevel")
    report: WCLReportRef | None = None
    guild: WCLGuildInfo | None = None
    server: WCLServerInfo | None = None
    faction: int | None = None
    size: int | None = None
    bracket_data: int | None = Field(default=None, alias="bracketData")
    model_config = ConfigDict(populate_by_name=True)


class WCLZoneRanking(BaseModel):
    """Per-encounter ranking for a character."""

    encounter_id: int = Field(alias="encounterID")
    encounter_name: str = Field(alias="encounterName")
    best_amount: float = Field(alias="bestAmount")
    median_percent: float | None = Field(default=None, alias="medianPercent")
    total_kills: int = Field(alias="totalKills")
    rank_percent: float | None = Field(default=None, alias="rankPercent")
    model_config = ConfigDict(populate_by_name=True)


class WCLCharacter(BaseModel):
    """Character profile from WCL API."""

    id: int
    name: str
    class_id: int = Field(alias="classID")
    server: WCLServerInfo | None = None
    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Report-Level
# ---------------------------------------------------------------------------


class WCLZoneInfo(BaseModel):
    """Zone (raid instance) reference."""

    id: int
    name: str


class WCLReport(BaseModel):
    """Report metadata."""

    code: str
    title: str | None = None
    start_time: int = Field(alias="startTime")
    end_time: int = Field(alias="endTime")
    zone: WCLZoneInfo | None = None
    model_config = ConfigDict(populate_by_name=True)


class WCLFight(BaseModel):
    """A fight within a report."""

    id: int
    encounter_id: int = Field(alias="encounterID")
    name: str
    kill: bool | None = None
    duration: int | None = None  # ms
    boss_percentage: float | None = Field(default=None, alias="bossPercentage")
    avg_item_level: float | None = Field(default=None, alias="averageItemLevel")
    size: int | None = None
    difficulty: int | None = None
    model_config = ConfigDict(populate_by_name=True)


class WCLActor(BaseModel):
    """An actor (player/NPC) in a report."""

    id: int
    name: str
    type: str
    sub_type: str | None = Field(default=None, alias="subType")
    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Per-Player (CombatantInfo)
# ---------------------------------------------------------------------------


class WCLGem(BaseModel):
    """A gem socketed in a gear item."""

    id: int
    item_level: int = Field(alias="itemLevel")
    icon: str | None = None
    model_config = ConfigDict(populate_by_name=True)


class WCLGearItem(BaseModel):
    """A gear item from CombatantInfo."""

    id: int
    slot: int
    quality: int | None = None
    name: str | None = None
    item_level: int = Field(alias="itemLevel")
    permanent_enchant: int | None = Field(default=None, alias="permanentEnchant")
    permanent_enchant_name: str | None = Field(default=None, alias="permanentEnchantName")
    gems: list[WCLGem] = Field(default_factory=list)
    set_id: int | None = Field(default=None, alias="setID")
    model_config = ConfigDict(populate_by_name=True)


class WCLTalentEntry(BaseModel):
    """A talent entry from CombatantInfo."""

    guid: int
    type: int | None = None
    name: str | None = None
    ability_icon: str | None = Field(default=None, alias="abilityIcon")
    model_config = ConfigDict(populate_by_name=True)


class WCLAuraEntry(BaseModel):
    """A pre-pull buff/aura."""

    source: int | None = None
    ability: int | None = None
    stacks: int | None = None
    icon: str | None = None
    name: str | None = None


class WCLCombatantInfo(BaseModel):
    """Full combatant info from ``events(dataType: CombatantInfo)``."""

    source_id: int = Field(alias="sourceID")
    spec_id: int | None = Field(default=None, alias="specID")
    faction: int | None = None
    # Stats --- all optional since not every class has every stat
    strength: int | None = None
    agility: int | None = None
    stamina: int | None = None
    intellect: int | None = None
    spirit: int | None = None
    crit_melee: int | None = Field(default=None, alias="critMelee")
    crit_ranged: int | None = Field(default=None, alias="critRanged")
    crit_spell: int | None = Field(default=None, alias="critSpell")
    haste_melee: int | None = Field(default=None, alias="hasteMelee")
    haste_ranged: int | None = Field(default=None, alias="hasteRanged")
    haste_spell: int | None = Field(default=None, alias="hasteSpell")
    hit_melee: int | None = Field(default=None, alias="hitMelee")
    hit_ranged: int | None = Field(default=None, alias="hitRanged")
    hit_spell: int | None = Field(default=None, alias="hitSpell")
    expertise: int | None = None
    dodge: int | None = None
    parry: int | None = None
    block: int | None = None
    armor: int | None = None
    gear: list[WCLGearItem] = Field(default_factory=list)
    talents: list[WCLTalentEntry] = Field(default_factory=list)
    auras: list[WCLAuraEntry] = Field(default_factory=list)
    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Tables (damage / buff / cast summaries)
# ---------------------------------------------------------------------------


class WCLAbilitySummary(BaseModel):
    """An ability in a damage/cast table."""

    name: str
    total: int
    type: int | None = None


class WCLTargetSummary(BaseModel):
    """Damage dealt to a specific target."""

    name: str
    total: int


class WCLDamageEntry(BaseModel):
    """Per-player damage breakdown from damage table."""

    name: str
    id: int
    guid: int | None = None
    type: str | None = None
    icon: str | None = None
    item_level: float | None = Field(default=None, alias="itemLevel")
    total: int
    active_time: int | None = Field(default=None, alias="activeTime")
    active_time_reduced: int | None = Field(default=None, alias="activeTimeReduced")
    abilities: list[WCLAbilitySummary] = Field(default_factory=list)
    targets: list[WCLTargetSummary] = Field(default_factory=list)
    gear: list[WCLGearItem] = Field(default_factory=list)
    talents: list[WCLTalentEntry] = Field(default_factory=list)
    model_config = ConfigDict(populate_by_name=True)


class WCLBuffBand(BaseModel):
    """Time range for a buff application."""

    start_time: int = Field(alias="startTime")
    end_time: int = Field(alias="endTime")
    model_config = ConfigDict(populate_by_name=True)


class WCLBuffAura(BaseModel):
    """Buff uptime data from buff table."""

    name: str
    guid: int
    type: int | None = None
    ability_icon: str | None = Field(default=None, alias="abilityIcon")
    total_uptime: int = Field(alias="totalUptime")
    total_uses: int = Field(alias="totalUses")
    bands: list[WCLBuffBand] = Field(default_factory=list)
    model_config = ConfigDict(populate_by_name=True)


class WCLCastEntry(BaseModel):
    """Per-player cast breakdown from cast table."""

    name: str
    id: int
    type: str | None = None
    total: int  # total casts
    abilities: list[WCLAbilitySummary] = Field(default_factory=list)
    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Raw Events
# ---------------------------------------------------------------------------


class WCLDamageEvent(BaseModel):
    """A single raw damage event."""

    timestamp: int
    type: str = "damage"
    source_id: int = Field(alias="sourceID")
    target_id: int = Field(alias="targetID")
    ability_game_id: int = Field(alias="abilityGameID")
    fight: int | None = None
    hit_type: int = Field(alias="hitType")
    amount: int = 0
    mitigated: int | None = None
    unmitigated_amount: int | None = Field(default=None, alias="unmitigatedAmount")
    resisted: int | None = None
    buffs: str | None = None  # dot-separated buff IDs
    is_aoe: bool | None = Field(default=None, alias="isAoE")
    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Fight Rankings
# ---------------------------------------------------------------------------


class WCLFightRanking(BaseModel):
    """Per-player ranking within a fight."""

    name: str
    server_name: str | None = Field(default=None, alias="server")
    class_name: str = Field(alias="class")
    spec: str | None = None
    amount: float  # DPS
    bracket: int | None = None
    rank: int | None = None
    rank_percent: float | None = Field(default=None, alias="rankPercent")
    total_parses: int | None = Field(default=None, alias="totalParses")
    model_config = ConfigDict(populate_by_name=True)
