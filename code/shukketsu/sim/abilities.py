"""Static ability and poison definitions for TBC Rogue simulation.

Contains all ability data tables used by the combat engine. No simulation
logic — just data definitions and lookup helpers.
"""

from pydantic import BaseModel

from code.shukketsu.sim.models import AbilityFlag, PoisonType, RogueSpec, WeaponType


class AbilityDef(BaseModel):
    """Static definition of a Rogue ability."""

    name: str
    spell_id: int
    energy_cost: int
    flat_damage: float = 0.0
    weapon_multiplier: float = 1.0
    normalized: bool = False
    norm_speed: float = 2.4
    combo_points_generated: int = 0
    combo_points_consumed: bool = False  # True for finishers
    cooldown_ms: int = 0
    duration_ms: int = 0
    flags: set[AbilityFlag] = set()
    miss_refund_pct: float = 0.80
    ap_coefficient: float = 0.0
    bonus_per_combo_point: float = 0.0
    weapon_type_required: WeaponType | None = None


class PoisonDef(BaseModel):
    """Static definition of a Rogue poison."""

    name: str
    spell_id: int
    proc_chance_base: float
    damage_per_proc: float = 0.0
    damage_per_stack_tick: float = 0.0  # For Deadly
    max_stacks: int = 1
    tick_interval_ms: int = 0
    duration_ms: int = 0
    flags: set[AbilityFlag] = set()


# ---------------------------------------------------------------------------
# Ability registry
# ---------------------------------------------------------------------------

ABILITIES: dict[str, AbilityDef] = {
    # --- Builders ---
    "sinister_strike": AbilityDef(
        name="Sinister Strike",
        spell_id=26862,
        energy_cost=45,
        flat_damage=98,
        normalized=True,
        norm_speed=2.4,
        combo_points_generated=1,
        flags={
            AbilityFlag.BUILDER,
            AbilityFlag.PHYSICAL,
            AbilityFlag.NORMALIZED,
            AbilityFlag.MAIN_HAND,
            AbilityFlag.APPLIES_LETHALITY,
        },
    ),
    "backstab": AbilityDef(
        name="Backstab",
        spell_id=26863,
        energy_cost=60,
        flat_damage=170,
        weapon_multiplier=1.5,
        normalized=True,
        norm_speed=1.7,
        combo_points_generated=1,
        weapon_type_required=WeaponType.DAGGER,
        flags={
            AbilityFlag.BUILDER,
            AbilityFlag.PHYSICAL,
            AbilityFlag.NORMALIZED,
            AbilityFlag.MAIN_HAND,
            AbilityFlag.APPLIES_LETHALITY,
        },
    ),
    "mutilate": AbilityDef(
        name="Mutilate",
        spell_id=34413,
        energy_cost=60,
        flat_damage=101,
        weapon_multiplier=1.0,
        normalized=True,
        norm_speed=1.7,
        combo_points_generated=2,
        flags={
            AbilityFlag.BUILDER,
            AbilityFlag.PHYSICAL,
            AbilityFlag.NORMALIZED,
            AbilityFlag.MAIN_HAND,
            AbilityFlag.OFF_HAND,
            AbilityFlag.APPLIES_LETHALITY,
        },
    ),
    "hemorrhage": AbilityDef(
        name="Hemorrhage",
        spell_id=26864,
        energy_cost=35,
        weapon_multiplier=1.1,
        normalized=True,
        norm_speed=2.4,
        combo_points_generated=1,
        flags={
            AbilityFlag.BUILDER,
            AbilityFlag.PHYSICAL,
            AbilityFlag.NORMALIZED,
            AbilityFlag.MAIN_HAND,
            AbilityFlag.APPLIES_LETHALITY,
        },
    ),
    "shiv": AbilityDef(
        name="Shiv",
        spell_id=5938,
        energy_cost=20,
        combo_points_generated=1,
        miss_refund_pct=0.80,
        flags={
            AbilityFlag.BUILDER,
            AbilityFlag.PHYSICAL,
            AbilityFlag.OFF_HAND,
            AbilityFlag.CANNOT_BE_DODGED,
        },
    ),
    "ambush": AbilityDef(
        name="Ambush",
        spell_id=27441,
        energy_cost=60,
        flat_damage=290,
        weapon_multiplier=2.5,
        normalized=True,
        norm_speed=1.7,
        combo_points_generated=2,
        weapon_type_required=WeaponType.DAGGER,
        flags={
            AbilityFlag.BUILDER,
            AbilityFlag.PHYSICAL,
            AbilityFlag.NORMALIZED,
            AbilityFlag.MAIN_HAND,
            AbilityFlag.APPLIES_LETHALITY,
        },
    ),
    "garrote": AbilityDef(
        name="Garrote",
        spell_id=26884,
        energy_cost=50,
        combo_points_generated=1,
        duration_ms=18000,
        ap_coefficient=0.18,
        flags={
            AbilityFlag.BUILDER,
            AbilityFlag.PHYSICAL,
            AbilityFlag.SNAPSHOT,
        },
    ),
    "cheap_shot": AbilityDef(
        name="Cheap Shot",
        spell_id=1833,
        energy_cost=60,
        combo_points_generated=2,
        flags={
            AbilityFlag.BUILDER,
            AbilityFlag.PHYSICAL,
            AbilityFlag.CANNOT_BE_DODGED,
        },
    ),
    # --- Finishers ---
    "eviscerate": AbilityDef(
        name="Eviscerate",
        spell_id=26865,
        energy_cost=35,
        combo_points_consumed=True,
        bonus_per_combo_point=185,
        flat_damage=245,
        ap_coefficient=0.03,
        flags={
            AbilityFlag.FINISHER,
            AbilityFlag.PHYSICAL,
        },
    ),
    "envenom": AbilityDef(
        name="Envenom",
        spell_id=32684,
        energy_cost=35,
        combo_points_consumed=True,
        bonus_per_combo_point=210,
        flags={
            AbilityFlag.FINISHER,
            AbilityFlag.NATURE,
            AbilityFlag.IGNORES_ARMOR,
        },
    ),
    "rupture": AbilityDef(
        name="Rupture",
        spell_id=26867,
        energy_cost=25,
        combo_points_consumed=True,
        duration_ms=16000,
        ap_coefficient=0.04,
        bonus_per_combo_point=0.0,
        flags={
            AbilityFlag.FINISHER,
            AbilityFlag.PHYSICAL,
            AbilityFlag.SNAPSHOT,
        },
    ),
    "slice_and_dice": AbilityDef(
        name="Slice and Dice",
        spell_id=6774,
        energy_cost=25,
        combo_points_consumed=True,
        duration_ms=9000,
        bonus_per_combo_point=3000,
        flags={
            AbilityFlag.FINISHER,
        },
    ),
    "expose_armor": AbilityDef(
        name="Expose Armor",
        spell_id=26866,
        energy_cost=25,
        combo_points_consumed=True,
        duration_ms=30000,
        flags={AbilityFlag.FINISHER},
    ),
    # --- Cooldowns ---
    "blade_flurry": AbilityDef(
        name="Blade Flurry",
        spell_id=13877,
        energy_cost=25,
        cooldown_ms=120000,
        duration_ms=15000,
        flags={AbilityFlag.OFF_GCD},
    ),
    "adrenaline_rush": AbilityDef(
        name="Adrenaline Rush",
        spell_id=13750,
        energy_cost=0,
        cooldown_ms=300000,
        duration_ms=15000,
        flags={AbilityFlag.OFF_GCD},
    ),
    "cold_blood": AbilityDef(
        name="Cold Blood",
        spell_id=14177,
        energy_cost=0,
        cooldown_ms=180000,
        flags={AbilityFlag.OFF_GCD},
    ),
    "thistle_tea": AbilityDef(
        name="Thistle Tea",
        spell_id=9512,
        energy_cost=0,
        cooldown_ms=300000,
        flags={AbilityFlag.OFF_GCD},
    ),
    "premeditation": AbilityDef(
        name="Premeditation",
        spell_id=14183,
        energy_cost=0,
        cooldown_ms=120000,
        flags={AbilityFlag.OFF_GCD},
    ),
}


# ---------------------------------------------------------------------------
# Poison registry
# ---------------------------------------------------------------------------

POISONS: dict[str, PoisonDef] = {
    "instant_poison": PoisonDef(
        name="Instant Poison",
        spell_id=26891,
        proc_chance_base=0.20,
        damage_per_proc=200,
        flags={AbilityFlag.NATURE, AbilityFlag.IGNORES_ARMOR},
    ),
    "deadly_poison": PoisonDef(
        name="Deadly Poison",
        spell_id=27187,
        proc_chance_base=0.30,
        damage_per_stack_tick=180,
        max_stacks=5,
        tick_interval_ms=3000,
        duration_ms=12000,
        flags={AbilityFlag.NATURE, AbilityFlag.IGNORES_ARMOR},
    ),
    "wound_poison": PoisonDef(
        name="Wound Poison",
        spell_id=27189,
        proc_chance_base=0.50,
        flags={AbilityFlag.NATURE},
    ),
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

_POISON_TYPE_TO_KEY: dict[PoisonType, str] = {
    PoisonType.INSTANT: "instant_poison",
    PoisonType.DEADLY: "deadly_poison",
    PoisonType.WOUND: "wound_poison",
}


def get_ability(name: str) -> AbilityDef:
    """Get ability by name. Raises KeyError if not found."""
    return ABILITIES[name]


def get_poison(poison_type: PoisonType) -> PoisonDef:
    """Get poison definition from PoisonType enum.

    Args:
        poison_type: The PoisonType enum value.

    Returns:
        The matching PoisonDef.

    Raises:
        KeyError: If the poison type has no definition (e.g. ANESTHETIC, NONE).
    """
    key = _POISON_TYPE_TO_KEY.get(poison_type)
    if key is None:
        raise KeyError(f"No poison definition for {poison_type}")
    return POISONS[key]


def builders_for_spec(spec: RogueSpec) -> list[str]:
    """Return priority-ordered builder names for a given spec.

    Args:
        spec: The Rogue specialization.

    Returns:
        List of ability registry keys for the spec's builders.
    """
    match spec:
        case RogueSpec.COMBAT_SWORDS | RogueSpec.COMBAT_FISTS:
            return ["sinister_strike"]
        case RogueSpec.COMBAT_DAGGERS:
            return ["backstab"]
        case RogueSpec.ASSASSINATION_MUTILATE:
            return ["mutilate"]


def finisher_for_spec(spec: RogueSpec, *, has_rupture: bool, fight_remaining: float) -> str:
    """Return preferred damage finisher for a spec and situation.

    Args:
        spec: The Rogue specialization.
        has_rupture: Whether Rupture is already active on the target.
        fight_remaining: Seconds remaining in the fight.

    Returns:
        Ability registry key for the recommended finisher.
    """
    if spec == RogueSpec.ASSASSINATION_MUTILATE:
        return "envenom"
    # Combat specs: Eviscerate if Rupture is up or fight is nearly over
    if has_rupture or fight_remaining < 12:
        return "eviscerate"
    return "rupture"
