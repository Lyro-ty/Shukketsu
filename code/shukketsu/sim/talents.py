"""Talent definitions and modifier computation for TBC Rogue simulation.

Defines the ~25 most DPS-relevant talents across Assassination, Combat, and
Subtlety trees.  Provides functions to parse talent strings (e.g. "20/41/0"),
look up canonical spec templates, and compute immutable modifier snapshots
consumed by the combat engine.
"""

from pydantic import BaseModel

from code.shukketsu.sim.models import RogueSpec

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TalentDef(BaseModel):
    """Static definition of a talent."""

    name: str
    tree: str  # "assassination", "combat", "subtlety"
    tier: int
    column: int
    max_ranks: int
    effect_per_rank: dict[str, float]  # modifier_name -> value per rank


class TalentAllocation(BaseModel):
    """Parsed talent allocation from a talent string."""

    assassination: int
    combat: int
    subtlety: int
    points: dict[str, int]  # talent_name -> ranks allocated


class TalentModifiers(BaseModel, frozen=True):
    """Computed modifiers from a talent allocation. Immutable."""

    # Hit / Crit / Expertise
    bonus_crit_pct: float = 0.0
    bonus_hit_pct: float = 0.0
    bonus_expertise: float = 0.0

    # Sinister Strike
    ss_energy_reduction: int = 0
    ss_damage_bonus_pct: float = 0.0

    # Backstab
    bs_crit_bonus_pct: float = 0.0
    bs_damage_bonus_pct: float = 0.0

    # Mutilate
    mutilate_crit_bonus_pct: float = 0.0
    mutilate_damage_bonus_pct: float = 0.0

    # Eviscerate / Rupture
    evis_damage_bonus_pct: float = 0.0
    rupture_damage_bonus_pct: float = 0.0

    # Slice and Dice
    snd_duration_mult: float = 1.0

    # Lethality (secondary crit mod for BS, Mut, Ambush, SS, Hemo)
    lethality_secondary_mod: float = 0.0

    # Mace Spec (primary crit mod, simplified)
    crit_damage_primary_mod: float = 1.0

    # Find Weakness
    find_weakness_damage_pct: float = 0.0

    # Seal Fate
    seal_fate_proc_chance: float = 0.0

    # Ruthlessness
    ruthlessness_proc_chance: float = 0.0

    # Relentless Strikes
    relentless_strikes_per_cp: float = 0.0

    # Quick Recovery
    quick_recovery_refund_pct: float = 0.0

    # Combat Potency
    combat_potency_proc_chance: float = 0.0
    combat_potency_energy: float = 0.0

    # Sword Spec
    sword_spec_proc_chance: float = 0.0

    # Weapon specializations
    dagger_spec_crit_bonus: float = 0.0
    fist_spec_crit_bonus: float = 0.0

    # Dual Wield Spec
    dw_spec_oh_bonus_pct: float = 0.0

    # Poison talents
    imp_poisons_ranks: int = 0
    vile_poisons_pct: float = 0.0
    master_poisoner_hit_pct: float = 0.0

    # Damage multipliers
    murder_damage_pct: float = 0.0
    aggression_damage_pct: float = 0.0
    opportunity_damage_pct: float = 0.0
    surprise_attacks_damage_pct: float = 0.0
    surprise_attacks_finisher_undodgeable: bool = False

    # Stat multipliers
    vitality_agi_mult: float = 1.0
    sinister_calling_agi_mult: float = 1.0
    deadliness_ap_mult: float = 1.0

    # ArPen / Rupture
    serrated_blades_arpen: int = 0
    serrated_blades_rupture_pct: float = 0.0

    # Ability unlock booleans
    vigor: bool = False
    cold_blood: bool = False
    blade_flurry: bool = False
    adrenaline_rush: bool = False
    mutilate_talented: bool = False
    hemorrhage_talented: bool = False
    premeditation_talented: bool = False


# ---------------------------------------------------------------------------
# Talent definitions — DPS-relevant talents only
# ---------------------------------------------------------------------------

TALENT_DEFS: dict[str, TalentDef] = {
    # ===== Assassination =====
    "improved_eviscerate": TalentDef(
        name="Improved Eviscerate",
        tree="assassination",
        tier=1,
        column=1,
        max_ranks=3,
        effect_per_rank={"evis_damage_bonus_pct": 0.05},
    ),
    "malice": TalentDef(
        name="Malice",
        tree="assassination",
        tier=2,
        column=3,
        max_ranks=5,
        effect_per_rank={"bonus_crit_pct": 0.01},
    ),
    "ruthlessness": TalentDef(
        name="Ruthlessness",
        tree="assassination",
        tier=3,
        column=1,
        max_ranks=3,
        effect_per_rank={"ruthlessness_proc_chance": 0.20},
    ),
    "murder": TalentDef(
        name="Murder",
        tree="assassination",
        tier=4,
        column=1,
        max_ranks=2,
        effect_per_rank={"murder_damage_pct": 0.01},
    ),
    "relentless_strikes": TalentDef(
        name="Relentless Strikes",
        tree="assassination",
        tier=4,
        column=3,
        max_ranks=1,
        effect_per_rank={"relentless_strikes_per_cp": 5.0},
    ),
    "improved_poisons": TalentDef(
        name="Improved Poisons",
        tree="assassination",
        tier=4,
        column=2,
        max_ranks=5,
        effect_per_rank={"imp_poisons_ranks": 1},
    ),
    "lethality": TalentDef(
        name="Lethality",
        tree="assassination",
        tier=5,
        column=3,
        max_ranks=5,
        effect_per_rank={"lethality_secondary_mod": 0.06},
    ),
    "vile_poisons": TalentDef(
        name="Vile Poisons",
        tree="assassination",
        tier=6,
        column=2,
        max_ranks=3,
        effect_per_rank={"vile_poisons_pct": 0.07},
    ),
    "quick_recovery": TalentDef(
        name="Quick Recovery",
        tree="assassination",
        tier=6,
        column=3,
        max_ranks=2,
        effect_per_rank={"quick_recovery_refund_pct": 0.40},
    ),
    "seal_fate": TalentDef(
        name="Seal Fate",
        tree="assassination",
        tier=7,
        column=2,
        max_ranks=5,
        effect_per_rank={"seal_fate_proc_chance": 0.20},
    ),
    "master_poisoner": TalentDef(
        name="Master Poisoner",
        tree="assassination",
        tier=7,
        column=3,
        max_ranks=2,
        effect_per_rank={"master_poisoner_hit_pct": 0.01},
    ),
    "vigor": TalentDef(
        name="Vigor",
        tree="assassination",
        tier=8,
        column=2,
        max_ranks=1,
        effect_per_rank={"vigor": 1},
    ),
    "find_weakness": TalentDef(
        name="Find Weakness",
        tree="assassination",
        tier=9,
        column=1,
        max_ranks=3,
        effect_per_rank={"find_weakness_damage_pct": 0.02},
    ),
    "mutilate_talent": TalentDef(
        name="Mutilate",
        tree="assassination",
        tier=9,
        column=2,
        max_ranks=1,
        effect_per_rank={"mutilate_talented": 1},
    ),
    "surprise_attacks": TalentDef(
        name="Surprise Attacks",
        tree="assassination",
        tier=9,
        column=3,
        max_ranks=1,
        effect_per_rank={"surprise_attacks_damage_pct": 0.10, "surprise_attacks_finisher_undodgeable": 1},
    ),
    # ===== Combat =====
    "improved_sinister_strike": TalentDef(
        name="Improved Sinister Strike",
        tree="combat",
        tier=1,
        column=1,
        max_ranks=2,
        effect_per_rank={"ss_energy_reduction": 3},
    ),
    "precision": TalentDef(
        name="Precision",
        tree="combat",
        tier=4,
        column=2,
        max_ranks=5,
        effect_per_rank={"bonus_hit_pct": 0.01},
    ),
    "dual_wield_specialization": TalentDef(
        name="Dual Wield Specialization",
        tree="combat",
        tier=4,
        column=3,
        max_ranks=5,
        effect_per_rank={"dw_spec_oh_bonus_pct": 0.10},
    ),
    "blade_flurry_talent": TalentDef(
        name="Blade Flurry",
        tree="combat",
        tier=5,
        column=2,
        max_ranks=1,
        effect_per_rank={"blade_flurry": 1},
    ),
    "sword_specialization": TalentDef(
        name="Sword Specialization",
        tree="combat",
        tier=6,
        column=1,
        max_ranks=5,
        effect_per_rank={"sword_spec_proc_chance": 0.01},
    ),
    "fist_specialization": TalentDef(
        name="Fist Weapon Specialization",
        tree="combat",
        tier=6,
        column=2,
        max_ranks=5,
        effect_per_rank={"fist_spec_crit_bonus": 0.01},
    ),
    "dagger_specialization": TalentDef(
        name="Dagger Specialization",
        tree="combat",
        tier=6,
        column=3,
        max_ranks=5,
        effect_per_rank={"dagger_spec_crit_bonus": 0.01},
    ),
    "mace_specialization": TalentDef(
        name="Mace Specialization",
        tree="combat",
        tier=6,
        column=4,
        max_ranks=5,
        effect_per_rank={"crit_damage_primary_mod": 0.01},
    ),
    "weapon_expertise": TalentDef(
        name="Weapon Expertise",
        tree="combat",
        tier=6,
        column=5,
        max_ranks=2,
        effect_per_rank={"bonus_expertise": 5},
    ),
    "aggression": TalentDef(
        name="Aggression",
        tree="combat",
        tier=7,
        column=1,
        max_ranks=3,
        effect_per_rank={"aggression_damage_pct": 0.02},
    ),
    "vitality": TalentDef(
        name="Vitality",
        tree="combat",
        tier=7,
        column=3,
        max_ranks=2,
        effect_per_rank={"vitality_agi_mult": 0.02},
    ),
    "adrenaline_rush_talent": TalentDef(
        name="Adrenaline Rush",
        tree="combat",
        tier=8,
        column=2,
        max_ranks=1,
        effect_per_rank={"adrenaline_rush": 1},
    ),
    "combat_potency": TalentDef(
        name="Combat Potency",
        tree="combat",
        tier=9,
        column=2,
        max_ranks=5,
        effect_per_rank={"combat_potency_proc_chance": 0.04, "combat_potency_energy": 3.0},
    ),
    # ===== Subtlety =====
    "opportunity": TalentDef(
        name="Opportunity",
        tree="subtlety",
        tier=2,
        column=1,
        max_ranks=2,
        effect_per_rank={"opportunity_damage_pct": 0.10},
    ),
    "hemorrhage_talent": TalentDef(
        name="Hemorrhage",
        tree="subtlety",
        tier=5,
        column=2,
        max_ranks=1,
        effect_per_rank={"hemorrhage_talented": 1},
    ),
    "serrated_blades": TalentDef(
        name="Serrated Blades",
        tree="subtlety",
        tier=5,
        column=3,
        max_ranks=3,
        effect_per_rank={"serrated_blades_rupture_pct": 0.10},
    ),
    "premeditation_talent": TalentDef(
        name="Premeditation",
        tree="subtlety",
        tier=5,
        column=1,
        max_ranks=1,
        effect_per_rank={"premeditation_talented": 1},
    ),
    "sinister_calling": TalentDef(
        name="Sinister Calling",
        tree="subtlety",
        tier=7,
        column=2,
        max_ranks=5,
        effect_per_rank={"sinister_calling_agi_mult": 0.03},
    ),
    "deadliness": TalentDef(
        name="Deadliness",
        tree="subtlety",
        tier=8,
        column=3,
        max_ranks=5,
        effect_per_rank={"deadliness_ap_mult": 0.02},
    ),
}

# Fields that use a 1.0 base and accumulate additively (i.e. result = 1.0 + sum).
_MULTIPLICATIVE_FIELDS: frozenset[str] = frozenset(
    {
        "vitality_agi_mult",
        "sinister_calling_agi_mult",
        "deadliness_ap_mult",
        "snd_duration_mult",
        "crit_damage_primary_mod",
    }
)

# Fields stored as booleans on TalentModifiers (set True when value >= 1).
_BOOLEAN_FIELDS: frozenset[str] = frozenset(
    {
        "vigor",
        "cold_blood",
        "blade_flurry",
        "adrenaline_rush",
        "mutilate_talented",
        "hemorrhage_talented",
        "premeditation_talented",
        "surprise_attacks_finisher_undodgeable",
    }
)

# Fields stored as int on TalentModifiers.
_INT_FIELDS: frozenset[str] = frozenset(
    {
        "ss_energy_reduction",
        "imp_poisons_ranks",
        "serrated_blades_arpen",
        "bonus_expertise",
    }
)


# ---------------------------------------------------------------------------
# Spec templates — canonical DPS-relevant point distributions
# ---------------------------------------------------------------------------

_SPEC_TEMPLATES: dict[RogueSpec, dict[str, int]] = {
    RogueSpec.COMBAT_SWORDS: {
        # Assassination (20)
        "malice": 5,
        "ruthlessness": 3,
        "murder": 2,
        "relentless_strikes": 1,
        "lethality": 5,
        "improved_poisons": 4,
        # Combat (41)
        "improved_sinister_strike": 2,
        "precision": 5,
        "dual_wield_specialization": 5,
        "blade_flurry_talent": 1,
        "sword_specialization": 5,
        "weapon_expertise": 2,
        "aggression": 3,
        "vitality": 2,
        "adrenaline_rush_talent": 1,
        "combat_potency": 5,
    },
    RogueSpec.COMBAT_FISTS: {
        # Assassination (20)
        "malice": 5,
        "ruthlessness": 3,
        "murder": 2,
        "relentless_strikes": 1,
        "lethality": 5,
        "improved_poisons": 4,
        # Combat (41)
        "improved_sinister_strike": 2,
        "precision": 5,
        "dual_wield_specialization": 5,
        "blade_flurry_talent": 1,
        "fist_specialization": 5,
        "weapon_expertise": 2,
        "aggression": 3,
        "vitality": 2,
        "adrenaline_rush_talent": 1,
        "combat_potency": 5,
    },
    RogueSpec.COMBAT_DAGGERS: {
        # Assassination (20)
        "malice": 5,
        "ruthlessness": 3,
        "murder": 2,
        "relentless_strikes": 1,
        "lethality": 5,
        "improved_poisons": 4,
        # Combat (41)
        "precision": 5,
        "dual_wield_specialization": 5,
        "blade_flurry_talent": 1,
        "dagger_specialization": 5,
        "weapon_expertise": 2,
        "aggression": 3,
        "vitality": 2,
        "adrenaline_rush_talent": 1,
        "combat_potency": 5,
    },
    RogueSpec.ASSASSINATION_MUTILATE: {
        # Assassination (41)
        "improved_eviscerate": 2,
        "malice": 5,
        "ruthlessness": 3,
        "murder": 2,
        "relentless_strikes": 1,
        "improved_poisons": 5,
        "lethality": 5,
        "vile_poisons": 3,
        "quick_recovery": 2,
        "seal_fate": 5,
        "master_poisoner": 2,
        "vigor": 1,
        "find_weakness": 3,
        "mutilate_talent": 1,
        "surprise_attacks": 1,
        # Combat (20)
        "precision": 5,
        "dual_wield_specialization": 5,
    },
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def parse_talents(talent_string: str, spec: RogueSpec) -> TalentAllocation:
    """Parse a ``"X/Y/Z"`` talent string and fill from the spec template.

    Validates format and total points (max 61 for Level 70 TBC).

    Args:
        talent_string: Talent distribution like ``"20/41/0"``.
        spec: The Rogue specialization (determines which template to use).

    Returns:
        A TalentAllocation with tree totals and per-talent point mapping.

    Raises:
        ValueError: If the format is invalid or total exceeds 61.
    """
    parts = talent_string.strip().split("/")
    if len(parts) != 3:
        raise ValueError(f"Talent string must be 'X/Y/Z', got: {talent_string!r}")

    try:
        tree_totals = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Non-integer value in talent string: {talent_string!r}") from exc

    if any(t < 0 for t in tree_totals):
        raise ValueError(f"Negative talent points in: {talent_string!r}")

    total = sum(tree_totals)
    if total > 61:
        raise ValueError(f"Total talent points ({total}) exceeds maximum of 61")

    template = get_spec_template(spec)

    return TalentAllocation(
        assassination=tree_totals[0],
        combat=tree_totals[1],
        subtlety=tree_totals[2],
        points=dict(template),
    )


def compute_modifiers(allocation: TalentAllocation) -> TalentModifiers:
    """Compute all modifier values from a talent allocation.

    Pure function: iterates ``allocation.points``, looks up each talent's
    ``TalentDef``, multiplies ``effect_per_rank * ranks``, and accumulates
    into modifier fields.

    Multiplier fields (``vitality_agi_mult``, ``sinister_calling_agi_mult``,
    ``deadliness_ap_mult``, ``crit_damage_primary_mod``) start at 1.0 and
    add per-rank values additively.  Boolean fields are set to ``True``
    when the accumulated value >= 1.

    Args:
        allocation: A TalentAllocation with per-talent rank counts.

    Returns:
        An immutable TalentModifiers snapshot.
    """
    accum: dict[str, float] = {}

    for talent_name, ranks in allocation.points.items():
        if ranks <= 0:
            continue

        talent_def = TALENT_DEFS.get(talent_name)
        if talent_def is None:
            continue

        capped_ranks = min(ranks, talent_def.max_ranks)
        for field, value_per_rank in talent_def.effect_per_rank.items():
            accum[field] = accum.get(field, 0.0) + value_per_rank * capped_ranks

    # Build kwargs for TalentModifiers
    kwargs: dict[str, float | int | bool] = {}

    for field, total in accum.items():
        if field in _BOOLEAN_FIELDS:
            kwargs[field] = total >= 1
        elif field in _INT_FIELDS:
            kwargs[field] = int(total)
        elif field in _MULTIPLICATIVE_FIELDS:
            kwargs[field] = 1.0 + total
        else:
            kwargs[field] = total

    return TalentModifiers.model_validate(kwargs)


def get_spec_template(spec: RogueSpec) -> dict[str, int]:
    """Return canonical talent point distribution for a spec.

    Args:
        spec: The Rogue specialization.

    Returns:
        Dict of ``talent_name -> ranks`` for the standard build.

    Raises:
        ValueError: If the spec has no defined template.
    """
    template = _SPEC_TEMPLATES.get(spec)
    if template is None:
        raise ValueError(f"No talent template defined for spec: {spec}")
    return dict(template)
