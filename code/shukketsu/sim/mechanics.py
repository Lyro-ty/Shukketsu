"""TBC 2.4.3 combat formula functions.

All functions are pure (no state, no side effects). Constants are defined at
module level and correspond to WoW TBC patch 2.4.3 values for a Level 70
Rogue attacking a Level 73 raid boss.
"""

from code.shukketsu.sim.models import HitOutcome, WeaponType

# =============================================================================
# Rating conversions (Level 70)
# =============================================================================

HIT_RATING_PER_PCT = 15.77
CRIT_RATING_PER_PCT = 22.08
HASTE_RATING_PER_PCT = 15.77
EXPERTISE_RATING_PER_POINT = 3.9423
AP_PER_DPS = 14.0
AGI_PER_CRIT_PCT = 40.0
AGI_PER_AP = 1.0
STR_PER_AP = 1.0

# =============================================================================
# Boss (Level 73)
# =============================================================================

BASE_MISS_CHANCE = 0.08
HIT_SUPPRESSION = 0.01
DW_MISS_PENALTY = 0.19
BASE_DODGE_CHANCE = 0.065
BASE_PARRY_CHANCE = 0.14
BASE_GLANCING_CHANCE = 0.24
GLANCING_MULTIPLIER = 0.75
CRIT_SUPPRESSION = 0.048

# =============================================================================
# Armor
# =============================================================================

ARMOR_CONSTANT = 10557.5
MAX_ARMOR_REDUCTION = 0.75

# =============================================================================
# Weapon normalization speeds
# =============================================================================

NORM_SPEED_DAGGER = 1.7
NORM_SPEED_ONE_HAND = 2.4

# =============================================================================
# Energy
# =============================================================================

ENERGY_TICK_MS = 2020
ENERGY_PER_TICK = 20.2
BASE_MAX_ENERGY = 100
VIGOR_BONUS_ENERGY = 10

# =============================================================================
# Crit
# =============================================================================

MELEE_CRIT_MULTIPLIER = 2.0
SPELL_CRIT_MULTIPLIER = 1.5
ROGUE_BASE_CRIT_ADJUSTMENT = -0.003

# =============================================================================
# Rogue
# =============================================================================

ROGUE_THREAT_MULTIPLIER = 0.71
BUILDER_MISS_REFUND = 0.80

# =============================================================================
# Boss armor presets
# =============================================================================

BOSS_ARMOR_STANDARD = 7700
BOSS_ARMOR_CASTER = 6200

# =============================================================================
# Armor debuffs
# =============================================================================

SUNDER_ARMOR_PER_STACK = 520
EXPOSE_ARMOR_BASE = 2050
FAERIE_FIRE_ARMOR = 610
CURSE_OF_RECKLESSNESS_ARMOR = 800


# =============================================================================
# Combat formula functions
# =============================================================================


def resolve_white_hit(
    miss_chance: float,
    dodge_chance: float,
    glancing_chance: float,
    crit_chance: float,
    roll: float,
) -> HitOutcome:
    """Resolve a white (auto-attack) hit using the single-roll attack table.

    The single-roll table stacks outcomes in order: miss, dodge, glancing,
    crit, hit. Crit CAN be pushed off the table if earlier outcomes consume
    the probability space.

    Args:
        miss_chance: Probability of a miss [0, 1].
        dodge_chance: Probability of a dodge [0, 1].
        glancing_chance: Probability of a glancing blow [0, 1].
        crit_chance: Probability of a critical strike [0, 1].
        roll: Uniform random value in [0, 1).

    Returns:
        The resulting HitOutcome.
    """
    threshold = miss_chance
    if roll < threshold:
        return HitOutcome.MISS

    threshold += dodge_chance
    if roll < threshold:
        return HitOutcome.DODGE

    threshold += glancing_chance
    if roll < threshold:
        return HitOutcome.GLANCING

    threshold += crit_chance
    if roll < threshold:
        return HitOutcome.CRIT

    return HitOutcome.HIT


def resolve_yellow_hit(
    miss_chance: float,
    dodge_chance: float,
    crit_chance: float,
    hit_roll: float,
    crit_roll: float,
    *,
    can_be_dodged: bool = True,
) -> HitOutcome:
    """Resolve a yellow (special ability) hit using the two-roll system.

    Roll 1 determines miss/dodge. Roll 2 independently determines crit.
    Crit CANNOT be pushed off the table for yellow attacks.

    Args:
        miss_chance: Probability of a miss [0, 1].
        dodge_chance: Probability of a dodge [0, 1].
        crit_chance: Probability of a critical strike [0, 1].
        hit_roll: Uniform random value in [0, 1) for miss/dodge check.
        crit_roll: Uniform random value in [0, 1) for crit check.
        can_be_dodged: Whether the ability can be dodged (e.g., Backstab
            from behind cannot be parried but CAN be dodged; some abilities
            like Shiv cannot be dodged).

    Returns:
        The resulting HitOutcome.
    """
    # Roll 1: miss / dodge
    if hit_roll < miss_chance:
        return HitOutcome.MISS

    if can_be_dodged and hit_roll < miss_chance + dodge_chance:
        return HitOutcome.DODGE

    # Roll 2: crit (independent)
    if crit_roll < crit_chance:
        return HitOutcome.CRIT

    return HitOutcome.HIT


def calc_miss_chance(
    hit_rating: float,
    *,
    is_dual_wield: bool,
    is_yellow: bool,
    precision_ranks: int = 0,
) -> float:
    """Calculate effective miss chance against a Level 73 boss.

    Yellow attacks do not suffer the dual-wield miss penalty.

    Args:
        hit_rating: Total hit rating from gear.
        is_dual_wield: Whether the character is dual-wielding.
        is_yellow: Whether this is a yellow (special) attack.
        precision_ranks: Ranks of Precision talent (0-5, 1% hit per rank).

    Returns:
        Miss chance clamped to [0, 1].
    """
    base = BASE_MISS_CHANCE + HIT_SUPPRESSION  # 9% vs boss

    if is_dual_wield and not is_yellow:
        base += DW_MISS_PENALTY  # +19% for DW white hits

    hit_pct = hit_rating / HIT_RATING_PER_PCT / 100.0
    precision_pct = precision_ranks * 0.01

    result = base - hit_pct - precision_pct
    return max(0.0, min(1.0, result))


def calc_dodge_chance(
    expertise_rating: float,
    *,
    weapon_expertise_ranks: int = 0,
) -> float:
    """Calculate effective dodge chance against a Level 73 boss.

    Args:
        expertise_rating: Total expertise rating from gear.
        weapon_expertise_ranks: Ranks of Weapon Expertise talent (0-2,
            5 expertise per rank).

    Returns:
        Dodge chance clamped to [0, 1].
    """
    expertise_points = expertise_rating / EXPERTISE_RATING_PER_POINT
    expertise_points += weapon_expertise_ranks * 5.0

    # Each point of expertise reduces dodge by 0.25%
    dodge_reduction = expertise_points * 0.0025

    result = BASE_DODGE_CHANCE - dodge_reduction
    return max(0.0, min(1.0, result))


def calc_crit_chance(
    crit_rating: float,
    agility: float,
    *,
    base_crit: float = 0.0,
    talent_crit: float = 0.0,
    bonus_crit: float = 0.0,
    crit_suppression: bool = True,
) -> float:
    """Calculate effective crit chance against a target.

    Args:
        crit_rating: Total crit rating from gear.
        agility: Total agility after buffs.
        base_crit: Base crit chance from class (fraction, e.g., 0.0).
        talent_crit: Crit chance from talents (fraction, e.g., 0.05 for 5%).
        bonus_crit: Additional crit chance from buffs/debuffs (fraction).
        crit_suppression: Whether to apply crit suppression vs boss (-4.8%).

    Returns:
        Effective crit chance (can be negative if suppression dominates).
    """
    from_rating = crit_rating / CRIT_RATING_PER_PCT / 100.0
    from_agility = agility / AGI_PER_CRIT_PCT / 100.0

    total = base_crit + from_rating + from_agility + talent_crit + bonus_crit + ROGUE_BASE_CRIT_ADJUSTMENT

    if crit_suppression:
        total -= CRIT_SUPPRESSION

    return total


def calc_crit_multiplier(
    base_mult: float,
    *,
    primary_mod: float = 1.0,
    secondary_mod: float = 0.0,
    has_meta_gem: bool = False,
) -> float:
    """Calculate effective crit damage multiplier using the WoWSims formula.

    Formula: 1.0 + (base * primary - 1.0) * (1.0 + secondary)
    Meta gem (Relentless Earthstorm Diamond): primary *= 1.03.

    Args:
        base_mult: Base crit multiplier (2.0 for melee, 1.5 for spells).
        primary_mod: Primary crit damage modifier (e.g., 1.0 default).
        secondary_mod: Secondary crit damage modifier (e.g., Lethality 0.30).
        has_meta_gem: Whether the Relentless Earthstorm Diamond is active.

    Returns:
        Effective crit damage multiplier.
    """
    effective_primary = primary_mod
    if has_meta_gem:
        effective_primary *= 1.03

    return 1.0 + (base_mult * effective_primary - 1.0) * (1.0 + secondary_mod)


def calc_armor_reduction(
    armor: int,
    *,
    arpen: int = 0,
    sunder_stacks: int = 0,
    expose_armor_ranks: int = 0,
    faerie_fire: bool = False,
    curse_of_recklessness: bool = False,
) -> float:
    """Calculate the damage multiplier from armor mitigation.

    Debuffs reduce armor first, then armor penetration is applied, then the
    reduction formula. Returns a MULTIPLIER: 1.0 means no reduction, 0.25
    means maximum 75% reduction.

    Args:
        armor: Target base armor value.
        arpen: Armor penetration rating from gear.
        sunder_stacks: Number of Sunder Armor stacks (0-5).
        expose_armor_ranks: Rank of Expose Armor (0 or 1 for presence).
        faerie_fire: Whether Faerie Fire is active.
        curse_of_recklessness: Whether Curse of Recklessness is active.

    Returns:
        Damage multiplier clamped to [0.25, 1.0].
    """
    effective_armor = float(armor)

    # Apply debuffs (flat reduction)
    effective_armor -= min(sunder_stacks, 5) * SUNDER_ARMOR_PER_STACK
    if expose_armor_ranks > 0:
        effective_armor -= EXPOSE_ARMOR_BASE
    if faerie_fire:
        effective_armor -= FAERIE_FIRE_ARMOR
    if curse_of_recklessness:
        effective_armor -= CURSE_OF_RECKLESSNESS_ARMOR

    # Armor cannot go below 0 after debuffs
    effective_armor = max(0.0, effective_armor)

    # Apply armor penetration
    effective_armor = max(0.0, effective_armor - arpen)

    # Calculate reduction
    if effective_armor <= 0:
        return 1.0

    reduction = effective_armor / (effective_armor + ARMOR_CONSTANT)
    reduction = min(reduction, MAX_ARMOR_REDUCTION)

    return 1.0 - reduction


def calc_weapon_damage(
    min_dmg: float,
    max_dmg: float,
    speed: float,
    attack_power: float,
    *,
    normalized: bool = False,
    norm_speed: float = 2.4,
    roll: float = 0.5,
) -> float:
    """Calculate weapon damage for a single swing.

    Base weapon damage is interpolated between min and max using the roll.
    Attack power contributes AP/14 * speed (or norm_speed if normalized).

    Args:
        min_dmg: Weapon minimum damage.
        max_dmg: Weapon maximum damage.
        speed: Weapon base speed.
        attack_power: Total attack power.
        normalized: Whether to use normalized weapon speed for AP scaling.
        norm_speed: Normalization speed (1.7 for daggers, 2.4 for others).
        roll: Uniform random value in [0, 1] to interpolate min/max damage.

    Returns:
        Total weapon damage for this swing.
    """
    base_damage = min_dmg + (max_dmg - min_dmg) * roll
    effective_speed = norm_speed if normalized else speed
    ap_bonus = attack_power / AP_PER_DPS * effective_speed

    return base_damage + ap_bonus


def calc_haste_multiplier(haste_rating: float, *multiplicative_buffs: float) -> float:
    """Calculate the total haste multiplier from rating and buffs.

    All haste sources stack multiplicatively.

    Args:
        haste_rating: Total haste rating from gear.
        *multiplicative_buffs: Additional multiplicative haste factors
            (e.g., 1.30 for Slice and Dice).

    Returns:
        Combined haste multiplier (>= 1.0).
    """
    from_rating = 1.0 + haste_rating / (HASTE_RATING_PER_PCT * 100.0)

    result = from_rating
    for buff in multiplicative_buffs:
        result *= buff

    return result


def calc_effective_speed(base_speed: float, haste_multiplier: float) -> float:
    """Calculate effective weapon speed after haste.

    Args:
        base_speed: Weapon base speed in seconds.
        haste_multiplier: Combined haste multiplier from calc_haste_multiplier.

    Returns:
        Effective weapon speed in seconds.
    """
    return base_speed / haste_multiplier


def calc_poison_proc_chance(base_chance: float, imp_poisons_ranks: int = 0) -> float:
    """Calculate poison proc chance with Improved Poisons talent.

    Args:
        base_chance: Base proc chance for the poison (e.g., 0.20 for IP rank 7).
        imp_poisons_ranks: Ranks of Improved Poisons talent (0-5).

    Returns:
        Effective proc chance.
    """
    return base_chance + 0.02 * imp_poisons_ranks


def calc_ppm_proc_chance(ppm: float, weapon_speed: float) -> float:
    """Calculate per-hit proc chance from a PPM (procs per minute) value.

    Args:
        ppm: Procs per minute value.
        weapon_speed: Effective weapon speed in seconds.

    Returns:
        Per-hit proc chance.
    """
    return ppm * weapon_speed / 60.0


def calc_glancing_reduction() -> float:
    """Return the glancing blow damage multiplier vs a Level 73 boss.

    Returns:
        Glancing blow damage multiplier (0.75 for 3-level difference).
    """
    return GLANCING_MULTIPLIER


def calc_normalized_speed(weapon_type: WeaponType) -> float:
    """Return the normalized weapon speed for the given weapon type.

    Args:
        weapon_type: The weapon type.

    Returns:
        Normalized speed: 1.7 for daggers, 2.4 for all others.
    """
    if weapon_type == WeaponType.DAGGER:
        return NORM_SPEED_DAGGER
    return NORM_SPEED_ONE_HAND
