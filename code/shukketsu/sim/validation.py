"""Validation profiles for the TBC Rogue DPS simulation engine.

Canonical profiles with expected DPS ranges based on community
benchmarks (WoWSims, Shadowpanther, Elitist Jerks). Used for
regression testing and sanity checking.
"""

import logging
from typing import Any

from pydantic import BaseModel

from code.shukketsu.sim.buffs import get_preset
from code.shukketsu.sim.models import (
    BossConfig,
    GearSlot,
    PoisonConfig,
    Race,
    RogueSpec,
    SimConfig,
)

logger = logging.getLogger(__name__)


class ValidationProfile(BaseModel):
    """A canonical DPS validation profile.

    Contains a full SimConfig along with the expected DPS range from
    community benchmarks, used to detect regression or drift in the
    simulation engine.
    """

    name: str
    spec: RogueSpec
    phase: int
    config: SimConfig
    expected_dps_range: tuple[float, float]
    tolerance_pct: float = 2.0


class ValidationResult(BaseModel):
    """Result of running a validation profile against the sim engine."""

    profile_name: str
    our_dps: float
    expected_range: tuple[float, float]
    within_range: bool
    drift_pct: float


def _build_config(
    spec: RogueSpec,
    race: Race,
    talents: str,
    gear: dict[GearSlot, int],
    *,
    preset: str = "full_25man",
    fight_length: int = 300,
    iterations: int = 10000,
    **overrides: Any,
) -> SimConfig:
    """Build a SimConfig from components with sensible defaults.

    Args:
        spec: Rogue specialization.
        race: Character race.
        talents: Talent string in ``"X/Y/Z"`` format.
        gear: Mapping of gear slots to item IDs.
        preset: Raid buff preset name (e.g. ``"full_25man"``).
        fight_length: Fight duration in seconds.
        iterations: Number of sim iterations.
        **overrides: Additional SimConfig fields to override.

    Returns:
        A fully populated SimConfig.
    """
    buff_ids, debuff_ids, consumable_ids = get_preset(preset)
    config_data: dict[str, Any] = {
        "spec": spec,
        "race": race,
        "talents": talents,
        "gear": gear,
        "buffs": buff_ids,
        "consumables": consumable_ids,
        "boss": BossConfig(debuffs=debuff_ids),
        "poisons": PoisonConfig(),
        "fight_length": fight_length,
        "iterations": iterations,
        "raid_preset": preset,
    }
    config_data.update(overrides)
    return SimConfig(**config_data)


# ---------------------------------------------------------------------------
# Talent strings — "Assassination/Combat/Subtlety" point distribution
# ---------------------------------------------------------------------------

# Combat Swords: 20/41/0 (standard)
_COMBAT_SWORDS_TALENTS = "20/41/0"
# Combat Daggers: 20/41/0 (dagger variant)
_COMBAT_DAGGERS_TALENTS = "20/41/0"
# Combat Fists: 20/41/0 (fist variant)
_COMBAT_FISTS_TALENTS = "20/41/0"
# Mutilate: 41/20/0
_MUTILATE_TALENTS = "41/20/0"


# ---------------------------------------------------------------------------
# P1 BiS Combat Swords gear (all IDs verified in items.py)
# ---------------------------------------------------------------------------

_P1_COMBAT_SWORDS_GEAR: dict[GearSlot, int] = {
    GearSlot.HEAD: 29044,  # Netherblade Facemask
    GearSlot.NECK: 28762,  # Adornment of Stolen Souls
    GearSlot.SHOULDER: 29048,  # Netherblade Pauldrons
    GearSlot.BACK: 28672,  # Drape of the Dark Reavers
    GearSlot.CHEST: 29045,  # Netherblade Chestpiece
    GearSlot.WRIST: 29246,  # Nightfall Wristguards
    GearSlot.HANDS: 29047,  # Netherblade Gloves
    GearSlot.WAIST: 28828,  # Gronn-Stitched Girdle
    GearSlot.LEGS: 29046,  # Netherblade Breeches
    GearSlot.FEET: 28545,  # Edgewalker Longboots
    GearSlot.RING_1: 28757,  # Ring of a Thousand Marks
    GearSlot.RING_2: 28649,  # Garona's Signet Ring
    GearSlot.TRINKET_1: 28830,  # Dragonspine Trophy
    GearSlot.TRINKET_2: 29383,  # Bloodlust Brooch
    GearSlot.MAIN_HAND: 28189,  # Latro's Shifting Sword
    GearSlot.OFF_HAND: 28295,  # Gladiator's Shiv
    GearSlot.RANGED: 28772,  # Sunfury Bow of the Phoenix
}

# ---------------------------------------------------------------------------
# P3 BiS Combat Swords gear (Hyjal / BT tier)
# ---------------------------------------------------------------------------

_P3_COMBAT_SWORDS_GEAR: dict[GearSlot, int] = {
    GearSlot.HEAD: 31027,  # Slayer's Helm
    GearSlot.NECK: 30017,  # Telonicus's Pendant of Mayhem
    GearSlot.SHOULDER: 31030,  # Slayer's Shoulderpads
    GearSlot.BACK: 32323,  # Shadowmoon Destroyer's Drape
    GearSlot.CHEST: 31028,  # Slayer's Chestguard
    GearSlot.WRIST: 32324,  # Insidious Bands
    GearSlot.HANDS: 31026,  # Slayer's Handguards
    GearSlot.WAIST: 32348,  # Belt of One-Hundred Deaths
    GearSlot.LEGS: 31029,  # Slayer's Legguards
    GearSlot.FEET: 32366,  # Shadowmaster's Boots
    GearSlot.RING_1: 32497,  # Stormrage Signet Ring
    GearSlot.RING_2: 30834,  # Shapeshifter's Signet
    GearSlot.TRINKET_1: 28830,  # Dragonspine Trophy
    GearSlot.TRINKET_2: 32505,  # Madness of the Betrayer
    GearSlot.MAIN_HAND: 32837,  # Warglaive of Azzinoth MH
    GearSlot.OFF_HAND: 32471,  # Shard of Azzinoth
    GearSlot.RANGED: 30724,  # Barrel-Blade Longrifle
}

# ---------------------------------------------------------------------------
# P5 BiS Combat Swords gear (Sunwell tier)
# ---------------------------------------------------------------------------

_P5_COMBAT_SWORDS_GEAR: dict[GearSlot, int] = {
    GearSlot.HEAD: 34244,  # Duplicitous Guise
    GearSlot.NECK: 34177,  # Clutch of Demise
    GearSlot.SHOULDER: 31030,  # Slayer's Shoulderpads (no P5 shoulders in DB)
    GearSlot.BACK: 32323,  # Shadowmoon Destroyer's Drape (no P5 cloak in DB)
    GearSlot.CHEST: 33496,  # Nether Shadow Tunic (P4)
    GearSlot.WRIST: 32324,  # Insidious Bands (no P5 wrists in DB)
    GearSlot.HANDS: 31026,  # Slayer's Handguards (no P5 hands in DB)
    GearSlot.WAIST: 33503,  # Waistguard of the Great Beast (P4)
    GearSlot.LEGS: 31029,  # Slayer's Legguards (no P5 legs in DB)
    GearSlot.FEET: 32366,  # Shadowmaster's Boots (no P5 boots in DB)
    GearSlot.RING_1: 32497,  # Stormrage Signet Ring
    GearSlot.RING_2: 30834,  # Shapeshifter's Signet
    GearSlot.TRINKET_1: 28830,  # Dragonspine Trophy
    GearSlot.TRINKET_2: 33831,  # Berserker's Call
    GearSlot.MAIN_HAND: 34164,  # Mounting Vengeance
    GearSlot.OFF_HAND: 34346,  # Hammer of Judgement
    GearSlot.RANGED: 34334,  # Thori'dal, the Stars' Fury
}

# ---------------------------------------------------------------------------
# P1 Combat Daggers gear
# ---------------------------------------------------------------------------

_P1_COMBAT_DAGGERS_GEAR: dict[GearSlot, int] = {
    GearSlot.HEAD: 29044,  # Netherblade Facemask
    GearSlot.NECK: 28762,  # Adornment of Stolen Souls
    GearSlot.SHOULDER: 29048,  # Netherblade Pauldrons
    GearSlot.BACK: 28672,  # Drape of the Dark Reavers
    GearSlot.CHEST: 29045,  # Netherblade Chestpiece
    GearSlot.WRIST: 29246,  # Nightfall Wristguards
    GearSlot.HANDS: 29047,  # Netherblade Gloves
    GearSlot.WAIST: 28828,  # Gronn-Stitched Girdle
    GearSlot.LEGS: 29046,  # Netherblade Breeches
    GearSlot.FEET: 28545,  # Edgewalker Longboots
    GearSlot.RING_1: 28757,  # Ring of a Thousand Marks
    GearSlot.RING_2: 28649,  # Garona's Signet Ring
    GearSlot.TRINKET_1: 28830,  # Dragonspine Trophy
    GearSlot.TRINKET_2: 29383,  # Bloodlust Brooch
    GearSlot.MAIN_HAND: 28524,  # Emerald Ripper (dagger MH)
    GearSlot.OFF_HAND: 28295,  # Gladiator's Shiv (dagger OH)
    GearSlot.RANGED: 28772,  # Sunfury Bow of the Phoenix
}


# ---------------------------------------------------------------------------
# Validation profiles
# ---------------------------------------------------------------------------

VALIDATION_PROFILES: list[ValidationProfile] = [
    ValidationProfile(
        name="P1 BiS Combat Swords",
        spec=RogueSpec.COMBAT_SWORDS,
        phase=1,
        config=_build_config(
            RogueSpec.COMBAT_SWORDS,
            Race.HUMAN,
            _COMBAT_SWORDS_TALENTS,
            _P1_COMBAT_SWORDS_GEAR,
        ),
        expected_dps_range=(1000.0, 1600.0),
    ),
    ValidationProfile(
        name="P3 BiS Combat Swords",
        spec=RogueSpec.COMBAT_SWORDS,
        phase=3,
        config=_build_config(
            RogueSpec.COMBAT_SWORDS,
            Race.HUMAN,
            _COMBAT_SWORDS_TALENTS,
            _P3_COMBAT_SWORDS_GEAR,
        ),
        expected_dps_range=(1600.0, 2200.0),
    ),
    ValidationProfile(
        name="P5 BiS Combat Swords",
        spec=RogueSpec.COMBAT_SWORDS,
        phase=5,
        config=_build_config(
            RogueSpec.COMBAT_SWORDS,
            Race.HUMAN,
            _COMBAT_SWORDS_TALENTS,
            _P5_COMBAT_SWORDS_GEAR,
        ),
        expected_dps_range=(2000.0, 2800.0),
    ),
    ValidationProfile(
        name="P3 BiS Combat Fists",
        spec=RogueSpec.COMBAT_FISTS,
        phase=3,
        config=_build_config(
            RogueSpec.COMBAT_FISTS,
            Race.ORC,
            _COMBAT_FISTS_TALENTS,
            _P3_COMBAT_SWORDS_GEAR,
        ),
        expected_dps_range=(1500.0, 2100.0),
    ),
    ValidationProfile(
        name="P3 BiS Mutilate",
        spec=RogueSpec.ASSASSINATION_MUTILATE,
        phase=3,
        config=_build_config(
            RogueSpec.ASSASSINATION_MUTILATE,
            Race.BLOOD_ELF,
            _MUTILATE_TALENTS,
            _P3_COMBAT_SWORDS_GEAR,
        ),
        expected_dps_range=(1400.0, 2000.0),
        tolerance_pct=4.0,
    ),
    ValidationProfile(
        name="P5 BiS Mutilate",
        spec=RogueSpec.ASSASSINATION_MUTILATE,
        phase=5,
        config=_build_config(
            RogueSpec.ASSASSINATION_MUTILATE,
            Race.BLOOD_ELF,
            _MUTILATE_TALENTS,
            _P5_COMBAT_SWORDS_GEAR,
        ),
        expected_dps_range=(1800.0, 2600.0),
        tolerance_pct=4.0,
    ),
    ValidationProfile(
        name="P1 BiS Combat Daggers",
        spec=RogueSpec.COMBAT_DAGGERS,
        phase=1,
        config=_build_config(
            RogueSpec.COMBAT_DAGGERS,
            Race.NIGHT_ELF,
            _COMBAT_DAGGERS_TALENTS,
            _P1_COMBAT_DAGGERS_GEAR,
        ),
        expected_dps_range=(900.0, 1500.0),
    ),
    # No Windfury profile — Karazhan 10-man (no WF totem available)
    ValidationProfile(
        name="No WF Group (P3 Combat)",
        spec=RogueSpec.COMBAT_SWORDS,
        phase=3,
        config=_build_config(
            RogueSpec.COMBAT_SWORDS,
            Race.HUMAN,
            _COMBAT_SWORDS_TALENTS,
            _P3_COMBAT_SWORDS_GEAR,
            preset="karazhan_10man",
        ),
        expected_dps_range=(1300.0, 1900.0),
    ),
    # Solo / no buffs
    ValidationProfile(
        name="Solo No Buffs (P3 Combat)",
        spec=RogueSpec.COMBAT_SWORDS,
        phase=3,
        config=_build_config(
            RogueSpec.COMBAT_SWORDS,
            Race.HUMAN,
            _COMBAT_SWORDS_TALENTS,
            _P3_COMBAT_SWORDS_GEAR,
            preset="solo",
        ),
        expected_dps_range=(800.0, 1400.0),
    ),
]


def get_validation_profiles() -> list[ValidationProfile]:
    """Return all canonical validation profiles."""
    return VALIDATION_PROFILES


async def run_validation(runner: Any, profile: ValidationProfile) -> ValidationResult:
    """Run a validation profile and check against expected range.

    Args:
        runner: A SimRunner instance with an async ``sim_run`` method.
        profile: The validation profile to check.

    Returns:
        ValidationResult with pass/fail and drift metrics.
    """
    result = await runner.sim_run(profile.config)
    our_dps = result.dps_mean
    low, high = profile.expected_dps_range
    within = low <= our_dps <= high

    # Calculate drift from midpoint of expected range
    midpoint = (low + high) / 2
    drift_pct = ((our_dps - midpoint) / midpoint) * 100 if midpoint > 0 else 0.0

    logger.info(
        "Validation %s: DPS=%.1f range=(%.1f, %.1f) %s drift=%.1f%%",
        profile.name,
        our_dps,
        low,
        high,
        "PASS" if within else "FAIL",
        drift_pct,
    )

    return ValidationResult(
        profile_name=profile.name,
        our_dps=our_dps,
        expected_range=profile.expected_dps_range,
        within_range=within,
        drift_pct=drift_pct,
    )
