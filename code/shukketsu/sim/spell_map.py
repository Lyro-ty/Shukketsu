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
    26862: "sinister_strike",  # Rank 10 (TBC max)
    53: "backstab",
    26863: "backstab",  # Rank 10 (TBC max)
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
    27441: "ambush",  # Rank 7 (TBC max)
    11290: "garrote",
    26884: "garrote",  # Rank 8 (TBC max)
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
