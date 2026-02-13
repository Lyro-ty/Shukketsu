"""Character import parsing for SimC, SeventyUpgrades, and WoWSims formats.

Supports auto-detection of import format and conversion to SimConfig
for the simulation engine.
"""

import json
import logging
import re
from enum import StrEnum

from pydantic import BaseModel

from code.shukketsu.resilience.errors import InvalidSimConfigError
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


# --- Enums ---


class ImportFormat(StrEnum):
    """Supported character import formats."""

    SIMC = "simc"
    SEVENTYUPGRADES = "seventyupgrades"
    WOWSIMS = "wowsims"


# --- Models ---


class CharacterImport(BaseModel):
    """Parsed character data from an external tool."""

    spec: RogueSpec
    race: Race
    talents: str
    gear: dict[GearSlot, int]
    enchants: dict[GearSlot, int] = {}
    gems: dict[GearSlot, list[int]] = {}


# --- Slot Mappings ---

_SIMC_SLOT_MAP: dict[str, GearSlot] = {
    "head": GearSlot.HEAD,
    "neck": GearSlot.NECK,
    "shoulder": GearSlot.SHOULDER,
    "back": GearSlot.BACK,
    "chest": GearSlot.CHEST,
    "wrist": GearSlot.WRIST,
    "hands": GearSlot.HANDS,
    "waist": GearSlot.WAIST,
    "legs": GearSlot.LEGS,
    "feet": GearSlot.FEET,
    "finger1": GearSlot.RING_1,
    "finger2": GearSlot.RING_2,
    "trinket1": GearSlot.TRINKET_1,
    "trinket2": GearSlot.TRINKET_2,
    "main_hand": GearSlot.MAIN_HAND,
    "off_hand": GearSlot.OFF_HAND,
    "ranged": GearSlot.RANGED,
}

_70U_SLOT_MAP: dict[str, GearSlot] = {
    "Head": GearSlot.HEAD,
    "Neck": GearSlot.NECK,
    "Shoulder": GearSlot.SHOULDER,
    "Back": GearSlot.BACK,
    "Chest": GearSlot.CHEST,
    "Wrist": GearSlot.WRIST,
    "Hands": GearSlot.HANDS,
    "Waist": GearSlot.WAIST,
    "Legs": GearSlot.LEGS,
    "Feet": GearSlot.FEET,
    "Ring 1": GearSlot.RING_1,
    "Ring 2": GearSlot.RING_2,
    "Finger 1": GearSlot.RING_1,
    "Finger 2": GearSlot.RING_2,
    "Trinket 1": GearSlot.TRINKET_1,
    "Trinket 2": GearSlot.TRINKET_2,
    "Main Hand": GearSlot.MAIN_HAND,
    "Off Hand": GearSlot.OFF_HAND,
    "Ranged": GearSlot.RANGED,
}

_RACE_MAP: dict[str, Race] = {
    "orc": Race.ORC,
    "human": Race.HUMAN,
    "night_elf": Race.NIGHT_ELF,
    "nightelf": Race.NIGHT_ELF,
    "blood_elf": Race.BLOOD_ELF,
    "bloodelf": Race.BLOOD_ELF,
    "undead": Race.UNDEAD,
    "dwarf": Race.DWARF,
    "gnome": Race.GNOME,
    "troll": Race.TROLL,
}

_SPEC_MAP: dict[str, RogueSpec] = {
    "combat": RogueSpec.COMBAT_SWORDS,
    "combat_swords": RogueSpec.COMBAT_SWORDS,
    "combat_fists": RogueSpec.COMBAT_FISTS,
    "combat_daggers": RogueSpec.COMBAT_DAGGERS,
    "assassination": RogueSpec.ASSASSINATION_MUTILATE,
    "assassination_mutilate": RogueSpec.ASSASSINATION_MUTILATE,
    "mutilate": RogueSpec.ASSASSINATION_MUTILATE,
}


# --- Format Detection ---


def detect_format(raw: str) -> ImportFormat:
    """Auto-detect import format from raw text.

    Key=value pattern in first non-empty line -> SIMC (e.g. ``rogue="Name"``).
    Starts with '{' or '[' -> SEVENTYUPGRADES (JSON).
    Otherwise -> WOWSIMS (Base64).
    """
    stripped = raw.strip()
    if not stripped:
        return ImportFormat.WOWSIMS

    first_line = stripped.split("\n")[0].strip()

    if first_line.startswith(("{", "[")):
        return ImportFormat.SEVENTYUPGRADES
    # SimC lines have word=value pattern where '=' is followed by content
    # (Base64 only has '=' as trailing padding)
    if re.match(r"^[a-zA-Z_]\w*=.+", first_line):
        return ImportFormat.SIMC
    return ImportFormat.WOWSIMS


# --- Parsers ---


def parse_simc(raw: str) -> CharacterImport:
    """Parse /simc addon output (key=value lines).

    Extracts race, spec, talents, and gear slot IDs from the SimC
    export format. Maps SimC slot names to GearSlot enum values.

    Raises:
        InvalidSimConfigError: If required fields are missing or malformed.
    """
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]

    race: Race | None = None
    spec: RogueSpec | None = None
    talents: str = ""
    gear: dict[GearSlot, int] = {}

    for line in lines:
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        if key == "race":
            race_key = value.lower().strip('"')
            if race_key not in _RACE_MAP:
                raise InvalidSimConfigError(f"Unknown race: {value}")
            race = _RACE_MAP[race_key]
        elif key == "spec":
            spec_key = value.lower().strip('"')
            if spec_key not in _SPEC_MAP:
                raise InvalidSimConfigError(f"Unknown spec: {value}")
            spec = _SPEC_MAP[spec_key]
        elif key == "talents":
            talents = value.strip('"')
        elif key in _SIMC_SLOT_MAP:
            # Gear line: slot=,id=XXXXX or slot=,id=XXXXX,enchant_id=YYY,...
            item_id_match = re.search(r"id=(\d+)", value)
            if item_id_match:
                gear[_SIMC_SLOT_MAP[key]] = int(item_id_match.group(1))

    if spec is None:
        raise InvalidSimConfigError("Missing spec in SimC import")
    if race is None:
        raise InvalidSimConfigError("Missing race in SimC import")
    if not talents:
        raise InvalidSimConfigError("Missing talents in SimC import")

    return CharacterImport(spec=spec, race=race, talents=talents, gear=gear)


def parse_seventyupgrades(raw: str) -> CharacterImport:
    """Parse seventyupgrades.com JSON export.

    Extracts race, spec, talents, and gear from the JSON structure
    exported by seventyupgrades.com.

    Raises:
        InvalidSimConfigError: If JSON is malformed or missing required fields.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise InvalidSimConfigError(f"Invalid JSON in SeventyUpgrades import: {e}") from e

    # Extract race
    raw_race = data.get("race", "").lower().replace(" ", "_")
    if raw_race not in _RACE_MAP:
        raise InvalidSimConfigError(f"Unknown race: {data.get('race')}")
    race = _RACE_MAP[raw_race]

    # Extract spec
    raw_spec = data.get("spec", "").lower().replace(" ", "_")
    if raw_spec not in _SPEC_MAP:
        raise InvalidSimConfigError(f"Unknown spec: {data.get('spec')}")
    spec = _SPEC_MAP[raw_spec]

    # Extract talents
    talents = data.get("talents", "")
    if not talents:
        raise InvalidSimConfigError("Missing talents in SeventyUpgrades import")

    # Extract gear
    gear: dict[GearSlot, int] = {}
    for item in data.get("items", []):
        slot_name = item.get("slot", "")
        item_id = item.get("id")
        if slot_name in _70U_SLOT_MAP and item_id is not None:
            gear[_70U_SLOT_MAP[slot_name]] = int(item_id)

    return CharacterImport(spec=spec, race=race, talents=talents, gear=gear)


def parse_wowsims(raw: str) -> CharacterImport:
    """Parse WoWSims Base64 URL export.

    This format uses protobuf encoding and is not yet implemented.

    Raises:
        InvalidSimConfigError: Always, as this format is not yet supported.
    """
    raise InvalidSimConfigError("WoWSims import not yet supported")


def parse_import(raw: str) -> CharacterImport:
    """Auto-detect format and parse character import data.

    Args:
        raw: Raw import string (SimC text, JSON, or Base64).

    Returns:
        Parsed CharacterImport ready for config building.
    """
    fmt = detect_format(raw)
    if fmt == ImportFormat.SIMC:
        return parse_simc(raw)
    if fmt == ImportFormat.SEVENTYUPGRADES:
        return parse_seventyupgrades(raw)
    return parse_wowsims(raw)


def build_config(
    char_import: CharacterImport,
    *,
    preset: str = "full_25man",
    **overrides: int | str | bool | float,
) -> SimConfig:
    """Build SimConfig from import + buff preset + overrides.

    Combines a parsed CharacterImport with a raid buff preset and any
    additional simulation parameter overrides into a complete SimConfig.

    Args:
        char_import: Parsed character data.
        preset: Raid buff preset name (default "full_25man").
        **overrides: Additional SimConfig field overrides.

    Returns:
        Complete SimConfig ready for simulation.
    """
    buff_ids, debuff_ids, consumable_ids = get_preset(preset)

    config_data: dict[str, object] = {
        "spec": char_import.spec,
        "race": char_import.race,
        "talents": char_import.talents,
        "gear": char_import.gear,
        "enchants": char_import.enchants,
        "gems": char_import.gems,
        "buffs": buff_ids,
        "consumables": consumable_ids,
        "boss": BossConfig(debuffs=debuff_ids),
        "poisons": PoisonConfig(),
        "raid_preset": preset,
    }

    # Apply overrides (e.g. fight_length=180, iterations=5000)
    for key, value in overrides.items():
        config_data[key] = value

    return SimConfig(**config_data)  # type: ignore[arg-type]
