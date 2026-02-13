"""Tests for character import parsing (SimC, SeventyUpgrades, WoWSims)."""

import json

import pytest

from code.shukketsu.resilience.errors import InvalidSimConfigError
from code.shukketsu.sim.imports import (
    ImportFormat,
    build_config,
    detect_format,
    parse_import,
    parse_seventyupgrades,
    parse_simc,
    parse_wowsims,
)
from code.shukketsu.sim.models import GearSlot, Race, RogueSpec

SAMPLE_SIMC = """rogue="TestRogue"
level=70
race=orc
spec=combat
talents=20/41/0
head=,id=29044
neck=,id=28762
shoulder=,id=29048
back=,id=28672
chest=,id=29045
wrist=,id=28502
hands=,id=29047
waist=,id=28750
legs=,id=29046
feet=,id=28517
finger1=,id=28757
finger2=,id=29283
trinket1=,id=28830
trinket2=,id=28163
main_hand=,id=28573
off_hand=,id=28189
ranged=,id=28772
"""

SAMPLE_70U = json.dumps(
    {
        "race": "Orc",
        "class": "Rogue",
        "spec": "Combat",
        "talents": "20/41/0",
        "items": [
            {"slot": "Head", "id": 29044},
            {"slot": "Neck", "id": 28762},
            {"slot": "Trinket 1", "id": 28830},
            {"slot": "Trinket 2", "id": 28163},
            {"slot": "Main Hand", "id": 28573},
            {"slot": "Off Hand", "id": 28189},
        ],
    }
)


class TestDetectFormat:
    """Test auto-detection of import formats."""

    def test_simc_detected_by_equals(self) -> None:
        """Lines with '=' in first line -> SIMC."""
        assert detect_format(SAMPLE_SIMC) == ImportFormat.SIMC

    def test_json_detected_as_seventyupgrades(self) -> None:
        """String starting with '{' -> SEVENTYUPGRADES."""
        assert detect_format(SAMPLE_70U) == ImportFormat.SEVENTYUPGRADES

    def test_fallback_is_wowsims(self) -> None:
        """Non-SimC, non-JSON string -> WOWSIMS (Base64 fallback)."""
        assert detect_format("aGVsbG8gd29ybGQ=") == ImportFormat.WOWSIMS


class TestParseSimc:
    """Test SimC format parsing."""

    def test_extracts_gear_ids(self) -> None:
        """parse_simc extracts all gear slot IDs."""
        result = parse_simc(SAMPLE_SIMC)
        assert result.gear[GearSlot.HEAD] == 29044
        assert result.gear[GearSlot.MAIN_HAND] == 28573
        assert result.gear[GearSlot.TRINKET_1] == 28830
        assert result.gear[GearSlot.RANGED] == 28772

    def test_extracts_race(self) -> None:
        """parse_simc maps orc -> Race.ORC."""
        result = parse_simc(SAMPLE_SIMC)
        assert result.race == Race.ORC

    def test_extracts_talents(self) -> None:
        """parse_simc extracts the talents string."""
        result = parse_simc(SAMPLE_SIMC)
        assert result.talents == "20/41/0"

    def test_maps_slot_names(self) -> None:
        """parse_simc maps finger1 -> RING_1, trinket2 -> TRINKET_2."""
        result = parse_simc(SAMPLE_SIMC)
        assert GearSlot.RING_1 in result.gear
        assert GearSlot.RING_2 in result.gear
        assert GearSlot.TRINKET_2 in result.gear

    def test_invalid_simc_missing_spec(self) -> None:
        """Missing spec raises InvalidSimConfigError."""
        bad_input = 'rogue="Test"\nrace=orc\ntalents=20/41/0\n'
        with pytest.raises(InvalidSimConfigError, match="Missing spec"):
            parse_simc(bad_input)


class TestParseSeventyUpgrades:
    """Test SeventyUpgrades JSON parsing."""

    def test_extracts_from_json(self) -> None:
        """parse_seventyupgrades extracts gear from sample JSON."""
        result = parse_seventyupgrades(SAMPLE_70U)
        assert result.race == Race.ORC
        assert result.spec == RogueSpec.COMBAT_SWORDS
        assert result.talents == "20/41/0"
        assert result.gear[GearSlot.HEAD] == 29044
        assert result.gear[GearSlot.MAIN_HAND] == 28573


class TestParseWowsims:
    """Test WoWSims stub behavior."""

    def test_raises_not_supported(self) -> None:
        """parse_wowsims raises InvalidSimConfigError (stub)."""
        with pytest.raises(InvalidSimConfigError, match="not yet supported"):
            parse_wowsims("some_base64_data")


class TestParseImport:
    """Test auto-detection + delegation."""

    def test_auto_detects_simc(self) -> None:
        """parse_import auto-detects SimC format and delegates."""
        result = parse_import(SAMPLE_SIMC)
        assert result.race == Race.ORC
        assert result.spec == RogueSpec.COMBAT_SWORDS


class TestBuildConfig:
    """Test SimConfig building from imports."""

    def test_produces_valid_config(self) -> None:
        """build_config produces a SimConfig with preset buffs."""
        char = parse_simc(SAMPLE_SIMC)
        config = build_config(char, preset="full_25man")
        assert config.spec == RogueSpec.COMBAT_SWORDS
        assert config.race == Race.ORC
        assert "kings" in config.buffs
        assert "battle_shout" in config.buffs
        assert config.raid_preset == "full_25man"

    def test_applies_overrides(self) -> None:
        """build_config applies keyword overrides like fight_length."""
        char = parse_simc(SAMPLE_SIMC)
        config = build_config(char, preset="solo", fight_length=180)
        assert config.fight_length == 180
        assert config.buffs == []
