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
        """JSON without 'player' key -> SEVENTYUPGRADES."""
        assert detect_format(SAMPLE_70U) == ImportFormat.SEVENTYUPGRADES

    def test_fallback_is_wowsims(self) -> None:
        """Non-SimC, non-JSON string -> WOWSIMS (Base64 fallback)."""
        assert detect_format("aGVsbG8gd29ybGQ=") == ImportFormat.WOWSIMS

    def test_wowsims_json_detected(self) -> None:
        """JSON with 'player' key -> WOWSIMS."""
        wowsims_json = json.dumps({"player": {"name": "Test", "race": 2}})
        assert detect_format(wowsims_json) == ImportFormat.WOWSIMS


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


class TestWoWSimsImport:
    """Test WoWSims JSON import parsing."""

    def test_parse_basic(self) -> None:
        """parse_wowsims extracts name, race, and source from WoWSims JSON."""
        data = {
            "player": {
                "name": "TestRogue",
                "race": 2,  # Orc
                "class": 4,
                "equipment": {
                    "items": [
                        {"id": 28224},  # Head
                        {"id": 0},  # Neck (empty)
                    ]
                },
                "talentsString": "005303104-0520301050140150233151-05",
            },
            "encounter": {"duration": 300},
        }
        result = parse_wowsims(data)
        assert result.race == Race.ORC
        assert result.spec == RogueSpec.COMBAT_SWORDS
        assert GearSlot.HEAD in result.gear
        assert result.gear[GearSlot.HEAD] == 28224

    def test_spec_detection_combat(self) -> None:
        """Combat talent string is detected as COMBAT_SWORDS."""
        data = {
            "player": {
                "talentsString": "005303104-0520301050140150233151-05",
                "equipment": {"items": []},
            },
        }
        result = parse_wowsims(data)
        assert result.spec == RogueSpec.COMBAT_SWORDS

    def test_spec_detection_assassination(self) -> None:
        """Assassination-heavy talent string is detected as ASSASSINATION_MUTILATE."""
        data = {
            "player": {
                "talentsString": "05503012050150120531-05-0520020",
                "equipment": {"items": []},
            },
        }
        result = parse_wowsims(data)
        assert result.spec == RogueSpec.ASSASSINATION_MUTILATE

    def test_gear_mapping(self) -> None:
        """Items at correct indices are mapped to the right GearSlots."""
        items: list[dict[str, int]] = [{"id": 0}] * 17  # 17 slots
        items[14] = {"id": 28189}  # Main hand
        items[15] = {"id": 28572}  # Off hand
        data = {
            "player": {
                "equipment": {"items": items},
                "talentsString": "-0520301050140150233151-",
            },
        }
        result = parse_wowsims(data)
        assert GearSlot.MAIN_HAND in result.gear
        assert result.gear[GearSlot.MAIN_HAND] == 28189
        assert GearSlot.OFF_HAND in result.gear
        assert result.gear[GearSlot.OFF_HAND] == 28572

    def test_missing_player_raises(self) -> None:
        """Missing 'player' key raises InvalidSimConfigError."""
        with pytest.raises(InvalidSimConfigError, match="missing 'player' key"):
            parse_wowsims({})

    def test_talent_format_conversion(self) -> None:
        """Per-point WoWSims talent strings are summed to tree totals."""
        data = {
            "player": {
                "talentsString": "005-0520301-05",
                "equipment": {"items": []},
            },
        }
        result = parse_wowsims(data)
        assert "/" in result.talents
        assert "-" not in result.talents
        # 0+0+5=5, 0+5+2+0+3+0+1=11, 0+5=5
        assert result.talents == "5/11/5"

    def test_parse_from_json_string(self) -> None:
        """parse_wowsims accepts a raw JSON string as well as a dict."""
        data = {
            "player": {
                "race": 1,  # Human
                "equipment": {"items": [{"id": 29044}]},
                "talentsString": "-0520301050140150233151-",
            },
        }
        result = parse_wowsims(json.dumps(data))
        assert result.race == Race.HUMAN
        assert GearSlot.HEAD in result.gear

    def test_enchants_extracted(self) -> None:
        """Enchant IDs from WoWSims items are extracted."""
        data = {
            "player": {
                "equipment": {
                    "items": [{"id": 29044, "enchant": 2999}],
                },
                "talentsString": "-0520301-",
            },
        }
        result = parse_wowsims(data)
        assert GearSlot.HEAD in result.enchants
        assert result.enchants[GearSlot.HEAD] == 2999

    def test_gems_extracted(self) -> None:
        """Gem IDs from WoWSims items are extracted."""
        data = {
            "player": {
                "equipment": {
                    "items": [{"id": 29044, "gems": [24028, 24055]}],
                },
                "talentsString": "-0520301-",
            },
        }
        result = parse_wowsims(data)
        assert GearSlot.HEAD in result.gems
        assert result.gems[GearSlot.HEAD] == [24028, 24055]

    def test_unknown_race_defaults_to_human(self) -> None:
        """Unknown WoWSims race ID falls back to HUMAN."""
        data = {
            "player": {
                "race": 999,
                "equipment": {"items": []},
                "talentsString": "-0520301-",
            },
        }
        result = parse_wowsims(data)
        assert result.race == Race.HUMAN

    def test_empty_items_skipped(self) -> None:
        """Items with id=0 are skipped."""
        data = {
            "player": {
                "equipment": {"items": [{"id": 0}, {"id": 0}]},
                "talentsString": "-0520301-",
            },
        }
        result = parse_wowsims(data)
        assert len(result.gear) == 0

    def test_invalid_json_string_raises(self) -> None:
        """Invalid JSON string raises InvalidSimConfigError."""
        with pytest.raises(InvalidSimConfigError, match="Invalid JSON"):
            parse_wowsims("not valid json {{{")


class TestParseImport:
    """Test auto-detection + delegation."""

    def test_auto_detects_simc(self) -> None:
        """parse_import auto-detects SimC format and delegates."""
        result = parse_import(SAMPLE_SIMC)
        assert result.race == Race.ORC
        assert result.spec == RogueSpec.COMBAT_SWORDS

    def test_auto_detects_wowsims_json(self) -> None:
        """parse_import auto-detects WoWSims JSON and delegates."""
        wowsims_json = json.dumps(
            {
                "player": {
                    "race": 2,
                    "equipment": {"items": [{"id": 29044}]},
                    "talentsString": "-0520301050140150233151-",
                },
            }
        )
        result = parse_import(wowsims_json)
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
