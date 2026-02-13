"""Tests for WCL spell ID <-> sim ability name mapping."""

import pytest


class TestSpellIdToAbility:
    def test_sinister_strike_rank10(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_ability_name

        assert wcl_ability_name(1752) == "sinister_strike"

    def test_sinister_strike_other_rank(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_ability_name

        assert wcl_ability_name(11294) == "sinister_strike"

    def test_eviscerate(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_ability_name

        assert wcl_ability_name(26865) == "eviscerate"

    def test_unknown_spell(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_ability_name

        assert wcl_ability_name(99999) is None

    def test_blade_flurry(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_ability_name

        assert wcl_ability_name(13877) == "blade_flurry"

    def test_instant_poison(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_ability_name

        assert wcl_ability_name(26891) == "instant_poison"


class TestDisplayName:
    def test_sinister_strike_display(self) -> None:
        from code.shukketsu.sim.spell_map import sim_display_name

        assert sim_display_name("sinister_strike") == "Sinister Strike"

    def test_unknown_ability_returns_title_case(self) -> None:
        from code.shukketsu.sim.spell_map import sim_display_name

        assert sim_display_name("some_ability") == "Some Ability"


class TestBuffMapping:
    def test_kings(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_buff_name

        assert wcl_buff_name(25898) == "kings"

    def test_heroism(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_buff_name

        assert wcl_buff_name(32182) == "heroism"

    def test_unknown_buff(self) -> None:
        from code.shukketsu.sim.spell_map import wcl_buff_name

        assert wcl_buff_name(99999) is None


class TestAggregateAbilities:
    def test_aggregate_merges_ranks(self) -> None:
        """Multiple spell IDs for the same ability should merge."""
        from code.shukketsu.sim.spell_map import aggregate_wcl_abilities

        wcl_abilities = [
            {"guid": 1752, "name": "Sinister Strike", "total": 10000},
            {"guid": 11294, "name": "Sinister Strike", "total": 5000},
            {"guid": 26865, "name": "Eviscerate", "total": 8000},
        ]
        result = aggregate_wcl_abilities(wcl_abilities)
        assert result["sinister_strike"] == 15000
        assert result["eviscerate"] == 8000

    def test_aggregate_logs_unmapped_spells(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unknown spell IDs are logged as warnings."""
        import logging

        from code.shukketsu.sim.spell_map import aggregate_wcl_abilities

        with caplog.at_level(logging.WARNING, logger="code.shukketsu.sim.spell_map"):
            result = aggregate_wcl_abilities([{"guid": 99999, "name": "Mystery", "total": 42000}])
        assert len(result) == 0
        assert "99999" in caplog.text
