"""Tests for static ability and poison definitions."""

import pytest

from code.shukketsu.sim.abilities import (
    ABILITIES,
    POISONS,
    AbilityDef,
    PoisonDef,
    builders_for_spec,
    finisher_for_spec,
    get_ability,
    get_poison,
)
from code.shukketsu.sim.models import AbilityFlag, PoisonType, RogueSpec, WeaponType


class TestAbilityRegistry:
    """Tests for the ABILITIES registry and AbilityDef model."""

    def test_all_abilities_exist(self) -> None:
        """Every expected ability key is present in the registry."""
        expected = {
            "sinister_strike",
            "backstab",
            "mutilate",
            "hemorrhage",
            "shiv",
            "ambush",
            "garrote",
            "cheap_shot",
            "eviscerate",
            "envenom",
            "rupture",
            "slice_and_dice",
            "expose_armor",
            "blade_flurry",
            "adrenaline_rush",
            "cold_blood",
            "thistle_tea",
            "premeditation",
        }
        assert set(ABILITIES.keys()) == expected

    def test_ability_values_are_ability_def(self) -> None:
        """All registry values are AbilityDef instances."""
        for ability in ABILITIES.values():
            assert isinstance(ability, AbilityDef)

    @pytest.mark.parametrize(
        ("name", "cost"),
        [
            ("sinister_strike", 45),
            ("backstab", 60),
            ("mutilate", 60),
            ("hemorrhage", 35),
            ("shiv", 20),
            ("eviscerate", 35),
            ("envenom", 35),
            ("rupture", 25),
            ("slice_and_dice", 25),
            ("blade_flurry", 25),
            ("adrenaline_rush", 0),
            ("cold_blood", 0),
            ("thistle_tea", 0),
        ],
    )
    def test_energy_costs(self, name: str, cost: int) -> None:
        """Each ability has the correct energy cost."""
        assert ABILITIES[name].energy_cost == cost

    def test_builders_have_builder_flag(self) -> None:
        """All builders carry the BUILDER flag."""
        builders = [
            "sinister_strike",
            "backstab",
            "mutilate",
            "hemorrhage",
            "shiv",
            "ambush",
            "garrote",
            "cheap_shot",
        ]
        for name in builders:
            assert AbilityFlag.BUILDER in ABILITIES[name].flags, f"{name} missing BUILDER flag"

    def test_finishers_have_finisher_flag(self) -> None:
        """All finishers carry the FINISHER flag and consume combo points."""
        finishers = ["eviscerate", "envenom", "rupture", "slice_and_dice", "expose_armor"]
        for name in finishers:
            ability = ABILITIES[name]
            assert AbilityFlag.FINISHER in ability.flags, f"{name} missing FINISHER flag"
            assert ability.combo_points_consumed is True, f"{name} should consume combo points"

    def test_normalized_abilities_have_flag(self) -> None:
        """Abilities marked normalized=True also carry the NORMALIZED flag."""
        for name, ability in ABILITIES.items():
            if ability.normalized:
                assert AbilityFlag.NORMALIZED in ability.flags, f"{name} missing NORMALIZED flag"

    def test_cooldowns_have_off_gcd_flag(self) -> None:
        """Cooldown abilities are all off-GCD."""
        cooldowns = ["blade_flurry", "adrenaline_rush", "cold_blood", "thistle_tea", "premeditation"]
        for name in cooldowns:
            assert AbilityFlag.OFF_GCD in ABILITIES[name].flags, f"{name} missing OFF_GCD flag"

    def test_shiv_cannot_be_dodged(self) -> None:
        """Shiv has the CANNOT_BE_DODGED flag."""
        assert AbilityFlag.CANNOT_BE_DODGED in ABILITIES["shiv"].flags

    def test_mutilate_generates_two_combo_points(self) -> None:
        """Mutilate generates 2 combo points per use."""
        assert ABILITIES["mutilate"].combo_points_generated == 2

    def test_garrote_has_snapshot_flag(self) -> None:
        """Garrote snapshots stats at cast time."""
        assert AbilityFlag.SNAPSHOT in ABILITIES["garrote"].flags

    def test_ambush_requires_dagger(self) -> None:
        """Ambush requires a dagger in the main hand."""
        assert ABILITIES["ambush"].weapon_type_required == WeaponType.DAGGER

    def test_thistle_tea_zero_energy_cost(self) -> None:
        """Thistle Tea costs no energy (100 energy restore handled by combat engine)."""
        assert ABILITIES["thistle_tea"].energy_cost == 0

    def test_cooldown_durations(self) -> None:
        """Blade Flurry and Adrenaline Rush have correct cooldown and duration values."""
        bf = ABILITIES["blade_flurry"]
        assert bf.cooldown_ms == 120_000
        assert bf.duration_ms == 15_000

        ar = ABILITIES["adrenaline_rush"]
        assert ar.cooldown_ms == 300_000
        assert ar.duration_ms == 15_000

    def test_get_ability_returns_correct_def(self) -> None:
        """get_ability returns the matching AbilityDef."""
        ss = get_ability("sinister_strike")
        assert ss.name == "Sinister Strike"
        assert ss.energy_cost == 45

    def test_get_ability_raises_key_error(self) -> None:
        """get_ability raises KeyError for unknown abilities."""
        with pytest.raises(KeyError):
            get_ability("shadow_dance")


class TestPoisonRegistry:
    """Tests for the POISONS registry and PoisonDef model."""

    def test_all_poisons_exist(self) -> None:
        """Every expected poison key is present."""
        assert set(POISONS.keys()) == {"instant_poison", "deadly_poison", "wound_poison"}

    def test_poison_values_are_poison_def(self) -> None:
        """All registry values are PoisonDef instances."""
        for poison in POISONS.values():
            assert isinstance(poison, PoisonDef)

    def test_instant_poison_proc_chance(self) -> None:
        """Instant Poison has 20% base proc chance."""
        assert POISONS["instant_poison"].proc_chance_base == pytest.approx(0.20)

    def test_deadly_poison_stacking(self) -> None:
        """Deadly Poison stacks to 5 with 3s tick interval."""
        dp = POISONS["deadly_poison"]
        assert dp.max_stacks == 5
        assert dp.tick_interval_ms == 3000
        assert dp.duration_ms == 12000

    def test_wound_poison_proc_chance(self) -> None:
        """Wound Poison has 50% base proc chance."""
        assert POISONS["wound_poison"].proc_chance_base == pytest.approx(0.50)

    def test_get_poison_instant(self) -> None:
        """get_poison maps PoisonType.INSTANT to instant_poison."""
        ip = get_poison(PoisonType.INSTANT)
        assert ip.name == "Instant Poison"
        assert ip.damage_per_proc == 200

    def test_get_poison_raises_for_none_type(self) -> None:
        """get_poison raises KeyError for PoisonType.NONE."""
        with pytest.raises(KeyError):
            get_poison(PoisonType.NONE)


class TestSpecHelpers:
    """Tests for builders_for_spec and finisher_for_spec."""

    def test_combat_swords_builder(self) -> None:
        """Combat Swords uses Sinister Strike."""
        assert builders_for_spec(RogueSpec.COMBAT_SWORDS) == ["sinister_strike"]

    def test_combat_fists_builder(self) -> None:
        """Combat Fists uses Sinister Strike."""
        assert builders_for_spec(RogueSpec.COMBAT_FISTS) == ["sinister_strike"]

    def test_combat_daggers_builder(self) -> None:
        """Combat Daggers uses Backstab."""
        assert builders_for_spec(RogueSpec.COMBAT_DAGGERS) == ["backstab"]

    def test_assassination_mutilate_builder(self) -> None:
        """Assassination Mutilate uses Mutilate."""
        assert builders_for_spec(RogueSpec.ASSASSINATION_MUTILATE) == ["mutilate"]

    def test_finisher_mutilate_spec_returns_envenom(self) -> None:
        """Assassination always uses Envenom regardless of Rupture state."""
        assert finisher_for_spec(RogueSpec.ASSASSINATION_MUTILATE, has_rupture=False, fight_remaining=200) == "envenom"
        assert finisher_for_spec(RogueSpec.ASSASSINATION_MUTILATE, has_rupture=True, fight_remaining=200) == "envenom"

    def test_finisher_combat_with_rupture(self) -> None:
        """Combat uses Eviscerate when Rupture is already active."""
        assert finisher_for_spec(RogueSpec.COMBAT_SWORDS, has_rupture=True, fight_remaining=200) == "eviscerate"

    def test_finisher_combat_short_fight(self) -> None:
        """Combat uses Eviscerate when fight is nearly over."""
        assert finisher_for_spec(RogueSpec.COMBAT_SWORDS, has_rupture=False, fight_remaining=10) == "eviscerate"

    def test_finisher_combat_no_rupture_long_fight(self) -> None:
        """Combat uses Rupture when it's not up and fight has time left."""
        assert finisher_for_spec(RogueSpec.COMBAT_SWORDS, has_rupture=False, fight_remaining=60) == "rupture"
