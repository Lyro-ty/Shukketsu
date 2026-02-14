"""Tests for talent definitions, parsing, and modifier computation."""

import pytest

from code.shukketsu.sim.models import RogueSpec
from code.shukketsu.sim.talents import (
    TALENT_DEFS,
    TalentAllocation,
    TalentModifiers,
    compute_modifiers,
    get_spec_template,
    parse_talents,
)

# =============================================================================
# parse_talents
# =============================================================================


class TestParseTalents:
    """Tests for parse_talents function."""

    def test_valid_combat_swords(self) -> None:
        alloc = parse_talents("20/41/0", RogueSpec.COMBAT_SWORDS)
        assert alloc.assassination == 20
        assert alloc.combat == 41
        assert alloc.subtlety == 0
        assert isinstance(alloc.points, dict)
        assert len(alloc.points) > 0

    def test_valid_mutilate(self) -> None:
        alloc = parse_talents("41/20/0", RogueSpec.ASSASSINATION_MUTILATE)
        assert alloc.assassination == 41
        assert alloc.combat == 20
        assert alloc.subtlety == 0
        assert "mutilate_talent" in alloc.points
        assert alloc.points["mutilate_talent"] == 1

    def test_rejects_more_than_61_points(self) -> None:
        with pytest.raises(ValueError, match="exceeds maximum of 61"):
            parse_talents("30/30/2", RogueSpec.COMBAT_SWORDS)

    def test_rejects_invalid_format_two_parts(self) -> None:
        with pytest.raises(ValueError, match="must be 'X/Y/Z'"):
            parse_talents("20/41", RogueSpec.COMBAT_SWORDS)

    def test_rejects_non_integer(self) -> None:
        with pytest.raises(ValueError, match="Non-integer"):
            parse_talents("20/abc/0", RogueSpec.COMBAT_SWORDS)

    def test_rejects_negative_points(self) -> None:
        with pytest.raises(ValueError, match="Negative"):
            parse_talents("20/-1/0", RogueSpec.COMBAT_SWORDS)

    def test_exactly_61_points_is_valid(self) -> None:
        alloc = parse_talents("20/41/0", RogueSpec.COMBAT_SWORDS)
        assert alloc.assassination + alloc.combat + alloc.subtlety == 61


# =============================================================================
# get_spec_template
# =============================================================================


class TestGetSpecTemplate:
    """Tests for get_spec_template function."""

    def test_combat_swords_template(self) -> None:
        tpl = get_spec_template(RogueSpec.COMBAT_SWORDS)
        assert "sword_specialization" in tpl
        assert tpl["sword_specialization"] == 5
        assert "precision" in tpl
        assert tpl["precision"] == 5

    def test_combat_swords_has_improved_snd(self) -> None:
        tpl = get_spec_template(RogueSpec.COMBAT_SWORDS)
        assert "improved_slice_and_dice" in tpl
        assert tpl["improved_slice_and_dice"] == 3

    def test_combat_fists_template(self) -> None:
        tpl = get_spec_template(RogueSpec.COMBAT_FISTS)
        assert "fist_specialization" in tpl
        assert tpl["fist_specialization"] == 5
        # Should NOT have sword spec
        assert "sword_specialization" not in tpl

    def test_combat_daggers_template(self) -> None:
        tpl = get_spec_template(RogueSpec.COMBAT_DAGGERS)
        assert "dagger_specialization" in tpl
        assert tpl["dagger_specialization"] == 5

    def test_mutilate_template(self) -> None:
        tpl = get_spec_template(RogueSpec.ASSASSINATION_MUTILATE)
        assert "mutilate_talent" in tpl
        assert tpl["mutilate_talent"] == 1
        assert "seal_fate" in tpl
        assert tpl["seal_fate"] == 5
        assert "vigor" in tpl
        assert tpl["vigor"] == 1

    def test_all_specs_have_templates(self) -> None:
        for spec in RogueSpec:
            tpl = get_spec_template(spec)
            assert isinstance(tpl, dict)
            assert len(tpl) > 0

    def test_template_returns_copy(self) -> None:
        tpl1 = get_spec_template(RogueSpec.COMBAT_SWORDS)
        tpl2 = get_spec_template(RogueSpec.COMBAT_SWORDS)
        assert tpl1 == tpl2
        assert tpl1 is not tpl2


# =============================================================================
# compute_modifiers — Combat Swords
# =============================================================================


class TestComputeModifiersCombatSwords:
    """Tests for compute_modifiers with a standard Combat Swords build."""

    @pytest.fixture()
    def combat_swords_mods(self) -> TalentModifiers:
        alloc = parse_talents("20/41/0", RogueSpec.COMBAT_SWORDS)
        return compute_modifiers(alloc)

    def test_precision_5_5(self, combat_swords_mods: TalentModifiers) -> None:
        assert combat_swords_mods.bonus_hit_pct == pytest.approx(0.05)

    def test_sword_spec_5_5(self, combat_swords_mods: TalentModifiers) -> None:
        assert combat_swords_mods.sword_spec_proc_chance == pytest.approx(0.05)

    def test_combat_potency_5_5(self, combat_swords_mods: TalentModifiers) -> None:
        assert combat_swords_mods.combat_potency_proc_chance == pytest.approx(0.20)
        assert combat_swords_mods.combat_potency_energy == pytest.approx(15.0)

    def test_lethality_5_5(self, combat_swords_mods: TalentModifiers) -> None:
        assert combat_swords_mods.lethality_secondary_mod == pytest.approx(0.30)

    def test_malice_5_5(self, combat_swords_mods: TalentModifiers) -> None:
        assert combat_swords_mods.bonus_crit_pct == pytest.approx(0.05)

    def test_weapon_expertise_2_2(self, combat_swords_mods: TalentModifiers) -> None:
        assert combat_swords_mods.bonus_expertise == 10

    def test_dw_spec_5_5(self, combat_swords_mods: TalentModifiers) -> None:
        assert combat_swords_mods.dw_spec_oh_bonus_pct == pytest.approx(0.50)

    def test_blade_flurry_unlocked(self, combat_swords_mods: TalentModifiers) -> None:
        assert combat_swords_mods.blade_flurry is True

    def test_adrenaline_rush_unlocked(self, combat_swords_mods: TalentModifiers) -> None:
        assert combat_swords_mods.adrenaline_rush is True

    def test_ruthlessness_3_3(self, combat_swords_mods: TalentModifiers) -> None:
        assert combat_swords_mods.ruthlessness_proc_chance == pytest.approx(0.60)

    def test_aggression_3_3(self, combat_swords_mods: TalentModifiers) -> None:
        assert combat_swords_mods.aggression_damage_pct == pytest.approx(0.06)

    def test_vitality_2_2(self, combat_swords_mods: TalentModifiers) -> None:
        # vitality_agi_mult is multiplicative: 1.0 + 0.02*2 = 1.04
        assert combat_swords_mods.vitality_agi_mult == pytest.approx(1.04)


# =============================================================================
# compute_modifiers — Assassination Mutilate
# =============================================================================


class TestComputeModifiersMutilate:
    """Tests for compute_modifiers with a standard Mutilate build."""

    @pytest.fixture()
    def mutilate_mods(self) -> TalentModifiers:
        alloc = parse_talents("41/20/0", RogueSpec.ASSASSINATION_MUTILATE)
        return compute_modifiers(alloc)

    def test_seal_fate_5_5(self, mutilate_mods: TalentModifiers) -> None:
        assert mutilate_mods.seal_fate_proc_chance == pytest.approx(1.0)

    def test_find_weakness_3_3(self, mutilate_mods: TalentModifiers) -> None:
        assert mutilate_mods.find_weakness_damage_pct == pytest.approx(0.06)

    def test_mutilate_talented(self, mutilate_mods: TalentModifiers) -> None:
        assert mutilate_mods.mutilate_talented is True

    def test_vigor_enabled(self, mutilate_mods: TalentModifiers) -> None:
        assert mutilate_mods.vigor is True

    def test_surprise_attacks(self, mutilate_mods: TalentModifiers) -> None:
        assert mutilate_mods.surprise_attacks_damage_pct == pytest.approx(0.10)
        assert mutilate_mods.surprise_attacks_finisher_undodgeable is True

    def test_vile_poisons_3_3(self, mutilate_mods: TalentModifiers) -> None:
        assert mutilate_mods.vile_poisons_pct == pytest.approx(0.21)

    def test_improved_poisons_5_5(self, mutilate_mods: TalentModifiers) -> None:
        assert mutilate_mods.imp_poisons_ranks == 5

    def test_quick_recovery_2_2(self, mutilate_mods: TalentModifiers) -> None:
        assert mutilate_mods.quick_recovery_refund_pct == pytest.approx(0.80)

    def test_master_poisoner_2_2(self, mutilate_mods: TalentModifiers) -> None:
        assert mutilate_mods.master_poisoner_hit_pct == pytest.approx(0.02)

    def test_murder_2_2(self, mutilate_mods: TalentModifiers) -> None:
        assert mutilate_mods.murder_damage_pct == pytest.approx(0.02)

    def test_no_sword_spec(self, mutilate_mods: TalentModifiers) -> None:
        assert mutilate_mods.sword_spec_proc_chance == 0.0

    def test_no_adrenaline_rush(self, mutilate_mods: TalentModifiers) -> None:
        assert mutilate_mods.adrenaline_rush is False


# =============================================================================
# TalentModifiers — immutability and defaults
# =============================================================================


class TestTalentModifiers:
    """Tests for TalentModifiers model properties."""

    def test_frozen_model(self) -> None:
        mods = TalentModifiers()
        with pytest.raises(Exception):  # noqa: B017
            mods.bonus_crit_pct = 0.10  # type: ignore[misc]

    def test_empty_allocation_returns_defaults(self) -> None:
        alloc = TalentAllocation(assassination=0, combat=0, subtlety=0, points={})
        mods = compute_modifiers(alloc)
        assert mods.bonus_crit_pct == 0.0
        assert mods.bonus_hit_pct == 0.0
        assert mods.bonus_expertise == 0
        assert mods.sword_spec_proc_chance == 0.0
        assert mods.combat_potency_proc_chance == 0.0
        assert mods.lethality_secondary_mod == 0.0
        assert mods.vigor is False
        assert mods.blade_flurry is False
        assert mods.adrenaline_rush is False
        assert mods.mutilate_talented is False
        assert mods.vitality_agi_mult == 1.0
        assert mods.sinister_calling_agi_mult == 1.0
        assert mods.deadliness_ap_mult == 1.0
        assert mods.crit_damage_primary_mod == 1.0


# =============================================================================
# Individual talent effects
# =============================================================================


class TestIndividualTalentEffects:
    """Tests that individual talents produce correct magnitude values."""

    def test_single_rank_malice(self) -> None:
        alloc = TalentAllocation(assassination=1, combat=0, subtlety=0, points={"malice": 1})
        mods = compute_modifiers(alloc)
        assert mods.bonus_crit_pct == pytest.approx(0.01)

    def test_improved_sinister_strike_2_2(self) -> None:
        alloc = TalentAllocation(assassination=0, combat=2, subtlety=0, points={"improved_sinister_strike": 2})
        mods = compute_modifiers(alloc)
        assert mods.ss_energy_reduction == 6

    def test_relentless_strikes_1_1(self) -> None:
        alloc = TalentAllocation(assassination=1, combat=0, subtlety=0, points={"relentless_strikes": 1})
        mods = compute_modifiers(alloc)
        assert mods.relentless_strikes_per_cp == pytest.approx(20.0)

    def test_serrated_blades_3_3(self) -> None:
        alloc = TalentAllocation(assassination=0, combat=0, subtlety=3, points={"serrated_blades": 3})
        mods = compute_modifiers(alloc)
        assert mods.serrated_blades_rupture_pct == pytest.approx(0.30)

    def test_deadliness_5_5_multiplicative(self) -> None:
        alloc = TalentAllocation(assassination=0, combat=0, subtlety=5, points={"deadliness": 5})
        mods = compute_modifiers(alloc)
        # deadliness_ap_mult starts at 1.0, adds 0.02*5 = 0.10 -> 1.10
        assert mods.deadliness_ap_mult == pytest.approx(1.10)

    def test_sinister_calling_5_5_multiplicative(self) -> None:
        alloc = TalentAllocation(assassination=0, combat=0, subtlety=5, points={"sinister_calling": 5})
        mods = compute_modifiers(alloc)
        # sinister_calling_agi_mult starts at 1.0, adds 0.03*5 = 0.15 -> 1.15
        assert mods.sinister_calling_agi_mult == pytest.approx(1.15)

    def test_ranks_capped_at_max(self) -> None:
        """Allocating more ranks than max_ranks should cap at the maximum."""
        alloc = TalentAllocation(assassination=10, combat=0, subtlety=0, points={"malice": 10})
        mods = compute_modifiers(alloc)
        # malice max_ranks=5, so 10 ranks should be capped to 5 -> 0.05
        assert mods.bonus_crit_pct == pytest.approx(0.05)

    def test_improved_slice_and_dice_3_3(self) -> None:
        alloc = TalentAllocation(assassination=0, combat=3, subtlety=0, points={"improved_slice_and_dice": 3})
        mods = compute_modifiers(alloc)
        # snd_duration_mult starts at 1.0, adds 0.15*3 = 0.45 -> 1.45
        assert mods.snd_duration_mult == pytest.approx(1.45)

    def test_unknown_talent_ignored(self) -> None:
        alloc = TalentAllocation(assassination=0, combat=0, subtlety=0, points={"nonexistent_talent": 5})
        mods = compute_modifiers(alloc)
        # Should return defaults without error
        assert mods.bonus_crit_pct == 0.0


# =============================================================================
# Talent definitions registry
# =============================================================================


class TestTalentDefs:
    """Tests for the TALENT_DEFS registry."""

    def test_registry_has_entries(self) -> None:
        assert len(TALENT_DEFS) >= 25

    def test_all_trees_represented(self) -> None:
        trees = {t.tree for t in TALENT_DEFS.values()}
        assert trees == {"assassination", "combat", "subtlety"}

    def test_max_ranks_positive(self) -> None:
        for name, talent in TALENT_DEFS.items():
            assert talent.max_ranks >= 1, f"{name} has max_ranks < 1"

    def test_effect_per_rank_non_empty(self) -> None:
        for name, talent in TALENT_DEFS.items():
            assert len(talent.effect_per_rank) >= 1, f"{name} has no effects"
