"""Tests for TBC 2.4.3 combat formula functions."""

import pytest

from code.shukketsu.sim.mechanics import (
    BASE_DODGE_CHANCE,
    BOSS_ARMOR_STANDARD,
    CRIT_SUPPRESSION,
    GLANCING_MULTIPLIER,
    MELEE_CRIT_MULTIPLIER,
    NORM_SPEED_DAGGER,
    NORM_SPEED_ONE_HAND,
    ROGUE_BASE_CRIT_ADJUSTMENT,
    calc_armor_reduction,
    calc_crit_chance,
    calc_crit_multiplier,
    calc_dodge_chance,
    calc_effective_speed,
    calc_glancing_reduction,
    calc_haste_multiplier,
    calc_miss_chance,
    calc_normalized_speed,
    calc_poison_proc_chance,
    calc_ppm_proc_chance,
    calc_weapon_damage,
    resolve_white_hit,
    resolve_yellow_hit,
)
from code.shukketsu.sim.models import HitOutcome, WeaponType

# =============================================================================
# resolve_white_hit
# =============================================================================


class TestResolveWhiteHit:
    """Tests for the single-roll white hit attack table."""

    def test_miss_at_zero_roll(self) -> None:
        result = resolve_white_hit(0.08, 0.065, 0.24, 0.20, 0.0)
        assert result == HitOutcome.MISS

    def test_dodge_at_miss_boundary(self) -> None:
        result = resolve_white_hit(0.08, 0.065, 0.24, 0.20, 0.08)
        assert result == HitOutcome.DODGE

    def test_glancing_at_dodge_boundary(self) -> None:
        # miss=0.08, dodge=0.065 => dodge threshold=0.145; use 0.15 to land in glancing
        result = resolve_white_hit(0.08, 0.065, 0.24, 0.20, 0.15)
        assert result == HitOutcome.GLANCING

    def test_crit_at_glancing_boundary(self) -> None:
        result = resolve_white_hit(0.08, 0.065, 0.24, 0.20, 0.385)
        assert result == HitOutcome.CRIT

    def test_hit_at_crit_boundary(self) -> None:
        result = resolve_white_hit(0.08, 0.065, 0.24, 0.20, 0.585)
        assert result == HitOutcome.HIT

    def test_crit_pushed_off_table(self) -> None:
        """When miss + dodge + glancing > 1.0, crit is entirely pushed off the table."""
        # miss=0.40, dodge=0.40, glancing=0.24 => sum=1.04 (overflows)
        # Crit threshold would be 1.04 + 0.20 = 1.24 — unreachable
        # A roll at 0.50 lands in dodge; but a roll that somehow passes glancing
        # (impossible since glancing threshold is already 1.04) would be HIT, not CRIT.
        # More practically: roll at 0.80 lands in glancing (0.80 < 1.04)
        result = resolve_white_hit(0.40, 0.40, 0.24, 0.50, 0.80)
        assert result == HitOutcome.GLANCING  # crit never reached

    def test_crit_fully_pushed_off(self) -> None:
        """When miss+dodge+glancing >= 1.0, crit is completely gone."""
        # 0.50 + 0.30 + 0.24 = 1.04, crit pushed off entirely
        result = resolve_white_hit(0.50, 0.30, 0.24, 0.20, 0.99)
        assert result == HitOutcome.GLANCING


# =============================================================================
# resolve_yellow_hit
# =============================================================================


class TestResolveYellowHit:
    """Tests for the two-roll yellow hit system."""

    def test_miss_on_hit_roll(self) -> None:
        result = resolve_yellow_hit(0.09, 0.065, 0.30, 0.05, 0.0)
        assert result == HitOutcome.MISS

    def test_dodge_on_hit_roll(self) -> None:
        result = resolve_yellow_hit(0.09, 0.065, 0.30, 0.10, 0.0)
        assert result == HitOutcome.DODGE

    def test_crit_on_independent_roll(self) -> None:
        """Crit roll is independent — cannot be pushed off for yellow."""
        result = resolve_yellow_hit(0.09, 0.065, 0.30, 0.50, 0.10)
        assert result == HitOutcome.CRIT

    def test_hit_when_both_rolls_pass(self) -> None:
        result = resolve_yellow_hit(0.09, 0.065, 0.30, 0.50, 0.50)
        assert result == HitOutcome.HIT

    def test_cannot_be_dodged_flag(self) -> None:
        """When can_be_dodged=False, dodge is skipped even if roll is in dodge range."""
        # Roll 0.10 is past miss (0.09) but within miss+dodge (0.155)
        result = resolve_yellow_hit(0.09, 0.065, 0.30, 0.10, 0.50, can_be_dodged=False)
        assert result == HitOutcome.HIT


# =============================================================================
# calc_miss_chance
# =============================================================================


class TestCalcMissChance:
    """Tests for miss chance calculation."""

    def test_zero_hit_rating_dual_wield(self) -> None:
        """DW white: 8% + 1% suppression + 19% penalty = 28%."""
        miss = calc_miss_chance(0.0, is_dual_wield=True, is_yellow=False)
        assert miss == pytest.approx(0.28, abs=1e-6)

    def test_zero_hit_rating_two_hand(self) -> None:
        """2H white: 8% + 1% suppression = 9%."""
        miss = calc_miss_chance(0.0, is_dual_wield=False, is_yellow=False)
        assert miss == pytest.approx(0.09, abs=1e-6)

    def test_yellow_no_dw_penalty(self) -> None:
        """Yellow: 8% + 1% suppression = 9%, no DW penalty."""
        miss = calc_miss_chance(0.0, is_dual_wield=True, is_yellow=True)
        assert miss == pytest.approx(0.09, abs=1e-6)

    def test_with_precision_talent(self) -> None:
        """5 ranks of Precision = 5% hit."""
        miss = calc_miss_chance(0.0, is_dual_wield=False, is_yellow=False, precision_ranks=5)
        assert miss == pytest.approx(0.04, abs=1e-6)

    def test_at_hit_cap_clamped_to_zero(self) -> None:
        """Excessive hit rating should clamp to 0% miss."""
        # 500 rating = 500/15.77 = ~31.7% hit, more than enough
        miss = calc_miss_chance(500.0, is_dual_wield=True, is_yellow=False)
        assert miss == 0.0


# =============================================================================
# calc_dodge_chance
# =============================================================================


class TestCalcDodgeChance:
    """Tests for dodge chance calculation."""

    def test_zero_expertise(self) -> None:
        dodge = calc_dodge_chance(0.0)
        assert dodge == pytest.approx(BASE_DODGE_CHANCE, abs=1e-6)

    def test_with_expertise_rating(self) -> None:
        """Some expertise rating reduces dodge."""
        dodge = calc_dodge_chance(25.0)
        # 25 / 3.9423 = ~6.341 expertise points, * 0.0025 = ~0.01585 reduction
        expected = BASE_DODGE_CHANCE - (25.0 / 3.9423) * 0.0025
        assert dodge == pytest.approx(expected, abs=1e-6)

    def test_with_weapon_expertise_talent(self) -> None:
        """2 ranks = 10 expertise points = 2.5% dodge reduction."""
        dodge = calc_dodge_chance(0.0, weapon_expertise_ranks=2)
        expected = BASE_DODGE_CHANCE - 10 * 0.0025
        assert dodge == pytest.approx(expected, abs=1e-6)

    def test_at_dodge_cap_clamped_to_zero(self) -> None:
        """Excessive expertise clamps dodge to 0%."""
        dodge = calc_dodge_chance(200.0, weapon_expertise_ranks=2)
        assert dodge == 0.0


# =============================================================================
# calc_crit_chance
# =============================================================================


class TestCalcCritChance:
    """Tests for crit chance calculation."""

    def test_base_case_only_suppression(self) -> None:
        """With no gear/agi, just rogue adjustment and crit suppression."""
        crit = calc_crit_chance(0.0, 0.0)
        expected = ROGUE_BASE_CRIT_ADJUSTMENT - CRIT_SUPPRESSION
        assert crit == pytest.approx(expected, abs=1e-6)

    def test_with_agility(self) -> None:
        """400 agi = 400/40/100 = 10% crit from agi."""
        crit = calc_crit_chance(0.0, 400.0, crit_suppression=False)
        expected = 0.10 + ROGUE_BASE_CRIT_ADJUSTMENT
        assert crit == pytest.approx(expected, abs=1e-6)

    def test_with_crit_rating(self) -> None:
        """220.8 crit rating = 10% crit."""
        crit = calc_crit_chance(220.8, 0.0, crit_suppression=False)
        expected = 0.10 + ROGUE_BASE_CRIT_ADJUSTMENT
        assert crit == pytest.approx(expected, abs=1e-6)

    def test_rogue_base_adjustment_applied(self) -> None:
        """Rogue base crit adjustment is always applied (-0.3%)."""
        crit = calc_crit_chance(0.0, 0.0, crit_suppression=False)
        assert crit == pytest.approx(ROGUE_BASE_CRIT_ADJUSTMENT, abs=1e-6)


# =============================================================================
# calc_crit_multiplier
# =============================================================================


class TestCalcCritMultiplier:
    """Tests for crit damage multiplier calculation."""

    def test_base_melee_crit(self) -> None:
        """Base melee crit: 1 + (2.0 * 1.0 - 1.0) * (1.0 + 0.0) = 2.0."""
        mult = calc_crit_multiplier(MELEE_CRIT_MULTIPLIER)
        assert mult == pytest.approx(2.0, abs=1e-6)

    def test_with_lethality(self) -> None:
        """Lethality 5/5 = 30% secondary: 1 + (2.0 - 1.0) * 1.30 = 2.30."""
        mult = calc_crit_multiplier(MELEE_CRIT_MULTIPLIER, secondary_mod=0.30)
        assert mult == pytest.approx(2.30, abs=1e-6)

    def test_with_meta_gem(self) -> None:
        """Meta gem: primary *= 1.03. 1 + (2.0 * 1.03 - 1.0) * 1.0 = 2.06."""
        mult = calc_crit_multiplier(MELEE_CRIT_MULTIPLIER, has_meta_gem=True)
        assert mult == pytest.approx(2.06, abs=1e-6)

    def test_combined_lethality_and_meta(self) -> None:
        """Lethality + meta: 1 + (2.0 * 1.03 - 1.0) * 1.30 = 2.378."""
        mult = calc_crit_multiplier(MELEE_CRIT_MULTIPLIER, secondary_mod=0.30, has_meta_gem=True)
        expected = 1.0 + (2.0 * 1.03 - 1.0) * 1.30
        assert mult == pytest.approx(expected, abs=1e-6)


# =============================================================================
# calc_armor_reduction
# =============================================================================


class TestCalcArmorReduction:
    """Tests for armor reduction calculation."""

    def test_standard_boss_no_debuffs(self) -> None:
        """7700 armor: reduction = 7700 / (7700 + 10557.5) = ~0.4218."""
        mult = calc_armor_reduction(BOSS_ARMOR_STANDARD)
        expected_reduction = 7700.0 / (7700.0 + 10557.5)
        assert mult == pytest.approx(1.0 - expected_reduction, abs=1e-4)

    def test_with_full_debuffs(self) -> None:
        """Full debuffs: 5 sunder + EA + FF + CoR."""
        mult = calc_armor_reduction(
            BOSS_ARMOR_STANDARD,
            sunder_stacks=5,
            expose_armor_ranks=1,
            faerie_fire=True,
            curse_of_recklessness=True,
        )
        debuffed = 7700 - 5 * 520 - 2050 - 610 - 800
        debuffed = max(0.0, debuffed)
        expected_reduction = debuffed / (debuffed + 10557.5)
        assert mult == pytest.approx(1.0 - expected_reduction, abs=1e-4)

    def test_armor_cap_75_percent(self) -> None:
        """Very high armor should cap at 75% reduction (multiplier 0.25)."""
        mult = calc_armor_reduction(100000)
        assert mult == pytest.approx(0.25, abs=1e-4)

    def test_zero_armor_no_reduction(self) -> None:
        """Zero armor = no reduction = multiplier 1.0."""
        mult = calc_armor_reduction(0)
        assert mult == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# calc_weapon_damage
# =============================================================================


class TestCalcWeaponDamage:
    """Tests for weapon damage calculation."""

    def test_normal_swing_midpoint(self) -> None:
        """Normal swing with roll=0.5, 0 AP."""
        dmg = calc_weapon_damage(100.0, 200.0, 2.7, 0.0, roll=0.5)
        assert dmg == pytest.approx(150.0, abs=1e-6)

    def test_normalized_dagger(self) -> None:
        """Normalized dagger: AP scales with 1.7 speed instead of weapon speed."""
        dmg = calc_weapon_damage(50.0, 100.0, 1.8, 1400.0, normalized=True, norm_speed=NORM_SPEED_DAGGER, roll=0.5)
        expected = 75.0 + 1400.0 / 14.0 * 1.7
        assert dmg == pytest.approx(expected, abs=1e-6)

    def test_normalized_sword(self) -> None:
        """Normalized sword: AP scales with 2.4 speed."""
        dmg = calc_weapon_damage(80.0, 150.0, 2.7, 1400.0, normalized=True, norm_speed=NORM_SPEED_ONE_HAND, roll=0.5)
        expected = 115.0 + 1400.0 / 14.0 * 2.4
        assert dmg == pytest.approx(expected, abs=1e-6)

    def test_with_ap_scaling(self) -> None:
        """AP contributes AP/14 * speed to damage."""
        dmg = calc_weapon_damage(100.0, 100.0, 2.7, 1400.0, roll=0.0)
        expected = 100.0 + 1400.0 / 14.0 * 2.7
        assert dmg == pytest.approx(expected, abs=1e-6)


# =============================================================================
# calc_haste_multiplier
# =============================================================================


class TestCalcHasteMultiplier:
    """Tests for haste multiplier calculation."""

    def test_zero_rating_no_buffs(self) -> None:
        mult = calc_haste_multiplier(0.0)
        assert mult == pytest.approx(1.0, abs=1e-6)

    def test_with_snd(self) -> None:
        """Slice and Dice 30% haste as a multiplicative buff."""
        mult = calc_haste_multiplier(0.0, 1.30)
        assert mult == pytest.approx(1.30, abs=1e-6)

    def test_multiplicative_stacking(self) -> None:
        """Rating + SND + Blade Flurry haste all multiply."""
        # 157.7 rating = 10% from rating => 1.10
        # SND = 1.30, another buff = 1.20
        mult = calc_haste_multiplier(157.7, 1.30, 1.20)
        expected = 1.10 * 1.30 * 1.20
        assert mult == pytest.approx(expected, abs=1e-2)


# =============================================================================
# calc_effective_speed
# =============================================================================


class TestCalcEffectiveSpeed:
    """Tests for effective weapon speed calculation."""

    def test_no_haste(self) -> None:
        speed = calc_effective_speed(2.7, 1.0)
        assert speed == pytest.approx(2.7, abs=1e-6)

    def test_with_haste(self) -> None:
        speed = calc_effective_speed(2.7, 1.30)
        assert speed == pytest.approx(2.7 / 1.30, abs=1e-4)


# =============================================================================
# calc_poison_proc_chance
# =============================================================================


class TestCalcPoisonProcChance:
    """Tests for poison proc chance calculation."""

    def test_base_chance_no_talent(self) -> None:
        chance = calc_poison_proc_chance(0.20, 0)
        assert chance == pytest.approx(0.20, abs=1e-6)

    def test_with_imp_poisons_5(self) -> None:
        """5 ranks = +10% flat."""
        chance = calc_poison_proc_chance(0.20, 5)
        assert chance == pytest.approx(0.30, abs=1e-6)


# =============================================================================
# calc_ppm_proc_chance
# =============================================================================


class TestCalcPpmProcChance:
    """Tests for PPM proc chance calculation."""

    def test_known_value(self) -> None:
        """1 PPM at 2.7 speed = 2.7/60 = 0.045."""
        chance = calc_ppm_proc_chance(1.0, 2.7)
        assert chance == pytest.approx(2.7 / 60.0, abs=1e-6)

    def test_mongoose_ppm(self) -> None:
        """Mongoose ~1.0 PPM at 2.7 speed."""
        chance = calc_ppm_proc_chance(1.0, 2.7)
        assert chance == pytest.approx(0.045, abs=1e-3)


# =============================================================================
# calc_glancing_reduction and calc_normalized_speed
# =============================================================================


class TestGlancingReduction:
    """Tests for glancing blow reduction."""

    def test_returns_glancing_multiplier(self) -> None:
        assert calc_glancing_reduction() == GLANCING_MULTIPLIER


class TestCalcNormalizedSpeed:
    """Tests for normalized weapon speed lookup."""

    def test_dagger_returns_1_7(self) -> None:
        assert calc_normalized_speed(WeaponType.DAGGER) == NORM_SPEED_DAGGER

    def test_sword_returns_2_4(self) -> None:
        assert calc_normalized_speed(WeaponType.SWORD) == NORM_SPEED_ONE_HAND

    def test_fist_returns_2_4(self) -> None:
        assert calc_normalized_speed(WeaponType.FIST) == NORM_SPEED_ONE_HAND

    def test_mace_returns_2_4(self) -> None:
        assert calc_normalized_speed(WeaponType.MACE) == NORM_SPEED_ONE_HAND
