"""Tests for the TBC Rogue rotation state-machine engine."""

import random

import pytest

from code.shukketsu.sim.models import RogueSpec
from code.shukketsu.sim.rotation import (
    RotationAction,
    RotationContext,
    RotationEngine,
    RotationState,
)
from code.shukketsu.sim.talents import TalentModifiers

# =============================================================================
# Helpers
# =============================================================================


def _make_ctx(
    *,
    combo_points: int = 0,
    energy: int = 80,
    max_energy: int = 100,
    snd_remaining_ms: int = 20000,
    rupture_remaining_ms: int = 0,
    ea_remaining_ms: int = 0,
    dp_remaining_ms: int = 10000,
    dp_stacks: int = 5,
    ar_active: bool = False,
    bf_active: bool = False,
    bf_ready: bool = False,
    ar_ready: bool = False,
    cb_ready: bool = False,
    tea_ready: bool = False,
    premeditation_ready: bool = False,
    fight_remaining_ms: int = 200000,
    target_count: int = 1,
    is_stealthed: bool = False,
    gcd_ready_at_ms: int = 0,
    current_time_ms: int = 5000,
) -> RotationContext:
    """Build a RotationContext with sensible defaults; override any field."""
    return RotationContext(
        combo_points=combo_points,
        energy=energy,
        max_energy=max_energy,
        snd_remaining_ms=snd_remaining_ms,
        rupture_remaining_ms=rupture_remaining_ms,
        ea_remaining_ms=ea_remaining_ms,
        dp_remaining_ms=dp_remaining_ms,
        dp_stacks=dp_stacks,
        ar_active=ar_active,
        bf_active=bf_active,
        bf_ready=bf_ready,
        ar_ready=ar_ready,
        cb_ready=cb_ready,
        tea_ready=tea_ready,
        premeditation_ready=premeditation_ready,
        fight_remaining_ms=fight_remaining_ms,
        target_count=target_count,
        is_stealthed=is_stealthed,
        gcd_ready_at_ms=gcd_ready_at_ms,
        current_time_ms=current_time_ms,
    )


def _combat_swords_mods() -> TalentModifiers:
    """Standard 20/41/0 combat swords modifiers."""
    return TalentModifiers(
        adrenaline_rush=True,
        blade_flurry=True,
        cold_blood=False,
    )


def _mut_mods() -> TalentModifiers:
    """Standard 41/20/0 mutilate modifiers."""
    return TalentModifiers(
        cold_blood=True,
        mutilate_talented=True,
        vigor=True,
    )


def _combat_daggers_mods() -> TalentModifiers:
    """Standard combat daggers modifiers."""
    return TalentModifiers(
        adrenaline_rush=True,
        blade_flurry=True,
        cold_blood=False,
    )


# =============================================================================
# RotationState enum
# =============================================================================


class TestRotationState:
    """Verify RotationState enum values."""

    def test_all_states_exist(self) -> None:
        assert len(RotationState) == 7
        assert RotationState.OPENER == "opener"
        assert RotationState.SLICE_ASAP == "slice_asap"
        assert RotationState.DISPATCH == "dispatch"


# =============================================================================
# RotationContext model
# =============================================================================


class TestRotationContext:
    """Verify RotationContext model validation."""

    def test_context_validates_with_all_fields(self) -> None:
        ctx = _make_ctx()
        assert ctx.combo_points == 0
        assert ctx.energy == 80
        assert ctx.max_energy == 100
        assert ctx.fight_remaining_ms == 200000

    def test_context_is_frozen(self) -> None:
        ctx = _make_ctx()
        with pytest.raises(Exception):  # noqa: B017
            ctx.energy = 50  # type: ignore[misc]


# =============================================================================
# Opener phase
# =============================================================================


class TestOpenerState:
    """Tests for the OPENER state transitions."""

    def test_opener_returns_garrote_when_stealthed(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        ctx = _make_ctx(is_stealthed=True, snd_remaining_ms=0)
        action = engine.decide(ctx)
        assert action.ability_name == "garrote"
        assert action.target == "boss"

    def test_opener_with_combo_points_transitions_to_slice_asap(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        ctx = _make_ctx(combo_points=1, is_stealthed=False, snd_remaining_ms=0)
        action = engine.decide(ctx)
        # Should transition through SLICE_ASAP and return SND.
        assert action.ability_name == "slice_and_dice"

    def test_opener_without_stealth_or_cps_dispatches(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        ctx = _make_ctx(combo_points=0, is_stealthed=False, snd_remaining_ms=0)
        action = engine.decide(ctx)
        # No CPs, SND down -> build a CP first.
        assert action.ability_name == "sinister_strike"


# =============================================================================
# Slice-and-Dice ASAP
# =============================================================================


class TestSliceAsapState:
    """Tests for the SLICE_ASAP state."""

    def test_slice_asap_returns_snd_at_1cp(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        # Force engine into SLICE_ASAP state.
        engine._state = RotationState.SLICE_ASAP
        ctx = _make_ctx(combo_points=1, snd_remaining_ms=0)
        action = engine.decide(ctx)
        assert action.ability_name == "slice_and_dice"
        assert action.target == "self"

    def test_slice_asap_returns_snd_at_5cp(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.SLICE_ASAP
        ctx = _make_ctx(combo_points=5, snd_remaining_ms=0)
        action = engine.decide(ctx)
        assert action.ability_name == "slice_and_dice"


# =============================================================================
# DISPATCH — SND management
# =============================================================================


class TestDispatchSnd:
    """Tests for SND-related DISPATCH decisions."""

    def test_dispatch_with_snd_expired_uses_snd(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(combo_points=3, snd_remaining_ms=0)
        action = engine.decide(ctx)
        assert action.ability_name == "slice_and_dice"

    def test_dispatch_with_snd_expired_no_cps_builds(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(combo_points=0, snd_remaining_ms=0)
        action = engine.decide(ctx)
        assert action.ability_name == "sinister_strike"


# =============================================================================
# DISPATCH — finisher selection
# =============================================================================


class TestDispatchFinisher:
    """Tests for finisher selection in DISPATCH."""

    def test_enough_cps_and_snd_active_uses_damage_finisher(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(combo_points=5, snd_remaining_ms=20000, energy=80)
        action = engine.decide(ctx)
        # Combat with no rupture active -> rupture.
        assert action.ability_name in {"eviscerate", "rupture"}

    def test_low_cps_builds(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(combo_points=2, snd_remaining_ms=20000)
        action = engine.decide(ctx)
        assert action.ability_name == "sinister_strike"

    def test_combat_rupture_not_active_uses_rupture(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(
            combo_points=5,
            snd_remaining_ms=20000,
            rupture_remaining_ms=0,
            fight_remaining_ms=200000,
            energy=80,
        )
        action = engine.decide(ctx)
        assert action.ability_name == "rupture"

    def test_mutilate_uses_envenom(self) -> None:
        engine = RotationEngine(RogueSpec.ASSASSINATION_MUTILATE, _mut_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(combo_points=4, snd_remaining_ms=20000, energy=80)
        action = engine.decide(ctx)
        assert action.ability_name == "envenom"


# =============================================================================
# Energy pooling
# =============================================================================


class TestEnergyPooling:
    """Tests for energy pooling behavior."""

    def test_no_finisher_at_40_energy(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(combo_points=5, snd_remaining_ms=20000, energy=40)
        action = engine.decide(ctx)
        # Should pool energy (40 < 50 threshold) -> wait.
        assert action.wait_for_energy is True

    def test_finisher_at_40_energy_during_ar(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(combo_points=5, snd_remaining_ms=20000, energy=40, ar_active=True)
        action = engine.decide(ctx)
        # During AR, threshold drops to 30 -- 40 >= 30, so finisher fires.
        assert action.ability_name in {"eviscerate", "rupture"}
        assert action.wait_for_energy is False


# =============================================================================
# Builder selection
# =============================================================================


class TestBuilderSelection:
    """Tests for spec-based builder selection."""

    def test_combat_swords_uses_sinister_strike(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(combo_points=1, snd_remaining_ms=20000)
        action = engine.decide(ctx)
        assert action.ability_name == "sinister_strike"

    def test_assassination_mutilate_uses_mutilate(self) -> None:
        engine = RotationEngine(RogueSpec.ASSASSINATION_MUTILATE, _mut_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(combo_points=1, snd_remaining_ms=20000)
        action = engine.decide(ctx)
        assert action.ability_name == "mutilate"

    def test_combat_daggers_uses_backstab(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_DAGGERS, _combat_daggers_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(combo_points=1, snd_remaining_ms=20000)
        action = engine.decide(ctx)
        assert action.ability_name == "backstab"


# =============================================================================
# Cooldown timing
# =============================================================================


class TestCooldownTiming:
    """Tests for off-GCD cooldown decisions."""

    def test_ar_offered_at_low_energy_with_snd(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(energy=60, ar_ready=True, snd_remaining_ms=20000)
        action = engine.decide(ctx)
        assert action.ability_name == "adrenaline_rush"

    def test_ar_not_offered_at_high_energy(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(energy=100, ar_ready=True, snd_remaining_ms=20000)
        action = engine.decide(ctx)
        assert action.ability_name != "adrenaline_rush"

    def test_bf_offered_when_snd_active(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(bf_ready=True, snd_remaining_ms=20000, ar_ready=False)
        action = engine.decide(ctx)
        assert action.ability_name == "blade_flurry"

    def test_tea_offered_at_low_energy(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(energy=0, max_energy=100, tea_ready=True, snd_remaining_ms=20000)
        action = engine.decide(ctx)
        assert action.ability_name == "thistle_tea"

    def test_tea_not_offered_at_high_energy(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(energy=10, max_energy=100, tea_ready=True, snd_remaining_ms=20000)
        action = engine.decide(ctx)
        # energy=10 > max-100=0, so tea is not offered.
        assert action.ability_name != "thistle_tea"


# =============================================================================
# Shiv for Deadly Poison (Mutilate only)
# =============================================================================


class TestShivDeadlyPoison:
    """Tests for DP maintenance via Shiv."""

    def test_shiv_when_dp_low_mutilate(self) -> None:
        engine = RotationEngine(RogueSpec.ASSASSINATION_MUTILATE, _mut_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(
            combo_points=1,
            snd_remaining_ms=20000,
            dp_remaining_ms=1000,
            dp_stacks=5,
        )
        action = engine.decide(ctx)
        assert action.ability_name == "shiv"

    def test_no_shiv_for_combat_spec(self) -> None:
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(
            combo_points=1,
            snd_remaining_ms=20000,
            dp_remaining_ms=1000,
        )
        action = engine.decide(ctx)
        # Combat swords should build, not shiv.
        assert action.ability_name == "sinister_strike"


# =============================================================================
# SND refresh buffer
# =============================================================================


class TestSndRefreshBuffer:
    """Tests for the SND refresh buffer formula."""

    def test_refresh_at_1cp_when_snd_below_3200ms(self) -> None:
        """At 1 CP, buffer = 4000 - 1*800 = 3200ms. Refresh when SND < 3200."""
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(combo_points=1, snd_remaining_ms=3000, energy=80)
        action = engine.decide(ctx)
        assert action.ability_name == "slice_and_dice"

    def test_no_refresh_at_1cp_when_snd_above_buffer(self) -> None:
        """At 1 CP, buffer = 3200ms. SND at 5000ms should not trigger refresh."""
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(combo_points=1, snd_remaining_ms=5000, energy=80)
        action = engine.decide(ctx)
        # Should just build, not refresh SND yet.
        assert action.ability_name == "sinister_strike"

    def test_at_5cp_buffer_is_zero(self) -> None:
        """At 5 CP, buffer = 4000 - 5*800 = 0ms. Only refresh when actually expired."""
        engine = RotationEngine(RogueSpec.COMBAT_SWORDS, _combat_swords_mods())
        engine._state = RotationState.DISPATCH
        ctx = _make_ctx(combo_points=5, snd_remaining_ms=100, energy=80)
        action = engine.decide(ctx)
        # SND at 100ms > 0ms buffer. Should use damage finisher instead.
        assert action.ability_name in {"eviscerate", "rupture"}


# =============================================================================
# State machine robustness
# =============================================================================


class TestStateMachineRobustness:
    """Fuzz / robustness tests ensuring the engine always returns an action."""

    def test_random_contexts_always_return_action(self) -> None:
        """Run 200 random contexts and verify every call returns a RotationAction."""
        rng = random.Random(42)
        specs = list(RogueSpec)
        mod_map = {
            RogueSpec.COMBAT_SWORDS: _combat_swords_mods(),
            RogueSpec.COMBAT_FISTS: _combat_swords_mods(),
            RogueSpec.COMBAT_DAGGERS: _combat_daggers_mods(),
            RogueSpec.ASSASSINATION_MUTILATE: _mut_mods(),
        }

        for _ in range(200):
            spec = rng.choice(specs)
            engine = RotationEngine(spec, mod_map[spec])
            engine._state = rng.choice(list(RotationState))
            ctx = _make_ctx(
                combo_points=rng.randint(0, 5),
                energy=rng.randint(0, 120),
                max_energy=rng.choice([100, 110]),
                snd_remaining_ms=rng.randint(0, 30000),
                rupture_remaining_ms=rng.randint(0, 16000),
                ea_remaining_ms=rng.randint(0, 30000),
                dp_remaining_ms=rng.randint(0, 12000),
                dp_stacks=rng.randint(0, 5),
                ar_active=rng.choice([True, False]),
                bf_active=rng.choice([True, False]),
                bf_ready=rng.choice([True, False]),
                ar_ready=rng.choice([True, False]),
                cb_ready=rng.choice([True, False]),
                tea_ready=rng.choice([True, False]),
                premeditation_ready=rng.choice([True, False]),
                fight_remaining_ms=rng.randint(0, 300000),
                target_count=rng.randint(1, 3),
                is_stealthed=rng.choice([True, False]),
                gcd_ready_at_ms=rng.randint(0, 5000),
                current_time_ms=rng.randint(0, 300000),
            )
            action = engine.decide(ctx)
            assert isinstance(action, RotationAction), f"Expected RotationAction, got {type(action)}"
            assert action.ability_name, "ability_name must not be empty"
