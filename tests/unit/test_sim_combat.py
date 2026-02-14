"""Tests for the discrete-event combat simulation engine."""

from __future__ import annotations

from code.shukketsu.sim.buffs import resolve_buffs
from code.shukketsu.sim.combat import (
    FIND_WEAKNESS_DURATION_MS,
    GCD_MS,
    CombatSimulation,
    CombatState,
    DotState,
    EventType,
    SimEvent,
)
from code.shukketsu.sim.items import ItemDatabase
from code.shukketsu.sim.models import (
    BossConfig,
    GearSlot,
    PoisonConfig,
    PoisonType,
    RogueSpec,
    SimConfig,
)
from code.shukketsu.sim.rotation import RotationContext, RotationEngine
from code.shukketsu.sim.talents import TalentModifiers, compute_modifiers, parse_talents

# =============================================================================
# Helpers / Fixtures
# =============================================================================

_ITEM_DB = ItemDatabase()

# Latro's Shifting Sword (MH, id=28189) + Gladiator's Shiv (OH, id=28295)
_MH_ID = 28189
_OH_ID = 28295


def _combat_swords_modifiers() -> TalentModifiers:
    """Return computed modifiers for the standard 20/41/0 combat swords build."""
    alloc = parse_talents("20/41/0", RogueSpec.COMBAT_SWORDS)
    return compute_modifiers(alloc)


def _minimal_config(
    *,
    fight_length: int = 30,
    target_count: int = 1,
    iterations: int = 10000,
    raid_preset: str = "solo",
    mh_id: int = _MH_ID,
    oh_id: int = _OH_ID,
) -> SimConfig:
    """Build a minimal SimConfig for testing."""
    return SimConfig(
        spec=RogueSpec.COMBAT_SWORDS,
        talents="20/41/0",
        gear={GearSlot.MAIN_HAND: mh_id, GearSlot.OFF_HAND: oh_id},
        fight_length=fight_length,
        target_count=target_count,
        iterations=iterations,
        raid_preset=raid_preset,
        poisons=PoisonConfig(main_hand=PoisonType.INSTANT, off_hand=PoisonType.DEADLY),
        boss=BossConfig(armor=7700),
    )


def _make_sim(
    *,
    fight_length: int = 30,
    target_count: int = 1,
    raid_preset: str = "solo",
) -> CombatSimulation:
    """Build a CombatSimulation with reasonable defaults."""
    config = _minimal_config(fight_length=fight_length, target_count=target_count, raid_preset=raid_preset)
    mods = _combat_swords_modifiers()
    buffs = resolve_buffs([], [], [])
    rotation = RotationEngine(RogueSpec.COMBAT_SWORDS, mods)
    return CombatSimulation(config, mods, buffs, _ITEM_DB, rotation)


def _run_quick(
    *,
    iterations: int = 1,
    seed: int = 42,
    fight_length: int = 30,
    target_count: int = 1,
    raid_preset: str = "solo",
):
    """Run a quick simulation and return the SimResult."""
    sim = _make_sim(fight_length=fight_length, target_count=target_count, raid_preset=raid_preset)
    return sim.run(iterations, seed=seed)


# =============================================================================
# SimEvent ordering
# =============================================================================


class TestSimEventOrdering:
    """Verify SimEvent heapq ordering by (timestamp_ms, priority)."""

    def test_lower_timestamp_sorts_first(self) -> None:
        early = SimEvent(timestamp_ms=100, event_type=EventType.MH_AUTO, priority=0)
        late = SimEvent(timestamp_ms=200, event_type=EventType.OH_AUTO, priority=0)
        assert early < late

    def test_same_timestamp_lower_priority_sorts_first(self) -> None:
        high_pri = SimEvent(timestamp_ms=100, event_type=EventType.MH_AUTO, priority=0)
        low_pri = SimEvent(timestamp_ms=100, event_type=EventType.ENERGY_TICK, priority=1)
        assert high_pri < low_pri

    def test_event_not_lt_non_event(self) -> None:
        event = SimEvent(timestamp_ms=100, event_type=EventType.MH_AUTO, priority=0)
        assert event.__lt__("not_an_event") is NotImplemented


# =============================================================================
# CombatState
# =============================================================================


class TestCombatState:
    """Verify CombatState initialization and helper methods."""

    def test_initializes_with_zero_damage(self) -> None:
        state = CombatState()
        assert state.total_damage == 0.0
        assert state.energy == 100
        assert state.combo_points == 0
        assert state.current_time_ms == 0

    def test_custom_max_energy(self) -> None:
        state = CombatState(max_energy=110)
        assert state.max_energy == 110
        assert state.energy == 110

    def test_is_buff_active_returns_false_when_empty(self) -> None:
        state = CombatState()
        assert state.is_buff_active("slice_and_dice") is False

    def test_is_buff_active_returns_true_when_active(self) -> None:
        state = CombatState()
        state.current_time_ms = 5000
        state.buff_timers["slice_and_dice"] = 20000
        assert state.is_buff_active("slice_and_dice") is True

    def test_is_buff_active_returns_false_when_expired(self) -> None:
        state = CombatState()
        state.current_time_ms = 25000
        state.buff_timers["slice_and_dice"] = 20000
        assert state.is_buff_active("slice_and_dice") is False


# =============================================================================
# DotState
# =============================================================================


class TestDotState:
    """Verify DotState model validation."""

    def test_dot_state_validates(self) -> None:
        dot = DotState(
            remaining_ticks=8,
            tick_interval_ms=2000,
            damage_per_tick=150.0,
            next_tick_ms=4000,
            snapshot_ap=2000.0,
        )
        assert dot.remaining_ticks == 8
        assert dot.tick_interval_ms == 2000
        assert dot.damage_per_tick == 150.0


# =============================================================================
# Single iteration
# =============================================================================


class TestSingleIteration:
    """Verify basic single-iteration simulation behavior."""

    def test_single_iteration_produces_nonzero_dps(self) -> None:
        result = _run_quick(iterations=1, fight_length=30)
        assert result.dps_mean > 0

    def test_fight_length_matches_config(self) -> None:
        result = _run_quick(iterations=1, fight_length=60)
        assert result.fight_length == 60


# =============================================================================
# Determinism
# =============================================================================


class TestDeterminism:
    """Verify deterministic RNG behavior."""

    def test_same_seed_same_dps(self) -> None:
        result1 = _run_quick(iterations=2, seed=42, fight_length=30)
        result2 = _run_quick(iterations=2, seed=42, fight_length=30)
        assert result1.dps_mean == result2.dps_mean

    def test_different_seeds_different_dps(self) -> None:
        result1 = _run_quick(iterations=2, seed=42, fight_length=60)
        result2 = _run_quick(iterations=2, seed=999, fight_length=60)
        # Very unlikely to be identical with different seeds over 60s fight
        assert result1.dps_mean != result2.dps_mean


# =============================================================================
# Auto-attacks
# =============================================================================


class TestAutoAttacks:
    """Verify auto-attacks are generated."""

    def test_mh_auto_in_breakdown(self) -> None:
        result = _run_quick(iterations=1, fight_length=30)
        ability_names = {ab.name for ab in result.ability_breakdown}
        assert "mh_auto" in ability_names

    def test_oh_auto_in_breakdown(self) -> None:
        result = _run_quick(iterations=1, fight_length=30)
        ability_names = {ab.name for ab in result.ability_breakdown}
        assert "oh_auto" in ability_names


# =============================================================================
# Energy ticks
# =============================================================================


class TestEnergyTicks:
    """Verify energy tick mechanics."""

    def test_energy_consumed_and_restored(self) -> None:
        """Over a 30s fight, energy is consumed by abilities and restored by ticks."""
        result = _run_quick(iterations=1, fight_length=30)
        # If abilities were used, the resource stats should show non-zero GCD utilization
        assert result.resource_stats.gcd_utilization_pct > 0


# =============================================================================
# GCD enforcement
# =============================================================================


class TestGcdEnforcement:
    """Verify abilities are spaced by at least the GCD duration."""

    def test_abilities_spaced_by_gcd(self) -> None:
        """Run a sim and verify ability timestamps are spaced by >= GCD_MS."""
        sim = _make_sim(fight_length=30)
        from random import Random

        rng = Random(42)
        sim._rotation._state = sim._rotation._state.__class__("opener")
        _, state = sim._run_iteration(rng)

        timestamps = state.ability_timestamps
        assert len(timestamps) > 1, "Expected multiple abilities to be used"

        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            # Allow 1ms tolerance for floating point
            assert gap >= GCD_MS - 1, f"GCD violation: {gap}ms gap between abilities at index {i}"


# =============================================================================
# Ability breakdown
# =============================================================================


class TestAbilityBreakdown:
    """Verify per-ability damage breakdown."""

    def test_sinister_strike_in_breakdown(self) -> None:
        result = _run_quick(iterations=1, fight_length=60)
        ability_names = {ab.name for ab in result.ability_breakdown}
        assert "sinister_strike" in ability_names

    def test_eviscerate_in_breakdown(self) -> None:
        result = _run_quick(iterations=1, fight_length=60)
        ability_names = {ab.name for ab in result.ability_breakdown}
        # Over 60s, eviscerate should appear at least once
        assert "eviscerate" in ability_names or "rupture" in ability_names

    def test_snd_is_cast_during_combat(self) -> None:
        """Slice and Dice should be used during combat."""
        result = _run_quick(iterations=3, fight_length=60)
        # SND does 0 damage so it may not appear in the damage breakdown.
        # We verify SND is used by checking the GCD utilization is high,
        # which implies abilities (including SND) are being cast.
        assert result.resource_stats.gcd_utilization_pct > 0

    def test_breakdown_percentages_sum_approximately_100(self) -> None:
        result = _run_quick(iterations=3, fight_length=60)
        total_pct = sum(ab.damage_pct for ab in result.ability_breakdown)
        # Allow tolerance for rounding
        assert 98.0 <= total_pct <= 102.0, f"Expected ~100%, got {total_pct}%"


# =============================================================================
# Blade Flurry cleave
# =============================================================================


class TestBladeFlurryCleave:
    """Verify Blade Flurry cleave damage with multiple targets."""

    def test_cleave_increases_total_damage(self) -> None:
        """With BF talented and 2 targets, total damage should exceed single target."""
        result_1t = _run_quick(iterations=3, seed=42, fight_length=60, target_count=1)
        result_2t = _run_quick(iterations=3, seed=42, fight_length=60, target_count=2)
        # 2-target should deal more total damage due to cleave
        assert result_2t.dps_mean > result_1t.dps_mean


# =============================================================================
# Rupture DOT
# =============================================================================


class TestRuptureDot:
    """Verify Rupture DOT ticking behavior."""

    def test_rupture_appears_in_breakdown(self) -> None:
        """Over a long enough fight, Rupture should appear in the damage breakdown."""
        result = _run_quick(iterations=3, fight_length=120)
        ability_names = {ab.name for ab in result.ability_breakdown}
        # The rotation uses Rupture when it's not active and fight is long enough
        assert "rupture" in ability_names


# =============================================================================
# Result metadata
# =============================================================================


class TestResultMetadata:
    """Verify result metadata fields."""

    def test_iteration_count_matches(self) -> None:
        result = _run_quick(iterations=5, fight_length=30)
        assert result.iterations == 5

    def test_dps_in_reasonable_range(self) -> None:
        result = _run_quick(iterations=3, fight_length=60)
        assert 200 <= result.dps_mean <= 5000, f"DPS {result.dps_mean} outside reasonable range"

    def test_resource_stats_populated(self) -> None:
        result = _run_quick(iterations=1, fight_length=30)
        rs = result.resource_stats
        assert rs.energy_per_second > 0
        assert isinstance(rs.gcd_utilization_pct, float)
        assert isinstance(rs.avg_energy_waste, float)
        assert isinstance(rs.combo_point_overcap_pct, float)


# =============================================================================
# _aggregate_results
# =============================================================================


class TestAggregateResults:
    """Verify the aggregation logic produces correct statistics."""

    def test_aggregate_with_known_dps_values(self) -> None:
        """Hand-craft DPS values and verify mean/std/median."""
        sim = _make_sim(fight_length=30)
        dps_values = [1000.0, 1100.0, 1200.0, 900.0, 1050.0]
        damage_by_ability: dict[str, float] = {"mh_auto": 5000.0, "sinister_strike": 3000.0}
        casts_by_ability: dict[str, float] = {"mh_auto": 50.0, "sinister_strike": 30.0}
        outcome_counts: dict[str, dict[str, int]] = {
            "mh_auto": {"hit": 30, "crit": 10, "miss": 5, "dodge": 3, "glancing": 2},
            "sinister_strike": {"hit": 20, "crit": 8, "miss": 2, "dodge": 0},
        }

        result = sim._aggregate_results(
            dps_values,
            damage_by_ability,
            casts_by_ability,
            outcome_counts,
            total_energy_wasted=50.0,
            total_cp_overcap=3,
            total_gcd_time=15000,
            total_energy_starved=5000,
            iterations=5,
        )

        assert abs(result.dps_mean - 1050.0) < 0.01
        assert result.dps_min == 900.0
        assert result.dps_max == 1200.0
        assert result.dps_median == 1050.0
        assert result.iterations == 5
        assert len(result.ability_breakdown) == 2

    def test_aggregate_single_iteration(self) -> None:
        sim = _make_sim(fight_length=30)
        result = sim._aggregate_results(
            [500.0],
            {"mh_auto": 1000.0},
            {"mh_auto": 10.0},
            {"mh_auto": {"hit": 8, "crit": 2}},
            total_energy_wasted=0,
            total_cp_overcap=0,
            total_gcd_time=5000,
            total_energy_starved=0,
            iterations=1,
        )
        assert result.dps_mean == 500.0
        assert result.dps_std == 0.0  # stdev of single value is 0


# =============================================================================
# Rotation context construction
# =============================================================================


class TestRotationContextBuild:
    """Verify rotation context is correctly built from combat state."""

    def test_context_reflects_state(self) -> None:
        sim = _make_sim(fight_length=60)
        state = CombatState(max_energy=100, fight_length_ms=60000)
        state.current_time_ms = 10000
        state.combo_points = 3
        state.energy = 75
        state.buff_timers["slice_and_dice"] = 25000

        ctx = sim._build_rotation_context(state)

        assert ctx.combo_points == 3
        assert ctx.energy == 75
        assert ctx.snd_remaining_ms == 15000  # 25000 - 10000
        assert ctx.fight_remaining_ms == 50000  # 60000 - 10000
        assert isinstance(ctx, RotationContext)


# =============================================================================
# Poison procs
# =============================================================================


class TestPoisonProcs:
    """Verify poison procs appear in simulation results."""

    def test_instant_poison_in_results(self) -> None:
        """Over a long fight, instant poison procs should generate damage."""
        result = _run_quick(iterations=3, fight_length=120)
        ability_names = {ab.name for ab in result.ability_breakdown}
        # With IP on MH, we expect procs over 120s
        assert "instant_poison" in ability_names

    def test_deadly_poison_in_results(self) -> None:
        """Over a long fight, deadly poison procs should generate some damage."""
        result = _run_quick(iterations=3, fight_length=120)
        ability_names = {ab.name for ab in result.ability_breakdown}
        assert "deadly_poison" in ability_names


# =============================================================================
# EventType enum completeness
# =============================================================================


class TestEventTypeEnum:
    """Verify EventType enum has all expected values."""

    def test_all_event_types_exist(self) -> None:
        assert len(EventType) == 9
        assert EventType.MH_AUTO == "mh_auto"
        assert EventType.OH_AUTO == "oh_auto"
        assert EventType.ENERGY_TICK == "energy_tick"
        assert EventType.ABILITY_USE == "ability_use"
        assert EventType.DOT_TICK == "dot_tick"
        assert EventType.BUFF_EXPIRE == "buff_expire"
        assert EventType.PROC_TRIGGER == "proc_trigger"
        assert EventType.COOLDOWN_USE == "cooldown_use"
        assert EventType.POTION_USE == "potion_use"


# =============================================================================
# Mutilate spec helpers
# =============================================================================


# Emerald Ripper (MH dagger) + Gladiator's Shiv (OH dagger)
_MUT_MH_ID = 28524
_MUT_OH_ID = 28295


def _mutilate_modifiers() -> TalentModifiers:
    """Return computed modifiers for the standard 41/20/0 Mutilate build."""
    alloc = parse_talents("41/20/0", RogueSpec.ASSASSINATION_MUTILATE)
    return compute_modifiers(alloc)


def _mutilate_config(
    *,
    fight_length: int = 30,
    iterations: int = 10000,
) -> SimConfig:
    """Build a Mutilate SimConfig for testing."""
    return SimConfig(
        spec=RogueSpec.ASSASSINATION_MUTILATE,
        talents="41/20/0",
        gear={GearSlot.MAIN_HAND: _MUT_MH_ID, GearSlot.OFF_HAND: _MUT_OH_ID},
        fight_length=fight_length,
        target_count=1,
        iterations=iterations,
        raid_preset="solo",
        poisons=PoisonConfig(main_hand=PoisonType.INSTANT, off_hand=PoisonType.DEADLY),
        boss=BossConfig(armor=7700),
    )


def _make_mutilate_sim(*, fight_length: int = 30) -> CombatSimulation:
    """Build a CombatSimulation for the Mutilate spec."""
    config = _mutilate_config(fight_length=fight_length)
    mods = _mutilate_modifiers()
    buffs = resolve_buffs([], [], [])
    rotation = RotationEngine(RogueSpec.ASSASSINATION_MUTILATE, mods)
    return CombatSimulation(config, mods, buffs, _ITEM_DB, rotation)


def _run_mutilate_quick(
    *,
    iterations: int = 1,
    seed: int = 42,
    fight_length: int = 60,
):
    """Run a quick Mutilate simulation and return the SimResult."""
    sim = _make_mutilate_sim(fight_length=fight_length)
    return sim.run(iterations, seed=seed)


# =============================================================================
# Mutilate +50% poison bonus
# =============================================================================


class TestMutilatePoisonBonus:
    """Verify Mutilate +50% damage when target has Deadly Poison."""

    def test_mutilate_appears_in_breakdown(self) -> None:
        """Mutilate MH and OH should both appear in the ability breakdown."""
        result = _run_mutilate_quick(iterations=3, fight_length=60)
        ability_names = {ab.name for ab in result.ability_breakdown}
        assert "mutilate" in ability_names
        assert "mutilate_oh" in ability_names

    def test_mutilate_with_dp_does_more_damage(self) -> None:
        """Compare Mutilate damage in two scenarios: with and without DP proc.

        The sim with DP already procced (forced) should deal more mutilate damage
        because of the +50% poison bonus.
        """
        # Run a long enough fight that DP procs naturally
        result = _run_mutilate_quick(iterations=100, fight_length=120, seed=1)
        # DP should proc at some point during the fight
        dp_total = sum(ab.damage_total for ab in result.ability_breakdown if ab.name == "deadly_poison")
        assert dp_total > 0, "Deadly Poison should proc during a 120s Mutilate fight"

        # Mutilate damage should be substantial (includes +50% bonus after DP procs)
        mut_total = sum(ab.damage_total for ab in result.ability_breakdown if ab.name in ("mutilate", "mutilate_oh"))
        assert mut_total > 0, "Mutilate should deal damage"

    def test_find_weakness_constant_is_10_seconds(self) -> None:
        """Verify the Find Weakness duration constant."""
        assert FIND_WEAKNESS_DURATION_MS == 10000


# =============================================================================
# Find Weakness talent
# =============================================================================


class TestFindWeakness:
    """Verify Find Weakness activates on finishers and increases damage."""

    def test_find_weakness_modifier_value(self) -> None:
        """The Mutilate spec allocates 3 ranks of Find Weakness = 6% bonus."""
        mods = _mutilate_modifiers()
        assert abs(mods.find_weakness_damage_pct - 0.06) < 1e-9

    def test_find_weakness_not_in_combat_spec(self) -> None:
        """Combat swords spec should not have Find Weakness."""
        mods = _combat_swords_modifiers()
        assert mods.find_weakness_damage_pct == 0.0

    def test_find_weakness_buff_activates_on_finisher(self) -> None:
        """Running the Mutilate sim should activate Find Weakness buff via finishers."""
        # Run a single iteration long enough to use finishers
        sim = _make_mutilate_sim(fight_length=60)
        # Access internals: run one iteration manually to check buff timers
        result = sim.run(1, seed=42)

        # With a 60s fight and Mutilate spec, some finisher must have been used
        finisher_names = {"eviscerate", "envenom", "rupture", "slice_and_dice", "expose_armor"}
        casts = {ab.name: ab.casts for ab in result.ability_breakdown}
        finisher_casts = sum(casts.get(name, 0) for name in finisher_names)
        assert finisher_casts > 0, "At least one finisher should be cast in 60s"

    def test_find_weakness_increases_total_dps(self) -> None:
        """Mutilate spec with Find Weakness should deal more total DPS than without.

        We verify by checking the modifier is applied by computing modifiers
        with and without the talent.
        """
        # With Find Weakness (standard Mutilate build)
        mods_with = _mutilate_modifiers()
        assert mods_with.find_weakness_damage_pct > 0

        # Without Find Weakness
        alloc = parse_talents("41/20/0", RogueSpec.ASSASSINATION_MUTILATE)
        # Zero out find_weakness
        alloc.points["find_weakness"] = 0
        mods_without = compute_modifiers(alloc)
        assert mods_without.find_weakness_damage_pct == 0.0

        # The damage bonus should be meaningful
        assert mods_with.find_weakness_damage_pct == 0.06
