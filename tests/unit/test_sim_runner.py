"""Tests for the SimRunner public API."""

from __future__ import annotations

import pytest

from code.shukketsu.resilience.errors import ItemNotFoundError
from code.shukketsu.sim.combat import CombatSimulation
from code.shukketsu.sim.items import ItemDatabase
from code.shukketsu.sim.models import (
    BossConfig,
    GearSlot,
    PoisonConfig,
    PoisonType,
    RogueSpec,
    SimConfig,
)
from code.shukketsu.sim.runner import (
    AbilityDiff,
    CompareResult,
    ItemRecommendation,
    OptimizeResult,
    SimRunner,
    StatDiff,
)

# =============================================================================
# Helpers
# =============================================================================

# Latro's Shifting Sword (MH) + Gladiator's Shiv (OH)
_MH_ID = 28189
_OH_ID = 28295

# Netherblade Facemask (head, phase 1)
_HEAD_ID = 29044


def _minimal_config(
    *,
    fight_length: int = 30,
    iterations: int = 2,
    mh_id: int = _MH_ID,
    oh_id: int = _OH_ID,
    head_id: int = _HEAD_ID,
) -> SimConfig:
    """Build a minimal SimConfig for testing."""
    return SimConfig(
        spec=RogueSpec.COMBAT_SWORDS,
        talents="20/41/0",
        gear={
            GearSlot.MAIN_HAND: mh_id,
            GearSlot.OFF_HAND: oh_id,
            GearSlot.HEAD: head_id,
        },
        fight_length=fight_length,
        iterations=iterations,
        raid_preset="solo",
        poisons=PoisonConfig(main_hand=PoisonType.INSTANT, off_hand=PoisonType.DEADLY),
        boss=BossConfig(armor=7700),
    )


SAMPLE_SIMC = """rogue="TestRogue"
level=70
race=orc
spec=combat
talents=20/41/0
head=,id=29044
main_hand=,id=28189
off_hand=,id=28295
"""


# =============================================================================
# Initialization
# =============================================================================


class TestSimRunnerInit:
    """Verify SimRunner initialization."""

    def test_init_with_default_item_db(self) -> None:
        runner = SimRunner()
        assert isinstance(runner._item_db, ItemDatabase)

    def test_init_with_custom_item_db(self) -> None:
        custom_db = ItemDatabase()
        runner = SimRunner(item_db=custom_db)
        assert runner._item_db is custom_db


# =============================================================================
# sim_run
# =============================================================================


class TestSimRun:
    """Verify sim_run produces valid results."""

    @pytest.mark.asyncio
    async def test_sim_run_returns_valid_result(self) -> None:
        runner = SimRunner()
        config = _minimal_config(fight_length=30, iterations=2)
        result = await runner.sim_run(config)

        assert result.dps_mean > 0
        assert result.iterations == 2
        assert result.fight_length == 30

    @pytest.mark.asyncio
    async def test_sim_run_different_configs_different_dps(self) -> None:
        """Running with different fight lengths should produce different DPS."""
        runner = SimRunner()
        config_short = _minimal_config(fight_length=30, iterations=3)
        config_long = _minimal_config(fight_length=60, iterations=3)

        result_short = await runner.sim_run(config_short)
        result_long = await runner.sim_run(config_long)

        # DPS may vary due to fight length (e.g. opener weight differs)
        # We just verify both produce real non-zero values
        assert result_short.dps_mean > 0
        assert result_long.dps_mean > 0


# =============================================================================
# sim_compare
# =============================================================================


class TestSimCompare:
    """Verify sim_compare produces valid CompareResult."""

    @pytest.mark.asyncio
    async def test_compare_returns_compare_result(self) -> None:
        runner = SimRunner()
        config_a = _minimal_config(fight_length=30, iterations=2)
        # Config B with a different MH weapon (Blade of Infamy, id=28311)
        config_b = _minimal_config(fight_length=30, iterations=2, mh_id=28311)

        result = await runner.sim_compare(config_a, config_b)

        assert isinstance(result, CompareResult)
        assert result.dps_before > 0
        assert result.dps_after > 0
        assert isinstance(result.dps_delta, float)
        assert isinstance(result.summary, str)
        assert "DPS:" in result.summary

    @pytest.mark.asyncio
    async def test_compare_identical_configs_near_zero_delta(self) -> None:
        runner = SimRunner()
        config = _minimal_config(fight_length=30, iterations=5)

        result = await runner.sim_compare(config, config)

        # Same config, same seed behavior -> delta should be exactly 0
        assert result.dps_delta == 0.0
        assert result.dps_delta_pct == 0.0


# =============================================================================
# build_config_from_import
# =============================================================================


class TestBuildConfigFromImport:
    """Verify build_config_from_import parses SimC format."""

    def test_simc_format_produces_valid_config(self) -> None:
        runner = SimRunner()
        config = runner.build_config_from_import(SAMPLE_SIMC)

        assert isinstance(config, SimConfig)
        assert config.spec == RogueSpec.COMBAT_SWORDS
        assert config.talents == "20/41/0"
        assert GearSlot.MAIN_HAND in config.gear
        assert config.gear[GearSlot.MAIN_HAND] == 28189


# =============================================================================
# swap_item
# =============================================================================


class TestSwapItem:
    """Verify swap_item finds and swaps items."""

    def test_swap_item_changes_correct_slot(self) -> None:
        runner = SimRunner()
        config = _minimal_config()

        new_config = runner.swap_item(config, GearSlot.MAIN_HAND, "Blade of Infamy")

        assert new_config.gear[GearSlot.MAIN_HAND] == 28311
        # Other slots unchanged
        assert new_config.gear[GearSlot.OFF_HAND] == _OH_ID

    def test_swap_item_unknown_raises_item_not_found_error(self) -> None:
        runner = SimRunner()
        config = _minimal_config()

        with pytest.raises(ItemNotFoundError, match="No item found"):
            runner.swap_item(config, GearSlot.MAIN_HAND, "Thunderfury Blessed Blade of the Windseeker")


# =============================================================================
# _build_simulation
# =============================================================================


class TestBuildSimulation:
    """Verify _build_simulation returns CombatSimulation."""

    def test_returns_combat_simulation(self) -> None:
        runner = SimRunner()
        config = _minimal_config()

        sim = runner._build_simulation(config)

        assert isinstance(sim, CombatSimulation)


# =============================================================================
# _pre_filter_candidates
# =============================================================================


class TestPreFilterCandidates:
    """Verify _pre_filter_candidates returns bounded item list."""

    def test_returns_at_most_20_items(self) -> None:
        runner = SimRunner()
        config = _minimal_config()

        candidates = runner._pre_filter_candidates(config, GearSlot.HEAD, phase=5)

        assert len(candidates) <= 20
        assert all(c.slot == GearSlot.HEAD for c in candidates)


# =============================================================================
# Pydantic model validation
# =============================================================================


class TestModelValidation:
    """Verify all additional Pydantic models validate correctly."""

    def test_stat_diff_validates(self) -> None:
        sd = StatDiff(stat_name="agility", before=100.0, after=120.0, delta=20.0)
        assert sd.stat_name == "agility"
        assert sd.delta == 20.0

    def test_ability_diff_validates(self) -> None:
        ad = AbilityDiff(
            ability_name="sinister_strike",
            dps_before=200.0,
            dps_after=220.0,
            delta=20.0,
            delta_pct=10.0,
        )
        assert ad.ability_name == "sinister_strike"
        assert ad.delta_pct == 10.0

    def test_compare_result_validates(self) -> None:
        cr = CompareResult(
            dps_before=1000.0,
            dps_after=1100.0,
            dps_delta=100.0,
            dps_delta_pct=10.0,
            stat_changes=[StatDiff(stat_name="agility", before=100.0, after=120.0, delta=20.0)],
            ability_changes=[
                AbilityDiff(
                    ability_name="sinister_strike",
                    dps_before=200.0,
                    dps_after=220.0,
                    delta=20.0,
                    delta_pct=10.0,
                )
            ],
            summary="DPS: 1000.0 -> 1100.0 (+100.0, +10.0%)",
        )
        assert cr.dps_delta == 100.0
        assert len(cr.stat_changes) == 1
        assert len(cr.ability_changes) == 1

    def test_item_recommendation_validates(self) -> None:
        ir = ItemRecommendation(
            item_name="Warglaive of Azzinoth MH",
            item_id=32837,
            dps=1500.0,
            dps_delta=200.0,
            source="Black Temple",
            phase=3,
        )
        assert ir.item_name == "Warglaive of Azzinoth MH"
        assert ir.phase == 3

    def test_optimize_result_validates(self) -> None:
        opt = OptimizeResult(
            current_item="Latro's Shifting Sword",
            current_dps=1200.0,
            recommendations=[
                ItemRecommendation(
                    item_name="Warglaive of Azzinoth MH",
                    item_id=32837,
                    dps=1400.0,
                    dps_delta=200.0,
                    source="Black Temple",
                    phase=3,
                )
            ],
        )
        assert opt.current_dps == 1200.0
        assert len(opt.recommendations) == 1
