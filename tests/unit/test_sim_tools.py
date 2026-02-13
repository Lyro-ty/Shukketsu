"""Tests for simulation analysis tools."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from code.shukketsu.tools.analysis.sim_compare import SimCompareTool
from code.shukketsu.tools.analysis.sim_optimize import SimOptimizeTool
from code.shukketsu.tools.analysis.sim_run import SimRunTool


def _make_mock_sim_result(**overrides: Any) -> MagicMock:
    """Create a mock SimResult with sensible defaults."""
    result = MagicMock()
    result.dps_mean = overrides.get("dps_mean", 1500.0)
    result.dps_median = overrides.get("dps_median", 1480.0)
    result.dps_std = overrides.get("dps_std", 120.0)
    result.dps_min = overrides.get("dps_min", 1100.0)
    result.dps_max = overrides.get("dps_max", 1900.0)
    result.iterations = overrides.get("iterations", 10000)
    result.fight_length = overrides.get("fight_length", 300)

    ab1 = MagicMock()
    ab1.name = "Sinister Strike"
    ab1.damage_pct = 35.0
    ab1.casts = 120.0
    ab1.crit_pct = 28.0

    ab2 = MagicMock()
    ab2.name = "Eviscerate"
    ab2.damage_pct = 20.0
    ab2.casts = 25.0
    ab2.crit_pct = 32.0

    result.ability_breakdown = [ab1, ab2]
    return result


def _make_mock_runner() -> MagicMock:
    """Create a mock SimRunner."""
    runner = MagicMock()
    runner.build_config_from_import = MagicMock(return_value=MagicMock())
    runner.sim_run = AsyncMock(return_value=_make_mock_sim_result())
    runner.stat_weights = AsyncMock(return_value=[])
    runner.sim_compare = AsyncMock()
    runner.sim_optimize = AsyncMock()
    runner.swap_item = MagicMock()
    return runner


class TestSimRunTool:
    def test_name_and_description(self) -> None:
        tool = SimRunTool()
        assert tool.name == "sim_run"
        assert "simulation" in tool.description.lower() or "sim" in tool.description.lower()

    def test_parameters_schema_has_import_string(self) -> None:
        tool = SimRunTool()
        assert "import_string" in tool.parameters_schema

    def test_parameters_schema_has_compute_stat_weights(self) -> None:
        tool = SimRunTool()
        assert "compute_stat_weights" in tool.parameters_schema

    @pytest.mark.asyncio
    async def test_execute_returns_observation_string(self) -> None:
        """Mock SimRunner and verify formatted output."""
        runner = _make_mock_runner()
        tool = SimRunTool(runner=runner)

        result = await tool.execute({"import_string": "rogue=Test"})

        assert "Mean DPS: 1500.0" in result
        assert "Sinister Strike" in result
        runner.build_config_from_import.assert_called_once()
        runner.sim_run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_missing_import_string(self) -> None:
        """Tool should return error for missing import_string."""
        tool = SimRunTool(runner=_make_mock_runner())
        result = await tool.execute({})
        assert "Error" in result
        assert "import_string" in result

    @pytest.mark.asyncio
    async def test_execute_with_stat_weights(self) -> None:
        """When compute_stat_weights is True, stat weights appear in output."""
        runner = _make_mock_runner()
        weight = MagicMock()
        weight.stat = "hit_rating"
        weight.ep_value = 2.15
        weight.dps_per_point = 0.43
        weight.is_capped = False
        runner.stat_weights = AsyncMock(return_value=[weight])

        tool = SimRunTool(runner=runner)
        result = await tool.execute({"import_string": "rogue=Test", "compute_stat_weights": True})

        assert "Stat Weights" in result
        assert "hit_rating" in result
        runner.stat_weights.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_passes_overrides(self) -> None:
        """Overrides like fight_length should be passed to build_config_from_import."""
        runner = _make_mock_runner()
        tool = SimRunTool(runner=runner)

        await tool.execute({"import_string": "rogue=Test", "fight_length": 180, "iterations": 5000})

        call_kwargs = runner.build_config_from_import.call_args
        assert call_kwargs[1]["fight_length"] == 180
        assert call_kwargs[1]["iterations"] == 5000


class TestSimCompareTool:
    def test_name_and_description(self) -> None:
        tool = SimCompareTool()
        assert tool.name == "sim_compare"
        assert "compare" in tool.description.lower()

    def test_parameters_schema_has_swap_slot(self) -> None:
        tool = SimCompareTool()
        assert "swap_slot" in tool.parameters_schema

    def test_parameters_schema_has_swap_item(self) -> None:
        tool = SimCompareTool()
        assert "swap_item" in tool.parameters_schema

    @pytest.mark.asyncio
    async def test_execute_returns_comparison_string(self) -> None:
        """Mock SimRunner.sim_compare and verify formatted output."""
        runner = _make_mock_runner()

        compare_result = MagicMock()
        compare_result.dps_before = 1500.0
        compare_result.dps_after = 1550.0
        compare_result.dps_delta = 50.0
        compare_result.dps_delta_pct = 3.3
        compare_result.summary = "DPS: 1500.0 -> 1550.0 (+50.0, +3.3%)"
        compare_result.stat_changes = []
        compare_result.ability_changes = []
        runner.sim_compare = AsyncMock(return_value=compare_result)

        tool = SimCompareTool(runner=runner)
        result = await tool.execute(
            {
                "import_string": "rogue=Test",
                "swap_slot": "trinket_1",
                "swap_item": "Dragonspine Trophy",
            }
        )

        assert "1500.0" in result
        assert "1550.0" in result
        assert "+50.0" in result

    @pytest.mark.asyncio
    async def test_execute_missing_required_params(self) -> None:
        """Tool should return error for missing required params."""
        tool = SimCompareTool(runner=_make_mock_runner())

        result = await tool.execute({"import_string": "test"})
        assert "Error" in result
        assert "swap_slot" in result

    @pytest.mark.asyncio
    async def test_execute_invalid_slot(self) -> None:
        """Tool should return error for invalid gear slot."""
        tool = SimCompareTool(runner=_make_mock_runner())
        result = await tool.execute(
            {
                "import_string": "rogue=Test",
                "swap_slot": "invalid_slot",
                "swap_item": "Test Item",
            }
        )
        assert "Error" in result
        assert "Invalid gear slot" in result


class TestSimOptimizeTool:
    def test_name_and_description(self) -> None:
        tool = SimOptimizeTool()
        assert tool.name == "sim_optimize"
        assert "best" in tool.description.lower() or "optimize" in tool.description.lower()

    def test_parameters_schema_has_slot(self) -> None:
        tool = SimOptimizeTool()
        assert "slot" in tool.parameters_schema

    def test_parameters_schema_has_phase(self) -> None:
        tool = SimOptimizeTool()
        assert "phase" in tool.parameters_schema

    @pytest.mark.asyncio
    async def test_execute_returns_ranked_results(self) -> None:
        """Mock SimRunner.sim_optimize and verify formatted output."""
        runner = _make_mock_runner()

        rec = MagicMock()
        rec.item_name = "Dragonspine Trophy"
        rec.item_id = 28830
        rec.dps = 1550.0
        rec.dps_delta = 50.0
        rec.source = "Gruul's Lair"
        rec.phase = 1

        optimize_result = MagicMock()
        optimize_result.current_item = "Abacus of Violent Odds"
        optimize_result.current_dps = 1500.0
        optimize_result.recommendations = [rec]
        runner.sim_optimize = AsyncMock(return_value=optimize_result)

        tool = SimOptimizeTool(runner=runner)
        result = await tool.execute(
            {
                "import_string": "rogue=Test",
                "slot": "trinket_1",
            }
        )

        assert "Dragonspine Trophy" in result
        assert "1550.0" in result
        assert "Abacus of Violent Odds" in result

    @pytest.mark.asyncio
    async def test_execute_missing_slot(self) -> None:
        """Tool should return error for missing slot."""
        tool = SimOptimizeTool(runner=_make_mock_runner())
        result = await tool.execute({"import_string": "rogue=Test"})
        assert "Error" in result
        assert "slot" in result
