"""Tests for sim validation comparison engine."""

import pytest

from code.shukketsu.sim.comparator import (
    AbilityMetrics,
    MetricDrift,
    ValidationReport,
    WCLFightMetrics,
    compute_drift,
)


class TestMetricDrift:
    def test_pass_within_threshold(self) -> None:
        d = MetricDrift(
            metric_name="dps",
            sim_value=1050,
            wcl_value=1000,
            absolute_delta=50,
            relative_pct=5.0,
            status="pass",
        )
        assert d.status == "pass"

    def test_fail_above_threshold(self) -> None:
        d = MetricDrift(
            metric_name="dps",
            sim_value=1200,
            wcl_value=1000,
            absolute_delta=200,
            relative_pct=20.0,
            status="fail",
        )
        assert d.status == "fail"


class TestWCLFightMetrics:
    def test_creation(self) -> None:
        m = WCLFightMetrics(
            total_damage=300000,
            active_dps=1000.0,
            fight_duration_ms=300000,
            ability_breakdown={
                "sinister_strike": AbilityMetrics(
                    damage_total=150000,
                    damage_pct=50.0,
                    cast_count=120,
                )
            },
            buff_uptimes={"slice_and_dice": 0.95},
            proc_counts={"combat_potency": 45},
        )
        assert m.active_dps == 1000.0
        assert "sinister_strike" in m.ability_breakdown


class TestComputeDrift:
    """Tests for the compute_drift static function."""

    def test_pass(self) -> None:
        drift = compute_drift("dps", 1020.0, 1000.0, threshold_pass=5.0, threshold_warn=10.0)
        assert drift.status == "pass"
        assert abs(drift.relative_pct - 2.0) < 0.1

    def test_warn(self) -> None:
        drift = compute_drift("dps", 1080.0, 1000.0, threshold_pass=5.0, threshold_warn=10.0)
        assert drift.status == "warn"

    def test_fail(self) -> None:
        drift = compute_drift("dps", 1200.0, 1000.0, threshold_pass=5.0, threshold_warn=10.0)
        assert drift.status == "fail"

    def test_zero_wcl_value(self) -> None:
        """Zero WCL value should not divide by zero."""
        drift = compute_drift("dps", 100.0, 0.0, threshold_pass=5.0, threshold_warn=10.0)
        assert drift.status == "fail"

    def test_negative_drift(self) -> None:
        """Sim below WCL is still measured by absolute relative pct."""
        drift = compute_drift("dps", 800.0, 1000.0, threshold_pass=5.0, threshold_warn=10.0)
        assert drift.status == "fail"
        assert drift.relative_pct == pytest.approx(-20.0)


class TestValidationReport:
    def test_report_aggregation(self) -> None:
        report = ValidationReport(
            character_name="Lyroo",
            total_fights=10,
            included_fights=8,
            excluded_fights=2,
            per_fight=[],
            per_boss={},
            overall_dps_drift_pct=3.5,
            overall_status="pass",
            timestamp="2026-02-13T00:00:00",
        )
        assert report.overall_status == "pass"
