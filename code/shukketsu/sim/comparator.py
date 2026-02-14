"""Comparison engine for sim validation.

Measures drift between SimResult and WCL fight data across DPS,
ability breakdown, buff uptimes, and proc rates.
"""

import json
import logging
import sqlite3
from datetime import UTC, datetime

from pydantic import BaseModel

from code.shukketsu import config
from code.shukketsu.sim.models import SimResult
from code.shukketsu.sim.spell_map import aggregate_wcl_abilities, wcl_buff_name

logger = logging.getLogger(__name__)


class AbilityMetrics(BaseModel):
    """Per-ability metrics from WCL."""

    damage_total: int
    damage_pct: float
    cast_count: int = 0


class WCLFightMetrics(BaseModel):
    """Extracted WCL fight metrics for comparison."""

    total_damage: int
    active_dps: float
    fight_duration_ms: int
    ability_breakdown: dict[str, AbilityMetrics]
    buff_uptimes: dict[str, float]
    proc_counts: dict[str, int]


class MetricDrift(BaseModel):
    """Drift measurement for a single metric."""

    metric_name: str
    sim_value: float
    wcl_value: float
    absolute_delta: float
    relative_pct: float
    status: str


class FightValidation(BaseModel):
    """Validation result for a single fight."""

    report_code: str
    fight_id: int
    encounter_name: str
    fight_duration_ms: int
    dps_drift: MetricDrift
    ability_drifts: list[MetricDrift]
    buff_drifts: list[MetricDrift]
    overall_status: str


class BossAggregate(BaseModel):
    """Aggregated validation across fights for one boss."""

    encounter_name: str
    fight_count: int
    avg_sim_dps: float
    avg_wcl_dps: float
    avg_drift_pct: float
    status: str


class ValidationReport(BaseModel):
    """Full validation report."""

    character_name: str
    total_fights: int
    included_fights: int
    excluded_fights: int
    per_fight: list[FightValidation]
    per_boss: dict[str, BossAggregate]
    overall_dps_drift_pct: float
    overall_status: str
    timestamp: str


def compute_drift(
    metric_name: str,
    sim_value: float,
    wcl_value: float,
    *,
    threshold_pass: float,
    threshold_warn: float,
) -> MetricDrift:
    """Compute drift between sim and WCL values.

    Args:
        metric_name: Human-readable name for the metric.
        sim_value: Value from simulation.
        wcl_value: Value from WCL data.
        threshold_pass: Absolute relative % below which status is "pass".
        threshold_warn: Absolute relative % below which status is "warn".

    Returns:
        MetricDrift with computed status.
    """
    absolute_delta = sim_value - wcl_value
    if wcl_value != 0:
        relative_pct = (sim_value - wcl_value) / wcl_value * 100
    else:
        relative_pct = 100.0 if sim_value > 0 else 0.0

    abs_pct = abs(relative_pct)
    if abs_pct <= threshold_pass:
        status = "pass"
    elif abs_pct <= threshold_warn:
        status = "warn"
    else:
        status = "fail"

    return MetricDrift(
        metric_name=metric_name,
        sim_value=sim_value,
        wcl_value=wcl_value,
        absolute_delta=absolute_delta,
        relative_pct=relative_pct,
        status=status,
    )


class SimComparator:
    """Compares SimResult against WCL fight data."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def extract_wcl_metrics(
        self,
        report_code: str,
        fight_id: int,
        player_name: str,
        duration_ms: int,
    ) -> WCLFightMetrics:
        """Extract WCL metrics from database for one fight.

        Args:
            report_code: WCL report code.
            fight_id: Fight ID within the report.
            player_name: Player name to filter damage data.
            duration_ms: Fight duration in milliseconds.

        Returns:
            WCLFightMetrics extracted from the database.
        """
        dmg_row = self._conn.execute(
            "SELECT total_damage, active_time_ms, abilities_json "
            "FROM wcl_damage WHERE report_code = ? AND fight_id = ? AND player_name = ?",
            (report_code, fight_id, player_name),
        ).fetchone()

        total_damage = dmg_row["total_damage"] if dmg_row else 0
        active_time_ms = (dmg_row["active_time_ms"] or 0) if dmg_row else 0
        active_dps = total_damage / (active_time_ms / 1000) if active_time_ms > 0 else 0.0

        abilities_raw = json.loads(dmg_row["abilities_json"]) if dmg_row else []
        ability_breakdown: dict[str, AbilityMetrics] = {}
        for name, total in aggregate_wcl_abilities(abilities_raw).items():
            ability_breakdown[name] = AbilityMetrics(
                damage_total=total,
                damage_pct=(total / total_damage * 100) if total_damage > 0 else 0.0,
            )

        buff_rows = self._conn.execute(
            "SELECT buff_guid, total_uptime_ms FROM wcl_buffs WHERE report_code = ? AND fight_id = ?",
            (report_code, fight_id),
        ).fetchall()
        buff_uptimes: dict[str, float] = {}
        for br in buff_rows:
            buff = wcl_buff_name(br["buff_guid"])
            if buff is not None:
                buff_uptimes[buff] = br["total_uptime_ms"] / duration_ms if duration_ms > 0 else 0.0

        return WCLFightMetrics(
            total_damage=total_damage,
            active_dps=active_dps,
            fight_duration_ms=duration_ms,
            ability_breakdown=ability_breakdown,
            buff_uptimes=buff_uptimes,
            proc_counts={},
        )

    def compare_fight(
        self,
        sim_result: SimResult,
        wcl_metrics: WCLFightMetrics,
        report_code: str,
        fight_id: int,
        encounter_name: str,
    ) -> FightValidation:
        """Compare a single sim result against WCL metrics.

        Args:
            sim_result: Output from the simulation engine.
            wcl_metrics: Extracted WCL fight metrics.
            report_code: WCL report code.
            fight_id: Fight ID within the report.
            encounter_name: Boss/encounter name.

        Returns:
            FightValidation with per-metric drift analysis.
        """
        dps_drift = compute_drift(
            "dps",
            sim_result.dps_mean,
            wcl_metrics.active_dps,
            threshold_pass=config.VALIDATION_DPS_THRESHOLD,
            threshold_warn=config.VALIDATION_DPS_WARN_THRESHOLD,
        )

        ability_drifts: list[MetricDrift] = []
        for name, wcl_ab in wcl_metrics.ability_breakdown.items():
            sim_ab = next((a for a in sim_result.ability_breakdown if a.name == name), None)
            sim_pct = sim_ab.damage_pct if sim_ab else 0.0
            ability_drifts.append(
                compute_drift(
                    name,
                    sim_pct,
                    wcl_ab.damage_pct,
                    threshold_pass=config.VALIDATION_ABILITY_THRESHOLD,
                    threshold_warn=config.VALIDATION_ABILITY_THRESHOLD * 2,
                )
            )

        buff_drifts: list[MetricDrift] = []
        for buff_name, wcl_uptime in wcl_metrics.buff_uptimes.items():
            sim_uptime = sim_result.buff_uptimes.get(buff_name, 0.0)
            buff_drifts.append(
                compute_drift(
                    buff_name,
                    sim_uptime * 100,
                    wcl_uptime * 100,
                    threshold_pass=config.VALIDATION_BUFF_THRESHOLD,
                    threshold_warn=config.VALIDATION_BUFF_THRESHOLD * 2,
                )
            )

        statuses = [dps_drift.status] + [d.status for d in ability_drifts] + [d.status for d in buff_drifts]
        if "fail" in statuses:
            overall = "fail"
        elif "warn" in statuses:
            overall = "warn"
        else:
            overall = "pass"

        return FightValidation(
            report_code=report_code,
            fight_id=fight_id,
            encounter_name=encounter_name,
            fight_duration_ms=wcl_metrics.fight_duration_ms,
            dps_drift=dps_drift,
            ability_drifts=ability_drifts,
            buff_drifts=buff_drifts,
            overall_status=overall,
        )

    def build_report(
        self,
        validations: list[FightValidation],
        character_name: str,
        total_fights: int,
        excluded_fights: int,
    ) -> ValidationReport:
        """Build aggregated validation report.

        Args:
            validations: List of per-fight validation results.
            character_name: Name of the character being validated.
            total_fights: Total fights considered (included + excluded).
            excluded_fights: Number of fights excluded from validation.

        Returns:
            ValidationReport with per-boss aggregation and overall status.
        """
        per_boss: dict[str, list[FightValidation]] = {}
        for v in validations:
            per_boss.setdefault(v.encounter_name, []).append(v)

        boss_aggs: dict[str, BossAggregate] = {}
        for boss_name, fights in per_boss.items():
            avg_sim = sum(f.dps_drift.sim_value for f in fights) / len(fights)
            avg_wcl = sum(f.dps_drift.wcl_value for f in fights) / len(fights)
            avg_drift = sum(f.dps_drift.relative_pct for f in fights) / len(fights)
            abs_drift = abs(avg_drift)
            if abs_drift <= config.VALIDATION_DPS_THRESHOLD:
                status = "pass"
            elif abs_drift <= config.VALIDATION_DPS_WARN_THRESHOLD:
                status = "warn"
            else:
                status = "fail"
            boss_aggs[boss_name] = BossAggregate(
                encounter_name=boss_name,
                fight_count=len(fights),
                avg_sim_dps=avg_sim,
                avg_wcl_dps=avg_wcl,
                avg_drift_pct=avg_drift,
                status=status,
            )

        all_drifts = [v.dps_drift.relative_pct for v in validations]
        overall_drift = sum(all_drifts) / len(all_drifts) if all_drifts else 0.0
        abs_overall = abs(overall_drift)
        if abs_overall <= config.VALIDATION_DPS_THRESHOLD:
            overall_status = "pass"
        elif abs_overall <= config.VALIDATION_DPS_WARN_THRESHOLD:
            overall_status = "warn"
        else:
            overall_status = "fail"

        return ValidationReport(
            character_name=character_name,
            total_fights=total_fights,
            included_fights=len(validations),
            excluded_fights=excluded_fights,
            per_fight=validations,
            per_boss=boss_aggs,
            overall_dps_drift_pct=overall_drift,
            overall_status=overall_status,
            timestamp=datetime.now(UTC).isoformat(),
        )
