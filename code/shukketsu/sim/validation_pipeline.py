"""Validation pipeline orchestrator.

Orchestrates: FightFilter -> WCLBridge -> SimRunner -> SimComparator -> store report.
Uses asyncio.to_thread for CPU-bound sim parallelism.
"""

import asyncio
import logging
import sqlite3
from collections.abc import Callable

from code.shukketsu.sim.comparator import SimComparator, ValidationReport
from code.shukketsu.sim.fight_filter import FightFilter, ValidatedFight
from code.shukketsu.sim.models import SimConfig, SimResult
from code.shukketsu.sim.wcl_bridge import WCLBridge

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """Orchestrates WCL validation: filter -> bridge -> sim -> compare -> store."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    async def run_validation(
        self,
        character_name: str,
        *,
        race: str = "human",
        on_progress: Callable[[str], None] | None = None,
    ) -> ValidationReport:
        """Run full validation for a character.

        Args:
            character_name: WCL player name to validate.
            race: Player race (lowercase).
            on_progress: Optional callback for progress messages.

        Returns:
            ValidationReport with per-fight and per-boss metrics.
        """

        def _progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)
            logger.info(msg)

        _progress(f"Filtering fights for {character_name}...")
        fight_filter = FightFilter()
        all_fights = fight_filter.filter_fights(self._conn, character_name)
        included = [f for f in all_fights if f.included]
        excluded_count = len(all_fights) - len(included)

        if not included:
            _progress("No valid fights found.")
            comparator = SimComparator(self._conn)
            report = comparator.build_report([], character_name, len(all_fights), excluded_count)
            self._store_report(report)
            return report

        _progress(f"Building configs for {len(included)} fights...")
        bridge = WCLBridge(self._conn)
        configs: list[tuple[ValidatedFight, SimConfig]] = []
        for fight in included:
            try:
                sim_config = bridge.build_config(
                    fight.report_code,
                    fight.fight_id,
                    fight.source_id,
                    race=race,
                )
                configs.append((fight, sim_config))
            except Exception:
                logger.warning(
                    "Failed to build config for fight %d in %s",
                    fight.fight_id,
                    fight.report_code,
                    exc_info=True,
                )

        if not configs:
            _progress("No configs could be built.")
            comparator = SimComparator(self._conn)
            report = comparator.build_report([], character_name, len(all_fights), excluded_count)
            self._store_report(report)
            return report

        _progress(f"Running {len(configs)} simulations...")
        sim_results = await self._run_sims_threaded(configs, bridge)

        _progress("Comparing results...")
        comparator = SimComparator(self._conn)
        validations = []
        for (fight, _), sim_result in zip(configs, sim_results):
            wcl_metrics = comparator.extract_wcl_metrics(
                fight.report_code,
                fight.fight_id,
                character_name,
                fight.duration_ms,
            )
            validation = comparator.compare_fight(
                sim_result,
                wcl_metrics,
                fight.report_code,
                fight.fight_id,
                fight.encounter_name,
            )
            validations.append(validation)

        report = comparator.build_report(validations, character_name, len(all_fights), excluded_count)
        self._store_report(report)
        _progress(f"Validation complete: {report.overall_status} ({report.overall_dps_drift_pct:.1f}% drift)")
        return report

    async def _run_sims_threaded(
        self,
        configs: list[tuple[ValidatedFight, SimConfig]],
        bridge: WCLBridge,
    ) -> list[SimResult]:
        """Run sims using asyncio.to_thread (GIL-limited but safe with synthetic items)."""
        from code.shukketsu.sim.buffs import resolve_buffs
        from code.shukketsu.sim.combat import CombatSimulation
        from code.shukketsu.sim.rotation import RotationEngine
        from code.shukketsu.sim.talents import compute_modifiers, parse_talents

        async def run_one(sim_config: SimConfig) -> SimResult:
            talent_tree = parse_talents(sim_config.talents, sim_config.spec)
            modifiers = compute_modifiers(talent_tree)
            buffs = resolve_buffs(sim_config.buffs, sim_config.boss.debuffs, sim_config.consumables)
            rotation = RotationEngine(sim_config.spec, modifiers)
            sim = CombatSimulation(sim_config, modifiers, buffs, bridge.item_db, rotation)
            return await asyncio.to_thread(sim.run, sim_config.iterations)

        tasks = [run_one(cfg) for _, cfg in configs]
        return await asyncio.gather(*tasks)

    def _store_report(self, report: ValidationReport) -> None:
        """Store report in validation_runs table."""
        self._conn.execute(
            """INSERT INTO validation_runs
               (character_name, run_type, total_fights, included_fights,
                overall_dps_drift_pct, overall_status, report_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                report.character_name,
                "wcl",
                report.total_fights,
                report.included_fights,
                report.overall_dps_drift_pct,
                report.overall_status,
                report.model_dump_json(),
            ),
        )
        self._conn.commit()
