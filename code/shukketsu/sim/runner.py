"""Public API runner for the TBC Rogue DPS simulation engine.

Ties together all sim components (items, talents, buffs, rotation, combat)
into a single high-level interface for running simulations, comparing configs,
computing stat weights, and optimizing gear.
"""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from typing import Any

from pydantic import BaseModel

from code.shukketsu.config import SIM_STAT_WEIGHT_DELTA
from code.shukketsu.resilience.errors import ItemNotFoundError
from code.shukketsu.sim.buffs import get_preset, resolve_buffs
from code.shukketsu.sim.combat import CombatSimulation
from code.shukketsu.sim.imports import build_config, parse_import
from code.shukketsu.sim.items import ItemDatabase
from code.shukketsu.sim.models import (
    GearSlot,
    Item,
    SimConfig,
    SimResult,
    StatWeight,
)
from code.shukketsu.sim.rotation import RotationEngine
from code.shukketsu.sim.talents import compute_modifiers, parse_talents

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stats to run delta-sim on for stat weights. Order determines output order.
_STAT_WEIGHT_STATS: list[str] = [
    "hit_rating",
    "expertise_rating",
    "haste_rating",
    "crit_rating",
    "agility",
    "strength",
    "attack_power",
    "armor_penetration",
]

# Soft caps for TBC Rogue stats (approximate).
_STAT_CAPS: dict[str, float] = {
    "hit_rating": 142.0,  # 9% hit for dual-wield specials
    "expertise_rating": 99.0,  # 6.5% dodge reduction (25 expertise = 99 rating)
}

# Rough EP weights for pre-filtering gear candidates.
_APPROX_EP: dict[str, float] = {
    "agility": 2.0,
    "strength": 1.1,
    "attack_power": 1.0,
    "crit_rating": 1.8,
    "hit_rating": 2.2,
    "haste_rating": 1.6,
    "expertise_rating": 2.0,
    "armor_penetration": 1.2,
    "stamina": 0.0,
}


# ---------------------------------------------------------------------------
# Additional result models
# ---------------------------------------------------------------------------


class StatDiff(BaseModel):
    """Difference in a single stat between two configs."""

    stat_name: str
    before: float
    after: float
    delta: float


class AbilityDiff(BaseModel):
    """Difference in per-ability DPS between two sim results."""

    ability_name: str
    dps_before: float
    dps_after: float
    delta: float
    delta_pct: float


class CompareResult(BaseModel):
    """Result of comparing two simulation configs."""

    dps_before: float
    dps_after: float
    dps_delta: float
    dps_delta_pct: float
    stat_changes: list[StatDiff]
    ability_changes: list[AbilityDiff]
    summary: str


class ItemRecommendation(BaseModel):
    """A single item recommendation from slot optimization."""

    item_name: str
    item_id: int
    dps: float
    dps_delta: float
    source: str
    phase: int


class OptimizeResult(BaseModel):
    """Result of optimizing a single gear slot."""

    current_item: str
    current_dps: float
    recommendations: list[ItemRecommendation]


# ---------------------------------------------------------------------------
# SimRunner
# ---------------------------------------------------------------------------


class SimRunner:
    """High-level simulation runner that ties all sim components together.

    Provides methods for running sims, comparing configs, computing stat
    weights, and optimizing gear slots.

    Args:
        item_db: Item database instance. Uses default if not provided.
    """

    def __init__(self, item_db: ItemDatabase | None = None) -> None:
        self._item_db = item_db or ItemDatabase()

    # -- Public async API ---------------------------------------------------

    async def sim_run(self, config: SimConfig, *, item_db: ItemDatabase | None = None) -> SimResult:
        """Run a simulation and return the result.

        CPU-bound work is offloaded to a thread via asyncio.to_thread.

        Args:
            config: Full simulation configuration.
            item_db: Optional override item database (used for stat weight patching).

        Returns:
            Aggregated SimResult with DPS statistics and breakdowns.
        """
        sim = self._build_simulation(config, item_db=item_db)
        return await asyncio.to_thread(sim.run, config.iterations)

    async def sim_compare(self, config_a: SimConfig, config_b: SimConfig) -> CompareResult:
        """Compare two configs by running both in parallel.

        Args:
            config_a: The "before" configuration.
            config_b: The "after" configuration.

        Returns:
            CompareResult with DPS delta, stat changes, and ability changes.
        """
        result_a, result_b = await asyncio.gather(
            self.sim_run(config_a),
            self.sim_run(config_b),
        )

        dps_delta = result_b.dps_mean - result_a.dps_mean
        dps_delta_pct = (dps_delta / result_a.dps_mean * 100) if result_a.dps_mean > 0 else 0.0

        # Compute stat changes from gear
        stats_a = self._sum_gear_stats(config_a)
        stats_b = self._sum_gear_stats(config_b)
        all_stat_keys = sorted(set(stats_a.keys()) | set(stats_b.keys()))
        stat_changes: list[StatDiff] = []
        for key in all_stat_keys:
            val_a = stats_a.get(key, 0.0)
            val_b = stats_b.get(key, 0.0)
            if val_a != val_b:
                stat_changes.append(StatDiff(stat_name=key, before=val_a, after=val_b, delta=val_b - val_a))

        # Compute ability DPS changes
        breakdown_a = {ab.name: ab for ab in result_a.ability_breakdown}
        breakdown_b = {ab.name: ab for ab in result_b.ability_breakdown}
        all_abilities = sorted(set(breakdown_a.keys()) | set(breakdown_b.keys()))
        ability_changes: list[AbilityDiff] = []
        fight_length_a = config_a.fight_length or 1
        fight_length_b = config_b.fight_length or 1
        for name in all_abilities:
            dps_a = breakdown_a[name].damage_total / fight_length_a if name in breakdown_a else 0.0
            dps_b = breakdown_b[name].damage_total / fight_length_b if name in breakdown_b else 0.0
            ab_delta = dps_b - dps_a
            ab_delta_pct = (ab_delta / dps_a * 100) if dps_a > 0 else 0.0
            if abs(ab_delta) > 0.01:
                ability_changes.append(
                    AbilityDiff(
                        ability_name=name,
                        dps_before=dps_a,
                        dps_after=dps_b,
                        delta=ab_delta,
                        delta_pct=ab_delta_pct,
                    )
                )

        sign = "+" if dps_delta >= 0 else ""
        summary = (
            f"DPS: {result_a.dps_mean:.1f} -> {result_b.dps_mean:.1f} "
            f"({sign}{dps_delta:.1f}, {sign}{dps_delta_pct:.1f}%)"
        )

        return CompareResult(
            dps_before=result_a.dps_mean,
            dps_after=result_b.dps_mean,
            dps_delta=dps_delta,
            dps_delta_pct=dps_delta_pct,
            stat_changes=stat_changes,
            ability_changes=ability_changes,
            summary=summary,
        )

    async def sim_optimize(
        self,
        config: SimConfig,
        slot: GearSlot,
        *,
        top_n: int = 5,
        phase: int = 5,
    ) -> OptimizeResult:
        """Find the best items for a gear slot via simulation.

        Pre-filters candidates using EP heuristic, then sims each one.

        Args:
            config: Base simulation configuration.
            slot: Gear slot to optimize.
            top_n: Number of top recommendations to return.
            phase: Maximum content phase for item filtering.

        Returns:
            OptimizeResult with current item info and ranked recommendations.
        """
        # Get current item info
        current_id = config.gear.get(slot)
        current_item_name = "Unknown"
        if current_id is not None:
            item = self._item_db.get_item(current_id)
            if item is not None:
                current_item_name = item.name

        # Run base sim
        base_result = await self.sim_run(config)

        # Pre-filter candidates
        candidates = self._pre_filter_candidates(config, slot, phase)

        # Sim each candidate
        recommendations: list[ItemRecommendation] = []
        for candidate in candidates:
            if candidate.id == current_id:
                continue
            try:
                new_gear = dict(config.gear)
                new_gear[slot] = candidate.id
                new_config = config.model_copy(update={"gear": new_gear})
                result = await self.sim_run(new_config)
                recommendations.append(
                    ItemRecommendation(
                        item_name=candidate.name,
                        item_id=candidate.id,
                        dps=result.dps_mean,
                        dps_delta=result.dps_mean - base_result.dps_mean,
                        source="",
                        phase=candidate.phase,
                    )
                )
            except (ValueError, KeyError) as exc:
                logger.debug("Skipping candidate %s: %s", candidate.name, exc)

        recommendations.sort(key=lambda r: r.dps, reverse=True)
        return OptimizeResult(
            current_item=current_item_name,
            current_dps=base_result.dps_mean,
            recommendations=recommendations[:top_n],
        )

    async def stat_weights(
        self,
        config: SimConfig,
        *,
        delta: int | None = None,
    ) -> list[StatWeight]:
        """Compute stat equivalence points via delta-simulation.

        Runs the base config, then re-runs with +delta of each stat.
        Normalizes all values relative to AP = 1.0 EP.

        Args:
            config: Base simulation configuration.
            delta: Stat delta amount. Defaults to SIM_STAT_WEIGHT_DELTA.

        Returns:
            List of StatWeight objects with EP values and cap flags.
        """
        stat_delta = delta if delta is not None else SIM_STAT_WEIGHT_DELTA

        # Use reduced iterations for faster stat weight calculation
        reduced_iters = max(500, config.iterations // 10)
        base_config = config.model_copy(update={"iterations": reduced_iters})

        base_result = await self.sim_run(base_config)
        base_dps = base_result.dps_mean

        # Run delta sims for each stat
        weights: list[StatWeight] = []
        for stat_name in _STAT_WEIGHT_STATS:
            patched_db = self._patched_item_db(base_config, stat_name, stat_delta)
            delta_result = await self.sim_run(base_config, item_db=patched_db)

            dps_per_point = (delta_result.dps_mean - base_dps) / stat_delta if stat_delta > 0 else 0.0

            is_capped = False
            if stat_name in _STAT_CAPS:
                current_value = self._get_stat_total(config, stat_name)
                if current_value >= _STAT_CAPS[stat_name]:
                    is_capped = True

            weights.append(
                StatWeight(
                    stat=stat_name,
                    ep_value=0.0,
                    dps_per_point=dps_per_point,
                    is_capped=is_capped,
                )
            )

        # Normalize to AP = 1.0 EP
        ap_dpp = next((w.dps_per_point for w in weights if w.stat == "attack_power"), 0.0)
        if ap_dpp > 0:
            for w in weights:
                w.ep_value = w.dps_per_point / ap_dpp
        else:
            for w in weights:
                w.ep_value = w.dps_per_point

        return weights

    # -- Public sync API ----------------------------------------------------

    def build_config_from_import(self, raw: str, **overrides: Any) -> SimConfig:
        """Parse a character import string and build a SimConfig.

        Args:
            raw: Raw import string (SimC, SeventyUpgrades JSON, or WoWSims Base64).
            **overrides: Additional SimConfig field overrides.

        Returns:
            Complete SimConfig ready for simulation.
        """
        char_import = parse_import(raw)
        return build_config(char_import, **overrides)

    def swap_item(self, config: SimConfig, slot: GearSlot, item_query: str) -> SimConfig:
        """Search for an item and return a new config with it equipped.

        Args:
            config: Base simulation configuration.
            slot: Gear slot to swap.
            item_query: Case-insensitive item name search string.

        Returns:
            New SimConfig with the item swapped into the specified slot.

        Raises:
            ItemNotFoundError: If no item matches the query.
        """
        matches = self._item_db.search(item_query, slot=slot)
        if not matches:
            matches = self._item_db.search(item_query)
        if not matches:
            raise ItemNotFoundError(f"No item found matching '{item_query}'")

        item = matches[0]
        new_gear = dict(config.gear)
        new_gear[slot] = item.id
        return config.model_copy(update={"gear": new_gear})

    # -- Internal helpers ---------------------------------------------------

    def _build_simulation(self, config: SimConfig, *, item_db: ItemDatabase | None = None) -> CombatSimulation:
        """Create a fully initialized CombatSimulation from a SimConfig.

        Args:
            config: Full simulation configuration.
            item_db: Optional override item database.

        Returns:
            Ready-to-run CombatSimulation instance.
        """
        db = item_db or self._item_db

        allocation = parse_talents(config.talents, config.spec)
        modifiers = compute_modifiers(allocation)

        buff_ids, debuff_ids, consumable_ids = get_preset(config.raid_preset)
        all_buffs = list(set(buff_ids + config.buffs))
        all_consumables = list(set(consumable_ids + config.consumables))
        all_debuffs = list(set(debuff_ids + config.boss.debuffs))
        resolved_buffs = resolve_buffs(all_buffs, all_debuffs, all_consumables)

        rotation = RotationEngine(config.spec, modifiers, expose_armor=config.expose_armor)

        return CombatSimulation(config, modifiers, resolved_buffs, db, rotation)

    def _pre_filter_candidates(self, config: SimConfig, slot: GearSlot, phase: int) -> list[Item]:
        """Get top ~20 items for a slot sorted by approximate EP score.

        Args:
            config: Base config (for context).
            slot: Gear slot to find items for.
            phase: Maximum content phase.

        Returns:
            List of up to 20 Item objects sorted by EP estimate.
        """
        items = self._item_db.items_for_slot(slot, phase=phase)

        scored: list[tuple[float, Item]] = []
        for item in items:
            ep_score = sum(item.stats.get(stat, 0.0) * weight for stat, weight in _APPROX_EP.items())
            scored.append((ep_score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:20]]

    def _patched_item_db(self, config: SimConfig, stat_name: str, delta: int) -> ItemDatabase:
        """Create a deep copy of the item database with +delta to one stat on an equipped item.

        Finds a suitable non-weapon gear piece and adds the stat delta to it.

        Args:
            config: Simulation configuration.
            stat_name: Name of the stat to increase.
            delta: Amount to add.

        Returns:
            Deep-copied ItemDatabase with the stat delta applied.
        """
        patched_db = deepcopy(self._item_db)

        # Find a suitable item to patch (prefer armor slots over weapons)
        target_id: int | None = None
        for slot in [GearSlot.HEAD, GearSlot.CHEST, GearSlot.LEGS, GearSlot.HANDS, GearSlot.SHOULDER]:
            if slot in config.gear:
                target_id = config.gear[slot]
                break

        if target_id is None:
            for slot, item_id in config.gear.items():
                if slot not in (GearSlot.MAIN_HAND, GearSlot.OFF_HAND):
                    target_id = item_id
                    break

        if target_id is not None:
            item = patched_db.get_item(target_id)
            if item is not None:
                new_stats = dict(item.stats)
                new_stats[stat_name] = new_stats.get(stat_name, 0.0) + delta
                patched_item = item.model_copy(update={"stats": new_stats})
                patched_db.register_item(patched_item)

        return patched_db

    def _sum_gear_stats(self, config: SimConfig) -> dict[str, float]:
        """Sum all stats from equipped gear items.

        Args:
            config: Simulation configuration.

        Returns:
            Dict of stat_name -> total_value from all gear.
        """
        stats: dict[str, float] = {}
        for item_id in config.gear.values():
            item = self._item_db.get_item(item_id)
            if item is not None:
                for stat_name, value in item.stats.items():
                    stats[stat_name] = stats.get(stat_name, 0.0) + value
        return stats

    def _get_stat_total(self, config: SimConfig, stat_name: str) -> float:
        """Sum a specific stat from all equipped gear.

        Args:
            config: Simulation configuration.
            stat_name: The stat to sum.

        Returns:
            Total value of the stat across all gear.
        """
        total = 0.0
        for item_id in config.gear.values():
            item = self._item_db.get_item(item_id)
            if item is not None:
                total += item.stats.get(stat_name, 0.0)
        return total
