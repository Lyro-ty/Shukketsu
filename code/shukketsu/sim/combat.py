"""Discrete-event combat loop for TBC Rogue DPS simulation.

The heart of the simulation engine: schedules and dispatches combat events
(auto-attacks, energy ticks, ability uses, DOT ticks, buff expirations,
proc triggers) on a priority queue. Each iteration runs a deterministic
fight from time 0 to fight_length, accumulating damage and resource stats.
"""

from __future__ import annotations

import heapq
import logging
import statistics
from collections import defaultdict
from enum import StrEnum
from random import Random

from pydantic import BaseModel

from code.shukketsu.sim.abilities import ABILITIES, AbilityFlag, get_ability, get_poison
from code.shukketsu.sim.buffs import ResolvedBuffs
from code.shukketsu.sim.items import ItemDatabase
from code.shukketsu.sim.mechanics import (
    AP_PER_DPS,
    BASE_GLANCING_CHANCE,
    ENERGY_PER_TICK,
    ENERGY_TICK_MS,
    MELEE_CRIT_MULTIPLIER,
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
    calc_weapon_damage,
    resolve_white_hit,
    resolve_yellow_hit,
)
from code.shukketsu.sim.models import (
    AbilityBreakdown,
    GearSlot,
    HitOutcome,
    Item,
    PoisonType,
    ResourceStats,
    SimConfig,
    SimResult,
)
from code.shukketsu.sim.rotation import RotationContext, RotationEngine
from code.shukketsu.sim.talents import TalentModifiers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GCD_MS: int = 1000
"""Global cooldown duration for Rogues in milliseconds."""

SWORD_SPEC_ICD_MS: int = 500
"""Internal cooldown for Sword Specialization procs in milliseconds."""

SND_HASTE_MULTIPLIER: float = 1.30
"""Slice and Dice haste multiplier (30% attack speed increase)."""

BF_HASTE_MULTIPLIER: float = 1.20
"""Blade Flurry haste multiplier (20% attack speed increase)."""

OH_DAMAGE_MULTIPLIER: float = 0.50
"""Off-hand attacks deal 50% of weapon damage."""

RUPTURE_TICK_INTERVAL_MS: int = 2000
"""Rupture ticks every 2 seconds."""

RUPTURE_BASE_DPT: float = 70.0
"""Base damage per tick for Rupture rank 7."""

RUPTURE_CP_DPT: float = 18.0
"""Additional damage per tick per combo point for Rupture."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EventType(StrEnum):
    """Types of events in the combat simulation event queue."""

    MH_AUTO = "mh_auto"
    OH_AUTO = "oh_auto"
    ENERGY_TICK = "energy_tick"
    ABILITY_USE = "ability_use"
    DOT_TICK = "dot_tick"
    BUFF_EXPIRE = "buff_expire"
    PROC_TRIGGER = "proc_trigger"
    COOLDOWN_USE = "cooldown_use"
    POTION_USE = "potion_use"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SimEvent(BaseModel):
    """A single event on the simulation priority queue."""

    timestamp_ms: int
    event_type: EventType
    priority: int = 0
    data: dict[str, object] = {}

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SimEvent):
            return NotImplemented
        if self.timestamp_ms != other.timestamp_ms:
            return self.timestamp_ms < other.timestamp_ms
        return self.priority < other.priority


class DotState(BaseModel):
    """Tracking state for a damage-over-time effect."""

    remaining_ticks: int
    tick_interval_ms: int
    damage_per_tick: float
    next_tick_ms: int
    snapshot_ap: float


# ---------------------------------------------------------------------------
# CombatState (mutable, NOT a BaseModel)
# ---------------------------------------------------------------------------


class CombatState:
    """Mutable combat state tracking all simulation variables for one iteration.

    This is a regular class (not Pydantic BaseModel) because it is mutated
    frequently throughout the combat loop.
    """

    def __init__(self, max_energy: int = 100, fight_length_ms: int = 300000) -> None:
        self.current_time_ms: int = 0
        self.fight_length_ms: int = fight_length_ms
        self.energy: int = max_energy  # Start with full energy
        self.max_energy: int = max_energy
        self.combo_points: int = 0
        self.health_pct: float = 1.0
        self.gcd_ready_at_ms: int = 0
        self.mh_swing_at_ms: int = 0
        self.oh_swing_at_ms: int = 0
        self.next_energy_tick_ms: int = 0

        # Buff / DOT timers
        self.buff_timers: dict[str, int] = {}  # buff_name -> expires_at_ms
        self.dot_timers: dict[str, DotState] = {}
        self.proc_icds: dict[str, int] = {}  # proc_name -> available_at_ms
        self.sword_spec_icd_ms: int = 0  # Available at this time

        # Cooldown tracking
        self.cooldown_ready: dict[str, int] = {}  # ability_name -> ready_at_ms

        # Result accumulators
        self.damage_by_ability: dict[str, float] = defaultdict(float)
        self.casts_by_ability: dict[str, int] = defaultdict(int)
        self.outcome_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.proc_counts: dict[str, int] = defaultdict(int)
        self.proc_uptime_ms: dict[str, int] = defaultdict(int)
        self.total_damage: float = 0.0
        self.energy_wasted: float = 0.0
        self.cp_overcap: int = 0
        self.gcd_time_ms: int = 0
        self.energy_starved_ms: int = 0

        # Tracking for ability timing (GCD enforcement)
        self.ability_timestamps: list[int] = []

    def is_buff_active(self, name: str) -> bool:
        """Check if a buff is currently active."""
        return name in self.buff_timers and self.buff_timers[name] > self.current_time_ms


# ---------------------------------------------------------------------------
# CombatSimulation
# ---------------------------------------------------------------------------


class CombatSimulation:
    """Discrete-event combat simulation for TBC Rogue DPS.

    Runs N iterations of a deterministic fight, collecting per-ability damage,
    proc uptimes, and resource statistics. Results are aggregated into a
    SimResult for analysis.

    Args:
        config: Full simulation configuration.
        modifiers: Computed talent modifiers.
        resolved_buffs: Aggregated raid buff effects.
        item_db: Item database for gear lookup.
        rotation: Rotation engine for ability decisions.
    """

    def __init__(
        self,
        config: SimConfig,
        modifiers: TalentModifiers,
        resolved_buffs: ResolvedBuffs,
        item_db: ItemDatabase,
        rotation: RotationEngine,
    ) -> None:
        self._config = config
        self._modifiers = modifiers
        self._buffs = resolved_buffs
        self._item_db = item_db
        self._rotation = rotation
        self._fight_length_ms = config.fight_length * 1000

        # Resolve gear
        self._mh_item, self._oh_item = self._resolve_weapons()
        # _resolve_weapons() validates these are not None
        assert self._mh_item.weapon is not None, "MH weapon validated in _resolve_weapons"
        assert self._oh_item.weapon is not None, "OH weapon validated in _resolve_weapons"
        self._mh_weapon = self._mh_item.weapon
        self._oh_weapon = self._oh_item.weapon

        # Compute base character stats from gear + buffs + talents
        self._base_stats = self._compute_base_stats()

        # Pre-compute combat values that don't change mid-fight
        self._armor_mult = self._compute_armor_mult()
        self._miss_chance_white = calc_miss_chance(
            self._base_stats.get("hit_rating", 0.0),
            is_dual_wield=True,
            is_yellow=False,
            precision_ranks=int(self._modifiers.bonus_hit_pct * 100),
        )
        self._miss_chance_yellow = calc_miss_chance(
            self._base_stats.get("hit_rating", 0.0),
            is_dual_wield=True,
            is_yellow=True,
            precision_ranks=int(self._modifiers.bonus_hit_pct * 100),
        )
        self._dodge_chance = calc_dodge_chance(
            self._base_stats.get("expertise_rating", 0.0),
            weapon_expertise_ranks=int(self._modifiers.bonus_expertise / 5) if self._modifiers.bonus_expertise else 0,
        )
        self._crit_chance = calc_crit_chance(
            self._base_stats.get("crit_rating", 0.0),
            self._base_stats.get("agility", 0.0),
            talent_crit=self._modifiers.bonus_crit_pct,
            bonus_crit=self._base_stats.get("melee_crit_pct", 0.0),
        )
        self._crit_mult = calc_crit_multiplier(
            MELEE_CRIT_MULTIPLIER,
            primary_mod=self._modifiers.crit_damage_primary_mod,
            secondary_mod=self._modifiers.lethality_secondary_mod,
        )

        # Pre-compute normalized speeds for abilities
        self._mh_norm_speed = calc_normalized_speed(self._mh_weapon.weapon_type)

        # Poison setup
        self._mh_poison_type = config.poisons.main_hand
        self._oh_poison_type = config.poisons.off_hand
        self._ip_proc_chance = calc_poison_proc_chance(0.20, self._modifiers.imp_poisons_ranks)
        self._dp_proc_chance = calc_poison_proc_chance(0.30, self._modifiers.imp_poisons_ranks)

        # Event queue (per-iteration, set in _run_iteration)
        self._queue: list[SimEvent] = []

    def run(self, iterations: int, *, seed: int = 42) -> SimResult:
        """Run the simulation for a number of iterations.

        Args:
            iterations: Number of fight iterations to simulate.
            seed: Base random seed for reproducibility.

        Returns:
            Aggregated SimResult with DPS statistics and breakdowns.
        """
        iteration_dps: list[float] = []
        all_damage_by_ability: dict[str, float] = defaultdict(float)
        all_casts_by_ability: dict[str, float] = defaultdict(float)
        all_outcome_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total_energy_wasted: float = 0.0
        total_cp_overcap: int = 0
        total_gcd_time: int = 0
        total_energy_starved: int = 0

        for i in range(iterations):
            rng = Random(i * 1000 + seed)
            # Reset rotation state for each iteration
            self._rotation._state = self._rotation._state.__class__("opener")
            dps, state = self._run_iteration(rng)
            iteration_dps.append(dps)

            # Accumulate stats
            for ability, dmg in state.damage_by_ability.items():
                all_damage_by_ability[ability] += dmg
            for ability, casts in state.casts_by_ability.items():
                all_casts_by_ability[ability] += casts
            for ability, outcomes in state.outcome_counts.items():
                for outcome, count in outcomes.items():
                    all_outcome_counts[ability][outcome] += count
            total_energy_wasted += state.energy_wasted
            total_cp_overcap += state.cp_overcap
            total_gcd_time += state.gcd_time_ms
            total_energy_starved += state.energy_starved_ms

        return self._aggregate_results(
            iteration_dps,
            all_damage_by_ability,
            all_casts_by_ability,
            all_outcome_counts,
            total_energy_wasted,
            total_cp_overcap,
            total_gcd_time,
            total_energy_starved,
            iterations,
        )

    def _run_iteration(self, rng: Random) -> tuple[float, CombatState]:
        """Run a single fight iteration.

        Args:
            rng: Seeded random number generator for this iteration.

        Returns:
            Tuple of (DPS for this iteration, final CombatState).
        """
        max_energy = 110 if self._modifiers.vigor else 100
        state = CombatState(max_energy=max_energy, fight_length_ms=self._fight_length_ms)
        self._queue = []

        # Schedule initial events
        # MH auto at time 0
        self._schedule(SimEvent(timestamp_ms=0, event_type=EventType.MH_AUTO, priority=0))

        # OH auto at random 0-50% of OH speed offset
        oh_speed_ms = int(self._oh_weapon.speed * 1000)
        oh_offset = rng.randint(0, oh_speed_ms // 2)
        self._schedule(SimEvent(timestamp_ms=oh_offset, event_type=EventType.OH_AUTO, priority=0))

        # First energy tick at random offset [0, ENERGY_TICK_MS)
        first_tick = rng.randint(0, ENERGY_TICK_MS - 1)
        state.next_energy_tick_ms = first_tick
        self._schedule(SimEvent(timestamp_ms=first_tick, event_type=EventType.ENERGY_TICK, priority=1))

        # Initialize cooldowns as ready
        if self._modifiers.adrenaline_rush:
            state.cooldown_ready["adrenaline_rush"] = 0
        if self._modifiers.blade_flurry:
            state.cooldown_ready["blade_flurry"] = 0

        # Main event loop
        while self._queue:
            event = heapq.heappop(self._queue)

            if event.timestamp_ms > self._fight_length_ms:
                break

            state.current_time_ms = event.timestamp_ms
            self._dispatch_event(event, state, rng)

            # After each event, try to use abilities from rotation
            self._try_use_ability(state, rng)

        fight_seconds = self._fight_length_ms / 1000.0
        dps = state.total_damage / fight_seconds if fight_seconds > 0 else 0.0
        return dps, state

    def _schedule(self, event: SimEvent) -> None:
        """Push an event onto the priority queue.

        Args:
            event: The SimEvent to schedule.
        """
        heapq.heappush(self._queue, event)

    def _dispatch_event(self, event: SimEvent, state: CombatState, rng: Random) -> None:
        """Route an event to the appropriate handler.

        Args:
            event: The event to handle.
            state: Current mutable combat state.
            rng: Random number generator.
        """
        match event.event_type:
            case EventType.MH_AUTO:
                self._handle_mh_auto(state, rng)
            case EventType.OH_AUTO:
                self._handle_oh_auto(state, rng)
            case EventType.ENERGY_TICK:
                self._handle_energy_tick(state)
            case EventType.ABILITY_USE:
                ability_name = str(event.data.get("ability_name", ""))
                if ability_name:
                    self._handle_ability(ability_name, state, rng)
            case EventType.DOT_TICK:
                dot_name = str(event.data.get("dot_name", ""))
                if dot_name:
                    self._handle_dot_tick(dot_name, state)
            case EventType.BUFF_EXPIRE:
                buff_name = str(event.data.get("buff_name", ""))
                if buff_name:
                    self._handle_buff_expire(buff_name, state)
            case EventType.PROC_TRIGGER:
                proc_name = str(event.data.get("proc_name", ""))
                if proc_name:
                    self._handle_proc(proc_name, state, rng)

    def _handle_mh_auto(self, state: CombatState, rng: Random) -> None:
        """Handle a main-hand auto-attack.

        Args:
            state: Current combat state.
            rng: Random number generator.
        """
        # Resolve hit outcome
        outcome = resolve_white_hit(
            self._miss_chance_white,
            self._dodge_chance,
            BASE_GLANCING_CHANCE,
            max(0.0, self._crit_chance),
            rng.random(),
        )

        state.outcome_counts["mh_auto"][outcome.value] += 1
        state.casts_by_ability["mh_auto"] += 1

        if outcome not in (HitOutcome.MISS, HitOutcome.DODGE):
            # Calculate damage
            base_dmg = calc_weapon_damage(
                self._mh_weapon.min_damage,
                self._mh_weapon.max_damage,
                self._mh_weapon.speed,
                self._base_stats["attack_power"],
                roll=rng.random(),
            )
            actual = self._apply_damage("mh_auto", base_dmg, outcome, state)

            # Blade Flurry cleave
            if state.is_buff_active("blade_flurry") and self._config.target_count >= 2:
                cleave_dmg = actual  # Full damage to second target
                state.damage_by_ability["blade_flurry_cleave"] += cleave_dmg
                state.total_damage += cleave_dmg

            # Check procs (poisons, sword spec, combat potency from OH only)
            self._check_procs("mh_auto", outcome, state, rng)
            self._apply_poison("mh", state, rng)

        # Schedule next MH auto
        haste_mult = self._current_haste_multiplier(state)
        effective_speed_s = calc_effective_speed(self._mh_weapon.speed, haste_mult)
        next_swing_ms = state.current_time_ms + int(effective_speed_s * 1000)
        if next_swing_ms <= self._fight_length_ms:
            self._schedule(SimEvent(timestamp_ms=next_swing_ms, event_type=EventType.MH_AUTO, priority=0))
        state.mh_swing_at_ms = next_swing_ms

    def _handle_oh_auto(self, state: CombatState, rng: Random) -> None:
        """Handle an off-hand auto-attack.

        Args:
            state: Current combat state.
            rng: Random number generator.
        """
        outcome = resolve_white_hit(
            self._miss_chance_white,
            self._dodge_chance,
            BASE_GLANCING_CHANCE,
            max(0.0, self._crit_chance),
            rng.random(),
        )

        state.outcome_counts["oh_auto"][outcome.value] += 1
        state.casts_by_ability["oh_auto"] += 1

        if outcome not in (HitOutcome.MISS, HitOutcome.DODGE):
            base_dmg = calc_weapon_damage(
                self._oh_weapon.min_damage,
                self._oh_weapon.max_damage,
                self._oh_weapon.speed,
                self._base_stats["attack_power"],
                roll=rng.random(),
            )
            # Off-hand penalty
            oh_bonus = 1.0 + self._modifiers.dw_spec_oh_bonus_pct
            base_dmg *= OH_DAMAGE_MULTIPLIER * oh_bonus

            actual = self._apply_damage("oh_auto", base_dmg, outcome, state)

            # Blade Flurry cleave
            if state.is_buff_active("blade_flurry") and self._config.target_count >= 2:
                cleave_dmg = actual
                state.damage_by_ability["blade_flurry_cleave"] += cleave_dmg
                state.total_damage += cleave_dmg

            # Combat Potency proc (OH hits only)
            if self._modifiers.combat_potency_proc_chance > 0 and outcome in (HitOutcome.HIT, HitOutcome.CRIT):
                if rng.random() < self._modifiers.combat_potency_proc_chance:
                    gained = int(self._modifiers.combat_potency_energy)
                    state.energy = min(state.max_energy, state.energy + gained)
                    state.proc_counts["combat_potency"] += 1

            self._check_procs("oh_auto", outcome, state, rng)
            self._apply_poison("oh", state, rng)

        # Schedule next OH auto
        haste_mult = self._current_haste_multiplier(state)
        effective_speed_s = calc_effective_speed(self._oh_weapon.speed, haste_mult)
        next_swing_ms = state.current_time_ms + int(effective_speed_s * 1000)
        if next_swing_ms <= self._fight_length_ms:
            self._schedule(SimEvent(timestamp_ms=next_swing_ms, event_type=EventType.OH_AUTO, priority=0))
        state.oh_swing_at_ms = next_swing_ms

    def _handle_energy_tick(self, state: CombatState) -> None:
        """Handle an energy tick event.

        Args:
            state: Current combat state.
        """
        tick_amount = int(ENERGY_PER_TICK)
        # Adrenaline Rush doubles energy regen
        if state.is_buff_active("adrenaline_rush"):
            tick_amount *= 2

        new_energy = state.energy + tick_amount
        if new_energy > state.max_energy:
            state.energy_wasted += new_energy - state.max_energy
            new_energy = state.max_energy
        state.energy = new_energy

        # Schedule next energy tick
        next_tick = state.current_time_ms + ENERGY_TICK_MS
        if next_tick <= self._fight_length_ms:
            state.next_energy_tick_ms = next_tick
            self._schedule(SimEvent(timestamp_ms=next_tick, event_type=EventType.ENERGY_TICK, priority=1))

    def _handle_ability(self, ability_name: str, state: CombatState, rng: Random) -> None:
        """Execute an ability.

        Args:
            ability_name: Registry key of the ability to use.
            state: Current combat state.
            rng: Random number generator.
        """
        ability_def = get_ability(ability_name)

        # -- Off-GCD cooldown abilities --
        if ability_name == "adrenaline_rush":
            self._activate_buff("adrenaline_rush", ability_def.duration_ms, state)
            state.cooldown_ready["adrenaline_rush"] = state.current_time_ms + ability_def.cooldown_ms
            state.casts_by_ability["adrenaline_rush"] += 1
            return

        if ability_name == "blade_flurry":
            old_haste = self._current_haste_multiplier(state)
            self._activate_buff("blade_flurry", ability_def.duration_ms, state)
            new_haste = self._current_haste_multiplier(state)
            self._adjust_swing_timers(old_haste, new_haste, state)
            state.cooldown_ready["blade_flurry"] = state.current_time_ms + ability_def.cooldown_ms
            state.casts_by_ability["blade_flurry"] += 1
            # BF costs 25 energy
            state.energy = max(0, state.energy - ability_def.energy_cost)
            return

        if ability_name == "cold_blood":
            state.buff_timers["cold_blood"] = state.current_time_ms + 30000  # Lasts until next finisher
            state.casts_by_ability["cold_blood"] += 1
            return

        if ability_name == "thistle_tea":
            state.energy = min(state.max_energy, state.energy + 100)
            state.casts_by_ability["thistle_tea"] += 1
            return

        if ability_name == "premeditation":
            self._grant_combo_points(2, state)
            state.casts_by_ability["premeditation"] += 1
            return

        # -- Slice and Dice --
        if ability_name == "slice_and_dice":
            cp = max(1, state.combo_points)
            base_dur = ability_def.duration_ms + int(ability_def.bonus_per_combo_point) * cp
            snd_dur = int(base_dur * self._modifiers.snd_duration_mult)

            old_haste = self._current_haste_multiplier(state)
            self._activate_buff("slice_and_dice", snd_dur, state)
            new_haste = self._current_haste_multiplier(state)
            self._adjust_swing_timers(old_haste, new_haste, state)

            state.energy = max(0, state.energy - ability_def.energy_cost)
            state.combo_points = 0
            state.casts_by_ability["slice_and_dice"] += 1

            # Relentless Strikes energy refund
            self._check_relentless_strikes(cp, state, rng)

            # Trigger GCD
            state.gcd_ready_at_ms = state.current_time_ms + GCD_MS
            state.gcd_time_ms += GCD_MS
            state.ability_timestamps.append(state.current_time_ms)
            return

        # -- Rupture (DOT finisher) --
        if ability_name == "rupture":
            cp = max(1, state.combo_points)

            # Rupture duration: 6 + 2*CP seconds
            duration_ms = (6 + 2 * cp) * 1000
            num_ticks = duration_ms // RUPTURE_TICK_INTERVAL_MS

            # Damage per tick: base + CP bonus + AP coefficient
            dpt = RUPTURE_BASE_DPT + RUPTURE_CP_DPT * cp + self._base_stats["attack_power"] * 0.04
            dpt *= 1.0 + self._modifiers.rupture_damage_bonus_pct
            dpt *= 1.0 + self._modifiers.murder_damage_pct

            dot_state = DotState(
                remaining_ticks=num_ticks,
                tick_interval_ms=RUPTURE_TICK_INTERVAL_MS,
                damage_per_tick=dpt,
                next_tick_ms=state.current_time_ms + RUPTURE_TICK_INTERVAL_MS,
                snapshot_ap=self._base_stats["attack_power"],
            )
            state.dot_timers["rupture"] = dot_state

            # Schedule first tick
            self._schedule(
                SimEvent(
                    timestamp_ms=dot_state.next_tick_ms,
                    event_type=EventType.DOT_TICK,
                    priority=2,
                    data={"dot_name": "rupture"},
                )
            )

            outcome = resolve_yellow_hit(
                self._miss_chance_yellow,
                self._dodge_chance,
                max(0.0, self._crit_chance),
                rng.random(),
                rng.random(),
            )
            state.outcome_counts["rupture"][outcome.value] += 1
            state.casts_by_ability["rupture"] += 1

            if outcome in (HitOutcome.MISS, HitOutcome.DODGE):
                # Rupture missed/dodged - remove DOT
                del state.dot_timers["rupture"]
                # Partial energy refund on miss
                if outcome == HitOutcome.MISS:
                    state.energy = min(state.max_energy, state.energy + int(ability_def.energy_cost * 0.8))
            else:
                state.energy = max(0, state.energy - ability_def.energy_cost)

            # Relentless Strikes
            self._check_relentless_strikes(cp, state, rng)
            state.combo_points = 0
            state.gcd_ready_at_ms = state.current_time_ms + GCD_MS
            state.gcd_time_ms += GCD_MS
            state.ability_timestamps.append(state.current_time_ms)
            return

        # -- Eviscerate (direct damage finisher) --
        if ability_name == "eviscerate":
            cp = max(1, state.combo_points)
            base_dmg = ability_def.flat_damage + ability_def.bonus_per_combo_point * cp

            # Apply talent bonuses
            dmg_mult = 1.0
            dmg_mult *= 1.0 + self._modifiers.evis_damage_bonus_pct
            dmg_mult *= 1.0 + self._modifiers.aggression_damage_pct
            dmg_mult *= 1.0 + self._modifiers.murder_damage_pct
            base_dmg *= dmg_mult

            # Add AP scaling (normalized MH)
            base_dmg += self._base_stats["attack_power"] / AP_PER_DPS * self._mh_norm_speed

            # Cold Blood check
            effective_crit = self._crit_chance
            if state.is_buff_active("cold_blood"):
                effective_crit = 1.0
                del state.buff_timers["cold_blood"]

            outcome = resolve_yellow_hit(
                self._miss_chance_yellow,
                self._dodge_chance,
                max(0.0, effective_crit),
                rng.random(),
                rng.random(),
            )

            state.outcome_counts["eviscerate"][outcome.value] += 1
            state.casts_by_ability["eviscerate"] += 1

            if outcome in (HitOutcome.MISS, HitOutcome.DODGE):
                if outcome == HitOutcome.MISS:
                    state.energy = min(state.max_energy, state.energy + int(ability_def.energy_cost * 0.8))
            else:
                state.energy = max(0, state.energy - ability_def.energy_cost)
                self._apply_damage("eviscerate", base_dmg, outcome, state)

            self._check_relentless_strikes(cp, state, rng)
            state.combo_points = 0
            state.gcd_ready_at_ms = state.current_time_ms + GCD_MS
            state.gcd_time_ms += GCD_MS
            state.ability_timestamps.append(state.current_time_ms)
            return

        # -- Sinister Strike / Backstab / Mutilate / Hemorrhage (builders) --
        if ability_name in ("sinister_strike", "backstab", "mutilate", "hemorrhage"):
            # Calculate energy cost with talent reductions
            energy_cost = ability_def.energy_cost
            if ability_name == "sinister_strike":
                energy_cost -= self._modifiers.ss_energy_reduction

            # Hit check
            outcome = resolve_yellow_hit(
                self._miss_chance_yellow,
                self._dodge_chance,
                max(0.0, self._crit_chance),
                rng.random(),
                rng.random(),
            )

            state.outcome_counts[ability_name][outcome.value] += 1
            state.casts_by_ability[ability_name] += 1

            if outcome in (HitOutcome.MISS, HitOutcome.DODGE):
                if outcome == HitOutcome.MISS:
                    state.energy = min(state.max_energy, state.energy + int(energy_cost * ability_def.miss_refund_pct))
                else:
                    state.energy = max(0, state.energy - energy_cost)
            else:
                state.energy = max(0, state.energy - energy_cost)

                # Calculate damage
                base_dmg = calc_weapon_damage(
                    self._mh_weapon.min_damage,
                    self._mh_weapon.max_damage,
                    self._mh_weapon.speed,
                    self._base_stats["attack_power"],
                    normalized=ability_def.normalized,
                    norm_speed=ability_def.norm_speed,
                    roll=rng.random(),
                )
                base_dmg *= ability_def.weapon_multiplier
                base_dmg += ability_def.flat_damage

                # Apply talent damage bonuses
                dmg_mult = 1.0
                if ability_name == "sinister_strike":
                    dmg_mult *= 1.0 + self._modifiers.aggression_damage_pct
                dmg_mult *= 1.0 + self._modifiers.murder_damage_pct

                base_dmg *= dmg_mult

                self._apply_damage(ability_name, base_dmg, outcome, state)
                self._grant_combo_points(ability_def.combo_points_generated, state)

                # Sword Spec and poison procs on yellow hits
                self._check_procs(ability_name, outcome, state, rng)
                self._apply_poison("mh", state, rng)

                # Seal Fate: extra CP on crit for builders
                if outcome == HitOutcome.CRIT and self._modifiers.seal_fate_proc_chance > 0:
                    if rng.random() < self._modifiers.seal_fate_proc_chance:
                        self._grant_combo_points(1, state)
                        state.proc_counts["seal_fate"] += 1

            state.gcd_ready_at_ms = state.current_time_ms + GCD_MS
            state.gcd_time_ms += GCD_MS
            state.ability_timestamps.append(state.current_time_ms)
            return

    def _handle_dot_tick(self, dot_name: str, state: CombatState) -> None:
        """Handle a DOT tick event.

        Args:
            dot_name: Name of the DOT (e.g., "rupture").
            state: Current combat state.
        """
        dot = state.dot_timers.get(dot_name)
        if dot is None:
            return

        # Apply damage (DOT ticks always hit, apply armor reduction for physical)
        damage = dot.damage_per_tick * self._armor_mult
        state.damage_by_ability[dot_name] += damage
        state.total_damage += damage
        state.outcome_counts[dot_name]["tick"] += 1

        dot.remaining_ticks -= 1
        if dot.remaining_ticks > 0:
            dot.next_tick_ms = state.current_time_ms + dot.tick_interval_ms
            self._schedule(
                SimEvent(
                    timestamp_ms=dot.next_tick_ms,
                    event_type=EventType.DOT_TICK,
                    priority=2,
                    data={"dot_name": dot_name},
                )
            )
        else:
            del state.dot_timers[dot_name]

    def _handle_buff_expire(self, buff_name: str, state: CombatState) -> None:
        """Handle a buff expiration event.

        Args:
            buff_name: Name of the buff expiring.
            state: Current combat state.
        """
        # Only expire if the buff hasn't been refreshed
        if buff_name in state.buff_timers and state.buff_timers[buff_name] <= state.current_time_ms:
            old_haste = self._current_haste_multiplier(state)
            del state.buff_timers[buff_name]
            new_haste = self._current_haste_multiplier(state)

            if buff_name in ("slice_and_dice", "blade_flurry", "adrenaline_rush"):
                self._adjust_swing_timers(old_haste, new_haste, state)

    def _handle_proc(self, proc_name: str, state: CombatState, rng: Random) -> None:
        """Handle a proc trigger event.

        Args:
            proc_name: Name of the proc.
            state: Current combat state.
            rng: Random number generator.
        """
        state.proc_counts[proc_name] += 1

    def _apply_damage(self, ability_name: str, base_damage: float, outcome: HitOutcome, state: CombatState) -> float:
        """Apply damage from an attack, accounting for outcome modifiers and armor.

        Args:
            ability_name: Name of the ability dealing damage.
            base_damage: Pre-mitigation damage.
            outcome: The hit outcome (crit, glancing, etc.).
            state: Current combat state.

        Returns:
            The actual damage dealt after all modifiers.
        """
        damage = base_damage

        # Apply armor reduction for physical abilities
        damage *= self._armor_mult

        # Apply outcome modifier
        if outcome == HitOutcome.CRIT:
            damage *= self._crit_mult
        elif outcome == HitOutcome.GLANCING:
            damage *= calc_glancing_reduction()

        state.damage_by_ability[ability_name] += damage
        state.total_damage += damage
        return damage

    def _check_procs(self, source: str, outcome: HitOutcome, state: CombatState, rng: Random) -> None:
        """Check for proc effects from a successful attack.

        Args:
            source: Name of the attack that triggered the check.
            outcome: Hit outcome of the triggering attack.
            state: Current combat state.
            rng: Random number generator.
        """
        if outcome in (HitOutcome.MISS, HitOutcome.DODGE):
            return

        # Sword Spec: extra MH swing on any hit with swords
        is_white = source in ("mh_auto", "oh_auto")
        is_sword_spec_source = source not in ("sword_spec",)  # Sword spec can't trigger itself

        if (
            self._modifiers.sword_spec_proc_chance > 0
            and is_sword_spec_source
            and self._mh_weapon.weapon_type.value == "sword"
            and state.current_time_ms >= state.sword_spec_icd_ms
        ):
            if rng.random() < self._modifiers.sword_spec_proc_chance:
                state.sword_spec_icd_ms = state.current_time_ms + SWORD_SPEC_ICD_MS
                state.proc_counts["sword_spec"] += 1

                # Extra MH swing
                ss_outcome = resolve_white_hit(
                    self._miss_chance_white,
                    self._dodge_chance,
                    BASE_GLANCING_CHANCE,
                    max(0.0, self._crit_chance),
                    rng.random(),
                )
                state.outcome_counts["sword_spec"][ss_outcome.value] += 1
                state.casts_by_ability["sword_spec"] += 1

                if ss_outcome not in (HitOutcome.MISS, HitOutcome.DODGE):
                    ss_dmg = calc_weapon_damage(
                        self._mh_weapon.min_damage,
                        self._mh_weapon.max_damage,
                        self._mh_weapon.speed,
                        self._base_stats["attack_power"],
                        roll=rng.random(),
                    )
                    self._apply_damage("sword_spec", ss_dmg, ss_outcome, state)
                    # Sword spec procs can trigger poisons (not self)
                    self._apply_poison("mh", state, rng)

        # Windfury Totem proc (white hits only, not from WF or sword spec procs)
        if is_white and "wf_totem" in self._buffs.active_buff_ids and source not in ("wf_proc", "sword_spec"):
            wf_icd_ready = state.proc_icds.get("wf_totem", 0) <= state.current_time_ms
            if wf_icd_ready:
                for proc in self._buffs.active_procs:
                    if proc.effect.get("extra_attacks", 0) > 0:
                        if rng.random() < proc.rate:
                            state.proc_icds["wf_totem"] = state.current_time_ms + int(proc.icd * 1000)
                            state.proc_counts["wf_totem"] += 1

                            # Extra attack with bonus AP (simplified)
                            wf_dmg = calc_weapon_damage(
                                self._mh_weapon.min_damage,
                                self._mh_weapon.max_damage,
                                self._mh_weapon.speed,
                                self._base_stats["attack_power"] + 445,  # WF totem bonus AP
                                roll=rng.random(),
                            )
                            wf_outcome = resolve_white_hit(
                                self._miss_chance_white,
                                self._dodge_chance,
                                BASE_GLANCING_CHANCE,
                                max(0.0, self._crit_chance),
                                rng.random(),
                            )
                            state.outcome_counts["wf_proc"][wf_outcome.value] += 1
                            state.casts_by_ability["wf_proc"] += 1
                            if wf_outcome not in (HitOutcome.MISS, HitOutcome.DODGE):
                                self._apply_damage("wf_proc", wf_dmg, wf_outcome, state)
                                self._apply_poison("mh", state, rng)
                        break

    def _apply_poison(self, hand: str, state: CombatState, rng: Random) -> None:
        """Check and apply poison proc for a weapon hand.

        Args:
            hand: "mh" or "oh".
            state: Current combat state.
            rng: Random number generator.
        """
        poison_type = self._mh_poison_type if hand == "mh" else self._oh_poison_type

        if poison_type == PoisonType.NONE:
            return

        if poison_type == PoisonType.INSTANT:
            if rng.random() < self._ip_proc_chance:
                poison_def = get_poison(PoisonType.INSTANT)
                damage = poison_def.damage_per_proc
                damage *= 1.0 + self._modifiers.vile_poisons_pct
                # Nature damage, ignores armor
                state.damage_by_ability["instant_poison"] += damage
                state.total_damage += damage
                state.proc_counts["instant_poison"] += 1

        elif poison_type == PoisonType.DEADLY:
            if rng.random() < self._dp_proc_chance:
                state.proc_counts["deadly_poison"] += 1
                # Simplified: just add average tick damage
                poison_def = get_poison(PoisonType.DEADLY)
                damage = poison_def.damage_per_stack_tick * 0.5  # Simplified avg
                damage *= 1.0 + self._modifiers.vile_poisons_pct
                state.damage_by_ability["deadly_poison"] += damage
                state.total_damage += damage

    def _adjust_swing_timers(self, old_haste: float, new_haste: float, state: CombatState) -> None:
        """Adjust in-progress swing timers when haste changes.

        Uses proportional adjustment: remaining = remaining * old_speed / new_speed.

        Args:
            old_haste: Previous haste multiplier.
            new_haste: New haste multiplier.
            state: Current combat state.
        """
        if old_haste == new_haste or old_haste <= 0:
            return

        ratio = old_haste / new_haste

        # Adjust MH swing timer
        mh_remaining = state.mh_swing_at_ms - state.current_time_ms
        if mh_remaining > 0:
            new_remaining = max(1, int(mh_remaining * ratio))
            state.mh_swing_at_ms = state.current_time_ms + new_remaining

        # Adjust OH swing timer
        oh_remaining = state.oh_swing_at_ms - state.current_time_ms
        if oh_remaining > 0:
            new_remaining = max(1, int(oh_remaining * ratio))
            state.oh_swing_at_ms = state.current_time_ms + new_remaining

    def _try_use_ability(self, state: CombatState, rng: Random) -> None:
        """Consult the rotation engine and execute an ability if possible.

        Called after every event dispatch. Checks GCD readiness and energy
        availability before executing.

        Args:
            state: Current combat state.
            rng: Random number generator.
        """
        # Can't act if GCD is not ready
        if state.current_time_ms < state.gcd_ready_at_ms:
            return

        ctx = self._build_rotation_context(state)
        action = self._rotation.decide(ctx)

        if action.wait_for_energy:
            # Track energy starved time
            return

        ability_name = action.ability_name

        if ability_name == "wait":
            return

        # Check if ability is in the registry
        if ability_name not in ABILITIES:
            return

        ability_def = ABILITIES[ability_name]

        # Off-GCD abilities: execute immediately regardless of GCD
        if AbilityFlag.OFF_GCD in ability_def.flags:
            # Check cooldown readiness
            if ability_name in state.cooldown_ready:
                if state.current_time_ms < state.cooldown_ready[ability_name]:
                    return
            self._handle_ability(ability_name, state, rng)
            return

        # GCD abilities: check energy
        energy_cost = ability_def.energy_cost
        if ability_name == "sinister_strike":
            energy_cost -= self._modifiers.ss_energy_reduction

        if state.energy >= energy_cost:
            self._handle_ability(ability_name, state, rng)

    def _build_rotation_context(self, state: CombatState) -> RotationContext:
        """Build a RotationContext from the current CombatState.

        Args:
            state: Current mutable combat state.

        Returns:
            Immutable rotation context snapshot.
        """
        snd_remaining = max(0, state.buff_timers.get("slice_and_dice", 0) - state.current_time_ms)
        rupture_remaining = 0
        if "rupture" in state.dot_timers:
            remaining_ticks = state.dot_timers["rupture"].remaining_ticks
            rupture_remaining = remaining_ticks * state.dot_timers["rupture"].tick_interval_ms

        ar_active = state.is_buff_active("adrenaline_rush")
        bf_active = state.is_buff_active("blade_flurry")

        ar_ready = (
            "adrenaline_rush" in state.cooldown_ready
            and state.current_time_ms >= state.cooldown_ready["adrenaline_rush"]
            and not ar_active
        )
        bf_ready = (
            "blade_flurry" in state.cooldown_ready
            and state.current_time_ms >= state.cooldown_ready["blade_flurry"]
            and not bf_active
        )

        fight_remaining = max(0, self._fight_length_ms - state.current_time_ms)

        return RotationContext(
            combo_points=state.combo_points,
            energy=state.energy,
            max_energy=state.max_energy,
            snd_remaining_ms=snd_remaining,
            rupture_remaining_ms=rupture_remaining,
            ea_remaining_ms=0,
            dp_remaining_ms=0,
            dp_stacks=0,
            ar_active=ar_active,
            bf_active=bf_active,
            bf_ready=bf_ready,
            ar_ready=ar_ready,
            cb_ready=False,
            tea_ready=False,
            premeditation_ready=False,
            fight_remaining_ms=fight_remaining,
            target_count=self._config.target_count,
            is_stealthed=False,
            gcd_ready_at_ms=state.gcd_ready_at_ms,
            current_time_ms=state.current_time_ms,
        )

    def _aggregate_results(
        self,
        iteration_dps: list[float],
        all_damage_by_ability: dict[str, float],
        all_casts_by_ability: dict[str, float],
        all_outcome_counts: dict[str, dict[str, int]],
        total_energy_wasted: float,
        total_cp_overcap: int,
        total_gcd_time: int,
        total_energy_starved: int,
        iterations: int,
    ) -> SimResult:
        """Aggregate per-iteration results into a final SimResult.

        Args:
            iteration_dps: List of DPS values from each iteration.
            all_damage_by_ability: Accumulated damage by ability across iterations.
            all_casts_by_ability: Accumulated casts by ability across iterations.
            all_outcome_counts: Accumulated outcome counts across iterations.
            total_energy_wasted: Total energy wasted across all iterations.
            total_cp_overcap: Total CP overcap across all iterations.
            total_gcd_time: Total GCD time across all iterations.
            total_energy_starved: Total energy starved time across all iterations.
            iterations: Number of iterations run.

        Returns:
            Complete SimResult.
        """
        n = len(iteration_dps)
        dps_mean = statistics.mean(iteration_dps) if n > 0 else 0.0
        dps_std = statistics.stdev(iteration_dps) if n > 1 else 0.0
        dps_median = statistics.median(iteration_dps) if n > 0 else 0.0
        dps_min = min(iteration_dps) if n > 0 else 0.0
        dps_max = max(iteration_dps) if n > 0 else 0.0

        # Build ability breakdown
        total_damage = sum(all_damage_by_ability.values())
        ability_breakdown: list[AbilityBreakdown] = []

        for ability_name, damage in sorted(all_damage_by_ability.items(), key=lambda x: -x[1]):
            if damage <= 0:
                continue

            casts_total = all_casts_by_ability.get(ability_name, 0)
            casts_avg = casts_total / iterations if iterations > 0 else 0

            # Compute outcome percentages
            outcomes = all_outcome_counts.get(ability_name, {})
            total_outcomes = sum(outcomes.values())
            hit_count = outcomes.get("hit", 0) + outcomes.get("tick", 0)
            crit_count = outcomes.get("crit", 0)
            miss_count = outcomes.get("miss", 0)
            dodge_count = outcomes.get("dodge", 0)
            glancing_count = outcomes.get("glancing", 0)

            ability_breakdown.append(
                AbilityBreakdown(
                    name=ability_name,
                    damage_total=damage / iterations if iterations > 0 else 0,
                    damage_pct=(damage / total_damage * 100) if total_damage > 0 else 0,
                    casts=casts_avg,
                    hit_pct=(hit_count / total_outcomes * 100) if total_outcomes > 0 else 0,
                    crit_pct=(crit_count / total_outcomes * 100) if total_outcomes > 0 else 0,
                    miss_pct=(miss_count / total_outcomes * 100) if total_outcomes > 0 else 0,
                    dodge_pct=(dodge_count / total_outcomes * 100) if total_outcomes > 0 else 0,
                    glancing_pct=(glancing_count / total_outcomes * 100) if total_outcomes > 0 else 0,
                )
            )

        # Resource stats
        fight_seconds = self._config.fight_length
        total_fight_ms = self._fight_length_ms * iterations

        resource_stats = ResourceStats(
            energy_per_second=ENERGY_PER_TICK / (ENERGY_TICK_MS / 1000),
            avg_energy_waste=total_energy_wasted / iterations if iterations > 0 else 0,
            combo_points_per_second=0.0,
            combo_point_overcap_pct=(total_cp_overcap / max(1, iterations)) * 100 / max(1, fight_seconds),
            gcd_utilization_pct=(total_gcd_time / total_fight_ms * 100) if total_fight_ms > 0 else 0,
            energy_starved_pct=(total_energy_starved / total_fight_ms * 100) if total_fight_ms > 0 else 0,
        )

        return SimResult(
            dps_mean=dps_mean,
            dps_std=dps_std,
            dps_median=dps_median,
            dps_min=dps_min,
            dps_max=dps_max,
            iterations=iterations,
            fight_length=self._config.fight_length,
            ability_breakdown=ability_breakdown,
            resource_stats=resource_stats,
            config=self._config,
        )

    # -- Helpers ---------------------------------------------------------------

    def _resolve_weapons(self) -> tuple[Item, Item]:
        """Resolve main-hand and off-hand weapon items from config.

        Returns:
            Tuple of (main_hand_item, off_hand_item).

        Raises:
            ValueError: If weapons are missing or have no weapon stats.
        """
        mh_id = self._config.gear.get(GearSlot.MAIN_HAND)
        oh_id = self._config.gear.get(GearSlot.OFF_HAND)

        if mh_id is None or oh_id is None:
            raise ValueError("Both main_hand and off_hand weapon IDs are required in gear config")

        mh_item = self._item_db.get_item(mh_id)
        oh_item = self._item_db.get_item(oh_id)

        if mh_item is None:
            raise ValueError(f"Main hand item ID {mh_id} not found in database")
        if oh_item is None:
            raise ValueError(f"Off hand item ID {oh_id} not found in database")
        if mh_item.weapon is None:
            raise ValueError(f"Main hand item {mh_item.name} has no weapon stats")
        if oh_item.weapon is None:
            raise ValueError(f"Off hand item {oh_item.name} has no weapon stats")

        return mh_item, oh_item

    def _compute_base_stats(self) -> dict[str, float]:
        """Sum stats from all equipped gear items and buffs.

        Returns:
            Dictionary of stat_name -> total_value.
        """
        stats: dict[str, float] = defaultdict(float)

        # Base Rogue stats (Level 70)
        stats["agility"] = 180.0
        stats["strength"] = 110.0
        stats["stamina"] = 100.0

        # Gear stats
        for slot, item_id in self._config.gear.items():
            item = self._item_db.get_item(item_id)
            if item is not None:
                for stat_name, value in item.stats.items():
                    stats[stat_name] += value

        # Buff stats
        for stat_name, value in self._buffs.flat_stats.items():
            stats[stat_name] += value

        # Percentage multipliers from buffs
        stats_pct = self._buffs.stat_multipliers.get("stats_pct", 0.0)
        if stats_pct > 0:
            stats["agility"] *= 1.0 + stats_pct
            stats["strength"] *= 1.0 + stats_pct
            stats["stamina"] *= 1.0 + stats_pct

        # Talent stat multipliers
        stats["agility"] *= self._modifiers.vitality_agi_mult
        stats["agility"] *= self._modifiers.sinister_calling_agi_mult

        # Compute attack power
        base_ap = stats["agility"] + stats["strength"] + stats.get("attack_power", 0.0)
        base_ap *= self._modifiers.deadliness_ap_mult
        stats["attack_power"] = base_ap

        # Melee crit from buffs
        stats["melee_crit_pct"] = self._buffs.stat_multipliers.get("melee_crit_pct", 0.0)

        return dict(stats)

    def _compute_armor_mult(self) -> float:
        """Compute armor damage reduction multiplier for the target boss.

        Returns:
            Damage multiplier from armor (e.g. 0.65 means 35% reduction).
        """
        return calc_armor_reduction(
            self._config.boss.armor,
            arpen=int(self._base_stats.get("armor_penetration", 0)),
            sunder_stacks=5 if "sunder_armor" in self._buffs.active_buff_ids else 0,
            faerie_fire="faerie_fire" in self._buffs.active_buff_ids,
            curse_of_recklessness="curse_of_recklessness" in self._buffs.active_buff_ids,
        )

    def _current_haste_multiplier(self, state: CombatState) -> float:
        """Calculate current haste multiplier based on active buffs.

        Args:
            state: Current combat state.

        Returns:
            Combined haste multiplier.
        """
        buffs: list[float] = []
        if state.is_buff_active("slice_and_dice"):
            buffs.append(SND_HASTE_MULTIPLIER)
        if state.is_buff_active("blade_flurry"):
            buffs.append(BF_HASTE_MULTIPLIER)

        return calc_haste_multiplier(self._base_stats.get("haste_rating", 0.0), *buffs)

    def _activate_buff(self, name: str, duration_ms: int, state: CombatState) -> None:
        """Activate a buff with the given duration.

        Args:
            name: Buff identifier.
            duration_ms: Duration in milliseconds.
            state: Current combat state.
        """
        expires_at = state.current_time_ms + duration_ms
        state.buff_timers[name] = expires_at

        # Schedule expiration event
        self._schedule(
            SimEvent(
                timestamp_ms=expires_at,
                event_type=EventType.BUFF_EXPIRE,
                priority=5,
                data={"buff_name": name},
            )
        )

    def _grant_combo_points(self, amount: int, state: CombatState) -> None:
        """Grant combo points, tracking overcap.

        Args:
            amount: Number of combo points to grant.
            state: Current combat state.
        """
        new_cp = state.combo_points + amount
        if new_cp > 5:
            state.cp_overcap += new_cp - 5
            new_cp = 5
        state.combo_points = new_cp

    def _check_relentless_strikes(self, combo_points: int, state: CombatState, rng: Random) -> None:
        """Check for Relentless Strikes energy refund on finisher use.

        Args:
            combo_points: Number of combo points consumed by the finisher.
            state: Current combat state.
            rng: Random number generator.
        """
        if self._modifiers.relentless_strikes_per_cp <= 0:
            return

        # 5 energy per CP chance (e.g. 25 energy at 5 CP)
        refund_chance = combo_points * self._modifiers.relentless_strikes_per_cp / 100.0
        if rng.random() < refund_chance:
            state.energy = min(state.max_energy, state.energy + 25)
            state.proc_counts["relentless_strikes"] += 1

        # Ruthlessness: chance of extra CP after finisher
        if self._modifiers.ruthlessness_proc_chance > 0:
            if rng.random() < self._modifiers.ruthlessness_proc_chance:
                self._grant_combo_points(1, state)
                state.proc_counts["ruthlessness"] += 1
