"""State-machine rotation engine for TBC Rogue simulation.

Implements per-spec rotation logic as a priority-based state machine.
The engine receives an immutable snapshot of the current combat state
and returns the next ability to use. No RNG or time advancement happens
here -- that is the combat engine's responsibility.
"""

import logging
from enum import StrEnum

from pydantic import BaseModel

from code.shukketsu.sim.models import RogueSpec

# TalentModifiers is frozen (immutable) -- safe to store as read-only reference.
from code.shukketsu.sim.talents import TalentModifiers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SND_REFRESH_BASE_MS: int = 4000
"""Base milliseconds for the SND refresh buffer formula."""

_SND_REFRESH_PER_CP_MS: int = 800
"""Milliseconds subtracted per combo point from the refresh buffer."""

_ENERGY_POOL_THRESHOLD: int = 50
"""Minimum energy to spend a finisher under normal conditions."""

_ENERGY_POOL_AR_THRESHOLD: int = 30
"""Minimum energy to spend a finisher during Adrenaline Rush."""

_AR_ENERGY_THRESHOLD: int = 85
"""Use AR when energy is at or below this value."""

_FINISHER_CP_COMBAT: int = 5
"""Combo points required before using a damage finisher (combat specs)."""

_FINISHER_CP_MUT: int = 4
"""Combo points required before using a damage finisher (mutilate -- 2 CP builder)."""

_DP_SHIV_BUFFER_MS: int = 2000
"""Refresh Deadly Poison via Shiv when remaining duration drops below this."""

_RUPTURE_MIN_FIGHT_MS: int = 12000
"""Don't apply Rupture if fewer than this many ms remain in the fight."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RotationState(StrEnum):
    """Rotation state-machine states."""

    OPENER = "opener"
    SLICE_ASAP = "slice_asap"
    DISPATCH = "dispatch"
    BUILD_FOR_SND = "build_for_snd"
    BUILD_FOR_EA = "build_for_ea"
    FILL_BEFORE_SND = "fill_before_snd"
    FILL_BEFORE_EA = "fill_before_ea"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RotationAction(BaseModel):
    """A single rotation decision returned by the engine."""

    ability_name: str
    target: str = "boss"
    wait_for_energy: bool = False


class RotationContext(BaseModel, frozen=True):
    """Read-only snapshot of the current combat state.

    Consumed by ``RotationEngine.decide()`` to make ability choices.
    All time values are in milliseconds.
    """

    combo_points: int
    energy: int
    max_energy: int
    snd_remaining_ms: int
    rupture_remaining_ms: int
    ea_remaining_ms: int
    dp_remaining_ms: int
    dp_stacks: int
    ar_active: bool
    bf_active: bool
    bf_ready: bool
    ar_ready: bool
    cb_ready: bool
    tea_ready: bool
    premeditation_ready: bool
    fight_remaining_ms: int
    target_count: int
    is_stealthed: bool
    gcd_ready_at_ms: int
    current_time_ms: int


# ---------------------------------------------------------------------------
# Rotation Engine
# ---------------------------------------------------------------------------


class RotationEngine:
    """Priority-based state-machine rotation for all Rogue specs.

    The engine maintains an internal state (e.g. OPENER, DISPATCH) and
    selects abilities based on the current ``RotationContext`` snapshot.
    Off-GCD cooldowns are checked first, then the state machine drives
    builder / finisher selection.

    Args:
        spec: The Rogue specialization determining builder/finisher choice.
        modifiers: Frozen talent modifiers affecting rotation decisions.
        expose_armor: Whether the Rogue is responsible for Expose Armor uptime.
    """

    def __init__(self, spec: RogueSpec, modifiers: TalentModifiers, *, expose_armor: bool = False) -> None:
        self._spec = spec
        self._modifiers = modifiers
        self._expose_armor = expose_armor
        self._state = RotationState.OPENER

        # Pre-compute finisher CP threshold based on spec.
        self._finisher_cp = _FINISHER_CP_MUT if spec == RogueSpec.ASSASSINATION_MUTILATE else _FINISHER_CP_COMBAT

    # -- public API ---------------------------------------------------------

    def decide(self, ctx: RotationContext) -> RotationAction:
        """Return the next ability to use given the current combat state.

        Evaluation order:
        1. Off-GCD cooldowns (AR, BF, Cold Blood, Thistle Tea, Premeditation).
        2. Opener ability when stealthed.
        3. Immediate Slice and Dice when in SLICE_ASAP with combo points.
        4. Delegate to ``_dispatch()`` for steady-state decisions.

        Args:
            ctx: Immutable snapshot of current combat state.

        Returns:
            A ``RotationAction`` describing which ability to use next.
        """
        # 1. Off-GCD cooldowns (checked every decision cycle).
        cd_action = self._check_cooldowns(ctx)
        if cd_action is not None:
            return cd_action

        # 2. Opener phase.
        if self._state == RotationState.OPENER:
            if ctx.is_stealthed:
                return RotationAction(ability_name="garrote", target="boss")
            # Left stealth already (or was never stealthed) -- move on.
            if ctx.combo_points >= 1:
                self._state = RotationState.SLICE_ASAP
            else:
                self._state = RotationState.DISPATCH

        # 3. Get SND up ASAP.
        if self._state == RotationState.SLICE_ASAP:
            if ctx.combo_points >= 1:
                self._state = RotationState.DISPATCH
                return RotationAction(ability_name="slice_and_dice", target="self")
            # No CPs yet -- fall through to dispatch to build one.
            self._state = RotationState.DISPATCH

        # 4. Steady-state dispatch.
        return self._dispatch(ctx)

    # -- internal state machine --------------------------------------------

    def _dispatch(self, ctx: RotationContext) -> RotationAction:
        """Central decision hub for the steady-state rotation.

        Priority:
        1. SND expired -> get it back up immediately.
        2. Expose Armor refresh (if responsible).
        3. SND approaching expiry (refresh buffer).
        4. Enough combo points -> damage finisher (energy-pool check).
        5. Shiv for Deadly Poison maintenance (Mutilate only).
        6. Build combo points.

        Args:
            ctx: Immutable snapshot of current combat state.

        Returns:
            A ``RotationAction`` for the next ability.
        """
        # --- Priority 1: SND expired ---
        if ctx.snd_remaining_ms <= 0:
            if ctx.combo_points >= 1:
                return RotationAction(ability_name="slice_and_dice", target="self")
            # No CPs -- build one first.
            return self._build_action(ctx)

        # --- Priority 2: Expose Armor ---
        # NOTE: Expose Armor is not supported in the sim engine because
        # _armor_mult is computed once at init and never recalculated.
        # The expose_armor config flag is accepted but ignored.

        # --- Priority 3: SND refresh buffer ---
        snd_buffer_ms = _SND_REFRESH_BASE_MS - ctx.combo_points * _SND_REFRESH_PER_CP_MS
        if ctx.snd_remaining_ms <= snd_buffer_ms:
            # SND is approaching expiry for our current CP count -- refresh now.
            if ctx.combo_points >= 1:
                return RotationAction(ability_name="slice_and_dice", target="self")
            return self._build_action(ctx)

        # --- Priority 4: Damage finisher with enough CPs ---
        if ctx.combo_points >= self._finisher_cp:
            if not self._should_pool_energy(ctx):
                finisher = self._select_finisher(ctx)
                return RotationAction(ability_name=finisher, target="boss")
            # Pooling energy -- wait instead of building/finishing.
            return RotationAction(ability_name="wait", target="boss", wait_for_energy=True)

        # --- Priority 5: Shiv for DP maintenance (Mutilate only) ---
        if (
            self._spec == RogueSpec.ASSASSINATION_MUTILATE
            and ctx.dp_remaining_ms < _DP_SHIV_BUFFER_MS
            and ctx.dp_remaining_ms >= 0
        ):
            return RotationAction(ability_name="shiv", target="boss")

        # --- Priority 6: Build combo points ---
        return self._build_action(ctx)

    def _check_cooldowns(self, ctx: RotationContext) -> RotationAction | None:
        """Check off-GCD cooldowns and return an action if one should fire.

        Args:
            ctx: Current combat state snapshot.

        Returns:
            A ``RotationAction`` for the cooldown, or ``None`` if no CD is ready.
        """
        # Premeditation: only from stealth, grants 2 free CPs.
        if ctx.is_stealthed and ctx.premeditation_ready and self._modifiers.premeditation_talented:
            return RotationAction(ability_name="premeditation", target="self")

        # Don't use combat CDs until SND is active.
        if ctx.snd_remaining_ms <= 0:
            return None

        # Adrenaline Rush: low energy, SND active.
        if self._should_use_cooldown("adrenaline_rush", ctx):
            return RotationAction(ability_name="adrenaline_rush", target="self")

        # Blade Flurry: SND active.
        if self._should_use_cooldown("blade_flurry", ctx):
            return RotationAction(ability_name="blade_flurry", target="self")

        # Cold Blood: pair with AR or use independently.
        if self._should_use_cooldown("cold_blood", ctx):
            return RotationAction(ability_name="cold_blood", target="self")

        # Thistle Tea: low energy.
        if self._should_use_cooldown("thistle_tea", ctx):
            return RotationAction(ability_name="thistle_tea", target="self")

        return None

    def _should_use_cooldown(self, cd_name: str, ctx: RotationContext) -> bool:
        """Determine whether a specific cooldown should be activated now.

        Args:
            cd_name: Registry key of the cooldown ability.
            ctx: Current combat state snapshot.

        Returns:
            True if the cooldown should be used immediately.
        """
        match cd_name:
            case "adrenaline_rush":
                return (
                    self._modifiers.adrenaline_rush
                    and ctx.ar_ready
                    and not ctx.ar_active
                    and ctx.energy <= _AR_ENERGY_THRESHOLD
                    and ctx.snd_remaining_ms > 0
                )
            case "blade_flurry":
                return self._modifiers.blade_flurry and ctx.bf_ready and not ctx.bf_active and ctx.snd_remaining_ms > 0
            case "cold_blood":
                return self._modifiers.cold_blood and ctx.cb_ready and (ctx.ar_active or not ctx.ar_ready)
            case "thistle_tea":
                return ctx.tea_ready and ctx.energy <= (ctx.max_energy - 100)
            case _:
                return False

    def _select_builder(self, ctx: RotationContext) -> str:
        """Choose the appropriate combo-point builder for the current spec.

        Args:
            ctx: Current combat state snapshot.

        Returns:
            Ability registry key for the chosen builder.
        """
        match self._spec:
            case RogueSpec.ASSASSINATION_MUTILATE:
                return "mutilate"
            case RogueSpec.COMBAT_DAGGERS:
                return "backstab"
            case RogueSpec.COMBAT_SWORDS | RogueSpec.COMBAT_FISTS:
                return "sinister_strike"
            case _:
                return "sinister_strike"

    def _select_finisher(self, ctx: RotationContext) -> str:
        """Choose the damage finisher based on spec and current debuffs.

        Args:
            ctx: Current combat state snapshot.

        Returns:
            Ability registry key for the chosen finisher.
        """
        # Assassination: always Envenom (consumes DP stacks for damage).
        if self._spec == RogueSpec.ASSASSINATION_MUTILATE:
            return "envenom"

        # Combat specs: Rupture if not active and fight is long enough.
        if ctx.rupture_remaining_ms <= 0 and ctx.fight_remaining_ms >= _RUPTURE_MIN_FIGHT_MS:
            return "rupture"

        return "eviscerate"

    def _should_pool_energy(self, ctx: RotationContext) -> bool:
        """Determine whether to delay a finisher to pool energy.

        Pooling ensures enough energy remains for the next builder after
        the finisher lands, avoiding dead GCDs. The threshold is lowered
        during Adrenaline Rush because energy regeneration is doubled.

        SND urgency overrides pooling -- if SND is about to drop, spend
        immediately.

        Args:
            ctx: Current combat state snapshot.

        Returns:
            True if the engine should wait for more energy before finishing.
        """
        # Never pool when SND is about to expire.
        snd_buffer_ms = _SND_REFRESH_BASE_MS - ctx.combo_points * _SND_REFRESH_PER_CP_MS
        if ctx.snd_remaining_ms <= snd_buffer_ms:
            return False

        threshold = _ENERGY_POOL_AR_THRESHOLD if ctx.ar_active else _ENERGY_POOL_THRESHOLD
        return ctx.energy < threshold

    def _build_action(self, ctx: RotationContext) -> RotationAction:
        """Return a builder action for the current spec.

        Args:
            ctx: Current combat state snapshot.

        Returns:
            A ``RotationAction`` using the spec's primary builder.
        """
        builder = self._select_builder(ctx)
        return RotationAction(ability_name=builder, target="boss")
