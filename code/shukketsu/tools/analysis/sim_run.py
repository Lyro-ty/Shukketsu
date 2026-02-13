"""Simulation run tool for the Analyst agent."""

import logging
from typing import Any

from code.shukketsu.tools.schemas import Tool

logger = logging.getLogger(__name__)


class SimRunTool(Tool):
    """Run a DPS simulation for a Rogue character.

    Accepts a character import string (SimC/WoWSims/SeventyUpgrades format)
    plus optional overrides. Returns mean DPS, ability breakdown, and
    optionally stat weights formatted as a readable observation string.
    """

    name = "sim_run"
    description = (
        "Run a DPS simulation for a Rogue character. "
        "Returns mean DPS, ability breakdown, and optionally stat weights. "
        "Requires an import_string with the character setup."
    )
    parameters_schema: dict[str, Any] = {
        "import_string": {"type": "string", "description": "Character import string (SimC, WoWSims, or 70upgrades)"},
        "buff_preset": {
            "type": "string",
            "description": "Buff preset name (default: full_25man)",
            "optional": True,
        },
        "boss_armor": {"type": "integer", "description": "Boss armor value override", "optional": True},
        "fight_length": {"type": "integer", "description": "Fight length in seconds", "optional": True},
        "iterations": {"type": "integer", "description": "Number of sim iterations", "optional": True},
        "compute_stat_weights": {
            "type": "boolean",
            "description": "Whether to compute stat weights (slower)",
            "optional": True,
        },
    }

    def __init__(self, runner: Any | None = None) -> None:
        """Initialize the tool with an optional SimRunner instance.

        Args:
            runner: A SimRunner instance. If None, one is created on first use.
        """
        self._runner = runner

    def _get_runner(self) -> Any:
        """Lazily create a SimRunner if not provided."""
        if self._runner is None:
            from code.shukketsu.sim.runner import SimRunner

            self._runner = SimRunner()
        return self._runner

    async def execute(self, tool_input: dict[str, Any]) -> str:
        """Execute a DPS simulation and return formatted results.

        Args:
            tool_input: Dictionary with import_string and optional overrides.

        Returns:
            Formatted string with DPS statistics and ability breakdown.
        """
        import_string = tool_input.get("import_string", "").strip()
        if not import_string:
            return "Error: 'import_string' parameter is required."

        runner = self._get_runner()

        # Build config from import string with overrides
        overrides: dict[str, Any] = {}
        if "buff_preset" in tool_input:
            overrides["raid_preset"] = tool_input["buff_preset"]
        if "boss_armor" in tool_input:
            overrides["boss"] = {"armor": tool_input["boss_armor"]}
        if "fight_length" in tool_input:
            overrides["fight_length"] = tool_input["fight_length"]
        if "iterations" in tool_input:
            overrides["iterations"] = tool_input["iterations"]

        try:
            config = runner.build_config_from_import(import_string, **overrides)
        except Exception as exc:
            return f"Error parsing import string: {exc}"

        try:
            result = await runner.sim_run(config)
        except Exception as exc:
            logger.exception("Simulation failed: %s", exc)
            return f"Error running simulation: {exc}"

        # Format the result
        parts = [
            f"Simulation Results ({result.iterations} iterations, {result.fight_length}s fight):",
            f"  Mean DPS: {result.dps_mean:.1f}",
            f"  Median DPS: {result.dps_median:.1f}",
            f"  Std Dev: {result.dps_std:.1f}",
            f"  Range: {result.dps_min:.1f} - {result.dps_max:.1f}",
            "",
            "Ability Breakdown:",
        ]
        for ab in sorted(result.ability_breakdown, key=lambda a: a.damage_pct, reverse=True):
            parts.append(f"  {ab.name}: {ab.damage_pct:.1f}% ({ab.casts:.0f} casts, {ab.crit_pct:.1f}% crit)")

        # Optionally compute stat weights
        compute_weights = tool_input.get("compute_stat_weights", False)
        if compute_weights:
            try:
                weights = await runner.stat_weights(config)
                parts.append("")
                parts.append("Stat Weights (EP, relative to 1 AP):")
                for w in sorted(weights, key=lambda sw: sw.ep_value, reverse=True):
                    cap_str = " [CAPPED]" if w.is_capped else ""
                    parts.append(f"  {w.stat}: {w.ep_value:.2f} EP ({w.dps_per_point:.3f} DPS/point){cap_str}")
            except Exception as exc:
                parts.append(f"\nStat weight computation failed: {exc}")

        return "\n".join(parts)
