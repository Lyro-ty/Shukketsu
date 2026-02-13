"""Simulation optimization tool for the Analyst agent."""

import logging
from typing import Any

from code.shukketsu.tools.schemas import Tool

logger = logging.getLogger(__name__)


class SimOptimizeTool(Tool):
    """Find the best items for a specific gear slot by simulating all candidates.

    Pre-filters items by phase and approximate EP score, then sims each
    candidate to produce a ranked list of recommendations.
    """

    name = "sim_optimize"
    description = (
        "Find the best items for a specific gear slot by simulating all candidates. "
        "Returns a ranked list of items with DPS values and deltas."
    )
    parameters_schema: dict[str, Any] = {
        "import_string": {"type": "string", "description": "Character import string for the base setup"},
        "slot": {"type": "string", "description": "Gear slot to optimize (e.g. trinket_1, main_hand)"},
        "phase": {"type": "integer", "description": "Maximum content phase (1-5, default 5)", "optional": True},
        "top_n": {"type": "integer", "description": "Number of top items to show (default 5)", "optional": True},
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
        """Execute a slot optimization and return ranked results.

        Args:
            tool_input: Dictionary with import_string, slot, and optional phase/top_n.

        Returns:
            Formatted string with ranked item recommendations.
        """
        import_string = tool_input.get("import_string", "").strip()
        slot_str = tool_input.get("slot", "").strip()

        if not import_string:
            return "Error: 'import_string' parameter is required."
        if not slot_str:
            return "Error: 'slot' parameter is required."

        phase = tool_input.get("phase", 5)
        top_n = tool_input.get("top_n", 5)

        runner = self._get_runner()

        try:
            config = runner.build_config_from_import(import_string)
        except Exception as exc:
            return f"Error parsing import string: {exc}"

        # Parse the gear slot
        from code.shukketsu.sim.models import GearSlot

        try:
            slot = GearSlot(slot_str)
        except ValueError:
            valid = ", ".join(s.value for s in GearSlot)
            return f"Error: Invalid gear slot '{slot_str}'. Valid slots: {valid}"

        # Run optimization
        try:
            result = await runner.sim_optimize(config, slot, top_n=top_n, phase=phase)
        except Exception as exc:
            logger.exception("Optimization failed: %s", exc)
            return f"Error running optimization: {exc}"

        # Format result
        parts = [
            f"Slot Optimization: {slot_str} (Phase {phase}, top {top_n})",
            f"  Current: {result.current_item} ({result.current_dps:.1f} DPS)",
            "",
            "Ranked Recommendations:",
        ]

        for i, rec in enumerate(result.recommendations, 1):
            sign = "+" if rec.dps_delta >= 0 else ""
            parts.append(f"  {i}. {rec.item_name} (P{rec.phase}): {rec.dps:.1f} DPS ({sign}{rec.dps_delta:.1f})")

        if not result.recommendations:
            parts.append("  No alternative items found for this slot and phase.")

        return "\n".join(parts)
