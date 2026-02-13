"""Simulation comparison tool for the Analyst agent."""

import logging
from typing import Any

from code.shukketsu.tools.schemas import Tool

logger = logging.getLogger(__name__)


class SimCompareTool(Tool):
    """Compare two gear setups by simulating each and showing the DPS difference.

    Builds a base config from the import string, swaps one item,
    and runs both configurations to show the impact.
    """

    name = "sim_compare"
    description = (
        "Compare two gear setups by swapping one item and simulating both. "
        "Shows DPS difference, stat changes, and ability breakdown changes."
    )
    parameters_schema: dict[str, Any] = {
        "import_string": {"type": "string", "description": "Character import string for the base setup"},
        "swap_slot": {"type": "string", "description": "Gear slot to swap (e.g. trinket_1, main_hand)"},
        "swap_item": {"type": "string", "description": "Name of the item to swap in"},
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
        """Execute a comparison simulation and return formatted results.

        Args:
            tool_input: Dictionary with import_string, swap_slot, and swap_item.

        Returns:
            Formatted string comparing DPS, stats, and ability changes.
        """
        import_string = tool_input.get("import_string", "").strip()
        swap_slot = tool_input.get("swap_slot", "").strip()
        swap_item = tool_input.get("swap_item", "").strip()

        if not import_string:
            return "Error: 'import_string' parameter is required."
        if not swap_slot:
            return "Error: 'swap_slot' parameter is required."
        if not swap_item:
            return "Error: 'swap_item' parameter is required."

        runner = self._get_runner()

        try:
            base_config = runner.build_config_from_import(import_string)
        except Exception as exc:
            return f"Error parsing import string: {exc}"

        # Parse the gear slot
        from code.shukketsu.sim.models import GearSlot

        try:
            slot = GearSlot(swap_slot)
        except ValueError:
            valid = ", ".join(s.value for s in GearSlot)
            return f"Error: Invalid gear slot '{swap_slot}'. Valid slots: {valid}"

        # Swap the item
        try:
            swapped_config = runner.swap_item(base_config, slot, swap_item)
        except Exception as exc:
            return f"Error swapping item: {exc}"

        # Run comparison
        try:
            compare = await runner.sim_compare(base_config, swapped_config)
        except Exception as exc:
            logger.exception("Comparison simulation failed: %s", exc)
            return f"Error running comparison: {exc}"

        # Format result
        sign = "+" if compare.dps_delta >= 0 else ""
        parts = [
            f"Comparison Result: {compare.summary}",
            "",
            f"  Before: {compare.dps_before:.1f} DPS",
            f"  After:  {compare.dps_after:.1f} DPS",
            f"  Delta:  {sign}{compare.dps_delta:.1f} DPS ({sign}{compare.dps_delta_pct:.1f}%)",
        ]

        if compare.stat_changes:
            parts.append("")
            parts.append("Stat Changes:")
            for sc in compare.stat_changes:
                s = "+" if sc.delta >= 0 else ""
                parts.append(f"  {sc.stat_name}: {sc.before:.0f} -> {sc.after:.0f} ({s}{sc.delta:.0f})")

        if compare.ability_changes:
            parts.append("")
            parts.append("Ability DPS Changes:")
            for ac in sorted(compare.ability_changes, key=lambda a: abs(a.delta), reverse=True)[:5]:
                s = "+" if ac.delta >= 0 else ""
                parts.append(f"  {ac.ability_name}: {s}{ac.delta:.1f} DPS ({s}{ac.delta_pct:.1f}%)")

        return "\n".join(parts)
