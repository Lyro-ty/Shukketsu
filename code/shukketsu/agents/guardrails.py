"""Agent guardrails: loop detection and budget enforcement."""

import json
import logging
from collections import Counter

from code.shukketsu import config

logger = logging.getLogger(__name__)


class LoopDetector:
    """Detects agent loops by inspecting the scratchpad.

    Checks for:
    1. Consecutive identical tool calls (agent stuck repeating)
    2. Total repeats of any tool+input pair (agent going in circles)
    3. Token budget exhaustion (estimated via char count / 4)
    """

    def __init__(
        self,
        *,
        max_consecutive_same: int = config.LOOP_MAX_CONSECUTIVE_SAME,
        max_total_repeats: int = config.LOOP_MAX_TOTAL_REPEATS,
        max_token_budget: int = config.MAX_TOTAL_TOKENS,
    ) -> None:
        self._max_consecutive_same = max_consecutive_same
        self._max_total_repeats = max_total_repeats
        self._max_token_budget = max_token_budget

    def check(self, scratchpad: list[dict]) -> str | None:
        """Check the scratchpad for loop patterns.

        Returns:
            Error message string if a loop is detected, None if OK.
        """
        if not scratchpad:
            return None

        # Check 1: Consecutive identical calls
        msg = self._check_consecutive(scratchpad)
        if msg:
            return msg

        # Check 2: Total repeats
        msg = self._check_total_repeats(scratchpad)
        if msg:
            return msg

        # Check 3: Token budget
        msg = self._check_token_budget(scratchpad)
        if msg:
            return msg

        return None

    def _make_key(self, entry: dict) -> str:
        """Create a hashable key from a scratchpad entry's tool call."""
        tool_name = entry.get("tool_name", "")
        tool_input = entry.get("tool_input", {})
        return f"{tool_name}:{json.dumps(tool_input, sort_keys=True)}"

    def _check_consecutive(self, scratchpad: list[dict]) -> str | None:
        """Check for N consecutive identical tool calls."""
        if len(scratchpad) < self._max_consecutive_same:
            return None

        tail = scratchpad[-self._max_consecutive_same :]
        keys = [self._make_key(e) for e in tail]
        if len(set(keys)) == 1:
            tool_name = tail[0].get("tool_name", "unknown")
            return f"Agent stuck: called '{tool_name}' with same input {self._max_consecutive_same} times consecutively"
        return None

    def _check_total_repeats(self, scratchpad: list[dict]) -> str | None:
        """Check if any tool+input pair appears N times total."""
        counts = Counter(self._make_key(e) for e in scratchpad)
        for key, count in counts.items():
            if count >= self._max_total_repeats:
                tool_name = key.split(":")[0]
                return f"Agent looping: repeated '{tool_name}' with same input {count} times"
        return None

    def _check_token_budget(self, scratchpad: list[dict]) -> str | None:
        """Check estimated token usage against budget."""
        total_chars = sum(
            len(str(e.get("reasoning", ""))) + len(str(e.get("tool_input", ""))) + len(str(e.get("observation", "")))
            for e in scratchpad
        )
        estimated_tokens = total_chars // 4
        if estimated_tokens > self._max_token_budget:
            return f"Token budget exceeded: ~{estimated_tokens} tokens used (budget: {self._max_token_budget})"
        return None
