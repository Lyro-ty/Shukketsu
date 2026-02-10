"""Tests for agent loop detection guardrails."""

from code.shukketsu.agents.guardrails import LoopDetector


def _entry(tool_name: str = "rag_search", tool_input: dict | None = None, observation: str = "result") -> dict:
    """Helper to build a scratchpad entry."""
    return {
        "reasoning": "thinking...",
        "tool_name": tool_name,
        "tool_input": tool_input or {"query": "hit cap"},
        "observation": observation,
    }


class TestLoopDetector:
    """Tests for the LoopDetector class."""

    def test_empty_scratchpad_returns_none(self) -> None:
        detector = LoopDetector()
        assert detector.check([]) is None

    def test_no_loop_returns_none(self) -> None:
        detector = LoopDetector()
        scratchpad = [
            _entry("rag_search", {"query": "hit cap"}),
            _entry("web_search", {"query": "rogue guide"}),
            _entry("rag_search", {"query": "combat talents"}),
        ]
        assert detector.check(scratchpad) is None

    def test_consecutive_identical_calls_detected(self) -> None:
        detector = LoopDetector(max_consecutive_same=3)
        scratchpad = [
            _entry("rag_search", {"query": "hit cap"}),
            _entry("rag_search", {"query": "hit cap"}),
            _entry("rag_search", {"query": "hit cap"}),
        ]
        result = detector.check(scratchpad)
        assert result is not None
        assert "consecutive" in result.lower() or "stuck" in result.lower()

    def test_below_consecutive_threshold_ok(self) -> None:
        detector = LoopDetector(max_consecutive_same=3)
        scratchpad = [
            _entry("rag_search", {"query": "hit cap"}),
            _entry("rag_search", {"query": "hit cap"}),
        ]
        assert detector.check(scratchpad) is None

    def test_non_consecutive_repeats_detected(self) -> None:
        detector = LoopDetector(max_total_repeats=3)
        scratchpad = [
            _entry("rag_search", {"query": "hit cap"}),
            _entry("web_search", {"query": "rogue guide"}),
            _entry("rag_search", {"query": "hit cap"}),
            _entry("web_search", {"query": "other"}),
            _entry("rag_search", {"query": "hit cap"}),
        ]
        result = detector.check(scratchpad)
        assert result is not None
        assert "repeat" in result.lower() or "loop" in result.lower()

    def test_different_inputs_not_flagged(self) -> None:
        detector = LoopDetector(max_consecutive_same=2, max_total_repeats=2)
        scratchpad = [
            _entry("rag_search", {"query": "hit cap"}),
            _entry("rag_search", {"query": "combat talents"}),
            _entry("rag_search", {"query": "sword spec"}),
        ]
        assert detector.check(scratchpad) is None

    def test_token_budget_exceeded(self) -> None:
        detector = LoopDetector(max_token_budget=100)
        long_observation = "x" * 500
        scratchpad = [_entry(observation=long_observation)]
        result = detector.check(scratchpad)
        assert result is not None
        assert "token" in result.lower() or "budget" in result.lower()

    def test_token_budget_ok_when_under_limit(self) -> None:
        detector = LoopDetector(max_token_budget=100_000)
        scratchpad = [_entry()]
        assert detector.check(scratchpad) is None
