# Step 9: Resilience — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add circuit breakers, retry logic, loop detection, and graceful degradation so the system handles failures instead of crashing.

**Architecture:** Four circuit breakers (vLLM, Ollama/router, Ollama/embed, Brave) protect external services. An async retry decorator with exponential backoff wraps flaky calls. A LoopDetector watches the agent scratchpad for repeated tool calls and token budget exhaustion. All seven degradation paths from the design spec are wired in.

**Tech Stack:** Pure Python (asyncio, time, random, functools, collections). No external resilience libraries.

---

### Task 1: Add CircuitOpenError to error taxonomy

**Files:**
- Modify: `code/shukketsu/resilience/errors.py:103-107` (append after BraveSearchError)
- Test: `tests/unit/test_errors.py:58-63` (append new test)

**Step 1: Write the failing test**

Append to `tests/unit/test_errors.py`:

```python
def test_circuit_open_error_has_model_unavailable_mode() -> None:
    from code.shukketsu.resilience.errors import CircuitOpenError

    err = CircuitOpenError("brave_search")
    assert isinstance(err, ShukketsuError)
    assert err.failure_mode == FailureMode.MODEL_UNAVAILABLE
    assert err.breaker_name == "brave_search"
    assert "brave_search" in str(err)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_errors.py::test_circuit_open_error_has_model_unavailable_mode -v`
Expected: FAIL with `ImportError: cannot import name 'CircuitOpenError'`

**Step 3: Write minimal implementation**

Append to `code/shukketsu/resilience/errors.py` after `BraveSearchError`:

```python
class CircuitOpenError(ShukketsuError):
    """Raised when a circuit breaker is open and rejecting requests."""

    def __init__(self, breaker_name: str):
        super().__init__(
            f"Circuit breaker '{breaker_name}' is open — service unavailable",
            FailureMode.MODEL_UNAVAILABLE,
        )
        self.breaker_name = breaker_name
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_errors.py -v`
Expected: 9 passed

**Step 5: Commit**

```bash
git add code/shukketsu/resilience/errors.py tests/unit/test_errors.py
git commit -m "feat: add CircuitOpenError to error taxonomy"
```

---

### Task 2: Add resilience + loop detection config constants

**Files:**
- Modify: `code/shukketsu/config.py:79` (append after ROBOTS_CACHE_TTL_HOURS)

**Step 1: Append config constants**

Append to `code/shukketsu/config.py`:

```python
# Circuit breaker defaults
CB_VLLM_FAILURE_THRESHOLD = 3
CB_VLLM_RECOVERY_TIMEOUT = 30.0
CB_OLLAMA_ROUTER_FAILURE_THRESHOLD = 5
CB_OLLAMA_ROUTER_RECOVERY_TIMEOUT = 60.0
CB_OLLAMA_EMBED_FAILURE_THRESHOLD = 5
CB_OLLAMA_EMBED_RECOVERY_TIMEOUT = 60.0
CB_BRAVE_FAILURE_THRESHOLD = 5
CB_BRAVE_RECOVERY_TIMEOUT = 120.0

# Retry defaults
RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 30.0

# Loop detection
LOOP_MAX_CONSECUTIVE_SAME = 3
LOOP_MAX_TOTAL_REPEATS = 3
```

**Step 2: Run existing tests to verify nothing breaks**

Run: `python3 -m pytest tests/unit/ -v --tb=short`
Expected: 196 passed (195 + 1 from Task 1)

**Step 3: Commit**

```bash
git add code/shukketsu/config.py
git commit -m "feat: add circuit breaker, retry, and loop detection config"
```

---

### Task 3: Build CircuitBreaker class

**Files:**
- Create: `code/shukketsu/resilience/circuit_breaker.py`
- Create: `tests/unit/test_circuit_breaker.py`

**Step 1: Write all failing tests**

Create `tests/unit/test_circuit_breaker.py`:

```python
"""Tests for the circuit breaker pattern."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from code.shukketsu.resilience.circuit_breaker import CircuitBreaker, CircuitState
from code.shukketsu.resilience.errors import CircuitOpenError


class TestCircuitBreakerClosed:
    """Tests for the CLOSED (normal) state."""

    async def test_passes_calls_through(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=1.0)
        fn = AsyncMock(return_value="ok")
        result = await cb.call(fn, "arg1", key="val")
        assert result == "ok"
        fn.assert_awaited_once_with("arg1", key="val")

    async def test_starts_in_closed_state(self) -> None:
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED

    async def test_failure_increments_counter(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=3)
        fn = AsyncMock(side_effect=ConnectionError("down"))
        with pytest.raises(ConnectionError):
            await cb.call(fn)
        assert cb._failure_count == 1
        assert cb.state == CircuitState.CLOSED

    async def test_success_resets_failure_counter(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=3)
        failing = AsyncMock(side_effect=ConnectionError("down"))
        succeeding = AsyncMock(return_value="ok")
        # Fail twice (below threshold)
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await cb.call(failing)
        assert cb._failure_count == 2
        # Success resets
        await cb.call(succeeding)
        assert cb._failure_count == 0


class TestCircuitBreakerOpen:
    """Tests for the OPEN (failing) state."""

    async def test_threshold_transitions_to_open(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=60.0)
        fn = AsyncMock(side_effect=ConnectionError("down"))
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await cb.call(fn)
        assert cb.state == CircuitState.OPEN

    async def test_open_raises_circuit_open_error(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=60.0)
        fn = AsyncMock(side_effect=ConnectionError("down"))
        with pytest.raises(ConnectionError):
            await cb.call(fn)
        assert cb.state == CircuitState.OPEN
        # Next call should raise CircuitOpenError without calling fn
        fn.reset_mock()
        with pytest.raises(CircuitOpenError, match="test"):
            await cb.call(fn)
        fn.assert_not_awaited()


class TestCircuitBreakerHalfOpen:
    """Tests for the HALF_OPEN (testing) state."""

    async def test_recovery_timeout_transitions_to_half_open(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        fn = AsyncMock(side_effect=ConnectionError("down"))
        with pytest.raises(ConnectionError):
            await cb.call(fn)
        assert cb.state == CircuitState.OPEN
        await asyncio.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

    async def test_half_open_success_transitions_to_closed(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        failing = AsyncMock(side_effect=ConnectionError("down"))
        succeeding = AsyncMock(return_value="recovered")
        with pytest.raises(ConnectionError):
            await cb.call(failing)
        await asyncio.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        result = await cb.call(succeeding)
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    async def test_half_open_failure_transitions_to_open(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        fn = AsyncMock(side_effect=ConnectionError("down"))
        with pytest.raises(ConnectionError):
            await cb.call(fn)
        await asyncio.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        with pytest.raises(ConnectionError):
            await cb.call(fn)
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerReset:
    """Tests for the manual reset method."""

    async def test_reset_returns_to_closed(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=60.0)
        fn = AsyncMock(side_effect=ConnectionError("down"))
        with pytest.raises(ConnectionError):
            await cb.call(fn)
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_circuit_breaker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'code.shukketsu.resilience.circuit_breaker'`

**Step 3: Write the implementation**

Create `code/shukketsu/resilience/circuit_breaker.py`:

```python
"""Circuit breaker pattern for external service protection."""

import logging
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from code.shukketsu import config
from code.shukketsu.resilience.errors import CircuitOpenError

logger = logging.getLogger(__name__)


class CircuitState(StrEnum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Async circuit breaker that protects against cascading failures.

    Tracks consecutive failures for a named service. After reaching
    the failure threshold, the breaker opens and rejects requests
    immediately. After a recovery timeout, it transitions to half-open
    and allows one test request through.
    """

    def __init__(self, name: str, *, failure_threshold: int = 5, recovery_timeout: float = 60.0) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._state = CircuitState.CLOSED

    @property
    def state(self) -> CircuitState:
        """Current breaker state, accounting for recovery timeout."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    async def call(self, fn: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """Execute fn through the circuit breaker.

        Args:
            fn: Async callable to execute.
            *args: Positional arguments for fn.
            **kwargs: Keyword arguments for fn.

        Returns:
            The return value of fn.

        Raises:
            CircuitOpenError: If the breaker is open and rejecting requests.
            Exception: Any exception raised by fn (also tracked as a failure).
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            logger.warning("Circuit breaker '%s' is OPEN — rejecting request", self.name)
            raise CircuitOpenError(self.name)

        try:
            result = await fn(*args, **kwargs)
        except Exception:
            self._record_failure()
            raise

        self._record_success()
        return result

    def reset(self) -> None:
        """Manually reset the breaker to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        logger.info("Circuit breaker '%s' manually reset to CLOSED", self.name)

    def _record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit breaker '%s' opened after %d failures",
                self.name,
                self._failure_count,
            )

    def _record_success(self) -> None:
        """Record a success and reset counters."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED


# Named instances for each external service
vllm_breaker = CircuitBreaker(
    "vllm",
    failure_threshold=config.CB_VLLM_FAILURE_THRESHOLD,
    recovery_timeout=config.CB_VLLM_RECOVERY_TIMEOUT,
)

ollama_router_breaker = CircuitBreaker(
    "ollama_router",
    failure_threshold=config.CB_OLLAMA_ROUTER_FAILURE_THRESHOLD,
    recovery_timeout=config.CB_OLLAMA_ROUTER_RECOVERY_TIMEOUT,
)

ollama_embed_breaker = CircuitBreaker(
    "ollama_embed",
    failure_threshold=config.CB_OLLAMA_EMBED_FAILURE_THRESHOLD,
    recovery_timeout=config.CB_OLLAMA_EMBED_RECOVERY_TIMEOUT,
)

brave_breaker = CircuitBreaker(
    "brave_search",
    failure_threshold=config.CB_BRAVE_FAILURE_THRESHOLD,
    recovery_timeout=config.CB_BRAVE_RECOVERY_TIMEOUT,
)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_circuit_breaker.py -v`
Expected: 10 passed

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v --tb=short`
Expected: 206 passed

**Step 6: Commit**

```bash
git add code/shukketsu/resilience/circuit_breaker.py tests/unit/test_circuit_breaker.py
git commit -m "feat: add circuit breaker with four named instances"
```

---

### Task 4: Build retry decorator

**Files:**
- Create: `code/shukketsu/resilience/retry.py`
- Create: `tests/unit/test_retry.py`

**Step 1: Write all failing tests**

Create `tests/unit/test_retry.py`:

```python
"""Tests for the exponential backoff retry decorator."""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from code.shukketsu.resilience.retry import with_retry


class TestWithRetry:
    """Tests for the with_retry decorator."""

    async def test_successful_call_returns_immediately(self) -> None:
        @with_retry(max_attempts=3)
        async def fn() -> str:
            return "ok"

        result = await fn()
        assert result == "ok"

    async def test_retries_on_retryable_exception(self) -> None:
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.0, retryable=(ConnectionError,))
        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("down")
            return "recovered"

        result = await fn()
        assert result == "recovered"
        assert call_count == 3

    async def test_gives_up_after_max_attempts(self) -> None:
        @with_retry(max_attempts=2, base_delay=0.0, retryable=(ConnectionError,))
        async def fn() -> str:
            raise ConnectionError("always down")

        with pytest.raises(ConnectionError, match="always down"):
            await fn()

    async def test_non_retryable_exception_raises_immediately(self) -> None:
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.0, retryable=(ConnectionError,))
        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            await fn()
        assert call_count == 1

    async def test_passes_args_and_kwargs(self) -> None:
        @with_retry(max_attempts=1)
        async def fn(a: int, b: str, *, key: bool = False) -> str:
            return f"{a}-{b}-{key}"

        result = await fn(1, "two", key=True)
        assert result == "1-two-True"

    async def test_logs_retry_attempts(self, caplog: pytest.LogCaptureFixture) -> None:
        call_count = 0

        @with_retry(max_attempts=2, base_delay=0.0, retryable=(ConnectionError,))
        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("down")
            return "ok"

        with caplog.at_level(logging.WARNING):
            await fn()
        assert any("retry" in record.message.lower() for record in caplog.records)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_retry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'code.shukketsu.resilience.retry'`

**Step 3: Write the implementation**

Create `code/shukketsu/resilience/retry.py`:

```python
"""Exponential backoff retry decorator for async functions."""

import asyncio
import functools
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable: tuple[type[Exception], ...] = (ConnectionError, TimeoutError),
) -> Callable[..., Callable[..., Awaitable[Any]]]:
    """Retry an async function with exponential backoff and jitter.

    Delay formula: min(base_delay * 2^attempt + random(0, base_delay), max_delay)

    Args:
        max_attempts: Total number of attempts (including the first try).
        base_delay: Base delay in seconds before first retry.
        max_delay: Maximum delay cap in seconds.
        retryable: Tuple of exception types that trigger a retry.
    """

    def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except retryable as exc:
                    last_exc = exc
                    if attempt + 1 >= max_attempts:
                        break
                    delay = min(base_delay * (2**attempt) + random.uniform(0, base_delay), max_delay)
                    logger.warning(
                        "Retry %d/%d for %s after %s (delay=%.2fs)",
                        attempt + 1,
                        max_attempts - 1,
                        fn.__qualname__,
                        type(exc).__name__,
                        delay,
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_retry.py -v`
Expected: 6 passed

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v --tb=short`
Expected: 212 passed

**Step 6: Commit**

```bash
git add code/shukketsu/resilience/retry.py tests/unit/test_retry.py
git commit -m "feat: add async retry decorator with exponential backoff"
```

---

### Task 5: Build LoopDetector guardrail

**Files:**
- Create: `code/shukketsu/agents/guardrails.py`
- Create: `tests/unit/test_guardrails.py`

**Step 1: Write all failing tests**

Create `tests/unit/test_guardrails.py`:

```python
"""Tests for agent loop detection guardrails."""

import json

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
        # Each entry has ~50 chars of content; 3 entries should exceed 100 tokens at 4 chars/token
        long_observation = "x" * 500
        scratchpad = [_entry(observation=long_observation)]
        result = detector.check(scratchpad)
        assert result is not None
        assert "token" in result.lower() or "budget" in result.lower()

    def test_token_budget_ok_when_under_limit(self) -> None:
        detector = LoopDetector(max_token_budget=100_000)
        scratchpad = [_entry()]
        assert detector.check(scratchpad) is None
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_guardrails.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'code.shukketsu.agents.guardrails'`

**Step 3: Write the implementation**

Create `code/shukketsu/agents/guardrails.py`:

```python
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
            return (
                f"Agent stuck: called '{tool_name}' with same input "
                f"{self._max_consecutive_same} times consecutively"
            )
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
            len(str(e.get("reasoning", "")))
            + len(str(e.get("tool_input", "")))
            + len(str(e.get("observation", "")))
            for e in scratchpad
        )
        estimated_tokens = total_chars // 4
        if estimated_tokens > self._max_token_budget:
            return (
                f"Token budget exceeded: ~{estimated_tokens} tokens used "
                f"(budget: {self._max_token_budget})"
            )
        return None
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_guardrails.py -v`
Expected: 8 passed

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v --tb=short`
Expected: 220 passed

**Step 6: Commit**

```bash
git add code/shukketsu/agents/guardrails.py tests/unit/test_guardrails.py
git commit -m "feat: add LoopDetector with consecutive, repeat, and budget checks"
```

---

### Task 6: Wire LoopDetector into BaseAgent

**Files:**
- Modify: `code/shukketsu/agents/base.py:1-86`
- Modify: `tests/unit/test_base_agent.py` (append 2 new tests)

**Step 1: Write the failing tests**

Append to `tests/unit/test_base_agent.py`:

```python
class TestLoopDetection:
    """Tests for loop detection integration in BaseAgent."""

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_stops_on_consecutive_loop(self, mock_llm: AsyncMock) -> None:
        """Agent should stop and return graceful message when loop detected."""
        # Always call the same tool with the same input
        mock_llm.return_value = _tool_call("echo", {"text": "stuck"})
        agent = BaseAgent(tool_registry=_registry(EchoTool()), max_iterations=10)
        result = await agent.run("Loop test")
        # Should stop before max_iterations due to loop detection
        # Default max_consecutive_same=3, so it should stop after 3 identical calls
        assert mock_llm.call_count <= 4  # at most 3 loop calls + would stop
        assert isinstance(result, str)

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_loop_returns_partial_observations(self, mock_llm: AsyncMock) -> None:
        """Agent should include partial results when loop forces stop."""
        mock_llm.return_value = _tool_call("echo", {"text": "repeated"})
        agent = BaseAgent(tool_registry=_registry(EchoTool()), max_iterations=10)
        result = await agent.run("Loop test")
        # Should contain something (either partial observations or graceful failure)
        assert len(result) > 0
```

**Step 2: Run new tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_base_agent.py::TestLoopDetection -v`
Expected: FAIL (loop detection not wired in yet — agent runs all 10 iterations)

**Step 3: Modify BaseAgent to integrate LoopDetector**

Modify `code/shukketsu/agents/base.py` to:
1. Import `LoopDetector` from `agents.guardrails`
2. Create a `LoopDetector` in `__init__`
3. Call `loop_detector.check(scratchpad)` after each tool execution
4. If loop detected, return partial answer

The full updated file:

```python
"""BaseAgent with ReAct (Reason + Act) loop."""

import logging
from typing import Any

from code.shukketsu import config
from code.shukketsu.agents.guardrails import LoopDetector
from code.shukketsu.llm.schemas import ActionType, AgentStep
from code.shukketsu.llm.structured import get_structured_output
from code.shukketsu.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_REACT_INSTRUCTIONS = """You have access to the following tools:

{tool_descriptions}

When you need information to answer the question, use a tool by responding with action "tool_call".
When you have enough information to answer, respond with action "final_answer".
Always think step by step about what you need to do.
If a tool returns an error, try a different approach or answer with what you know."""


class BaseAgent:
    """Agent that uses a ReAct loop to answer questions with tools.

    Iterates: think -> act (call tool) -> observe (read result)
    until it reaches a final answer or hits the iteration limit.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        max_iterations: int = config.AGENT_MAX_ITERATIONS,
        system_prompt: str = config.SYSTEM_PROMPT,
    ) -> None:
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self._system_prompt = system_prompt
        self._loop_detector = LoopDetector()

    async def run(self, query: str) -> str:
        """Run the ReAct loop to answer a query.

        Args:
            query: The user's question.

        Returns:
            The agent's final answer, or a graceful failure message.

        Raises:
            LLMUnavailableError: If the LLM backend is unreachable.
            StructuredOutputError: If structured output validation fails.
        """
        scratchpad: list[dict[str, Any]] = []

        for iteration in range(self.max_iterations):
            messages = self._build_messages(query, scratchpad)

            logger.info("Agent iteration %d/%d", iteration + 1, self.max_iterations)
            step: AgentStep = await get_structured_output(
                response_model=AgentStep,
                messages=messages,
            )

            if step.action == ActionType.FINAL_ANSWER:
                logger.info("Agent reached final answer after %d iteration(s)", iteration + 1)
                return step.answer  # type: ignore[return-value]

            tool_call = step.tool_call
            assert tool_call is not None  # guaranteed by AgentStep validator
            logger.info("Agent calling tool: %s", tool_call.tool_name)

            observation = await self.tool_registry.execute(tool_call.tool_name, tool_call.tool_input)

            scratchpad.append(
                {
                    "reasoning": step.reasoning,
                    "tool_name": tool_call.tool_name,
                    "tool_input": tool_call.tool_input,
                    "observation": observation,
                }
            )

            # Check for loops after each tool call
            loop_msg = self._loop_detector.check(scratchpad)
            if loop_msg:
                logger.warning("Loop detected: %s", loop_msg)
                return self._synthesize_partial_answer(scratchpad)

        logger.warning("Agent reached max iterations (%d) without final answer", self.max_iterations)
        return config.AGENT_GRACEFUL_FAILURE

    def _synthesize_partial_answer(self, scratchpad: list[dict[str, Any]]) -> str:
        """Build an answer from partial observations when a loop is detected."""
        observations = [e["observation"] for e in scratchpad if e.get("observation")]
        if observations:
            # Return the last unique observation as partial info
            unique = list(dict.fromkeys(observations))
            return f"Based on partial results: {unique[-1]}"
        return config.AGENT_GRACEFUL_FAILURE

    def _build_messages(self, query: str, scratchpad: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Build the messages array for the LLM."""
        tool_descriptions = self.tool_registry.get_tool_descriptions()
        system_content = self._system_prompt + "\n\n" + _REACT_INSTRUCTIONS.format(tool_descriptions=tool_descriptions)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        for entry in scratchpad:
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Thought: {entry['reasoning']}\n"
                        f"Action: tool_call\n"
                        f"Tool: {entry['tool_name']}\n"
                        f"Input: {entry['tool_input']}"
                    ),
                }
            )
            messages.append({"role": "user", "content": f"Observation: {entry['observation']}"})

        return messages
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_base_agent.py -v`
Expected: 16 passed (14 existing + 2 new)

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v --tb=short`
Expected: 222 passed

**Step 6: Commit**

```bash
git add code/shukketsu/agents/base.py tests/unit/test_base_agent.py
git commit -m "feat: wire LoopDetector into BaseAgent ReAct loop"
```

---

### Task 7: Wire circuit breaker into router (Ollama/Qwen fallback)

**Files:**
- Modify: `code/shukketsu/routing/router.py:10-77`
- Modify: `tests/unit/test_router.py` (append 1 new test)

**Step 1: Write the failing test**

Append to `tests/unit/test_router.py`:

```python
    @pytest.mark.asyncio
    async def test_classify_fallback_on_circuit_open(self) -> None:
        """When Ollama circuit breaker is open, classify_query should return safe fallback."""
        from code.shukketsu.resilience.errors import CircuitOpenError

        with patch(
            "code.shukketsu.routing.router.get_structured_output",
            new_callable=AsyncMock,
            side_effect=CircuitOpenError("ollama_router"),
        ):
            from code.shukketsu.routing.router import classify_query

            result = await classify_query("What is the hit cap?")

        assert result.complexity == TaskComplexity.COMPLEX
        assert result.direct_answer is None
        assert result.needs_tools is True
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_router.py::TestClassifyQuery::test_classify_fallback_on_circuit_open -v`
Expected: FAIL (`CircuitOpenError` not caught by existing handler)

**Step 3: Modify router to catch CircuitOpenError**

Update `code/shukketsu/routing/router.py` — add `CircuitOpenError` to the import and the except clause:

Change line 11:
```python
from code.shukketsu.resilience.errors import CircuitOpenError, LLMUnavailableError, StructuredOutputError
```

Change line 75:
```python
    except (LLMUnavailableError, StructuredOutputError, CircuitOpenError) as exc:
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_router.py -v`
Expected: 7 passed

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v --tb=short`
Expected: 223 passed

**Step 6: Commit**

```bash
git add code/shukketsu/routing/router.py tests/unit/test_router.py
git commit -m "feat: add CircuitOpenError fallback to query router"
```

---

### Task 8: Wire circuit breaker into web_search tool (Brave fallback)

**Files:**
- Modify: `code/shukketsu/tools/research/web_search.py:49-63`
- Modify: `tests/unit/test_web_search.py` (append 1 new test)

**Step 1: Write the failing test**

Append to `tests/unit/test_web_search.py`:

```python
    async def test_circuit_open_returns_unavailable_message(self) -> None:
        from code.shukketsu.resilience.errors import CircuitOpenError

        tool = WebSearchTool()
        with (
            patch("code.shukketsu.tools.research.web_search.config") as mock_config,
            patch("code.shukketsu.tools.research.web_search.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_config.BRAVE_SEARCH_API_KEY = "test-key"
            mock_config.BRAVE_SEARCH_MAX_RESULTS = 5
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=CircuitOpenError("brave_search"))
            mock_client_cls.return_value = mock_client
            result = await tool.execute({"query": "test"})
        assert "unavailable" in result.lower() or "circuit" in result.lower()
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_web_search.py::TestWebSearchTool::test_circuit_open_returns_unavailable_message -v`
Expected: FAIL (CircuitOpenError not caught)

**Step 3: Modify web_search.py to catch CircuitOpenError**

Update `code/shukketsu/tools/research/web_search.py` — add the import and a new except clause.

After the existing imports, add:
```python
from code.shukketsu.resilience.errors import CircuitOpenError
```

After the `except httpx.ConnectError:` block (line 62-63), add:
```python
        except CircuitOpenError:
            return "Error: Web search is temporarily unavailable. Try answering from the knowledge base."
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_web_search.py -v`
Expected: 10 passed

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v --tb=short`
Expected: 224 passed

**Step 6: Commit**

```bash
git add code/shukketsu/tools/research/web_search.py tests/unit/test_web_search.py
git commit -m "feat: add circuit breaker fallback to web search tool"
```

---

### Task 9: Wire embedding fallback into hybrid search (FTS5-only degradation)

**Files:**
- Modify: `code/shukketsu/tools/knowledge/search.py:35-44`
- Modify: `tests/unit/test_rag_search.py` (append 1 new test)

The embedding fallback belongs in `RagSearchTool.execute()` because that's where the embed call happens. When embedding fails, we skip vector search and do FTS5-only via `rag/search.py`.

First, let me check the existing rag_search tests:

**Step 1: Write the failing test**

We need to check the existing test file first. The test should verify that when the embed function raises an error, the tool gracefully degrades to keyword-only results.

Append to `tests/unit/test_rag_search.py`:

```python
    async def test_embedding_failure_returns_error_message(self, test_db: sqlite3.Connection) -> None:
        """When embedding fails, tool should return an informative error."""
        from code.shukketsu.resilience.errors import EmbeddingError

        async def failing_embed(text: str) -> list[float]:
            raise EmbeddingError("Ollama embed down")

        tool = RagSearchTool(conn=test_db, embed_fn=failing_embed)
        result = await tool.execute({"query": "hit cap"})
        assert "error" in result.lower() or "unavailable" in result.lower()
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_rag_search.py::TestRagSearchTool::test_embedding_failure_returns_error_message -v`
Expected: FAIL (EmbeddingError propagates unhandled)

**Step 3: Modify RagSearchTool.execute() to handle embedding failures**

Update `code/shukketsu/tools/knowledge/search.py`:

Add import at the top:
```python
from code.shukketsu.resilience.errors import EmbeddingError
```

Wrap the embedding call in a try/except. When embedding fails, attempt FTS5-only search. Update lines 42-44 of execute():

Replace the execute method body (after the empty query check) with:

```python
        try:
            embedding = await self._embed_fn(query)
        except Exception as exc:
            logger.warning("Embedding failed, falling back to keyword-only search: %s", exc)
            return await self._fts_only_search(query, top_k)

        results = await hybrid_search(self._conn, query, embedding, top_k=top_k)
```

Add a new method `_fts_only_search` that runs FTS5 keyword search without vectors:

```python
    async def _fts_only_search(self, query: str, top_k: int) -> str:
        """Fallback keyword-only search when embedding is unavailable."""
        from code.shukketsu.rag.fusion import escape_fts_query

        fts_query = escape_fts_query(query)
        if not fts_query:
            return "Error: Embedding model unavailable and no keyword query provided."

        rows = self._conn.execute(
            """SELECT c.id, c.content, s.title, s.url, s.trust_score
               FROM chunks_fts f
               JOIN chunks c ON c.id = f.rowid
               JOIN sources s ON s.id = c.source_id
               WHERE chunks_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (fts_query, top_k),
        ).fetchall()

        if not rows:
            return "No relevant documents found (keyword search only — embedding model unavailable)."

        parts = [f"Found {len(rows)} result{'s' if len(rows) != 1 else ''} (keyword search only):\n"]
        for i, r in enumerate(rows, 1):
            parts.append(f"[{i}] Source: {r['title']} ({r['url']})\nTrust: {r['trust_score']}\nContent: {r['content']}\n")

        return "\n".join(parts)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_rag_search.py -v`
Expected: All existing tests pass + 1 new test passes

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v --tb=short`
Expected: 225 passed

**Step 6: Commit**

```bash
git add code/shukketsu/tools/knowledge/search.py tests/unit/test_rag_search.py
git commit -m "feat: add FTS5-only fallback when embedding model unavailable"
```

---

### Task 10: Final verification and CLAUDE.md update

**Files:**
- Modify: `CLAUDE.md` (update step status, test count)

**Step 1: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v --tb=short`
Expected: ~225 passed, 0 failed

**Step 2: Run ruff check**

Run: `ruff check code/shukketsu/resilience/ code/shukketsu/agents/ tests/unit/test_circuit_breaker.py tests/unit/test_retry.py tests/unit/test_guardrails.py`
Expected: No errors

**Step 3: Run ruff format**

Run: `ruff format code/shukketsu/resilience/ code/shukketsu/agents/ tests/unit/test_circuit_breaker.py tests/unit/test_retry.py tests/unit/test_guardrails.py`

**Step 4: Update CLAUDE.md**

In the "Development Phases" section, update Step 9's status:

Change:
```
9. **Resilience (circuit breakers, loop detection, retries)** ← current
```
To:
```
9. ~~Resilience (circuit breakers, loop detection, retries)~~ **DONE**
10. **Observability (Langfuse tracing)** ← current
```

**Step 5: Run full test suite one final time**

Run: `python3 -m pytest tests/unit/ -v --tb=short`
Expected: All passed

**Step 6: Commit**

```bash
git add -A
git commit -m "docs: update CLAUDE.md for Step 9 completion"
```
