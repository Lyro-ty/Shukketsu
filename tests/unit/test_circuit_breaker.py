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


class TestCircuitBreakerConcurrency:
    """Tests for async-safety of the circuit breaker."""

    async def test_half_open_allows_only_one_concurrent_request(self) -> None:
        """In HALF_OPEN, only one request should pass; others get CircuitOpenError."""
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        # Trip the breaker
        fn = AsyncMock(side_effect=ConnectionError("down"))
        with pytest.raises(ConnectionError):
            await cb.call(fn)
        await asyncio.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

        # A slow function that simulates work
        async def slow_fn():
            await asyncio.sleep(0.05)
            return "ok"

        # Launch two concurrent calls
        tasks = [asyncio.create_task(cb.call(slow_fn)) for _ in range(2)]
        done = await asyncio.gather(*tasks, return_exceptions=True)

        # Exactly one should succeed, the other should get CircuitOpenError
        successes = [r for r in done if r == "ok"]
        errors = [r for r in done if isinstance(r, CircuitOpenError)]
        assert len(successes) == 1
        assert len(errors) == 1


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
