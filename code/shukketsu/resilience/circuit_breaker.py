"""Circuit breaker pattern for external service protection."""

import asyncio
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
        self._half_open_lock = asyncio.Lock()

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

        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_lock.locked():
                raise CircuitOpenError(self.name)
            async with self._half_open_lock:
                return await self._execute(fn, *args, **kwargs)

        return await self._execute(fn, *args, **kwargs)

    async def _execute(self, fn: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """Execute fn and record success/failure."""
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


def reset_all_breakers() -> None:
    """Reset all named circuit breakers to CLOSED. Used by test fixtures."""
    for breaker in (vllm_breaker, ollama_router_breaker, ollama_embed_breaker, brave_breaker):
        breaker.reset()
