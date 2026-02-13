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
            assert last_exc is not None, "retry loop must execute at least once"
            raise last_exc

        return wrapper

    return decorator
