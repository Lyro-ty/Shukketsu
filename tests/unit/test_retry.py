"""Tests for the exponential backoff retry decorator."""

import logging

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
