"""Tests for per-domain rate limiter."""

import asyncio
import time

from code.shukketsu.scraping.rate_limiter import RateLimiter

DOMAIN_POLICIES = {
    "fast.example.com": 0.1,
    "__default__": 0.2,
}


class TestRateLimiter:
    """Tests for RateLimiter."""

    async def test_first_request_passes_immediately(self) -> None:
        limiter = RateLimiter(policies=DOMAIN_POLICIES)
        start = time.monotonic()
        await limiter.acquire("fast.example.com")
        elapsed = time.monotonic() - start
        assert elapsed < 0.05

    async def test_second_request_waits_for_delay(self) -> None:
        limiter = RateLimiter(policies=DOMAIN_POLICIES)
        await limiter.acquire("fast.example.com")
        start = time.monotonic()
        await limiter.acquire("fast.example.com")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.08  # 0.1s policy minus timing tolerance

    async def test_different_domains_independent(self) -> None:
        limiter = RateLimiter(policies=DOMAIN_POLICIES)
        await limiter.acquire("fast.example.com")
        start = time.monotonic()
        await limiter.acquire("other.example.com")
        elapsed = time.monotonic() - start
        assert elapsed < 0.05  # No wait — different domain

    async def test_default_policy_used_for_unknown_domain(self) -> None:
        limiter = RateLimiter(policies=DOMAIN_POLICIES)
        await limiter.acquire("unknown.example.com")
        start = time.monotonic()
        await limiter.acquire("unknown.example.com")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.15  # __default__ = 0.2s minus tolerance

    async def test_concurrent_same_domain_serialized(self) -> None:
        limiter = RateLimiter(policies=DOMAIN_POLICIES)
        start = time.monotonic()
        await asyncio.gather(
            limiter.acquire("fast.example.com"),
            limiter.acquire("fast.example.com"),
            limiter.acquire("fast.example.com"),
        )
        elapsed = time.monotonic() - start
        # 3 requests with 0.1s delay between: first is instant, next two wait
        assert elapsed >= 0.15

    async def test_get_delay_returns_policy_value(self) -> None:
        limiter = RateLimiter(policies=DOMAIN_POLICIES)
        assert limiter.get_delay("fast.example.com") == 0.1
        assert limiter.get_delay("other.example.com") == 0.2
