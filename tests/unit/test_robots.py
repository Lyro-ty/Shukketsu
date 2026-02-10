"""Tests for robots.txt compliance checker."""

import time
from unittest.mock import AsyncMock, patch

import httpx

from code.shukketsu.scraping.robots import RobotsChecker

ROBOTS_DISALLOW_ALL = "User-agent: *\nDisallow: /"
ROBOTS_ALLOW_ALL = "User-agent: *\nAllow: /"
ROBOTS_DISALLOW_PRIVATE = "User-agent: *\nDisallow: /private/"


def _mock_response(text: str, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response."""
    return httpx.Response(status_code=status_code, text=text, request=httpx.Request("GET", "http://test"))


class TestRobotsChecker:
    """Tests for RobotsChecker."""

    async def test_allowed_url_returns_true(self) -> None:
        checker = RobotsChecker(cache_ttl_hours=1)
        with patch.object(checker, "_fetch_robots", new_callable=AsyncMock, return_value=ROBOTS_ALLOW_ALL):
            result = await checker.is_allowed("https://example.com/page")
            assert result is True

    async def test_disallowed_url_returns_false(self) -> None:
        checker = RobotsChecker(cache_ttl_hours=1)
        with patch.object(checker, "_fetch_robots", new_callable=AsyncMock, return_value=ROBOTS_DISALLOW_ALL):
            result = await checker.is_allowed("https://example.com/page")
            assert result is False

    async def test_partially_disallowed(self) -> None:
        checker = RobotsChecker(cache_ttl_hours=1)
        with patch.object(checker, "_fetch_robots", new_callable=AsyncMock, return_value=ROBOTS_DISALLOW_PRIVATE):
            assert await checker.is_allowed("https://example.com/public") is True
            assert await checker.is_allowed("https://example.com/private/data") is False

    async def test_missing_robots_defaults_to_allowed(self) -> None:
        checker = RobotsChecker(cache_ttl_hours=1)
        with patch.object(checker, "_fetch_robots", new_callable=AsyncMock, return_value=None):
            result = await checker.is_allowed("https://example.com/page")
            assert result is True

    async def test_cache_hit_skips_fetch(self) -> None:
        checker = RobotsChecker(cache_ttl_hours=1)
        mock_fetch = AsyncMock(return_value=ROBOTS_ALLOW_ALL)
        with patch.object(checker, "_fetch_robots", mock_fetch):
            await checker.is_allowed("https://example.com/page1")
            await checker.is_allowed("https://example.com/page2")
            # Same domain — should only fetch once
            mock_fetch.assert_called_once()

    async def test_expired_cache_refetches(self) -> None:
        checker = RobotsChecker(cache_ttl_hours=1)
        mock_fetch = AsyncMock(return_value=ROBOTS_ALLOW_ALL)
        with patch.object(checker, "_fetch_robots", mock_fetch):
            await checker.is_allowed("https://example.com/page")
            # Expire the cache manually
            domain = "example.com"
            checker._cache[domain] = (checker._cache[domain][0], time.monotonic() - 7200)
            await checker.is_allowed("https://example.com/page2")
            assert mock_fetch.call_count == 2
