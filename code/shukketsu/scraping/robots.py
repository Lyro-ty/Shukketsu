"""Robots.txt compliance checker with per-domain caching."""

import logging
import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from code.shukketsu import config

logger = logging.getLogger(__name__)


class RobotsChecker:
    """Check whether a URL is allowed by the site's robots.txt.

    Caches parsed robots.txt per domain for cache_ttl_hours.
    If robots.txt cannot be fetched (404, timeout), access is allowed
    by convention.
    """

    def __init__(self, cache_ttl_hours: int | None = None) -> None:
        ttl = cache_ttl_hours if cache_ttl_hours is not None else config.ROBOTS_CACHE_TTL_HOURS
        self._cache_ttl_seconds = ttl * 3600
        # Cache: domain -> (RobotFileParser, timestamp)
        self._cache: dict[str, tuple[RobotFileParser, float]] = {}

    async def is_allowed(self, url: str) -> bool:
        """Check whether our user-agent may fetch the given URL.

        Args:
            url: The full URL to check.

        Returns:
            True if allowed (or robots.txt unavailable), False if disallowed.
        """
        parsed = urlparse(url)
        domain = parsed.netloc

        parser = self._get_cached(domain)
        if parser is None:
            robots_text = await self._fetch_robots(domain, parsed.scheme)
            parser = RobotFileParser()
            if robots_text is not None:
                parser.parse(robots_text.splitlines())
            else:
                # No robots.txt → allow everything
                parser.allow_all = True
            self._cache[domain] = (parser, time.monotonic())

        return parser.can_fetch(config.SCRAPING_USER_AGENT, url)

    def _get_cached(self, domain: str) -> RobotFileParser | None:
        """Return cached parser if valid, None if missing or expired."""
        if domain not in self._cache:
            return None
        parser, cached_at = self._cache[domain]
        if time.monotonic() - cached_at > self._cache_ttl_seconds:
            return None
        return parser

    async def _fetch_robots(self, domain: str, scheme: str = "https") -> str | None:
        """Fetch robots.txt for a domain. Returns None on failure."""
        robots_url = f"{scheme}://{domain}/robots.txt"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(robots_url, headers={"User-Agent": config.SCRAPING_USER_AGENT})
                if response.status_code == 200:
                    return response.text
                logger.debug("robots.txt returned %d for %s", response.status_code, domain)
                return None
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            logger.debug("Failed to fetch robots.txt for %s: %s", domain, exc)
            return None
