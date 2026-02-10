"""Per-domain async rate limiter."""

import asyncio
import logging
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_POLICIES: dict[str, float] = {
    "warcraftlogs.com": 2.0,
    "wowhead.com": 3.0,
    "silentshadows.net": 5.0,
    "__default__": 3.0,
}


class RateLimiter:
    """Enforces per-domain delay between HTTP requests.

    Each domain has a configured minimum delay (seconds) between requests.
    Concurrent calls to the same domain are serialized via asyncio.Lock.
    """

    def __init__(self, policies: dict[str, float] | None = None) -> None:
        self._policies = policies if policies is not None else DEFAULT_POLICIES
        self._last_request: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def get_delay(self, domain: str) -> float:
        """Return the configured delay for a domain."""
        return self._policies.get(domain, self._policies.get("__default__", 3.0))

    async def acquire(self, domain_or_url: str) -> None:
        """Wait until a request to this domain is allowed.

        Args:
            domain_or_url: A domain name or full URL.
        """
        domain = self._extract_domain(domain_or_url)

        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()

        async with self._locks[domain]:
            delay = self.get_delay(domain)
            last = self._last_request.get(domain, 0.0)
            elapsed = time.monotonic() - last
            remaining = delay - elapsed

            if remaining > 0:
                logger.debug("Rate limiting %s: sleeping %.2fs", domain, remaining)
                await asyncio.sleep(remaining)

            self._last_request[domain] = time.monotonic()

    @staticmethod
    def _extract_domain(domain_or_url: str) -> str:
        """Extract domain from a URL or return as-is if already a domain."""
        if "://" in domain_or_url:
            return urlparse(domain_or_url).netloc
        return domain_or_url
