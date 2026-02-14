"""Web page fetcher with rate limiting and robots.txt compliance."""

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from code.shukketsu import config
from code.shukketsu.resilience.errors import RobotsDisallowedError, ScrapingError
from code.shukketsu.scraping.rate_limiter import RateLimiter
from code.shukketsu.scraping.robots import RobotsChecker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchResult:
    """Result of fetching a web page."""

    html: str
    status_code: int
    final_url: str


class WebFetcher:
    """Fetch web pages with rate limiting and robots.txt compliance.

    Composes RateLimiter and RobotsChecker so callers don't need to
    manage politeness policies themselves. Uses a shared httpx client
    for connection pooling across requests.
    """

    def __init__(self, rate_limiter: RateLimiter, robots_checker: RobotsChecker) -> None:
        self._rate_limiter = rate_limiter
        self._robots_checker = robots_checker
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=config.SCRAPING_DEFAULT_TIMEOUT,
                follow_redirects=True,
                max_redirects=5,
                headers={"User-Agent": config.SCRAPING_USER_AGENT},
            )
        return self._client

    async def close(self) -> None:
        """Close the shared HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def fetch(self, url: str) -> FetchResult:
        """Fetch a URL, respecting robots.txt and rate limits.

        Args:
            url: The URL to fetch.

        Returns:
            FetchResult with HTML content, status code, and final URL.

        Raises:
            RobotsDisallowedError: If robots.txt blocks the URL.
            ScrapingError: If the fetch fails (HTTP error, timeout, non-HTML).
        """
        # 1. Check robots.txt
        if not await self._robots_checker.is_allowed(url):
            raise RobotsDisallowedError(f"Blocked by robots.txt: {url}")

        # 2. Rate limit
        domain = urlparse(url).netloc
        await self._rate_limiter.acquire(domain)

        # 3. Fetch
        try:
            client = self._get_client()
            response = await client.get(url)
        except httpx.TransportError as exc:
            raise ScrapingError(f"Transport error fetching {url}: {exc}") from exc

        # 4. Check status
        if response.status_code >= 400:
            raise ScrapingError(f"HTTP {response.status_code} fetching {url}")

        # 4b. Check Content-Length header (response already in memory; early exit
        # before content-type checking and further processing)
        cl_header = response.headers.get("content-length")
        if cl_header and cl_header.isdigit() and int(cl_header) > config.SCRAPING_MAX_RESPONSE_BYTES:
            raise ScrapingError(f"Content-Length {cl_header} exceeds limit for {url}")

        # 5. Check content type
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            raise ScrapingError(f"Not an HTML page ({content_type}): {url}")

        # 6. Check response size
        content_length = len(response.content)
        if content_length > config.SCRAPING_MAX_RESPONSE_BYTES:
            raise ScrapingError(
                f"Response too large ({content_length} bytes, max {config.SCRAPING_MAX_RESPONSE_BYTES}): {url}"
            )

        final_url = str(response.url)
        logger.info("Fetched %s (%d bytes, final: %s)", url, content_length, final_url)

        return FetchResult(html=response.text, status_code=response.status_code, final_url=final_url)
