# Step 8: Web Search + Ingest Tools — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give the agent two new tools — `web_search` (Brave API) and `web_ingest` (fetch + extract + ingest) — plus the scraping infrastructure they depend on.

**Architecture:** Three scraping modules (rate limiter, robots checker, fetcher) compose into a `WebFetcher` that handles polite crawling. Two tool classes follow the existing `Tool` ABC pattern. `WebIngestTool` delegates to the existing `IngestPipeline` for chunking/embedding/storage. Trafilatura extracts article content as markdown to preserve heading structure for the chunker.

**Tech Stack:** httpx (HTTP client), trafilatura (content extraction), urllib.robotparser (robots.txt), asyncio (rate limiting)

---

## Task 1: Add Error Types + Config Constants

**Files:**
- Modify: `code/shukketsu/resilience/errors.py`
- Modify: `code/shukketsu/config.py`
- Test: `tests/unit/test_errors.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_errors.py`:

```python
def test_scraping_error_has_http_error_mode() -> None:
    from code.shukketsu.resilience.errors import ScrapingError

    err = ScrapingError("timeout")
    assert err.failure_mode == FailureMode.HTTP_ERROR

def test_robots_disallowed_error_has_rate_limited_mode() -> None:
    from code.shukketsu.resilience.errors import RobotsDisallowedError

    err = RobotsDisallowedError("blocked by robots.txt")
    assert err.failure_mode == FailureMode.RATE_LIMITED

def test_brave_search_error_has_tool_error_mode() -> None:
    from code.shukketsu.resilience.errors import BraveSearchError

    err = BraveSearchError("api key missing")
    assert err.failure_mode == FailureMode.TOOL_EXECUTION_ERROR
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_errors.py::test_scraping_error_has_http_error_mode tests/unit/test_errors.py::test_robots_disallowed_error_has_rate_limited_mode tests/unit/test_errors.py::test_brave_search_error_has_tool_error_mode -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `code/shukketsu/resilience/errors.py`:

```python
class ScrapingError(ShukketsuError):
    """Raised when a web page cannot be fetched."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.HTTP_ERROR)


class RobotsDisallowedError(ShukketsuError):
    """Raised when robots.txt disallows access to a URL."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.RATE_LIMITED)


class BraveSearchError(ShukketsuError):
    """Raised when the Brave Search API call fails."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.TOOL_EXECUTION_ERROR)
```

Add to `code/shukketsu/config.py`:

```python
# Scraping
BRAVE_SEARCH_MAX_RESULTS = int(os.getenv("BRAVE_SEARCH_MAX_RESULTS", "5"))
SCRAPING_USER_AGENT = "Shukketsu/0.1 (research bot)"
SCRAPING_DEFAULT_TIMEOUT = 15.0
SCRAPING_MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB
ROBOTS_CACHE_TTL_HOURS = 24
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_errors.py -v`
Expected: All error tests PASS

**Step 5: Commit**

```bash
git add code/shukketsu/resilience/errors.py code/shukketsu/config.py tests/unit/test_errors.py
git commit -m "feat(step8): add scraping error types and config constants"
```

---

## Task 2: Rate Limiter

**Files:**
- Create: `code/shukketsu/scraping/rate_limiter.py`
- Create: `tests/unit/test_rate_limiter.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_rate_limiter.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_rate_limiter.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Create `code/shukketsu/scraping/rate_limiter.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_rate_limiter.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add code/shukketsu/scraping/rate_limiter.py tests/unit/test_rate_limiter.py
git commit -m "feat(step8): add per-domain async rate limiter"
```

---

## Task 3: Robots.txt Checker

**Files:**
- Create: `code/shukketsu/scraping/robots.py`
- Create: `tests/unit/test_robots.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_robots.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_robots.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Create `code/shukketsu/scraping/robots.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_robots.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add code/shukketsu/scraping/robots.py tests/unit/test_robots.py
git commit -m "feat(step8): add robots.txt compliance checker with caching"
```

---

## Task 4: Web Fetcher

**Files:**
- Create: `code/shukketsu/scraping/fetcher.py`
- Create: `tests/unit/test_fetcher.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_fetcher.py`:

```python
"""Tests for the web page fetcher."""

from unittest.mock import AsyncMock, patch

import httpx

from code.shukketsu.resilience.errors import RobotsDisallowedError, ScrapingError
from code.shukketsu.scraping.fetcher import FetchResult, WebFetcher


def _make_fetcher(robots_allowed: bool = True) -> WebFetcher:
    """Create a WebFetcher with mocked dependencies."""
    rate_limiter = AsyncMock()
    robots_checker = AsyncMock()
    robots_checker.is_allowed = AsyncMock(return_value=robots_allowed)
    return WebFetcher(rate_limiter=rate_limiter, robots_checker=robots_checker)


def _mock_response(
    text: str = "<html><body>Hello</body></html>",
    status_code: int = 200,
    url: str = "https://example.com/page",
    content_type: str = "text/html",
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        text=text,
        request=httpx.Request("GET", url),
        headers={"content-type": content_type, "content-length": str(len(text.encode()))},
    )


class TestWebFetcher:
    """Tests for WebFetcher."""

    async def test_successful_fetch(self) -> None:
        fetcher = _make_fetcher()
        mock_resp = _mock_response(text="<html><body>content</body></html>")
        with patch("code.shukketsu.scraping.fetcher.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client
            result = await fetcher.fetch("https://example.com/page")
        assert isinstance(result, FetchResult)
        assert "content" in result.html
        assert result.status_code == 200

    async def test_robots_disallowed_raises(self) -> None:
        fetcher = _make_fetcher(robots_allowed=False)
        try:
            await fetcher.fetch("https://example.com/page")
            assert False, "Should have raised"
        except RobotsDisallowedError:
            pass

    async def test_http_error_raises_scraping_error(self) -> None:
        fetcher = _make_fetcher()
        mock_resp = _mock_response(status_code=403)
        with patch("code.shukketsu.scraping.fetcher.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client
            try:
                await fetcher.fetch("https://example.com/page")
                assert False, "Should have raised"
            except ScrapingError as exc:
                assert "403" in str(exc)

    async def test_timeout_raises_scraping_error(self) -> None:
        fetcher = _make_fetcher()
        with patch("code.shukketsu.scraping.fetcher.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            mock_client_cls.return_value = mock_client
            try:
                await fetcher.fetch("https://example.com/page")
                assert False, "Should have raised"
            except ScrapingError as exc:
                assert "timed out" in str(exc).lower()

    async def test_non_html_raises_scraping_error(self) -> None:
        fetcher = _make_fetcher()
        mock_resp = _mock_response(content_type="application/pdf")
        with patch("code.shukketsu.scraping.fetcher.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client
            try:
                await fetcher.fetch("https://example.com/file.pdf")
                assert False, "Should have raised"
            except ScrapingError as exc:
                assert "html" in str(exc).lower()

    async def test_final_url_tracked_after_redirect(self) -> None:
        fetcher = _make_fetcher()
        mock_resp = _mock_response(url="https://example.com/final")
        with patch("code.shukketsu.scraping.fetcher.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client
            result = await fetcher.fetch("https://example.com/old")
        assert result.final_url == "https://example.com/final"

    async def test_rate_limiter_called_before_fetch(self) -> None:
        fetcher = _make_fetcher()
        mock_resp = _mock_response()
        with patch("code.shukketsu.scraping.fetcher.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client
            await fetcher.fetch("https://example.com/page")
        fetcher._rate_limiter.acquire.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_fetcher.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Create `code/shukketsu/scraping/fetcher.py`:

```python
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
    manage politeness policies themselves.
    """

    def __init__(self, rate_limiter: RateLimiter, robots_checker: RobotsChecker) -> None:
        self._rate_limiter = rate_limiter
        self._robots_checker = robots_checker

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
            async with httpx.AsyncClient(
                timeout=config.SCRAPING_DEFAULT_TIMEOUT,
                follow_redirects=True,
                max_redirects=5,
            ) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": config.SCRAPING_USER_AGENT},
                )
        except httpx.TimeoutException as exc:
            raise ScrapingError(f"Timed out fetching {url}: {exc}") from exc
        except httpx.ConnectError as exc:
            raise ScrapingError(f"Connection error fetching {url}: {exc}") from exc

        # 4. Check status
        if response.status_code >= 400:
            raise ScrapingError(f"HTTP {response.status_code} fetching {url}")

        # 5. Check content type
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "text/xhtml" not in content_type:
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
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_fetcher.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add code/shukketsu/scraping/fetcher.py tests/unit/test_fetcher.py
git commit -m "feat(step8): add web fetcher with rate limiting and robots compliance"
```

---

## Task 5: Web Search Tool

**Files:**
- Create: `code/shukketsu/tools/research/web_search.py`
- Create: `tests/unit/test_web_search.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_web_search.py`:

```python
"""Tests for the web_search tool."""

from unittest.mock import AsyncMock, patch

import httpx

from code.shukketsu.tools.research.web_search import WebSearchTool

# Minimal Brave API response structure
BRAVE_RESPONSE = {
    "web": {
        "results": [
            {
                "title": "Combat Rogue Guide",
                "url": "https://example.com/guide",
                "description": "Complete guide to combat rogues in TBC.",
            },
            {
                "title": "Rogue DPS FAQ",
                "url": "https://example.com/faq",
                "description": "Frequently asked questions about rogue DPS.",
            },
        ]
    }
}

BRAVE_EMPTY_RESPONSE: dict = {"web": {"results": []}}


def _mock_brave_response(
    json_data: dict = BRAVE_RESPONSE,
    status_code: int = 200,
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("GET", "https://api.search.brave.com/res/v1/web/search"),
    )


class TestWebSearchTool:
    """Tests for WebSearchTool."""

    def test_has_correct_name(self) -> None:
        tool = WebSearchTool()
        assert tool.name == "web_search"

    def test_has_description(self) -> None:
        tool = WebSearchTool()
        assert len(tool.description) > 0

    def test_has_parameters_schema(self) -> None:
        tool = WebSearchTool()
        assert "query" in tool.parameters_schema

    async def test_empty_query_returns_error(self) -> None:
        tool = WebSearchTool()
        result = await tool.execute({"query": ""})
        assert "error" in result.lower()

    async def test_missing_api_key_returns_error(self) -> None:
        tool = WebSearchTool()
        with patch("code.shukketsu.tools.research.web_search.config") as mock_config:
            mock_config.BRAVE_SEARCH_API_KEY = ""
            mock_config.BRAVE_SEARCH_MAX_RESULTS = 5
            result = await tool.execute({"query": "test"})
        assert "api key" in result.lower()

    async def test_successful_search_formats_results(self) -> None:
        tool = WebSearchTool()
        mock_resp = _mock_brave_response()
        with (
            patch("code.shukketsu.tools.research.web_search.config") as mock_config,
            patch("code.shukketsu.tools.research.web_search.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_config.BRAVE_SEARCH_API_KEY = "test-key"
            mock_config.BRAVE_SEARCH_MAX_RESULTS = 5
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client
            result = await tool.execute({"query": "combat rogue"})
        assert "[1]" in result
        assert "[2]" in result
        assert "Combat Rogue Guide" in result
        assert "https://example.com/guide" in result

    async def test_no_results_returns_message(self) -> None:
        tool = WebSearchTool()
        mock_resp = _mock_brave_response(json_data=BRAVE_EMPTY_RESPONSE)
        with (
            patch("code.shukketsu.tools.research.web_search.config") as mock_config,
            patch("code.shukketsu.tools.research.web_search.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_config.BRAVE_SEARCH_API_KEY = "test-key"
            mock_config.BRAVE_SEARCH_MAX_RESULTS = 5
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client
            result = await tool.execute({"query": "nonexistent"})
        assert "no web results" in result.lower()

    async def test_http_error_returns_error_observation(self) -> None:
        tool = WebSearchTool()
        mock_resp = _mock_brave_response(status_code=429)
        with (
            patch("code.shukketsu.tools.research.web_search.config") as mock_config,
            patch("code.shukketsu.tools.research.web_search.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_config.BRAVE_SEARCH_API_KEY = "test-key"
            mock_config.BRAVE_SEARCH_MAX_RESULTS = 5
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client
            result = await tool.execute({"query": "test"})
        assert "rate limited" in result.lower() or "error" in result.lower()

    async def test_respects_count_parameter(self) -> None:
        tool = WebSearchTool()
        mock_resp = _mock_brave_response()
        with (
            patch("code.shukketsu.tools.research.web_search.config") as mock_config,
            patch("code.shukketsu.tools.research.web_search.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_config.BRAVE_SEARCH_API_KEY = "test-key"
            mock_config.BRAVE_SEARCH_MAX_RESULTS = 5
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client
            await tool.execute({"query": "test", "count": 3})
        call_kwargs = mock_client.get.call_args
        assert call_kwargs[1]["params"]["count"] == 3
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_web_search.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Create `code/shukketsu/tools/research/web_search.py`:

```python
"""Brave Search API tool for web search."""

import logging
from typing import Any

import httpx

from code.shukketsu import config
from code.shukketsu.tools.schemas import Tool

logger = logging.getLogger(__name__)

_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


class WebSearchTool(Tool):
    """Search the web using the Brave Search API.

    Returns a numbered list of results (title, URL, description) for the
    agent to evaluate. The agent can then use web_ingest to fetch and
    store promising pages.

    Note: JavaScript-rendered content may not appear in search snippets.
    """

    name = "web_search"
    description = (
        "Search the web for information not in the knowledge base. "
        "Returns titles, URLs, and snippets. Use web_ingest to fetch full content from promising URLs."
    )
    parameters_schema = {
        "query": {"type": "string", "description": "The search query"},
        "count": {"type": "integer", "description": "Number of results (default 5)", "optional": True},
    }

    async def execute(self, tool_input: dict[str, Any]) -> str:
        """Execute a web search via Brave Search API."""
        query = tool_input.get("query", "").strip()
        count = tool_input.get("count", config.BRAVE_SEARCH_MAX_RESULTS)

        if not query:
            return "Error: 'query' parameter is required."

        if not config.BRAVE_SEARCH_API_KEY:
            return "Error: Brave Search API key not configured. Set BRAVE_SEARCH_API_KEY in variables.env."

        logger.info("Web search: %r (count=%d)", query, count)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    _BRAVE_SEARCH_URL,
                    params={"q": query, "count": count},
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "X-Subscription-Token": config.BRAVE_SEARCH_API_KEY,
                    },
                )
        except httpx.TimeoutException:
            return "Error: Web search timed out. Try again later."
        except httpx.ConnectError:
            return "Error: Could not connect to Brave Search API."

        if response.status_code == 429:
            return "Error: Web search rate limited. Try again later."
        if response.status_code >= 400:
            return f"Error: Web search failed with HTTP {response.status_code}."

        data = response.json()
        results = data.get("web", {}).get("results", [])

        if not results:
            return f"No web results found for: {query}"

        parts = [f"Found {len(results)} web result{'s' if len(results) != 1 else ''} for \"{query}\":\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            desc = r.get("description", "No description available.")
            parts.append(f"[{i}] {title}\n    {url}\n    {desc}\n")

        return "\n".join(parts)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_web_search.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add code/shukketsu/tools/research/web_search.py tests/unit/test_web_search.py
git commit -m "feat(step8): add web_search tool using Brave Search API"
```

---

## Task 6: Web Ingest Tool

**Files:**
- Create: `code/shukketsu/tools/research/web_ingest.py`
- Create: `tests/unit/test_web_ingest.py`
- Modify: `requirements.txt` (replace beautifulsoup4 with trafilatura)

**Step 1: Update requirements.txt**

Replace `beautifulsoup4>=4.12` with `trafilatura>=2.0` in `requirements.txt`.

Run: `pip3 install trafilatura>=2.0 --break-system-packages`

**Step 2: Write the failing tests**

Create `tests/unit/test_web_ingest.py`:

```python
"""Tests for the web_ingest tool."""

import sqlite3
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

from code.shukketsu.ingest.pipeline import IngestResult
from code.shukketsu.resilience.errors import RobotsDisallowedError, ScrapingError
from code.shukketsu.tools.research.web_ingest import WebIngestTool


@dataclass(frozen=True)
class FakeFetchResult:
    html: str
    status_code: int
    final_url: str


SAMPLE_HTML = """
<html>
<head><title>Combat Rogue Guide</title></head>
<body>
<nav>Navigation links</nav>
<article>
<h1>Combat Rogue Guide</h1>
<p>The hit cap for combat rogues in TBC is 9% or 142 hit rating.</p>
</article>
<footer>Footer content</footer>
</body>
</html>
"""


def _make_tool(
    fetch_result: FakeFetchResult | None = None,
    fetch_error: Exception | None = None,
    ingest_result: IngestResult | None = None,
    extracted_text: str | None = "# Combat Rogue Guide\n\nThe hit cap is 9%.",
) -> WebIngestTool:
    """Create a WebIngestTool with mocked dependencies."""
    fetcher = AsyncMock()
    if fetch_error:
        fetcher.fetch = AsyncMock(side_effect=fetch_error)
    else:
        result = fetch_result or FakeFetchResult(
            html=SAMPLE_HTML, status_code=200, final_url="https://example.com/guide"
        )
        fetcher.fetch = AsyncMock(return_value=result)

    pipeline = AsyncMock()
    pipeline.ingest = AsyncMock(
        return_value=ingest_result or IngestResult(source_id=1, chunk_count=3, already_existed=False)
    )

    tool = WebIngestTool(fetcher=fetcher, pipeline=pipeline)

    # Mock trafilatura extraction
    with_extract = patch(
        "code.shukketsu.tools.research.web_ingest.trafilatura.extract",
        return_value=extracted_text,
    )
    with_bare = patch(
        "code.shukketsu.tools.research.web_ingest.trafilatura.bare_extraction",
        return_value={"title": "Combat Rogue Guide"} if extracted_text else {"title": None},
    )
    tool._extract_patch = with_extract
    tool._bare_patch = with_bare
    return tool


class TestWebIngestTool:
    """Tests for WebIngestTool."""

    def test_has_correct_name(self) -> None:
        tool = WebIngestTool(fetcher=AsyncMock(), pipeline=AsyncMock())
        assert tool.name == "web_ingest"

    def test_has_description(self) -> None:
        tool = WebIngestTool(fetcher=AsyncMock(), pipeline=AsyncMock())
        assert len(tool.description) > 0

    def test_has_parameters_schema(self) -> None:
        tool = WebIngestTool(fetcher=AsyncMock(), pipeline=AsyncMock())
        assert "url" in tool.parameters_schema

    async def test_empty_url_returns_error(self) -> None:
        tool = WebIngestTool(fetcher=AsyncMock(), pipeline=AsyncMock())
        result = await tool.execute({"url": ""})
        assert "error" in result.lower()

    async def test_invalid_url_returns_error(self) -> None:
        tool = WebIngestTool(fetcher=AsyncMock(), pipeline=AsyncMock())
        result = await tool.execute({"url": "not-a-url"})
        assert "error" in result.lower()

    async def test_successful_ingest(self) -> None:
        tool = _make_tool()
        with tool._extract_patch, tool._bare_patch:
            result = await tool.execute({"url": "https://example.com/guide"})
        assert "3 chunks" in result
        assert "example.com" in result

    async def test_dedup_returns_existing_message(self) -> None:
        tool = _make_tool(
            ingest_result=IngestResult(source_id=1, chunk_count=5, already_existed=True)
        )
        with tool._extract_patch, tool._bare_patch:
            result = await tool.execute({"url": "https://example.com/guide"})
        assert "already" in result.lower()

    async def test_empty_extraction_returns_error(self) -> None:
        tool = _make_tool(extracted_text=None)
        with tool._extract_patch, tool._bare_patch:
            result = await tool.execute({"url": "https://example.com/guide"})
        assert "could not extract" in result.lower()

    async def test_robots_blocked_returns_message(self) -> None:
        tool = _make_tool(fetch_error=RobotsDisallowedError("blocked"))
        with tool._extract_patch, tool._bare_patch:
            result = await tool.execute({"url": "https://example.com/private"})
        assert "robots.txt" in result.lower()

    async def test_fetch_failure_returns_error(self) -> None:
        tool = _make_tool(fetch_error=ScrapingError("HTTP 403"))
        with tool._extract_patch, tool._bare_patch:
            result = await tool.execute({"url": "https://example.com/page"})
        assert "failed to fetch" in result.lower() or "403" in result

    async def test_uses_final_url_for_ingest(self) -> None:
        tool = _make_tool(
            fetch_result=FakeFetchResult(
                html=SAMPLE_HTML, status_code=200, final_url="https://example.com/redirected"
            )
        )
        with tool._extract_patch, tool._bare_patch:
            await tool.execute({"url": "https://example.com/old"})
        # Check that pipeline.ingest was called with the final URL
        call_kwargs = tool._pipeline.ingest.call_args[1]
        assert call_kwargs["url"] == "https://example.com/redirected"

    async def test_auto_extracts_title(self) -> None:
        tool = _make_tool()
        with tool._extract_patch, tool._bare_patch:
            await tool.execute({"url": "https://example.com/guide"})
        call_kwargs = tool._pipeline.ingest.call_args[1]
        assert call_kwargs["title"] == "Combat Rogue Guide"
```

**Step 2b: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_web_ingest.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Create `code/shukketsu/tools/research/web_ingest.py`:

```python
"""Web ingest tool: fetch, extract, and ingest web content."""

import logging
from typing import Any
from urllib.parse import urlparse

import trafilatura

from code.shukketsu.ingest.pipeline import IngestPipeline
from code.shukketsu.resilience.errors import RobotsDisallowedError, ScrapingError
from code.shukketsu.scraping.fetcher import WebFetcher
from code.shukketsu.tools.schemas import Tool

logger = logging.getLogger(__name__)


class WebIngestTool(Tool):
    """Fetch a web page, extract its content, and ingest into the knowledge base.

    Uses trafilatura for content extraction (markdown output for better chunking).
    Delegates to the existing IngestPipeline for chunking, embedding, and storage.
    Respects robots.txt and per-domain rate limits via WebFetcher.

    Note: JavaScript-rendered content cannot be extracted. Pages that rely on
    JS frameworks (React, Angular) may yield empty extractions.
    """

    name = "web_ingest"
    description = (
        "Fetch a web page and ingest its content into the knowledge base for future retrieval. "
        "Use after web_search to store promising sources. "
        "Cannot extract content from JavaScript-rendered pages."
    )
    parameters_schema = {
        "url": {"type": "string", "description": "The URL to fetch and ingest"},
        "title": {"type": "string", "description": "Title for the source (auto-detected if omitted)", "optional": True},
    }

    def __init__(self, fetcher: WebFetcher, pipeline: IngestPipeline) -> None:
        self._fetcher = fetcher
        self._pipeline = pipeline

    async def execute(self, tool_input: dict[str, Any]) -> str:
        """Fetch a URL, extract content, and ingest into the knowledge base."""
        url = tool_input.get("url", "").strip()
        title_override = tool_input.get("title")

        # Validate URL
        if not url:
            return "Error: 'url' parameter is required."
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return "Error: URL must start with http:// or https://."

        # Fetch
        try:
            fetch_result = await self._fetcher.fetch(url)
        except RobotsDisallowedError:
            return f"Cannot fetch this URL: blocked by robots.txt ({urlparse(url).netloc})."
        except ScrapingError as exc:
            return f"Failed to fetch URL: {exc}"

        # Extract content as markdown (preserves headers for chunker)
        extracted = trafilatura.extract(
            fetch_result.html,
            output_format="markdown",
            include_links=False,
            include_comments=False,
        )

        if not extracted or not extracted.strip():
            return f"Could not extract meaningful content from {urlparse(url).netloc}. The page may use JavaScript rendering."

        # Auto-detect title if not provided
        title = title_override
        if not title:
            metadata = trafilatura.bare_extraction(fetch_result.html)
            title = metadata.get("title") if metadata else None
        if not title:
            title = urlparse(fetch_result.final_url).netloc

        # Ingest using the final URL (after redirects) for correct dedup
        result = await self._pipeline.ingest(
            text=extracted,
            url=fetch_result.final_url,
            title=title,
            source_type="web",
        )

        domain = urlparse(fetch_result.final_url).netloc
        if result.already_existed:
            return (
                f"Content from {domain} was already in the knowledge base "
                f"(source_id={result.source_id}, {result.chunk_count} chunks). No changes detected."
            )

        return (
            f"Ingested \"{title}\" from {domain}: "
            f"{result.chunk_count} chunks stored (source_id={result.source_id})."
        )
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_web_ingest.py -v`
Expected: All 12 tests PASS

**Step 5: Commit**

```bash
git add code/shukketsu/tools/research/web_ingest.py tests/unit/test_web_ingest.py requirements.txt
git commit -m "feat(step8): add web_ingest tool with trafilatura extraction"
```

---

## Task 7: Integration — Register Tools in Chat Handler

**Files:**
- Modify: `code/shukketsu/web/routers/chat.py` (the `_get_agent()` function)
- Test: Run all unit tests to verify nothing breaks

**Step 1: Update `_get_agent()` to register both new tools**

In `code/shukketsu/web/routers/chat.py`, replace the `_get_agent()` function:

```python
def _get_agent() -> BaseAgent:
    """Get or create the agent singleton.

    Lazy initialization avoids import-time side effects (DB connection,
    extension loading). The agent is created once and reused.
    """
    global _agent_instance  # noqa: PLW0603

    if _agent_instance is None:
        from code.shukketsu.agents.base import BaseAgent
        from code.shukketsu.db.connection import get_connection, init_db
        from code.shukketsu.ingest.embedder import get_embedder
        from code.shukketsu.ingest.pipeline import IngestPipeline
        from code.shukketsu.scraping.fetcher import WebFetcher
        from code.shukketsu.scraping.rate_limiter import RateLimiter
        from code.shukketsu.scraping.robots import RobotsChecker
        from code.shukketsu.tools.knowledge.search import RagSearchTool
        from code.shukketsu.tools.registry import ToolRegistry
        from code.shukketsu.tools.research.web_ingest import WebIngestTool
        from code.shukketsu.tools.research.web_search import WebSearchTool

        conn = get_connection()
        init_db(conn)
        embedder = get_embedder()

        registry = ToolRegistry()
        registry.register(RagSearchTool(conn=conn, embed_fn=embedder.embed_query))

        # Web search + ingest tools
        fetcher = WebFetcher(rate_limiter=RateLimiter(), robots_checker=RobotsChecker())
        pipeline = IngestPipeline(conn=conn, embedder=embedder)
        registry.register(WebSearchTool())
        registry.register(WebIngestTool(fetcher=fetcher, pipeline=pipeline))

        _agent_instance = BaseAgent(tool_registry=registry)

    return _agent_instance
```

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v`
Expected: All tests PASS (previous 152 + ~34 new = ~186)

**Step 3: Commit**

```bash
git add code/shukketsu/web/routers/chat.py
git commit -m "feat(step8): register web_search and web_ingest tools in chat handler"
```

---

## Task 8: Final Verification + CLAUDE.md Update

**Step 1: Run full test suite and count**

Run: `python3 -m pytest tests/unit/ -v --tb=short`
Expected: All tests pass, count should be ~186.

**Step 2: Run ruff check**

Run: `ruff check code/shukketsu/scraping/ code/shukketsu/tools/research/ tests/unit/test_rate_limiter.py tests/unit/test_robots.py tests/unit/test_fetcher.py tests/unit/test_web_search.py tests/unit/test_web_ingest.py`
Expected: No errors

**Step 3: Update CLAUDE.md**

Update the Step 8 line from `← current` to `**DONE**`, move the `← current` marker to Step 9. Update the test count. Add new key files to the project overview list.

**Step 4: Update memory**

Update `MEMORY.md` with Step 8 test count.

**Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Step 8 completion"
```

---

## Summary

| Task | What | New Tests | Files |
|------|------|-----------|-------|
| 1 | Error types + config | 3 | 2 modified |
| 2 | Rate limiter | 6 | 1 created, 1 test |
| 3 | Robots checker | 6 | 1 created, 1 test |
| 4 | Web fetcher | 7 | 1 created, 1 test |
| 5 | Web search tool | 9 | 1 created, 1 test |
| 6 | Web ingest tool | 12 | 1 created, 1 test, 1 modified |
| 7 | Integration | 0 | 1 modified |
| 8 | Verification | 0 | 1 modified |
| **Total** | | **~43** | **6 created, 5 modified, 6 tests** |
