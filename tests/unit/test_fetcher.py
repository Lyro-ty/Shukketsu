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
