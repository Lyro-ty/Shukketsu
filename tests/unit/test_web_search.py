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

    async def test_circuit_open_returns_unavailable_message(self) -> None:
        """When brave circuit breaker is open, tool returns fallback message."""
        from code.shukketsu.resilience.errors import CircuitOpenError

        tool = WebSearchTool()
        with (
            patch("code.shukketsu.tools.research.web_search.config") as mock_config,
            patch("code.shukketsu.tools.research.web_search.brave_breaker") as mock_breaker,
        ):
            mock_config.BRAVE_SEARCH_API_KEY = "test-key"
            mock_config.BRAVE_SEARCH_MAX_RESULTS = 5
            mock_breaker.call = AsyncMock(side_effect=CircuitOpenError("brave_search"))
            result = await tool.execute({"query": "test"})
        assert "unavailable" in result.lower()

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
