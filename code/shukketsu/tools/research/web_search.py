"""Brave Search API tool for web search."""

import logging
from typing import Any

import httpx

from code.shukketsu import config
from code.shukketsu.resilience.circuit_breaker import brave_breaker
from code.shukketsu.resilience.errors import CircuitOpenError
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

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self) -> None:
        """Close the shared HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

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
            client = self._get_client()
            response = await brave_breaker.call(
                client.get,
                _BRAVE_SEARCH_URL,
                params={"q": query, "count": count},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": config.BRAVE_SEARCH_API_KEY,
                },
            )
        except httpx.TransportError:
            return "Error: Could not connect to Brave Search API."
        except CircuitOpenError:
            return "Error: Web search is temporarily unavailable. Try answering from the knowledge base."

        if response.status_code == 429:
            return "Error: Web search rate limited. Try again later."
        if response.status_code >= 400:
            return f"Error: Web search failed with HTTP {response.status_code}."

        try:
            data = response.json()
        except ValueError:
            return "Error: Web search returned invalid response."
        results = data.get("web", {}).get("results", [])

        if not results:
            return f"No web results found for: {query}"

        parts = [f'Found {len(results)} web result{"s" if len(results) != 1 else ""} for "{query}":\n']
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            desc = r.get("description", "No description available.")
            parts.append(f"[{i}] {title}\n    {url}\n    {desc}\n")

        return "\n".join(parts)
