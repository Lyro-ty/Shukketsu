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
            return (
                f"Could not extract meaningful content from {urlparse(url).netloc}. "
                "The page may use JavaScript rendering."
            )

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

        return f'Ingested "{title}" from {domain}: {result.chunk_count} chunks stored (source_id={result.source_id}).'
