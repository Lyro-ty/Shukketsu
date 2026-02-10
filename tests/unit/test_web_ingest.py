"""Tests for the web_ingest tool."""

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
