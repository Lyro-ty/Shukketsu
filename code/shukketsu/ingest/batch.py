"""Batch ingest engine: ingest multiple URLs with concurrency control.

Provides a CLI for bulk content ingestion from a YAML manifest,
entity extraction as a separate resumable pass, and database statistics.

Supports two fetch modes:
- **http** (default): Uses WebFetcher + trafilatura for static HTML pages.
- **browser**: Uses Playwright to render JS-heavy pages (e.g. Wowhead) and
  extracts text from `[role="main"]` element. Much richer content but slower.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Page

import trafilatura

from code.shukketsu import config
from code.shukketsu.ingest.manifest import SourceEntry, load_manifest
from code.shukketsu.ingest.pipeline import ExtractFn, IngestPipeline, IngestResult
from code.shukketsu.rag.graph import GraphStore
from code.shukketsu.scraping.fetcher import FetchResult, WebFetcher

logger = logging.getLogger(__name__)


@dataclass
class SourceResult:
    """Result of ingesting a single source."""

    url: str
    title: str
    success: bool
    chunks: int = 0
    already_existed: bool = False
    error: str | None = None


@dataclass
class BatchReport:
    """Aggregate result of a batch ingest run."""

    results: list[SourceResult] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.success and not r.already_existed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.success and r.already_existed)

    @property
    def total_chunks(self) -> int:
        return sum(r.chunks for r in self.results if r.success and not r.already_existed)


class BatchIngest:
    """Ingest multiple sources with concurrency control."""

    def __init__(
        self,
        fetcher: WebFetcher,
        pipeline: IngestPipeline,
        concurrency: int = config.BATCH_CONCURRENCY,
    ) -> None:
        self._fetcher = fetcher
        self._pipeline = pipeline
        self._semaphore = asyncio.Semaphore(concurrency)

    async def ingest_all(self, sources: list[SourceEntry]) -> BatchReport:
        """Ingest all sources with concurrency limiting."""
        tasks = [self._ingest_one(source) for source in sources]
        raw_results: list[SourceResult | BaseException] = await asyncio.gather(*tasks, return_exceptions=True)

        report = BatchReport()
        for source, result in zip(sources, raw_results):
            if isinstance(result, BaseException):
                if not isinstance(result, Exception):
                    raise result  # Re-raise KeyboardInterrupt, SystemExit, etc.
                report.results.append(
                    SourceResult(
                        url=source.url,
                        title=source.title,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                report.results.append(result)
        return report

    async def _ingest_one(self, source: SourceEntry) -> SourceResult:
        """Ingest a single source, respecting the semaphore."""
        async with self._semaphore:
            return await self._fetch_and_ingest(source)

    async def _fetch_and_ingest(self, source: SourceEntry) -> SourceResult:
        """Fetch URL, extract content, ingest via pipeline."""
        logger.info("Fetching %s (%s)", source.url, source.title)

        try:
            fetch_result: FetchResult = await self._fetcher.fetch(source.url)
        except Exception as exc:
            return SourceResult(
                url=source.url,
                title=source.title,
                success=False,
                error=f"Fetch failed: {exc}",
            )

        # trafilatura.extract is CPU-bound — run in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        extracted = await loop.run_in_executor(
            None,
            functools.partial(
                trafilatura.extract,
                fetch_result.html,
                output_format="markdown",
                include_links=False,
                include_comments=False,
            ),
        )

        if not extracted or not extracted.strip():
            return SourceResult(
                url=source.url,
                title=source.title,
                success=False,
                error="Empty extraction (page may use JS rendering)",
            )

        try:
            result: IngestResult = await self._pipeline.ingest(
                text=extracted,
                url=fetch_result.final_url,
                title=source.title,
                source_type=source.source_type,
            )
        except Exception as exc:
            return SourceResult(
                url=source.url,
                title=source.title,
                success=False,
                error=f"Ingest failed: {exc}",
            )

        return SourceResult(
            url=source.url,
            title=source.title,
            success=True,
            chunks=result.chunk_count,
            already_existed=result.already_existed,
        )


class BrowserBatchIngest:
    """Ingest JS-rendered pages using Playwright.

    Navigates to each URL in a real browser, waits for content to render,
    then extracts text from the main content area. Sequential (one page at a
    time) since pages share a single browser instance.
    """

    def __init__(self, pipeline: IngestPipeline) -> None:
        self._pipeline = pipeline

    async def ingest_all(self, sources: list[SourceEntry]) -> BatchReport:
        """Ingest all sources using a headless browser."""
        from playwright.async_api import async_playwright

        report = BatchReport()
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()

            for source in sources:
                result = await self._fetch_and_ingest(page, source)
                report.results.append(result)

            await browser.close()
        return report

    async def _fetch_and_ingest(self, page: Page, source: SourceEntry) -> SourceResult:
        """Navigate to URL, extract rendered text, ingest."""
        logger.info("Browser fetching %s (%s)", source.url, source.title)

        try:
            await page.goto(source.url, wait_until="domcontentloaded", timeout=30000)
            # Wait for guide content to render (JS hydration)
            await page.wait_for_selector('[role="main"] h2, main h2, article h2', timeout=10000)
        except Exception as exc:
            return SourceResult(
                url=source.url,
                title=source.title,
                success=False,
                error=f"Browser navigation failed: {exc}",
            )

        # Extract text from main content area
        try:
            text = await page.evaluate(
                """() => {
                    const el = document.querySelector('[role="main"]')
                        || document.querySelector('main')
                        || document.querySelector('article')
                        || document.querySelector('.guide-body');
                    return el ? el.innerText : '';
                }"""
            )
        except Exception as exc:
            return SourceResult(
                url=source.url,
                title=source.title,
                success=False,
                error=f"Content extraction failed: {exc}",
            )

        if not text or not text.strip():
            # Fall back to trafilatura on page source
            html = await page.content()
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                None,
                functools.partial(
                    trafilatura.extract,
                    html,
                    output_format="markdown",
                    include_links=False,
                    include_comments=False,
                ),
            )

        if not text or not text.strip():
            return SourceResult(
                url=source.url,
                title=source.title,
                success=False,
                error="Empty extraction even with browser rendering",
            )

        final_url = page.url

        try:
            result: IngestResult = await self._pipeline.ingest(
                text=text,
                url=final_url,
                title=source.title,
                source_type=source.source_type,
            )
        except Exception as exc:
            return SourceResult(
                url=source.url,
                title=source.title,
                success=False,
                error=f"Ingest failed: {exc}",
            )

        return SourceResult(
            url=source.url,
            title=source.title,
            success=True,
            chunks=result.chunk_count,
            already_existed=result.already_existed,
        )


async def extract_entities_pass(
    conn: sqlite3.Connection,
    graph_store: GraphStore,
    extract_fn: ExtractFn,
) -> tuple[int, int, int]:
    """Run entity extraction on chunks that haven't been extracted yet.

    Finds chunks whose IDs are not referenced by any entity's source_chunk_id,
    runs the extract function on each, and stores results incrementally.

    Returns:
        Tuple of (chunks_processed, total_entities, total_relationships).
    """
    # Find chunk IDs not yet associated with any entity
    rows = conn.execute(
        """
        SELECT c.id, c.content
        FROM chunks c
        WHERE c.id NOT IN (SELECT DISTINCT source_chunk_id FROM entities WHERE source_chunk_id IS NOT NULL)
        ORDER BY c.id
        """
    ).fetchall()

    if not rows:
        logger.info("No unextracted chunks found")
        return 0, 0, 0

    logger.info("Found %d chunks without entities, starting extraction", len(rows))

    chunks_processed = 0
    total_entities = 0
    total_relationships = 0

    for row in rows:
        chunk_id = row["id"]
        content = row["content"]

        try:
            extraction = await extract_fn(content)
        except Exception as exc:
            logger.warning("Extraction failed for chunk %d: %s", chunk_id, exc)
            continue

        entity_id_map: dict[str, int] = {}
        for entity in extraction.entities:
            eid = graph_store.upsert_entity(
                entity.name,
                entity.entity_type,
                properties=entity.properties or None,
                source_chunk_id=chunk_id,
            )
            entity_id_map[entity.name] = eid
            total_entities += 1

        for rel in extraction.relationships:
            src_id = entity_id_map.get(rel.source)
            tgt_id = entity_id_map.get(rel.target)
            if src_id is None or tgt_id is None:
                continue
            graph_store.upsert_relationship(
                src_id,
                tgt_id,
                rel.relation_type,
                properties=rel.properties or None,
                source_chunk_id=chunk_id,
            )
            total_relationships += 1

        conn.commit()
        chunks_processed += 1
        logger.info(
            "Extracted chunk %d/%d: %d entities, %d relationships",
            chunks_processed,
            len(rows),
            len(extraction.entities),
            len(extraction.relationships),
        )

    return chunks_processed, total_entities, total_relationships


def print_db_stats(conn: sqlite3.Connection) -> None:
    """Print database content statistics."""
    source_count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    rel_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]

    print(f"\n{'=' * 40}")
    print("Database Statistics")
    print(f"{'=' * 40}")
    print(f"Sources:       {source_count}")
    print(f"Chunks:        {chunk_count}")
    print(f"Entities:      {entity_count}")
    print(f"Relationships: {rel_count}")

    if entity_count > 0:
        print(f"\n{'Entity Type Distribution':}")
        rows = conn.execute(
            """
            SELECT et.display_name, COUNT(e.id) as cnt
            FROM entities e JOIN entity_types et ON e.entity_type_id = et.id
            GROUP BY et.display_name ORDER BY cnt DESC
            """
        ).fetchall()
        for row in rows:
            print(f"  {row['display_name']:20s} {row['cnt']}")

    print()


def _print_report(report: BatchReport) -> None:
    """Print batch ingest report to stdout."""
    print(f"\n{'=' * 40}")
    print("Batch Ingest Report")
    print(f"{'=' * 40}")
    print(f"Succeeded: {report.succeeded} ({report.total_chunks} new chunks)")
    print(f"Skipped:   {report.skipped} (already existed)")
    print(f"Failed:    {report.failed}")

    if report.failed > 0:
        print("\nFailures:")
        for r in report.results:
            if not r.success:
                print(f"  {r.url}: {r.error}")
    print()


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Batch ingest sources from a YAML manifest",
        prog="python3 -m code.shukketsu.ingest.batch",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(config.MANIFEST_PATH),
        help="Path to manifest YAML (default: %(default)s)",
    )
    parser.add_argument("--priority", type=int, default=None, help="Filter to priority <= N")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--spec", type=str, default=None, help="Filter by spec")
    parser.add_argument("--browser", action="store_true", help="Use Playwright browser for JS-rendered pages")
    parser.add_argument("--extract", action="store_true", help="Enable entity extraction during ingest")
    parser.add_argument("--extract-only", action="store_true", help="Run entity extraction on existing chunks")
    parser.add_argument("--stats", action="store_true", help="Print database statistics")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested")
    return parser


async def _async_main(args: argparse.Namespace) -> None:
    """Async entry point for CLI."""
    from code.shukketsu.db.connection import get_connection, init_db
    from code.shukketsu.ingest.embedder import get_embedder
    from code.shukketsu.rag.entities import extract_entities_from_chunk
    from code.shukketsu.scraping.rate_limiter import RateLimiter
    from code.shukketsu.scraping.robots import RobotsChecker

    conn = get_connection(config.DB_PATH)
    init_db(conn)

    if args.stats:
        print_db_stats(conn)
        conn.close()
        return

    if args.extract_only:
        gs = GraphStore(conn)
        gs.seed_entity_types()
        processed, entities, rels = await extract_entities_pass(conn, gs, extract_entities_from_chunk)
        print(f"\nExtracted {entities} entities, {rels} relationships from {processed} chunks")
        print_db_stats(conn)
        conn.close()
        return

    # Load manifest
    manifest = load_manifest(args.manifest)
    sources = manifest.filter(
        max_priority=args.priority,
        category=args.category,
        spec=args.spec,
    )

    if not sources:
        print("No sources match the given filters")
        conn.close()
        return

    if args.dry_run:
        print(f"\nDry run: {len(sources)} sources would be ingested:\n")
        for s in sources:
            print(f"  [P{s.priority}] {s.title}")
            print(f"        {s.url}")
        conn.close()
        return

    # Build dependencies
    embedder = get_embedder()
    graph_store: GraphStore | None = GraphStore(conn) if args.extract else None
    extract_fn: ExtractFn | None = None
    if args.extract and graph_store is not None:
        graph_store.seed_entity_types()
        extract_fn = extract_entities_from_chunk

    pipeline = IngestPipeline(conn, embedder, graph_store=graph_store, extract_fn=extract_fn)

    fetcher: WebFetcher | None = None
    try:
        if args.browser:
            batch_engine: BatchIngest | BrowserBatchIngest = BrowserBatchIngest(pipeline)
        else:
            rate_limiter = RateLimiter()
            robots_checker = RobotsChecker()
            fetcher = WebFetcher(rate_limiter, robots_checker)
            batch_engine = BatchIngest(fetcher, pipeline)

        report = await batch_engine.ingest_all(sources)
        _print_report(report)
        print_db_stats(conn)
    finally:
        if fetcher is not None:
            await fetcher.close()
        conn.close()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
