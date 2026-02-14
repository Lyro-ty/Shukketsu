"""Content freshness checker: detect stale sources and verify content changes."""

import asyncio
import functools
import hashlib
import logging
import sqlite3
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx
import trafilatura
from pydantic import BaseModel

from code.shukketsu import config
from code.shukketsu.trust.scoring import TRUST_DELTAS, record_trust_event

logger = logging.getLogger(__name__)


class StaleSource(BaseModel):
    """A source that needs a freshness check."""

    id: int
    url: str
    content_hash: str | None
    check_interval_hours: int


class FreshnessResult(BaseModel):
    """Result of checking a single source for content changes."""

    source_id: int
    url: str
    changed: bool
    old_hash: str | None
    new_hash: str | None
    checked_at: str  # ISO 8601
    head_only: bool  # True if HEAD was sufficient (no full fetch needed)
    error: str | None = None


def find_stale_sources(conn: sqlite3.Connection) -> list[StaleSource]:
    """Find sources where now - last_checked > check_interval_hours.

    Sources with last_checked = NULL are always considered stale.
    Returns list ordered by staleness (most overdue first).
    """
    rows = conn.execute(
        """SELECT id, url, content_hash, check_interval_hours
           FROM sources
           WHERE last_checked IS NULL
              OR (julianday('now') - julianday(last_checked)) * 24 > check_interval_hours
           ORDER BY last_checked ASC NULLS FIRST""",
    ).fetchall()
    return [
        StaleSource(
            id=row["id"],
            url=row["url"],
            content_hash=row["content_hash"],
            check_interval_hours=row["check_interval_hours"],
        )
        for row in rows
    ]


def count_stale_sources(conn: sqlite3.Connection, source_urls: list[str]) -> int:
    """Count how many of the given URLs have is_stale = 1.

    Used by wiki article detail to show stale source warning badge.
    Returns 0 if source_urls is empty.
    """
    if not source_urls:
        return 0
    placeholders = ",".join("?" * len(source_urls))
    row = conn.execute(
        f"SELECT COUNT(*) FROM sources WHERE url IN ({placeholders}) AND is_stale = 1",  # noqa: S608
        source_urls,
    ).fetchone()
    return int(row[0])


async def check_source_freshness(
    source: StaleSource,
    conn: sqlite3.Connection,
    timeout: int | None = None,
    client: httpx.AsyncClient | None = None,
) -> FreshnessResult:
    """Check a single source for content changes.

    Flow:
    1. HEAD request for Last-Modified header
    2. If Last-Modified is before source's last_checked -> mark fresh (head_only)
    3. If header missing or stale -> full GET fetch
    4. Compare content hash
    5. Update DB accordingly
    """
    if timeout is None:
        timeout = config.FRESHNESS_HTTP_TIMEOUT
    now = datetime.now(UTC).isoformat()

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": config.SCRAPING_USER_AGENT},
            follow_redirects=True,
        )

    try:
        # Step 1: HEAD request
        head_resp = await client.head(source.url)

        if head_resp.status_code >= 400:
            try:
                record_trust_event(
                    conn,
                    source.id,
                    "dead_url",
                    TRUST_DELTAS["dead_url"],
                    details=f"HTTP {head_resp.status_code} for {source.url}",
                )
            except Exception:
                logger.warning("Failed to record dead_url trust event", exc_info=True)
            try:
                conn.execute("UPDATE sources SET last_checked = ? WHERE id = ?", (now, source.id))
                conn.commit()
            except Exception:
                conn.rollback()
                logger.warning("Failed to update last_checked for dead URL", exc_info=True)
            return FreshnessResult(
                source_id=source.id,
                url=source.url,
                changed=False,
                old_hash=source.content_hash,
                new_hash=None,
                checked_at=now,
                head_only=True,
                error=f"Source returned HTTP {head_resp.status_code}",
            )

        # Step 2: Check Last-Modified
        last_modified_str = head_resp.headers.get("Last-Modified")
        if last_modified_str:
            try:
                last_modified = parsedate_to_datetime(last_modified_str)
                row = conn.execute("SELECT last_checked FROM sources WHERE id = ?", (source.id,)).fetchone()
                if row and row["last_checked"]:
                    last_checked_dt = datetime.fromisoformat(row["last_checked"])
                    if last_modified.astimezone(UTC) < last_checked_dt.astimezone(UTC):
                        try:
                            conn.execute(
                                "UPDATE sources SET last_checked = ? WHERE id = ?",
                                (now, source.id),
                            )
                            conn.commit()
                        except Exception:
                            conn.rollback()
                            raise
                        return FreshnessResult(
                            source_id=source.id,
                            url=source.url,
                            changed=False,
                            old_hash=source.content_hash,
                            new_hash=source.content_hash,
                            checked_at=now,
                            head_only=True,
                        )
            except (ValueError, TypeError):
                pass  # Malformed header -- fall through to full fetch

        # Step 3: Full GET fetch + extract text (matches ingest pipeline hashing)
        get_resp = await client.get(source.url)
        # trafilatura.extract is CPU-bound — run in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        extracted = await loop.run_in_executor(
            None,
            functools.partial(
                trafilatura.extract,
                get_resp.text,
                output_format="markdown",
                include_links=False,
                include_comments=False,
            ),
        )
        if not extracted or not extracted.strip():
            try:
                conn.execute("UPDATE sources SET last_checked = ? WHERE id = ?", (now, source.id))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return FreshnessResult(
                source_id=source.id,
                url=source.url,
                changed=False,
                old_hash=source.content_hash,
                new_hash=None,
                checked_at=now,
                head_only=False,
                error="Content extraction returned empty — skipping hash comparison",
            )
        new_hash = hashlib.sha256(extracted.encode()).hexdigest()

    except Exception as exc:
        logger.warning("Freshness check failed for source %d (%s): %s", source.id, source.url, exc)
        return FreshnessResult(
            source_id=source.id,
            url=source.url,
            changed=False,
            old_hash=source.content_hash,
            new_hash=None,
            checked_at=now,
            head_only=False,
            error=str(exc),
        )
    finally:
        if owns_client:
            await client.aclose()

    # Step 4: Compare hashes
    if new_hash == source.content_hash:
        try:
            conn.execute("UPDATE sources SET last_checked = ? WHERE id = ?", (now, source.id))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return FreshnessResult(
            source_id=source.id,
            url=source.url,
            changed=False,
            old_hash=source.content_hash,
            new_hash=new_hash,
            checked_at=now,
            head_only=False,
        )

    # Step 5: Content changed
    try:
        conn.execute(
            """UPDATE sources
               SET content_hash_previous = content_hash,
                   content_hash = ?,
                   change_count = change_count + 1,
                   is_stale = 1,
                   last_checked = ?
               WHERE id = ?""",
            (new_hash, now, source.id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return FreshnessResult(
        source_id=source.id,
        url=source.url,
        changed=True,
        old_hash=source.content_hash,
        new_hash=new_hash,
        checked_at=now,
        head_only=False,
    )


async def run_freshness_sweep(
    conn: sqlite3.Connection,
    timeout: int | None = None,
) -> list[FreshnessResult]:
    """Find all stale sources and check each. Shares one httpx client across sources."""
    if timeout is None:
        timeout = config.FRESHNESS_HTTP_TIMEOUT
    stale = find_stale_sources(conn)
    results = []
    client = httpx.AsyncClient(
        timeout=timeout,
        headers={"User-Agent": config.SCRAPING_USER_AGENT},
        follow_redirects=True,
    )
    try:
        for source in stale:
            try:
                result = await check_source_freshness(source, conn, timeout=timeout, client=client)
            except Exception as exc:
                logger.warning("Freshness sweep: source %d (%s) crashed: %s", source.id, source.url, exc)
                result = FreshnessResult(
                    source_id=source.id,
                    url=source.url,
                    changed=False,
                    old_hash=source.content_hash,
                    new_hash=None,
                    checked_at=datetime.now(UTC).isoformat(),
                    head_only=False,
                    error=str(exc),
                )
            results.append(result)
    finally:
        await client.aclose()
    return results
