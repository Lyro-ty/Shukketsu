"""Content freshness checker: detect stale sources and verify content changes."""

import hashlib
import logging
import sqlite3
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx
from pydantic import BaseModel

from code.shukketsu import config

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

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": config.SCRAPING_USER_AGENT},
            follow_redirects=True,
        ) as client:
            # Step 1: HEAD request
            head_resp = await client.head(source.url)

            if head_resp.status_code >= 400:
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
                            conn.execute(
                                "UPDATE sources SET last_checked = ? WHERE id = ?",
                                (now, source.id),
                            )
                            conn.commit()
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

            # Step 3: Full GET fetch
            get_resp = await client.get(source.url)
            new_hash = hashlib.sha256(get_resp.text.encode()).hexdigest()

    except (httpx.HTTPError, OSError) as exc:
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

    # Step 4: Compare hashes
    if new_hash == source.content_hash:
        conn.execute("UPDATE sources SET last_checked = ? WHERE id = ?", (now, source.id))
        conn.commit()
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
    """Find all stale sources and check each. Returns results for all checked sources."""
    stale = find_stale_sources(conn)
    results = []
    for source in stale:
        result = await check_source_freshness(source, conn, timeout=timeout)
        results.append(result)
    return results
