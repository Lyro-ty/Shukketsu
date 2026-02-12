"""Content freshness checker: detect stale sources and verify content changes."""

import logging
import sqlite3

from pydantic import BaseModel

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
    return row[0]
