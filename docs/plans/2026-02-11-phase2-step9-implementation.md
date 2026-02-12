# Phase 2 Step 9: Content Freshness + Automated Backups — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add content freshness detection (find stale sources, verify via HEAD+fetch, flag changes) and SQLite backup management (create, verify integrity, prune) with manual API endpoints and startup sweep.

**Architecture:** Pure functions for freshness logic, a `BackupManager` class for backup CRUD, thin FastAPI routers for API endpoints. Domain-to-interval mapping in config. Ingest pipeline updated to set check intervals and track content changes. Wiki article detail shows stale source warning badge.

**Tech Stack:** httpx (async HTTP), sqlite3 (stdlib backup API), FastAPI routers, Pydantic models, pytest with mocked httpx responses.

---

## Task 1: Config additions

**Files:**
- Modify: `code/shukketsu/config.py:108-114`

**Step 1: Add freshness and backup config values**

Add after the reranker config block (line 114):

```python
# Freshness checking
FRESHNESS_CHECK_INTERVAL_HOURS = int(os.getenv("FRESHNESS_CHECK_INTERVAL_HOURS", "6"))
FRESHNESS_HTTP_TIMEOUT = int(os.getenv("FRESHNESS_HTTP_TIMEOUT", "10"))

# Domain → check_interval_hours mapping for new sources during ingest
DOMAIN_CHECK_INTERVALS: dict[str, int] = {
    "wowhead.com": 720,          # 30 days — guides update slowly
    "icy-veins.com": 720,        # 30 days
    "shadowpanther.net": 2160,   # 90 days — TBC content is static
    "silentshadows.net": 2160,   # 90 days
    "tbcdb.com": 2160,           # 90 days — database, patch-locked
    "warcraftlogs.com": 24,      # 1 day — rankings change constantly
}
DEFAULT_CHECK_INTERVAL_HOURS = int(os.getenv("DEFAULT_CHECK_INTERVAL_HOURS", "168"))


def get_check_interval(url: str) -> int:
    """Extract domain from URL, return check_interval_hours from mapping."""
    from urllib.parse import urlparse

    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return DOMAIN_CHECK_INTERVALS.get(domain, DEFAULT_CHECK_INTERVAL_HOURS)


# Backup
BACKUP_KEEP_COUNT = int(os.getenv("BACKUP_KEEP_COUNT", "7"))
```

**Step 2: Verify lint passes**

Run: `ruff check code/shukketsu/config.py && ruff format code/shukketsu/config.py`

**Step 3: Commit**

```bash
git add code/shukketsu/config.py
git commit -m "feat(config): add freshness and backup configuration"
```

---

## Task 2: Freshness checker — models and `find_stale_sources`

**Files:**
- Create: `code/shukketsu/freshness/checker.py`
- Create: `tests/unit/test_freshness_checker.py`

**Step 1: Write the failing tests for models and find_stale_sources**

```python
"""Tests for content freshness checker."""

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from code.shukketsu.freshness.checker import (
    FreshnessResult,
    StaleSource,
    find_stale_sources,
)


class TestStaleSourceModel:
    def test_valid_stale_source(self) -> None:
        s = StaleSource(id=1, url="https://example.com", content_hash="abc123", check_interval_hours=168)
        assert s.id == 1
        assert s.url == "https://example.com"

    def test_null_content_hash(self) -> None:
        s = StaleSource(id=1, url="https://example.com", content_hash=None, check_interval_hours=168)
        assert s.content_hash is None


class TestFreshnessResultModel:
    def test_unchanged_result(self) -> None:
        r = FreshnessResult(
            source_id=1, url="https://example.com", changed=False,
            old_hash="abc", new_hash="abc", checked_at="2026-02-11T00:00:00",
            head_only=False,
        )
        assert not r.changed
        assert r.error is None

    def test_error_result(self) -> None:
        r = FreshnessResult(
            source_id=1, url="https://example.com", changed=False,
            old_hash=None, new_hash=None, checked_at="2026-02-11T00:00:00",
            head_only=False, error="Timeout after 10s",
        )
        assert r.error == "Timeout after 10s"


class TestFindStaleSources:
    def test_returns_overdue_source(self, test_db: sqlite3.Connection) -> None:
        """Source past check_interval is returned."""
        old_time = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) "
            "VALUES (?, ?, ?, ?, ?)",
            ("https://example.com/old", "Old Guide", "hash1", 168, old_time),
        )
        test_db.commit()
        stale = find_stale_sources(test_db)
        assert len(stale) == 1
        assert stale[0].url == "https://example.com/old"

    def test_skips_fresh_source(self, test_db: sqlite3.Connection) -> None:
        """Source within check_interval is not returned."""
        recent_time = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) "
            "VALUES (?, ?, ?, ?, ?)",
            ("https://example.com/fresh", "Fresh Guide", "hash2", 168, recent_time),
        )
        test_db.commit()
        stale = find_stale_sources(test_db)
        assert len(stale) == 0

    def test_null_last_checked_is_stale(self, test_db: sqlite3.Connection) -> None:
        """Source with NULL last_checked is always stale."""
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours) "
            "VALUES (?, ?, ?, ?)",
            ("https://example.com/never", "Never Checked", "hash3", 168),
        )
        test_db.commit()
        stale = find_stale_sources(test_db)
        assert len(stale) == 1

    def test_empty_table(self, test_db: sqlite3.Connection) -> None:
        """No sources returns empty list."""
        stale = find_stale_sources(test_db)
        assert stale == []
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_freshness_checker.py -v`
Expected: FAIL with ImportError (checker.py doesn't exist yet)

**Step 3: Write the implementation**

Create `code/shukketsu/freshness/checker.py`:

```python
"""Content freshness checker: detect stale sources and verify content changes."""

import hashlib
import logging
import sqlite3
from datetime import UTC, datetime

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
    return row[0]
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_freshness_checker.py -v`
Expected: 8 passed

**Step 5: Commit**

```bash
git add code/shukketsu/freshness/checker.py tests/unit/test_freshness_checker.py
git commit -m "feat(freshness): add models, find_stale_sources, count_stale_sources"
```

---

## Task 3: Freshness checker — `check_source_freshness` and `count_stale_sources`

**Files:**
- Modify: `code/shukketsu/freshness/checker.py`
- Modify: `tests/unit/test_freshness_checker.py`

**Step 1: Write the failing tests**

Add to the test file:

```python
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from code.shukketsu.freshness.checker import (
    check_source_freshness,
    count_stale_sources,
    run_freshness_sweep,
)


def _make_stale(
    *,
    id: int = 1,
    url: str = "https://example.com/guide",
    content_hash: str = "oldhash",
    check_interval_hours: int = 168,
) -> StaleSource:
    return StaleSource(id=id, url=url, content_hash=content_hash, check_interval_hours=check_interval_hours)


class TestCheckSourceFreshness:
    async def test_head_last_modified_fresh(self, test_db: sqlite3.Connection) -> None:
        """Last-Modified before last_checked → head_only=True, changed=False."""
        now = datetime.now(UTC)
        old_time = (now - timedelta(hours=200)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) "
            "VALUES (?, ?, ?, ?, ?)",
            ("https://example.com/guide", "Guide", "oldhash", 168, old_time),
        )
        test_db.commit()
        source = _make_stale()

        # Last-Modified is before last_checked
        head_resp = httpx.Response(200, headers={"Last-Modified": "Sat, 01 Jan 2025 00:00:00 GMT"})
        with patch("code.shukketsu.freshness.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.head = AsyncMock(return_value=head_resp)
            mock_client_cls.return_value = mock_client

            result = await check_source_freshness(source, test_db)

        assert not result.changed
        assert result.head_only

    async def test_head_last_modified_stale(self, test_db: sqlite3.Connection) -> None:
        """Last-Modified after last_checked → triggers full fetch."""
        old_time = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) "
            "VALUES (?, ?, ?, ?, ?)",
            ("https://example.com/guide", "Guide", "oldhash", 168, old_time),
        )
        test_db.commit()
        source = _make_stale()

        # Last-Modified is very recent → content might have changed
        future_time = "Thu, 01 Jan 2099 00:00:00 GMT"
        head_resp = httpx.Response(200, headers={"Last-Modified": future_time})
        # Full fetch returns same content → unchanged
        old_content = "original content"
        old_hash = hashlib.sha256(old_content.encode()).hexdigest()
        get_resp = httpx.Response(200, text=old_content)

        # Re-insert with the correct hash
        test_db.execute("UPDATE sources SET content_hash = ? WHERE url = ?", (old_hash, source.url))
        test_db.commit()
        source = _make_stale(content_hash=old_hash)

        with patch("code.shukketsu.freshness.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.head = AsyncMock(return_value=head_resp)
            mock_client.get = AsyncMock(return_value=get_resp)
            mock_client_cls.return_value = mock_client

            result = await check_source_freshness(source, test_db)

        assert not result.changed
        assert not result.head_only

    async def test_head_no_headers_triggers_full_fetch(self, test_db: sqlite3.Connection) -> None:
        """No Last-Modified header → full fetch."""
        old_time = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) "
            "VALUES (?, ?, ?, ?, ?)",
            ("https://example.com/guide", "Guide", "oldhash", 168, old_time),
        )
        test_db.commit()
        source = _make_stale()

        head_resp = httpx.Response(200, headers={})  # No Last-Modified
        get_resp = httpx.Response(200, text="new content")

        with patch("code.shukketsu.freshness.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.head = AsyncMock(return_value=head_resp)
            mock_client.get = AsyncMock(return_value=get_resp)
            mock_client_cls.return_value = mock_client

            result = await check_source_freshness(source, test_db)

        assert result.changed  # "new content" hash != "oldhash"
        assert not result.head_only

    async def test_full_fetch_unchanged(self, test_db: sqlite3.Connection) -> None:
        """Same content hash after full fetch → changed=False, last_checked updated."""
        content = "same content"
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        old_time = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) "
            "VALUES (?, ?, ?, ?, ?)",
            ("https://example.com/guide", "Guide", content_hash, 168, old_time),
        )
        test_db.commit()
        source = _make_stale(content_hash=content_hash)

        head_resp = httpx.Response(200, headers={})
        get_resp = httpx.Response(200, text=content)

        with patch("code.shukketsu.freshness.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.head = AsyncMock(return_value=head_resp)
            mock_client.get = AsyncMock(return_value=get_resp)
            mock_client_cls.return_value = mock_client

            result = await check_source_freshness(source, test_db)

        assert not result.changed
        # last_checked should be updated
        row = test_db.execute("SELECT last_checked FROM sources WHERE id = ?", (source.id,)).fetchone()
        assert row["last_checked"] != old_time

    async def test_full_fetch_changed(self, test_db: sqlite3.Connection) -> None:
        """Different hash → changed=True, is_stale=1, content_hash_previous set."""
        old_time = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked, change_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("https://example.com/guide", "Guide", "oldhash", 168, old_time, 0),
        )
        test_db.commit()
        source = _make_stale()

        head_resp = httpx.Response(200, headers={})
        get_resp = httpx.Response(200, text="totally new content")

        with patch("code.shukketsu.freshness.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.head = AsyncMock(return_value=head_resp)
            mock_client.get = AsyncMock(return_value=get_resp)
            mock_client_cls.return_value = mock_client

            result = await check_source_freshness(source, test_db)

        assert result.changed
        row = test_db.execute(
            "SELECT content_hash, content_hash_previous, change_count, is_stale FROM sources WHERE id = ?",
            (source.id,),
        ).fetchone()
        assert row["content_hash_previous"] == "oldhash"
        assert row["change_count"] == 1
        assert row["is_stale"] == 1

    async def test_change_count_increments(self, test_db: sqlite3.Connection) -> None:
        """change_count goes up on each confirmed change."""
        old_time = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked, change_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("https://example.com/guide", "Guide", "oldhash", 168, old_time, 5),
        )
        test_db.commit()
        source = _make_stale()

        head_resp = httpx.Response(200, headers={})
        get_resp = httpx.Response(200, text="new content")

        with patch("code.shukketsu.freshness.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.head = AsyncMock(return_value=head_resp)
            mock_client.get = AsyncMock(return_value=get_resp)
            mock_client_cls.return_value = mock_client

            await check_source_freshness(source, test_db)

        row = test_db.execute("SELECT change_count FROM sources WHERE id = ?", (source.id,)).fetchone()
        assert row["change_count"] == 6

    async def test_network_error(self, test_db: sqlite3.Connection) -> None:
        """Network error → FreshnessResult.error set, source row unchanged."""
        old_time = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked, is_stale) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("https://example.com/guide", "Guide", "oldhash", 168, old_time, 0),
        )
        test_db.commit()
        source = _make_stale()

        with patch("code.shukketsu.freshness.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.head = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client_cls.return_value = mock_client

            result = await check_source_freshness(source, test_db)

        assert result.error is not None
        assert "Connection refused" in result.error
        # Source row unchanged
        row = test_db.execute("SELECT last_checked, is_stale FROM sources WHERE id = ?", (source.id,)).fetchone()
        assert row["last_checked"] == old_time
        assert row["is_stale"] == 0

    async def test_http_404(self, test_db: sqlite3.Connection) -> None:
        """HTTP 404 → error set, source unchanged."""
        old_time = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) "
            "VALUES (?, ?, ?, ?, ?)",
            ("https://example.com/gone", "Gone Page", "oldhash", 168, old_time),
        )
        test_db.commit()
        source = _make_stale(url="https://example.com/gone")

        head_resp = httpx.Response(404)

        with patch("code.shukketsu.freshness.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.head = AsyncMock(return_value=head_resp)
            mock_client_cls.return_value = mock_client

            result = await check_source_freshness(source, test_db)

        assert result.error is not None
        assert "404" in result.error


class TestCountStaleSources:
    def test_some_stale(self, test_db: sqlite3.Connection) -> None:
        """Returns correct count for mixed stale/fresh URLs."""
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, is_stale) VALUES (?, ?, ?, ?)",
            ("https://a.com", "A", "h1", 1),
        )
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, is_stale) VALUES (?, ?, ?, ?)",
            ("https://b.com", "B", "h2", 0),
        )
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, is_stale) VALUES (?, ?, ?, ?)",
            ("https://c.com", "C", "h3", 1),
        )
        test_db.commit()
        assert count_stale_sources(test_db, ["https://a.com", "https://b.com", "https://c.com"]) == 2

    def test_empty_list(self, test_db: sqlite3.Connection) -> None:
        """Empty URL list returns 0."""
        assert count_stale_sources(test_db, []) == 0
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_freshness_checker.py -v`
Expected: FAIL (functions not yet defined)

**Step 3: Implement `check_source_freshness` and `run_freshness_sweep`**

Add to `code/shukketsu/freshness/checker.py`:

```python
import httpx
from email.utils import parsedate_to_datetime

async def check_source_freshness(
    source: StaleSource, conn: sqlite3.Connection, timeout: int | None = None,
) -> FreshnessResult:
    """Check a single source for content changes.

    Flow:
    1. HEAD request for Last-Modified header
    2. If Last-Modified is before source's last_checked → mark fresh (head_only)
    3. If header missing or stale → full GET fetch
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
                    source_id=source.id, url=source.url, changed=False,
                    old_hash=source.content_hash, new_hash=None,
                    checked_at=now, head_only=True,
                    error=f"Source returned HTTP {head_resp.status_code}",
                )

            # Step 2: Check Last-Modified
            last_modified_str = head_resp.headers.get("Last-Modified")
            if last_modified_str:
                try:
                    last_modified = parsedate_to_datetime(last_modified_str)
                    # Get the source's last_checked from DB
                    row = conn.execute(
                        "SELECT last_checked FROM sources WHERE id = ?", (source.id,)
                    ).fetchone()
                    if row and row["last_checked"]:
                        last_checked_dt = datetime.fromisoformat(row["last_checked"])
                        if last_modified.astimezone(UTC) < last_checked_dt.astimezone(UTC):
                            # Content hasn't changed since our last check
                            conn.execute(
                                "UPDATE sources SET last_checked = ? WHERE id = ?",
                                (now, source.id),
                            )
                            conn.commit()
                            return FreshnessResult(
                                source_id=source.id, url=source.url, changed=False,
                                old_hash=source.content_hash, new_hash=source.content_hash,
                                checked_at=now, head_only=True,
                            )
                except (ValueError, TypeError):
                    pass  # Malformed header — fall through to full fetch

            # Step 3: Full GET fetch
            get_resp = await client.get(source.url)
            new_hash = hashlib.sha256(get_resp.text.encode()).hexdigest()

    except (httpx.HTTPError, OSError) as exc:
        return FreshnessResult(
            source_id=source.id, url=source.url, changed=False,
            old_hash=source.content_hash, new_hash=None,
            checked_at=now, head_only=False, error=str(exc),
        )

    # Step 4: Compare hashes
    if new_hash == source.content_hash:
        conn.execute("UPDATE sources SET last_checked = ? WHERE id = ?", (now, source.id))
        conn.commit()
        return FreshnessResult(
            source_id=source.id, url=source.url, changed=False,
            old_hash=source.content_hash, new_hash=new_hash,
            checked_at=now, head_only=False,
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
        source_id=source.id, url=source.url, changed=True,
        old_hash=source.content_hash, new_hash=new_hash,
        checked_at=now, head_only=False,
    )


async def run_freshness_sweep(
    conn: sqlite3.Connection, timeout: int | None = None,
) -> list[FreshnessResult]:
    """Find all stale sources and check each. Returns results for all checked sources."""
    stale = find_stale_sources(conn)
    results = []
    for source in stale:
        result = await check_source_freshness(source, conn, timeout=timeout)
        results.append(result)
    return results
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_freshness_checker.py -v`
Expected: 18 passed

**Step 5: Commit**

```bash
git add code/shukketsu/freshness/checker.py tests/unit/test_freshness_checker.py
git commit -m "feat(freshness): add check_source_freshness, run_freshness_sweep"
```

---

## Task 4: Backup manager

**Files:**
- Create: `code/shukketsu/backup/manager.py`
- Create: `tests/unit/test_backup_manager.py`

**Step 1: Write the failing tests**

```python
"""Tests for SQLite backup manager."""

import sqlite3
import time

import pytest

from code.shukketsu.backup.manager import BackupManager, BackupResult


@pytest.fixture
def backup_mgr(tmp_path) -> BackupManager:
    """BackupManager with a real SQLite DB and temp backup dir."""
    db_path = tmp_path / "test.db"
    # Create a minimal SQLite DB
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO test (value) VALUES ('hello')")
    conn.commit()
    conn.close()

    backup_dir = tmp_path / "backups"
    return BackupManager(db_path, backup_dir)


class TestCreateBackup:
    def test_produces_file(self, backup_mgr: BackupManager) -> None:
        result = backup_mgr.create_backup()
        from pathlib import Path

        assert Path(result.path).exists()
        assert result.size_bytes > 0

    def test_filename_format(self, backup_mgr: BackupManager) -> None:
        import re

        result = backup_mgr.create_backup()
        filename = result.path.split("/")[-1]
        assert re.match(r"shukketsu-\d{8}-\d{6}\.db$", filename)

    def test_integrity_ok(self, backup_mgr: BackupManager) -> None:
        result = backup_mgr.create_backup()
        assert result.integrity_ok


class TestVerifyIntegrity:
    def test_good_db(self, backup_mgr: BackupManager) -> None:
        result = backup_mgr.create_backup()
        from pathlib import Path

        assert backup_mgr.verify_integrity(Path(result.path))

    def test_bad_file(self, backup_mgr: BackupManager, tmp_path) -> None:
        bad_file = tmp_path / "corrupt.db"
        bad_file.write_text("this is not a database")
        assert not backup_mgr.verify_integrity(bad_file)


class TestListBackups:
    def test_sorted_newest_first(self, backup_mgr: BackupManager) -> None:
        backup_mgr.create_backup()
        time.sleep(1.1)  # Ensure different timestamp
        backup_mgr.create_backup()
        backups = backup_mgr.list_backups()
        assert len(backups) == 2
        assert backups[0].created_at >= backups[1].created_at


class TestPrune:
    def test_keeps_correct_count(self, backup_mgr: BackupManager) -> None:
        for _ in range(5):
            backup_mgr.create_backup()
            time.sleep(1.1)
        deleted = backup_mgr.prune_old_backups(keep=3)
        assert deleted == 2
        assert len(backup_mgr.list_backups()) == 3

    def test_prune_empty_directory(self, backup_mgr: BackupManager) -> None:
        deleted = backup_mgr.prune_old_backups(keep=3)
        assert deleted == 0
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_backup_manager.py -v`
Expected: FAIL with ImportError

**Step 3: Implement BackupManager**

Create `code/shukketsu/backup/manager.py`:

```python
"""SQLite online backup manager with integrity verification and pruning."""

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from code.shukketsu import config

logger = logging.getLogger(__name__)


class BackupResult(BaseModel):
    """Result of a backup operation."""

    path: str
    size_bytes: int
    created_at: str  # ISO 8601
    integrity_ok: bool


class BackupManager:
    """Create, verify, list, and prune SQLite database backups."""

    def __init__(self, db_path: Path, backup_dir: Path) -> None:
        self._db_path = db_path
        self._backup_dir = backup_dir
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self) -> BackupResult:
        """Create timestamped backup using sqlite3.Connection.backup()."""
        now = datetime.now(UTC)
        filename = f"shukketsu-{now.strftime('%Y%m%d-%H%M%S')}.db"
        backup_path = self._backup_dir / filename

        src = sqlite3.connect(str(self._db_path))
        dst = sqlite3.connect(str(backup_path))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()

        integrity_ok = self.verify_integrity(backup_path)
        size_bytes = backup_path.stat().st_size

        logger.info("Backup created: %s (%d bytes, integrity=%s)", backup_path, size_bytes, integrity_ok)
        return BackupResult(
            path=str(backup_path),
            size_bytes=size_bytes,
            created_at=now.isoformat(),
            integrity_ok=integrity_ok,
        )

    def verify_integrity(self, backup_path: Path) -> bool:
        """Run PRAGMA integrity_check on a backup file."""
        try:
            conn = sqlite3.connect(str(backup_path))
            result = conn.execute("PRAGMA integrity_check").fetchone()
            conn.close()
            return result is not None and result[0] == "ok"
        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            return False

    def list_backups(self) -> list[BackupResult]:
        """List existing backups sorted newest first."""
        backups = []
        for path in sorted(self._backup_dir.glob("shukketsu-*.db"), reverse=True):
            stat = path.stat()
            # Parse timestamp from filename: shukketsu-YYYYMMDD-HHMMSS.db
            stem = path.stem  # shukketsu-YYYYMMDD-HHMMSS
            try:
                ts_str = stem.replace("shukketsu-", "")
                created = datetime.strptime(ts_str, "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
            except ValueError:
                created = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            backups.append(
                BackupResult(
                    path=str(path),
                    size_bytes=stat.st_size,
                    created_at=created.isoformat(),
                    integrity_ok=True,  # Assumed OK since verified at creation
                )
            )
        return backups

    def prune_old_backups(self, keep: int | None = None) -> int:
        """Delete all but the most recent `keep` backups. Returns count deleted."""
        if keep is None:
            keep = config.BACKUP_KEEP_COUNT
        backups = list(sorted(self._backup_dir.glob("shukketsu-*.db"), reverse=True))
        to_delete = backups[keep:]
        for path in to_delete:
            path.unlink()
            logger.info("Pruned old backup: %s", path)
        return len(to_delete)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_backup_manager.py -v`
Expected: 8 passed

**Step 5: Commit**

```bash
git add code/shukketsu/backup/manager.py tests/unit/test_backup_manager.py
git commit -m "feat(backup): add BackupManager with create, verify, list, prune"
```

---

## Task 5: Freshness API routes

**Files:**
- Create: `code/shukketsu/web/routers/freshness.py`
- Create: `tests/unit/test_freshness_routes.py`

**Step 1: Write the failing tests**

```python
"""Tests for freshness API routes."""

import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


def _get_app():
    from code.shukketsu.web.app import app

    return app


@pytest.fixture
def freshness_db(tmp_path) -> sqlite3.Connection:
    """Cross-thread-safe DB for TestClient."""
    import sqlite_vec

    from code.shukketsu.db.connection import init_db

    db_path = tmp_path / "freshness_test.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def fclient(freshness_db: sqlite3.Connection):
    """TestClient with freshness DB dependency overridden."""
    from code.shukketsu.web.routers.freshness import _get_conn

    app = _get_app()
    app.dependency_overrides[_get_conn] = lambda: freshness_db

    # Patch run_freshness_sweep to use our DB
    with patch("code.shukketsu.web.routers.freshness.run_freshness_sweep") as mock_sweep:
        mock_sweep.return_value = []
        yield TestClient(app), mock_sweep, freshness_db

    app.dependency_overrides.clear()


class TestFreshnessRoutes:
    def test_post_check_returns_results(self, fclient) -> None:
        client, mock_sweep, db = fclient
        from code.shukketsu.freshness.checker import FreshnessResult

        mock_sweep.return_value = [
            FreshnessResult(
                source_id=1, url="https://example.com", changed=True,
                old_hash="old", new_hash="new", checked_at="2026-01-01T00:00:00",
                head_only=False,
            )
        ]
        resp = client.post("/api/freshness/check")
        assert resp.status_code == 200
        data = resp.json()
        assert data["checked"] == 1
        assert data["changed"] == 1

    def test_get_stale_returns_list(self, fclient) -> None:
        client, _, db = fclient
        db.execute(
            "INSERT INTO sources (url, title, content_hash, is_stale) VALUES (?, ?, ?, ?)",
            ("https://stale.com", "Stale", "hash", 1),
        )
        db.commit()
        resp = client.get("/api/freshness/stale")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["stale_sources"]) == 1
        assert data["stale_sources"][0]["url"] == "https://stale.com"

    def test_get_stale_empty(self, fclient) -> None:
        client, _, _ = fclient
        resp = client.get("/api/freshness/stale")
        assert resp.status_code == 200
        assert resp.json()["stale_sources"] == []

    def test_clear_stale_flag(self, fclient) -> None:
        client, _, db = fclient
        db.execute(
            "INSERT INTO sources (url, title, content_hash, is_stale) VALUES (?, ?, ?, ?)",
            ("https://stale.com", "Stale", "hash", 1),
        )
        db.commit()
        source_id = db.execute("SELECT id FROM sources WHERE url = ?", ("https://stale.com",)).fetchone()["id"]
        resp = client.post(f"/api/freshness/clear/{source_id}")
        assert resp.status_code == 200
        row = db.execute("SELECT is_stale FROM sources WHERE id = ?", (source_id,)).fetchone()
        assert row["is_stale"] == 0

    def test_clear_nonexistent_404(self, fclient) -> None:
        client, _, _ = fclient
        resp = client.post("/api/freshness/clear/9999")
        assert resp.status_code == 404
```

**Step 2: Implement freshness routes**

Create `code/shukketsu/web/routers/freshness.py`:

```python
"""Freshness check API routes."""

import logging
import sqlite3

from fastapi import APIRouter, Depends

from code.shukketsu import config
from code.shukketsu.freshness.checker import run_freshness_sweep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/freshness", tags=["freshness"])

_conn_instance: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    """Get or create DB connection singleton."""
    global _conn_instance  # noqa: PLW0603
    if _conn_instance is None:
        from code.shukketsu.db.connection import get_connection, init_db

        _conn_instance = get_connection()
        init_db(_conn_instance)
    return _conn_instance


@router.post("/check")
async def freshness_check(conn: sqlite3.Connection = Depends(_get_conn)) -> dict:
    """Run freshness sweep. Returns summary of results."""
    results = await run_freshness_sweep(conn)
    return {
        "checked": len(results),
        "changed": sum(1 for r in results if r.changed),
        "errors": sum(1 for r in results if r.error),
        "results": [r.model_dump() for r in results],
    }


@router.get("/stale")
async def list_stale_sources(conn: sqlite3.Connection = Depends(_get_conn)) -> dict:
    """List all currently stale sources (is_stale = 1)."""
    rows = conn.execute(
        "SELECT id, url, change_count, last_checked FROM sources WHERE is_stale = 1"
    ).fetchall()
    return {
        "stale_sources": [
            {"id": row["id"], "url": row["url"], "change_count": row["change_count"], "last_checked": row["last_checked"]}
            for row in rows
        ],
    }


@router.post("/clear/{source_id}")
async def clear_stale_flag(source_id: int, conn: sqlite3.Connection = Depends(_get_conn)) -> dict:
    """Clear is_stale flag for a source."""
    row = conn.execute("SELECT id FROM sources WHERE id = ?", (source_id,)).fetchone()
    if row is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")
    conn.execute("UPDATE sources SET is_stale = 0 WHERE id = ?", (source_id,))
    conn.commit()
    return {"cleared": True, "source_id": source_id}
```

**Step 3: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_freshness_routes.py -v`
Expected: 5 passed

**Step 4: Commit**

```bash
git add code/shukketsu/web/routers/freshness.py tests/unit/test_freshness_routes.py
git commit -m "feat(web): add freshness API routes"
```

---

## Task 6: Backup API routes

**Files:**
- Create: `code/shukketsu/web/routers/backup.py`
- Create: `tests/unit/test_backup_routes.py`

**Step 1: Write the failing tests**

```python
"""Tests for backup API routes."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from code.shukketsu.backup.manager import BackupManager, BackupResult


def _get_app():
    from code.shukketsu.web.app import app

    return app


@pytest.fixture
def backup_client(tmp_path):
    """TestClient with BackupManager mocked."""
    from code.shukketsu.web.routers.backup import _get_backup_manager

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE test (id INTEGER)")
    conn.commit()
    conn.close()

    backup_dir = tmp_path / "backups"
    mgr = BackupManager(db_path, backup_dir)

    app = _get_app()
    app.dependency_overrides[_get_backup_manager] = lambda: mgr
    yield TestClient(app), mgr
    app.dependency_overrides.clear()


class TestBackupRoutes:
    def test_post_create_backup(self, backup_client) -> None:
        client, _ = backup_client
        resp = client.post("/api/backup/create")
        assert resp.status_code == 200
        data = resp.json()
        assert data["integrity_ok"] is True
        assert data["size_bytes"] > 0

    def test_get_list_backups(self, backup_client) -> None:
        client, mgr = backup_client
        mgr.create_backup()
        resp = client.get("/api/backup/list")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert len(data["backups"]) == 1

    def test_post_prune_backups(self, backup_client) -> None:
        client, mgr = backup_client
        mgr.create_backup()
        resp = client.post("/api/backup/prune")
        assert resp.status_code == 200
        data = resp.json()
        assert "deleted" in data

    def test_prune_empty(self, backup_client) -> None:
        client, _ = backup_client
        resp = client.post("/api/backup/prune")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 0
```

**Step 2: Implement backup routes**

Create `code/shukketsu/web/routers/backup.py`:

```python
"""Backup management API routes."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends

from code.shukketsu import config
from code.shukketsu.backup.manager import BackupManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backup", tags=["backup"])

_mgr_instance: BackupManager | None = None


def _get_backup_manager() -> BackupManager:
    """Get or create BackupManager singleton."""
    global _mgr_instance  # noqa: PLW0603
    if _mgr_instance is None:
        _mgr_instance = BackupManager(config.DB_PATH, config.BACKUP_PATH)
    return _mgr_instance


@router.post("/create")
async def create_backup(mgr: BackupManager = Depends(_get_backup_manager)) -> dict:
    """Create a new backup."""
    result = mgr.create_backup()
    return result.model_dump()


@router.get("/list")
async def list_backups(mgr: BackupManager = Depends(_get_backup_manager)) -> dict:
    """List existing backups."""
    backups = mgr.list_backups()
    return {"backups": [b.model_dump() for b in backups], "count": len(backups)}


@router.post("/prune")
async def prune_backups(mgr: BackupManager = Depends(_get_backup_manager)) -> dict:
    """Prune old backups."""
    deleted = mgr.prune_old_backups()
    remaining = len(mgr.list_backups())
    return {"deleted": deleted, "remaining": remaining}
```

**Step 3: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_backup_routes.py -v`
Expected: 4 passed

**Step 4: Commit**

```bash
git add code/shukketsu/web/routers/backup.py tests/unit/test_backup_routes.py
git commit -m "feat(web): add backup API routes"
```

---

## Task 7: Mount routers + startup sweep in app.py

**Files:**
- Modify: `code/shukketsu/web/app.py`

**Step 1: Add router imports and startup sweep**

Modify `code/shukketsu/web/app.py` to import and mount the new routers, and add the freshness sweep to the lifespan:

The import block (top of file) becomes:

```python
from code.shukketsu.web.routers.backup import router as backup_router
from code.shukketsu.web.routers.chat import router as chat_router
from code.shukketsu.web.routers.freshness import router as freshness_router
from code.shukketsu.web.routers.wiki import router as wiki_router
```

The lifespan function becomes:

```python
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown hooks."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    init_langfuse()

    # Run freshness sweep on startup (non-blocking — stale sources just get flagged)
    try:
        from code.shukketsu.db.connection import get_connection, init_db
        from code.shukketsu.freshness.checker import run_freshness_sweep

        conn = get_connection()
        init_db(conn)
        results = await run_freshness_sweep(conn)
        stale_count = sum(1 for r in results if r.changed)
        logger.info("Freshness sweep: %d checked, %d changed", len(results), stale_count)
    except Exception:
        logger.exception("Freshness sweep failed on startup")

    yield
    flush_traces()
```

Add `logger = logging.getLogger(__name__)` near the top.

The router mounts become:

```python
app.include_router(chat_router)
app.include_router(wiki_router)
app.include_router(freshness_router)
app.include_router(backup_router)
```

**Step 2: Run lint and existing tests**

Run: `ruff check code/shukketsu/web/app.py && python3 -m pytest tests/unit/ -q --tb=short`
Expected: All 652+ tests pass, no lint errors

**Step 3: Commit**

```bash
git add code/shukketsu/web/app.py
git commit -m "feat(web): mount freshness and backup routers, add startup sweep"
```

---

## Task 8: Ingest pipeline freshness integration

**Files:**
- Modify: `code/shukketsu/ingest/pipeline.py:108-113`
- Create: `tests/unit/test_ingest_freshness.py`

**Step 1: Write the failing tests**

```python
"""Tests for ingest pipeline freshness integration."""

import hashlib
import sqlite3
from unittest.mock import AsyncMock

import pytest

from code.shukketsu.ingest.pipeline import IngestPipeline

EMBEDDING_DIM = 768


def _mock_embedder() -> AsyncMock:
    embedder = AsyncMock()

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[0.1] * EMBEDDING_DIM for _ in texts]

    embedder.embed_texts = AsyncMock(side_effect=fake_embed_texts)
    return embedder


class TestIngestFreshness:
    async def test_new_source_gets_domain_check_interval(self, test_db: sqlite3.Connection) -> None:
        """wowhead.com source gets 720h, unknown domain gets 168h."""
        pipeline = IngestPipeline(conn=test_db, embedder=_mock_embedder())

        await pipeline.ingest("Some guide content here.", "https://wowhead.com/tbc/guide", "WoWHead Guide")
        row = test_db.execute("SELECT check_interval_hours FROM sources WHERE url = ?", ("https://wowhead.com/tbc/guide",)).fetchone()
        assert row["check_interval_hours"] == 720

        await pipeline.ingest("Another guide.", "https://unknown-site.com/guide", "Unknown Guide")
        row = test_db.execute("SELECT check_interval_hours FROM sources WHERE url = ?", ("https://unknown-site.com/guide",)).fetchone()
        assert row["check_interval_hours"] == 168

    async def test_reingest_changed_content_updates_tracking(self, test_db: sqlite3.Connection) -> None:
        """On re-ingest with different content: content_hash_previous set, change_count incremented, is_stale cleared."""
        pipeline = IngestPipeline(conn=test_db, embedder=_mock_embedder())

        # First ingest
        await pipeline.ingest("Original content.", "https://example.com/guide", "Guide")
        row = test_db.execute("SELECT id, content_hash, change_count, is_stale FROM sources WHERE url = ?", ("https://example.com/guide",)).fetchone()
        original_hash = row["content_hash"]
        assert row["change_count"] == 0

        # Mark as stale (simulating freshness checker)
        test_db.execute("UPDATE sources SET is_stale = 1 WHERE id = ?", (row["id"],))
        test_db.commit()

        # Re-ingest with different content
        await pipeline.ingest("Updated content.", "https://example.com/guide", "Guide")
        row = test_db.execute("SELECT content_hash, content_hash_previous, change_count, is_stale FROM sources WHERE url = ?", ("https://example.com/guide",)).fetchone()
        assert row["content_hash_previous"] == original_hash
        assert row["content_hash"] != original_hash
        assert row["change_count"] == 1
        assert row["is_stale"] == 0

    async def test_reingest_same_content_no_tracking_change(self, test_db: sqlite3.Connection) -> None:
        """Unchanged content doesn't modify tracking columns."""
        pipeline = IngestPipeline(conn=test_db, embedder=_mock_embedder())

        await pipeline.ingest("Same content.", "https://example.com/guide", "Guide")
        result = await pipeline.ingest("Same content.", "https://example.com/guide", "Guide")
        assert result.already_existed

        row = test_db.execute("SELECT change_count, is_stale FROM sources WHERE url = ?", ("https://example.com/guide",)).fetchone()
        assert row["change_count"] == 0
        assert row["is_stale"] == 0
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_ingest_freshness.py -v`
Expected: FAIL (pipeline doesn't set check_interval_hours or tracking columns yet)

**Step 3: Modify the ingest pipeline**

In `code/shukketsu/ingest/pipeline.py`, make two changes:

**Change 1:** In the `else` branch (new source insert, around line 108-113), replace:

```python
            else:
                cursor = self._conn.execute(
                    "INSERT INTO sources (url, title, source_type, content_hash, fetched_at) VALUES (?, ?, ?, ?, ?)",
                    (url, title, source_type, content_hash, datetime.now(UTC).isoformat()),
                )
                source_id = cursor.lastrowid
```

With:

```python
            else:
                from code.shukketsu.config import get_check_interval

                now_iso = datetime.now(UTC).isoformat()
                check_interval = get_check_interval(url)
                cursor = self._conn.execute(
                    "INSERT INTO sources (url, title, source_type, content_hash, fetched_at, "
                    "check_interval_hours, last_checked) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (url, title, source_type, content_hash, now_iso, check_interval, now_iso),
                )
                source_id = cursor.lastrowid
```

**Change 2:** In the `if existing` branch where content differs (around line 100-107), replace:

```python
            if existing:
                source_id = existing["id"]
                self._delete_chunks_and_vectors(source_id)
                self._conn.execute(
                    "UPDATE sources SET title = ?, source_type = ?, content_hash = ?, "
                    "fetched_at = ?, chunk_count = 0 WHERE id = ?",
                    (title, source_type, content_hash, datetime.now(UTC).isoformat(), source_id),
                )
```

With:

```python
            if existing:
                source_id = existing["id"]
                self._delete_chunks_and_vectors(source_id)
                now_iso = datetime.now(UTC).isoformat()
                self._conn.execute(
                    "UPDATE sources SET title = ?, source_type = ?, "
                    "content_hash_previous = content_hash, content_hash = ?, "
                    "change_count = change_count + 1, is_stale = 0, "
                    "fetched_at = ?, last_checked = ?, chunk_count = 0 WHERE id = ?",
                    (title, source_type, content_hash, now_iso, now_iso, source_id),
                )
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_ingest_freshness.py -v`
Expected: 3 passed

Also run all existing tests to check for regressions:
Run: `python3 -m pytest tests/unit/ -q --tb=short`
Expected: All pass (existing pipeline tests should still pass since the new columns are additive)

**Step 5: Commit**

```bash
git add code/shukketsu/ingest/pipeline.py tests/unit/test_ingest_freshness.py
git commit -m "feat(ingest): set check_interval from domain, track content changes on re-ingest"
```

---

## Task 9: Wiki UI stale source warning

**Files:**
- Modify: `code/shukketsu/web/routers/wiki.py:90-119`
- Modify: `code/shukketsu/web/templates/wiki/article.html:53-69`

**Step 1: Modify wiki route to pass stale count**

In `code/shukketsu/web/routers/wiki.py`, in the `wiki_article()` handler, add stale source count computation. After line 113 (`meta = meta.model_copy(...)`) and before line 114 (`article_html = render_markdown(content)`), add:

```python
    # Compute stale source count for warning badge
    stale_source_count = 0
    if meta.sources:
        from code.shukketsu.freshness.checker import count_stale_sources

        source_urls = [s.url for s in meta.sources]
        stale_source_count = count_stale_sources(km._conn, source_urls)
```

Then update the template context dict to include `"stale_source_count": stale_source_count`.

The return statement becomes:

```python
    return _templates.TemplateResponse(
        request,
        "wiki/article.html",
        {"meta": meta, "article_html": article_html, "stale_source_count": stale_source_count},
    )
```

**Step 2: Add stale warning badge to article template**

In `code/shukketsu/web/templates/wiki/article.html`, after the Sources sidebar card (after line 69 `{% endif %}`), add:

```html
        <!-- Stale Source Warning -->
        {% if stale_source_count > 0 %}
        <div class="sidebar-card border-yellow-700 bg-yellow-900/20">
            <div class="flex items-center gap-2">
                <span class="text-yellow-400 text-sm font-medium">
                    {{ stale_source_count }} source{{ "s" if stale_source_count != 1 else "" }} may be outdated
                </span>
            </div>
        </div>
        {% endif %}
```

**Step 3: Run existing wiki route tests to verify no regressions**

Run: `python3 -m pytest tests/unit/test_wiki_routes.py -v`
Expected: All 16 pass (stale_source_count defaults to 0 so existing tests unaffected)

**Step 4: Commit**

```bash
git add code/shukketsu/web/routers/wiki.py code/shukketsu/web/templates/wiki/article.html
git commit -m "feat(wiki): add stale source warning badge on article detail"
```

---

## Task 10: Lint, type-check, and full test suite

**Files:** None (verification only)

**Step 1: Run ruff check and format**

Run: `ruff check code/ tests/ --fix && ruff format code/ tests/`

Fix any lint issues that arise. Common issues:
- Long lines (120 char limit)
- Import ordering
- Unused imports

**Step 2: Run mypy**

Run: `python3 -m mypy code/shukketsu/`

Fix any type errors. Likely candidates:
- `conn.execute()` return types
- `model_dump()` return type

**Step 3: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v`
Expected: ~686 tests passed

**Step 4: Final commit with any fixes**

```bash
# Only if there were lint/type fixes
git add -u
git commit -m "chore: lint and type fixes for Phase 2 Step 9"
```

---

## Task 11: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update test count and step status**

Change the step 9 line from:
```
9. Content Freshness + Automated Backups (staleness detection, SQLite backup) — DESIGN COMPLETE
```
To:
```
9. ~~Content Freshness + Automated Backups~~ — COMPLETE (N tests: freshness checker, backup manager, API routes, ingest integration, wiki stale badge)
```

Where N is the actual test count from the previous step.

Update the header line:
```
Phase 2 Steps 1-9 are complete (N unit tests).
```

Add the implementation plan to the plans table:
```
| `2026-02-11-phase2-step9-implementation.md` | Step 9 implementation plan (complete) |
```

Also update `Key files with real code` to include `freshness/checker.py`, `backup/manager.py`.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Phase 2 Step 9 completion"
```
