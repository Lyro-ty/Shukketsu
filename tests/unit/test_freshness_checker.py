"""Tests for content freshness checker."""

import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import httpx

from code.shukketsu.freshness.checker import (
    FreshnessResult,
    StaleSource,
    check_source_freshness,
    count_stale_sources,
    find_stale_sources,
    run_freshness_sweep,
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
            source_id=1,
            url="https://example.com",
            changed=False,
            old_hash="abc",
            new_hash="abc",
            checked_at="2026-02-11T00:00:00",
            head_only=False,
        )
        assert not r.changed
        assert r.error is None

    def test_error_result(self) -> None:
        r = FreshnessResult(
            source_id=1,
            url="https://example.com",
            changed=False,
            old_hash=None,
            new_hash=None,
            checked_at="2026-02-11T00:00:00",
            head_only=False,
            error="Timeout after 10s",
        )
        assert r.error == "Timeout after 10s"


class TestFindStaleSources:
    def test_returns_overdue_source(self, test_db: sqlite3.Connection) -> None:
        """Source past check_interval is returned."""
        old_time = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) VALUES (?, ?, ?, ?, ?)",
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
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) VALUES (?, ?, ?, ?, ?)",
            ("https://example.com/fresh", "Fresh Guide", "hash2", 168, recent_time),
        )
        test_db.commit()
        stale = find_stale_sources(test_db)
        assert len(stale) == 0

    def test_null_last_checked_is_stale(self, test_db: sqlite3.Connection) -> None:
        """Source with NULL last_checked is always stale."""
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours) VALUES (?, ?, ?, ?)",
            ("https://example.com/never", "Never Checked", "hash3", 168),
        )
        test_db.commit()
        stale = find_stale_sources(test_db)
        assert len(stale) == 1

    def test_empty_table(self, test_db: sqlite3.Connection) -> None:
        """No sources returns empty list."""
        stale = find_stale_sources(test_db)
        assert stale == []


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
        """Last-Modified before last_checked -> head_only=True, changed=False."""
        old_time = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) VALUES (?, ?, ?, ?, ?)",
            ("https://example.com/guide", "Guide", "oldhash", 168, old_time),
        )
        test_db.commit()
        source = _make_stale()

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

    async def test_head_no_headers_triggers_full_fetch(self, test_db: sqlite3.Connection) -> None:
        """No Last-Modified header -> full fetch."""
        old_time = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) VALUES (?, ?, ?, ?, ?)",
            ("https://example.com/guide", "Guide", "oldhash", 168, old_time),
        )
        test_db.commit()
        source = _make_stale()

        head_resp = httpx.Response(200, headers={})
        get_resp = httpx.Response(200, text="new content")

        with (
            patch("code.shukketsu.freshness.checker.httpx.AsyncClient") as mock_client_cls,
            patch("code.shukketsu.freshness.checker.trafilatura.extract", return_value="new content"),
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.head = AsyncMock(return_value=head_resp)
            mock_client.get = AsyncMock(return_value=get_resp)
            mock_client_cls.return_value = mock_client

            result = await check_source_freshness(source, test_db)

        assert result.changed
        assert not result.head_only

    async def test_full_fetch_unchanged(self, test_db: sqlite3.Connection) -> None:
        """Same content hash after full fetch -> changed=False, last_checked updated."""
        content = "same content"
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        old_time = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) VALUES (?, ?, ?, ?, ?)",
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
        row = test_db.execute("SELECT last_checked FROM sources WHERE id = ?", (source.id,)).fetchone()
        assert row["last_checked"] != old_time

    async def test_full_fetch_changed(self, test_db: sqlite3.Connection) -> None:
        """Different hash -> changed=True, is_stale=1, content_hash_previous set."""
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

        with (
            patch("code.shukketsu.freshness.checker.httpx.AsyncClient") as mock_client_cls,
            patch("code.shukketsu.freshness.checker.trafilatura.extract", return_value="totally new content"),
        ):
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

    async def test_network_error(self, test_db: sqlite3.Connection) -> None:
        """Network error -> FreshnessResult.error set, source row unchanged."""
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
        row = test_db.execute("SELECT last_checked, is_stale FROM sources WHERE id = ?", (source.id,)).fetchone()
        assert row["last_checked"] == old_time
        assert row["is_stale"] == 0

    async def test_http_404(self, test_db: sqlite3.Connection) -> None:
        """HTTP 404 -> error set, source unchanged."""
        old_time = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) VALUES (?, ?, ?, ?, ?)",
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


class TestRunFreshnessSweep:
    async def test_sweep_checks_all_stale(self, test_db: sqlite3.Connection) -> None:
        """Sweep finds and checks all stale sources."""
        old_time = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) VALUES (?, ?, ?, ?, ?)",
            ("https://example.com/a", "A", "hash_a", 168, old_time),
        )
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) VALUES (?, ?, ?, ?, ?)",
            ("https://example.com/b", "B", "hash_b", 168, old_time),
        )
        test_db.commit()

        head_resp = httpx.Response(200, headers={})
        get_resp = httpx.Response(200, text="same content")

        with patch("code.shukketsu.freshness.checker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.head = AsyncMock(return_value=head_resp)
            mock_client.get = AsyncMock(return_value=get_resp)
            mock_client_cls.return_value = mock_client

            results = await run_freshness_sweep(test_db)

        assert len(results) == 2

    async def test_sweep_empty(self, test_db: sqlite3.Connection) -> None:
        """Sweep with no stale sources returns empty list."""
        results = await run_freshness_sweep(test_db)
        assert results == []

    async def test_sweep_continues_after_source_crash(self, test_db: sqlite3.Connection) -> None:
        """If one source crashes, sweep still processes the remaining sources."""
        old_time = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) VALUES (?, ?, ?, ?, ?)",
            ("https://example.com/a", "A", "hash_a", 168, old_time),
        )
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) VALUES (?, ?, ?, ?, ?)",
            ("https://example.com/b", "B", "hash_b", 168, old_time),
        )
        test_db.commit()

        call_count = 0

        async def _mock_check(source, conn, timeout=None, client=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("trafilatura lxml crash")
            return FreshnessResult(
                source_id=source.id,
                url=source.url,
                changed=False,
                old_hash=source.content_hash,
                new_hash=source.content_hash,
                checked_at="2026-02-13T00:00:00",
                head_only=False,
            )

        with patch("code.shukketsu.freshness.checker.check_source_freshness", side_effect=_mock_check):
            results = await run_freshness_sweep(test_db)

        assert len(results) == 2
        assert results[0].error is not None
        assert "lxml crash" in results[0].error
        assert results[1].error is None


class TestCheckSourceFreshnessExceptionHandling:
    async def test_trafilatura_crash_returns_error_result(self, test_db: sqlite3.Connection) -> None:
        """trafilatura exceptions should be caught and return an error FreshnessResult."""
        old_time = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
        test_db.execute(
            "INSERT INTO sources (url, title, content_hash, check_interval_hours, last_checked) VALUES (?, ?, ?, ?, ?)",
            ("https://example.com/bad", "Bad Page", "oldhash", 168, old_time),
        )
        test_db.commit()
        source = _make_stale(url="https://example.com/bad")

        head_resp = httpx.Response(200, headers={})
        get_resp = httpx.Response(200, text="<html>bad html</html>")

        with (
            patch("code.shukketsu.freshness.checker.httpx.AsyncClient") as mock_client_cls,
            patch("code.shukketsu.freshness.checker.trafilatura.extract", side_effect=ValueError("lxml parse error")),
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.head = AsyncMock(return_value=head_resp)
            mock_client.get = AsyncMock(return_value=get_resp)
            mock_client_cls.return_value = mock_client

            result = await check_source_freshness(source, test_db)

        assert result.error is not None
        assert "lxml parse error" in result.error
        assert not result.changed
