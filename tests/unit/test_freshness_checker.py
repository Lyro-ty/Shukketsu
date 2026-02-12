"""Tests for content freshness checker."""

import sqlite3
from datetime import UTC, datetime, timedelta

from code.shukketsu.freshness.checker import (
    FreshnessResult,
    StaleSource,
    count_stale_sources,
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
