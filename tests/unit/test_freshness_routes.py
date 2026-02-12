"""Tests for freshness API routes."""

import sqlite3
from unittest.mock import AsyncMock, patch

import sqlite_vec
from fastapi.testclient import TestClient

from code.shukketsu.db.connection import init_db


def _get_app():
    from code.shukketsu.web.app import app

    return app


def _make_db(tmp_path) -> sqlite3.Connection:
    """Cross-thread-safe DB for TestClient."""
    db_path = tmp_path / "freshness_test.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    init_db(conn)
    return conn


class TestFreshnessRoutes:
    def test_post_check_returns_results(self, tmp_path) -> None:
        from code.shukketsu.freshness.checker import FreshnessResult
        from code.shukketsu.web.routers.freshness import _get_conn

        db = _make_db(tmp_path)
        app = _get_app()
        app.dependency_overrides[_get_conn] = lambda: db

        with patch("code.shukketsu.web.routers.freshness.run_freshness_sweep", new_callable=AsyncMock) as mock_sweep:
            mock_sweep.return_value = [
                FreshnessResult(
                    source_id=1,
                    url="https://example.com",
                    changed=True,
                    old_hash="old",
                    new_hash="new",
                    checked_at="2026-01-01T00:00:00",
                    head_only=False,
                )
            ]
            client = TestClient(app)
            resp = client.post("/api/freshness/check")

        app.dependency_overrides.clear()
        db.close()

        assert resp.status_code == 200
        data = resp.json()
        assert data["checked"] == 1
        assert data["changed"] == 1

    def test_get_stale_returns_list(self, tmp_path) -> None:
        from code.shukketsu.web.routers.freshness import _get_conn

        db = _make_db(tmp_path)
        app = _get_app()
        app.dependency_overrides[_get_conn] = lambda: db

        db.execute(
            "INSERT INTO sources (url, title, content_hash, is_stale) VALUES (?, ?, ?, ?)",
            ("https://stale.com", "Stale", "hash", 1),
        )
        db.commit()

        client = TestClient(app)
        resp = client.get("/api/freshness/stale")

        app.dependency_overrides.clear()
        db.close()

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["stale_sources"]) == 1
        assert data["stale_sources"][0]["url"] == "https://stale.com"

    def test_get_stale_empty(self, tmp_path) -> None:
        from code.shukketsu.web.routers.freshness import _get_conn

        db = _make_db(tmp_path)
        app = _get_app()
        app.dependency_overrides[_get_conn] = lambda: db

        client = TestClient(app)
        resp = client.get("/api/freshness/stale")

        app.dependency_overrides.clear()
        db.close()

        assert resp.status_code == 200
        assert resp.json()["stale_sources"] == []

    def test_clear_stale_flag(self, tmp_path) -> None:
        from code.shukketsu.web.routers.freshness import _get_conn

        db = _make_db(tmp_path)
        app = _get_app()
        app.dependency_overrides[_get_conn] = lambda: db

        db.execute(
            "INSERT INTO sources (url, title, content_hash, is_stale) VALUES (?, ?, ?, ?)",
            ("https://stale.com", "Stale", "hash", 1),
        )
        db.commit()
        source_id = db.execute("SELECT id FROM sources WHERE url = ?", ("https://stale.com",)).fetchone()["id"]

        client = TestClient(app)
        resp = client.post(f"/api/freshness/clear/{source_id}")

        row = db.execute("SELECT is_stale FROM sources WHERE id = ?", (source_id,)).fetchone()
        app.dependency_overrides.clear()
        db.close()

        assert resp.status_code == 200
        assert row["is_stale"] == 0

    def test_clear_nonexistent_404(self, tmp_path) -> None:
        from code.shukketsu.web.routers.freshness import _get_conn

        db = _make_db(tmp_path)
        app = _get_app()
        app.dependency_overrides[_get_conn] = lambda: db

        client = TestClient(app)
        resp = client.post("/api/freshness/clear/9999")

        app.dependency_overrides.clear()
        db.close()

        assert resp.status_code == 404
