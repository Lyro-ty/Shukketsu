"""Tests for backup API routes."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from code.shukketsu.backup.manager import BackupManager


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
