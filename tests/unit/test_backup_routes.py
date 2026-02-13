"""Tests for backup API routes."""

import sqlite3
from unittest.mock import patch

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


class TestBackupErrorHandling:
    """Tests for backup route error responses."""

    def test_create_backup_failure_returns_500(self, backup_client) -> None:
        """If backup creation raises, the route should return 500."""
        client, mgr = backup_client
        with patch.object(mgr, "create_backup", side_effect=RuntimeError("Disk full")):
            resp = client.post("/api/backup/create")
            assert resp.status_code == 500
            assert "failed" in resp.json()["detail"].lower()

    def test_list_backups_failure_returns_500(self, backup_client) -> None:
        """If listing backups raises, the route should return 500."""
        client, mgr = backup_client
        with patch.object(mgr, "list_backups", side_effect=RuntimeError("Permission denied")):
            resp = client.get("/api/backup/list")
            assert resp.status_code == 500
            assert "failed" in resp.json()["detail"].lower()

    def test_prune_backups_failure_returns_500(self, backup_client) -> None:
        """If pruning raises, the route should return 500."""
        client, mgr = backup_client
        with patch.object(mgr, "prune_old_backups", side_effect=RuntimeError("Unlink failed")):
            resp = client.post("/api/backup/prune")
            assert resp.status_code == 500
            assert "failed" in resp.json()["detail"].lower()
