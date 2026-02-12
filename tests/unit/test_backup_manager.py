"""Tests for SQLite backup manager."""

import sqlite3
import time

import pytest

from code.shukketsu.backup.manager import BackupManager


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
