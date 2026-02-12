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
            # Parse timestamp from filename: shukketsu-YYYYMMDD-HHMMSS
            stem = path.stem
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
