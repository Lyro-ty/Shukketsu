"""Backup management API routes."""

import logging

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
