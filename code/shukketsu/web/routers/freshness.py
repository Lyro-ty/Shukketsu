"""Freshness check API routes."""

import logging
import sqlite3

from fastapi import APIRouter, Depends, HTTPException

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
    rows = conn.execute("SELECT id, url, change_count, last_checked FROM sources WHERE is_stale = 1").fetchall()
    return {
        "stale_sources": [
            {
                "id": row["id"],
                "url": row["url"],
                "change_count": row["change_count"],
                "last_checked": row["last_checked"],
            }
            for row in rows
        ],
    }


@router.post("/clear/{source_id}")
async def clear_stale_flag(source_id: int, conn: sqlite3.Connection = Depends(_get_conn)) -> dict:
    """Clear is_stale flag for a source."""
    row = conn.execute("SELECT id FROM sources WHERE id = ?", (source_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Source not found: {source_id}")
    conn.execute("UPDATE sources SET is_stale = 0 WHERE id = ?", (source_id,))
    conn.commit()
    return {"cleared": True, "source_id": source_id}
