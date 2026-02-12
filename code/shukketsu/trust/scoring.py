"""Source trust scoring with evidence-based event model.

Trust is computed as: base_trust + sum(event_deltas), clamped [0.1, 1.0].
The legacy time-based decay function is preserved as _effective_trust_legacy.
"""

import sqlite3
from datetime import UTC, datetime, timedelta

SOURCE_TRUST = {
    "game_data": 1.0,
    "simulation": 0.9,
    "combat_logs": 0.85,
    "expert_guide": 0.75,
    "archived_theory": 0.7,
    "community": 0.5,
    "unknown": 0.3,
}

TRUST_DELTAS = {
    "contradiction": -0.1,
    "correction": -0.15,
    "dead_url": -0.6,
    "confirmation": 0.05,
    "corroboration": 0.05,
}


def effective_trust(
    base_trust: float,
    conn: sqlite3.Connection,
    source_id: int,
) -> float:
    """Compute trust from base + sum of evidence events, clamped [0.1, 1.0]."""
    row = conn.execute(
        "SELECT COALESCE(SUM(delta), 0.0) as total_delta FROM trust_events WHERE source_id = ?",
        (source_id,),
    ).fetchone()
    total_delta = float(row["total_delta"]) if row else 0.0
    return max(0.1, min(1.0, base_trust + total_delta))


def record_trust_event(
    conn: sqlite3.Connection,
    source_id: int,
    event_type: str,
    delta: float,
    details: str | None = None,
) -> None:
    """Insert a trust event into the audit trail."""
    conn.execute(
        "INSERT INTO trust_events (source_id, event_type, delta, details) VALUES (?, ?, ?, ?)",
        (source_id, event_type, delta, details),
    )
    conn.commit()


def get_trust_events(
    conn: sqlite3.Connection,
    source_id: int,
) -> list[dict]:
    """Get all trust events for a source, ordered by created_at."""
    rows = conn.execute(
        """SELECT id, event_type, delta, details, created_at
           FROM trust_events
           WHERE source_id = ?
           ORDER BY created_at ASC""",
        (source_id,),
    ).fetchall()
    return [
        {
            "id": row["id"],
            "event_type": row["event_type"],
            "delta": row["delta"],
            "details": row["details"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def _effective_trust_legacy(
    base_trust: float,
    fetched_at: datetime,
    max_age: timedelta,
    decay_factor: float,
) -> float:
    """Compute effective trust with time-based decay (legacy).

    Preserved for backward compatibility reference.
    """
    age = datetime.now(UTC) - fetched_at
    if age <= max_age:
        return base_trust
    periods_past = (age - max_age) / max_age
    return float(base_trust * (decay_factor**periods_past))
