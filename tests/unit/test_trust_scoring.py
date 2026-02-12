"""Tests for evidence-based trust scoring."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from code.shukketsu.trust.scoring import (
    SOURCE_TRUST,
    _effective_trust_legacy,
    effective_trust,
    get_trust_events,
    record_trust_event,
)


@pytest.fixture
def trust_db(tmp_path: Path) -> sqlite3.Connection:
    """Create a fresh test database with schema v4 for trust scoring tests."""
    from code.shukketsu.db.connection import get_connection, init_db

    conn = get_connection(tmp_path / "trust_test.db")
    init_db(conn)
    return conn


def _insert_source(conn: sqlite3.Connection, url: str = "https://example.com") -> int:
    """Insert a test source and return its id."""
    conn.execute("INSERT INTO sources (url, title, trust_score) VALUES (?, 'Test', 0.7)", (url,))
    conn.commit()
    row = conn.execute("SELECT id FROM sources WHERE url = ?", (url,)).fetchone()
    return int(row["id"])


class TestSourceTrust:
    """Tests for the SOURCE_TRUST mapping."""

    def test_game_data_highest(self) -> None:
        assert SOURCE_TRUST["game_data"] == 1.0

    def test_unknown_lowest(self) -> None:
        assert SOURCE_TRUST["unknown"] == 0.3

    def test_all_values_between_zero_and_one(self) -> None:
        for value in SOURCE_TRUST.values():
            assert 0.0 < value <= 1.0


class TestEffectiveTrustEvidence:
    """Tests for the evidence-based effective_trust function."""

    def test_effective_trust_no_events(self, trust_db: sqlite3.Connection) -> None:
        """Base trust returned unchanged when no events exist."""
        source_id = _insert_source(trust_db)
        result = effective_trust(0.7, trust_db, source_id)
        assert result == 0.7

    def test_effective_trust_negative_events(self, trust_db: sqlite3.Connection) -> None:
        """Contradiction events reduce trust."""
        source_id = _insert_source(trust_db)
        record_trust_event(trust_db, source_id, "contradiction", -0.1)
        result = effective_trust(0.7, trust_db, source_id)
        assert abs(result - 0.6) < 0.01

    def test_effective_trust_positive_events(self, trust_db: sqlite3.Connection) -> None:
        """Confirmation events increase trust."""
        source_id = _insert_source(trust_db)
        record_trust_event(trust_db, source_id, "confirmation", 0.05)
        result = effective_trust(0.7, trust_db, source_id)
        assert abs(result - 0.75) < 0.01

    def test_effective_trust_clamped_high(self, trust_db: sqlite3.Connection) -> None:
        """Trust cannot exceed 1.0."""
        source_id = _insert_source(trust_db)
        for _ in range(7):
            record_trust_event(trust_db, source_id, "confirmation", 0.05)
        result = effective_trust(0.9, trust_db, source_id)
        assert result == 1.0

    def test_effective_trust_clamped_low(self, trust_db: sqlite3.Connection) -> None:
        """Trust cannot go below 0.1."""
        source_id = _insert_source(trust_db)
        for _ in range(7):
            record_trust_event(trust_db, source_id, "contradiction", -0.1)
        result = effective_trust(0.3, trust_db, source_id)
        assert result == 0.1

    def test_dead_url_sets_minimum(self, trust_db: sqlite3.Connection) -> None:
        """dead_url event with large negative delta should floor trust at 0.1."""
        source_id = _insert_source(trust_db)
        record_trust_event(trust_db, source_id, "dead_url", -0.6, details="HTTP 404")
        result = effective_trust(0.7, trust_db, source_id)
        assert result == 0.1


class TestRecordTrustEvent:
    """Tests for record_trust_event."""

    def test_record_trust_event_inserts(self, trust_db: sqlite3.Connection) -> None:
        """Event should be stored in the trust_events table."""
        source_id = _insert_source(trust_db)
        record_trust_event(trust_db, source_id, "contradiction", -0.1, details="Claim X was wrong")
        row = trust_db.execute("SELECT * FROM trust_events WHERE source_id = ?", (source_id,)).fetchone()
        assert row is not None
        assert row["event_type"] == "contradiction"
        assert abs(row["delta"] - (-0.1)) < 0.001
        assert row["details"] == "Claim X was wrong"


class TestGetTrustEvents:
    """Tests for get_trust_events."""

    def test_get_trust_events_ordered(self, trust_db: sqlite3.Connection) -> None:
        """Events should be returned ordered by created_at."""
        source_id = _insert_source(trust_db)
        record_trust_event(trust_db, source_id, "confirmation", 0.05, details="first")
        record_trust_event(trust_db, source_id, "contradiction", -0.1, details="second")
        events = get_trust_events(trust_db, source_id)
        assert len(events) == 2
        assert events[0]["details"] == "first"
        assert events[1]["details"] == "second"

    def test_get_trust_events_empty(self, trust_db: sqlite3.Connection) -> None:
        """No events for source should return empty list."""
        source_id = _insert_source(trust_db)
        events = get_trust_events(trust_db, source_id)
        assert events == []


class TestLegacyFunction:
    """Tests for the preserved legacy decay function."""

    def test_legacy_function_preserved(self) -> None:
        """_effective_trust_legacy should still compute time-based decay."""
        now = datetime.now(UTC)
        result = _effective_trust_legacy(0.8, fetched_at=now, max_age=timedelta(days=30), decay_factor=0.5)
        assert result == 0.8

    def test_legacy_decay_applied(self) -> None:
        """Legacy function should apply decay after max_age."""
        now = datetime.now(UTC)
        fetched = now - timedelta(days=60)
        result = _effective_trust_legacy(0.8, fetched_at=fetched, max_age=timedelta(days=30), decay_factor=0.5)
        assert abs(result - 0.4) < 0.01
