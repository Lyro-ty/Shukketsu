"""Tests for trust scoring with time-based decay."""

from datetime import UTC, datetime, timedelta

from code.shukketsu.trust.scoring import SOURCE_TRUST, effective_trust


class TestSourceTrust:
    """Tests for the SOURCE_TRUST mapping."""

    def test_game_data_highest(self) -> None:
        assert SOURCE_TRUST["game_data"] == 1.0

    def test_unknown_lowest(self) -> None:
        assert SOURCE_TRUST["unknown"] == 0.3

    def test_all_values_between_zero_and_one(self) -> None:
        for value in SOURCE_TRUST.values():
            assert 0.0 < value <= 1.0


class TestEffectiveTrust:
    """Tests for the effective_trust decay function."""

    def test_within_max_age_returns_base(self) -> None:
        now = datetime.now(UTC)
        result = effective_trust(0.8, fetched_at=now, max_age=timedelta(days=30), decay_factor=0.5)
        assert result == 0.8

    def test_exactly_at_max_age_returns_base(self) -> None:
        now = datetime.now(UTC)
        fetched = now - timedelta(days=29, hours=23, minutes=59)  # just under max_age
        result = effective_trust(0.8, fetched_at=fetched, max_age=timedelta(days=30), decay_factor=0.5)
        assert result == 0.8

    def test_one_period_past_max_age_applies_decay(self) -> None:
        now = datetime.now(UTC)
        fetched = now - timedelta(days=60)  # 30 days past max_age of 30
        result = effective_trust(0.8, fetched_at=fetched, max_age=timedelta(days=30), decay_factor=0.5)
        assert abs(result - 0.4) < 0.01  # 0.8 * 0.5^1

    def test_two_periods_past_applies_double_decay(self) -> None:
        now = datetime.now(UTC)
        fetched = now - timedelta(days=90)  # 60 days past max_age of 30
        result = effective_trust(0.8, fetched_at=fetched, max_age=timedelta(days=30), decay_factor=0.5)
        assert abs(result - 0.2) < 0.01  # 0.8 * 0.5^2

    def test_result_is_float(self) -> None:
        now = datetime.now(UTC)
        result = effective_trust(1.0, fetched_at=now, max_age=timedelta(days=30), decay_factor=0.9)
        assert isinstance(result, float)

    def test_timezone_aware_fetched_at(self) -> None:
        """Ensure timezone-aware datetimes (as stored by IngestPipeline) work."""
        now = datetime.now(UTC)
        fetched = now - timedelta(hours=1)
        result = effective_trust(0.75, fetched_at=fetched, max_age=timedelta(days=7), decay_factor=0.5)
        assert result == 0.75
