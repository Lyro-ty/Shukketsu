# tests/unit/test_wcl_config.py
"""Tests for WCL configuration constants."""

from code.shukketsu import config


def test_wcl_endpoints_defined() -> None:
    assert config.WCL_TOKEN_URL == "https://www.warcraftlogs.com/oauth/token"
    assert "classic.warcraftlogs.com" in config.WCL_CLASSIC_ENDPOINT
    assert "fresh.warcraftlogs.com" in config.WCL_FRESH_ENDPOINT


def test_wcl_rate_limit_defaults() -> None:
    assert config.WCL_RATE_LIMIT_BUDGET == 3600
    assert config.WCL_RATE_LIMIT_BUFFER == 600
    assert config.WCL_RATE_CHECK_INTERVAL == 10


def test_wcl_query_timeout() -> None:
    assert config.WCL_QUERY_TIMEOUT == 30.0


def test_wcl_tracked_characters_default() -> None:
    assert isinstance(config.WCL_TRACKED_CHARACTERS, list)
    lyroo = config.WCL_TRACKED_CHARACTERS[0]
    assert lyroo["wcl_id"] == 104956434
    assert lyroo["name"] == "Lyroo"
    assert lyroo["endpoint"] == "fresh"


def test_wcl_circuit_breaker_config() -> None:
    assert config.CB_WCL_FAILURE_THRESHOLD == 3
    assert config.CB_WCL_RECOVERY_TIMEOUT == 60.0


def test_wcl_error_types_exist() -> None:
    from code.shukketsu.resilience.errors import (
        FailureMode,
        WCLAuthError,
        WCLQueryError,
        WCLRateLimitError,
    )

    assert FailureMode.WCL_API == "wcl_api"
    err = WCLAuthError("test")
    assert err.failure_mode == FailureMode.WCL_API
    rate_err = WCLRateLimitError("limit", points_reset_in=42.0)
    assert rate_err.points_reset_in == 42.0
    query_err = WCLQueryError("bad query")
    assert query_err.failure_mode == FailureMode.WCL_API
