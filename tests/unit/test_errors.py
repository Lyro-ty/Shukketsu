"""Tests for custom error classes."""

from code.shukketsu.resilience.errors import (
    FailureMode,
    ShukketsuError,
    ToolExecutionError,
    ToolNotFoundError,
)


class TestToolNotFoundError:
    """Tests for ToolNotFoundError."""

    def test_has_correct_failure_mode(self) -> None:
        err = ToolNotFoundError("no such tool: foo")
        assert err.failure_mode == FailureMode.TOOL_NOT_FOUND

    def test_inherits_shukketsu_error(self) -> None:
        err = ToolNotFoundError("no such tool")
        assert isinstance(err, ShukketsuError)


class TestToolExecutionError:
    """Tests for ToolExecutionError."""

    def test_has_correct_failure_mode(self) -> None:
        err = ToolExecutionError("tool failed")
        assert err.failure_mode == FailureMode.TOOL_EXECUTION_ERROR

    def test_inherits_shukketsu_error(self) -> None:
        err = ToolExecutionError("tool failed")
        assert isinstance(err, ShukketsuError)


def test_embedding_error() -> None:
    from code.shukketsu.resilience.errors import EmbeddingError, FailureMode

    err = EmbeddingError("Ollama unreachable")
    assert isinstance(err, ShukketsuError)
    assert err.failure_mode == FailureMode.EMBEDDING_ERROR
    assert "Ollama unreachable" in str(err)


def test_scraping_error_has_http_error_mode() -> None:
    from code.shukketsu.resilience.errors import ScrapingError

    err = ScrapingError("timeout")
    assert err.failure_mode == FailureMode.HTTP_ERROR


def test_robots_disallowed_has_access_denied_failure_mode() -> None:
    from code.shukketsu.resilience.errors import RobotsDisallowedError

    err = RobotsDisallowedError("blocked")
    assert err.failure_mode == FailureMode.ACCESS_DENIED


def test_brave_search_error_has_tool_error_mode() -> None:
    from code.shukketsu.resilience.errors import BraveSearchError

    err = BraveSearchError("api key missing")
    assert err.failure_mode == FailureMode.TOOL_EXECUTION_ERROR


def test_circuit_open_error_has_model_unavailable_mode() -> None:
    from code.shukketsu.resilience.errors import CircuitOpenError

    err = CircuitOpenError("brave_search")
    assert isinstance(err, ShukketsuError)
    assert err.failure_mode == FailureMode.MODEL_UNAVAILABLE
    assert err.breaker_name == "brave_search"
    assert "brave_search" in str(err)


class TestPhase4Errors:
    def test_wcl_bridge_error(self) -> None:
        from code.shukketsu.resilience.errors import FailureMode, WCLBridgeError

        err = WCLBridgeError("Missing weapon item 32837")
        assert err.failure_mode == FailureMode.SIM_VALIDATION
        assert "32837" in str(err)

    def test_log_parse_error(self) -> None:
        from code.shukketsu.resilience.errors import FailureMode, LogParseError

        err = LogParseError("Invalid CLEU line format")
        assert err.failure_mode == FailureMode.SIM_VALIDATION
        assert "CLEU" in str(err)

    def test_validation_pipeline_error(self) -> None:
        from code.shukketsu.resilience.errors import FailureMode, ValidationPipelineError

        err = ValidationPipelineError("No valid fights found")
        assert err.failure_mode == FailureMode.SIM_VALIDATION
