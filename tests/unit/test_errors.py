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


def test_embedding_error():
    from code.shukketsu.resilience.errors import EmbeddingError, FailureMode

    err = EmbeddingError("Ollama unreachable")
    assert isinstance(err, ShukketsuError)
    assert err.failure_mode == FailureMode.EMBEDDING_ERROR
    assert "Ollama unreachable" in str(err)
