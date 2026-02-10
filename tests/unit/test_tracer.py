"""Tests for observability/tracer.py — Langfuse initialization and flush."""

from unittest.mock import MagicMock, patch

import pytest


class TestInitLangfuse:
    """Tests for init_langfuse()."""

    def test_sets_env_vars_from_config(self, monkeypatch):
        """init_langfuse pushes config values into environment."""
        monkeypatch.setattr("code.shukketsu.config.LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setattr("code.shukketsu.config.LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setattr("code.shukketsu.config.LANGFUSE_HOST", "http://test:3000")
        monkeypatch.setattr("code.shukketsu.config.LANGFUSE_SAMPLE_RATE", 0.5)
        monkeypatch.setattr("code.shukketsu.config.LANGFUSE_TRACING_ENABLED", True)

        import os

        from code.shukketsu.observability.tracer import init_langfuse

        with patch("code.shukketsu.observability.tracer.get_client") as mock_get:
            init_langfuse()

        assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-test"
        assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-test"
        assert os.environ["LANGFUSE_HOST"] == "http://test:3000"
        assert os.environ["LANGFUSE_SAMPLE_RATE"] == "0.5"
        mock_get.assert_called_once()

    def test_noop_when_tracing_disabled(self, monkeypatch):
        """init_langfuse does nothing when LANGFUSE_TRACING_ENABLED is False."""
        monkeypatch.setattr("code.shukketsu.config.LANGFUSE_TRACING_ENABLED", False)

        from code.shukketsu.observability.tracer import init_langfuse

        with patch("code.shukketsu.observability.tracer.get_client") as mock_get:
            init_langfuse()

        mock_get.assert_not_called()

    def test_sets_tracing_enabled_env_var_false(self, monkeypatch):
        """When disabled, sets LANGFUSE_TRACING_ENABLED=false in env."""
        monkeypatch.setattr("code.shukketsu.config.LANGFUSE_TRACING_ENABLED", False)

        import os

        from code.shukketsu.observability.tracer import init_langfuse

        with patch("code.shukketsu.observability.tracer.get_client"):
            init_langfuse()

        assert os.environ.get("LANGFUSE_TRACING_ENABLED") == "false"


class TestFlushTraces:
    """Tests for flush_traces()."""

    def test_flush_no_error_when_not_initialized(self):
        """flush_traces doesn't crash when Langfuse was never initialized."""
        from code.shukketsu.observability.tracer import flush_traces

        flush_traces()  # Should not raise

    def test_flush_calls_client_flush(self):
        """flush_traces calls flush() on the Langfuse client."""
        from code.shukketsu.observability.tracer import flush_traces

        mock_client = MagicMock()
        with patch("code.shukketsu.observability.tracer.get_client", return_value=mock_client):
            flush_traces()

        mock_client.flush.assert_called_once()
