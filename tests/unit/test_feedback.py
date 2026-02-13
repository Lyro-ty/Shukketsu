"""Tests for chat feedback handling."""

from unittest.mock import AsyncMock, MagicMock, patch

from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity


def _get_app():
    from code.shukketsu.web.app import app

    return app


def _trivial_decision(answer: str = "Direct answer.") -> RoutingDecision:
    return RoutingDecision(
        complexity=TaskComplexity.TRIVIAL,
        category=TaskCategory.CONVERSATION,
        needs_tools=False,
        suggested_agent="general",
        direct_answer=answer,
    )


def _mock_agents(answer: str = "Test answer") -> tuple[MagicMock, MagicMock]:
    researcher = MagicMock()
    researcher.execute = AsyncMock(return_value=MagicMock(output=answer))
    orchestrator = MagicMock()
    orchestrator.execute = AsyncMock(return_value=MagicMock(output=answer))
    return researcher, orchestrator


class TestFeedbackMessage:
    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
    def test_feedback_accepted_silently(self, mock_get: MagicMock) -> None:
        """Feedback message should be accepted without error response."""
        from fastapi.testclient import TestClient

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "feedback", "score": 1, "trace_id": "trace-123"})
            # Send a real message to verify connection still works
            ws.send_json({"type": "message", "content": ""})
            msg = ws.receive_json()
            assert msg["type"] == "error"  # Empty message error, not feedback error

    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
    def test_feedback_without_trace_id_ignored(self, mock_get: MagicMock) -> None:
        """Feedback without trace_id should be silently ignored."""
        from fastapi.testclient import TestClient

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "feedback", "score": 1})
            ws.send_json({"type": "message", "content": ""})
            msg = ws.receive_json()
            assert msg["type"] == "error"

    @patch("code.shukketsu.web.routers.chat.get_client")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_done_includes_trace_id(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
        mock_get_client: MagicMock,
    ) -> None:
        """Done messages should include trace_id from Langfuse context."""
        mock_classify.return_value = _trivial_decision("Answer")
        mock_agents.return_value = _mock_agents()
        mock_client = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace-abc"
        mock_get_client.return_value = mock_client

        from fastapi.testclient import TestClient

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hi"})
            # May get routing status first
            msgs = []
            while True:
                msg = ws.receive_json()
                msgs.append(msg)
                if msg["type"] == "done":
                    break
            done = msgs[-1]
            assert done["type"] == "done"
            assert "trace_id" in done

    @patch("code.shukketsu.web.routers.chat.get_client")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_done_trace_id_none_when_disabled(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
        mock_get_client: MagicMock,
    ) -> None:
        """trace_id should be None when Langfuse tracing is disabled."""
        mock_classify.return_value = _trivial_decision("Answer")
        mock_agents.return_value = _mock_agents()
        mock_client = MagicMock()
        mock_client.get_current_trace_id.return_value = None
        mock_get_client.return_value = mock_client

        from fastapi.testclient import TestClient

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hi"})
            msgs = []
            while True:
                msg = ws.receive_json()
                msgs.append(msg)
                if msg["type"] == "done":
                    break
            done = msgs[-1]
            assert done["trace_id"] is None

    @patch("code.shukketsu.web.routers.chat.get_client")
    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
    def test_feedback_records_to_langfuse(
        self,
        mock_agents: MagicMock,
        mock_langfuse: MagicMock,
    ) -> None:
        """Thumbs up should call langfuse.score()."""
        mock_client = MagicMock()
        mock_langfuse.return_value = mock_client

        from fastapi.testclient import TestClient

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "feedback", "score": 1, "trace_id": "trace-xyz"})
            # Verify by sending another message
            ws.send_json({"type": "message", "content": ""})
            ws.receive_json()  # error (empty)

        mock_client.score.assert_called_once()
        call_kwargs = mock_client.score.call_args.kwargs
        assert call_kwargs["trace_id"] == "trace-xyz"
        assert call_kwargs["value"] == 1.0

    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
    def test_thumbs_down_value_zero(self, mock_get: MagicMock) -> None:
        """Thumbs down should send score=0."""
        with patch("code.shukketsu.web.routers.chat.get_client") as mock_langfuse:
            mock_client = MagicMock()
            mock_langfuse.return_value = mock_client

            from fastapi.testclient import TestClient

            client = TestClient(_get_app())
            with client.websocket_connect("/ws/chat") as ws:
                ws.receive_json()  # connected
                ws.send_json({"type": "feedback", "score": 0, "trace_id": "trace-xyz"})
                ws.send_json({"type": "message", "content": ""})
                ws.receive_json()

            call_kwargs = mock_client.score.call_args.kwargs
            assert call_kwargs["value"] == 0.0
