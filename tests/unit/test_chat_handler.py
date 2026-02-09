"""Tests for the WebSocket chat handler."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from code.shukketsu.resilience.errors import LLMUnavailableError


def _get_app():
    from code.shukketsu.web.app import app

    return app


def _mock_agent(answer: str = "Test answer") -> MagicMock:
    agent = MagicMock()
    agent.run = AsyncMock(return_value=answer)
    return agent


class TestWebSocketProtocol:
    """Tests for WebSocket connection and message protocol."""

    @patch("code.shukketsu.web.routers.chat._get_agent", return_value=_mock_agent())
    def test_connects_and_sends_status(self, mock_get: MagicMock) -> None:
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            msg = ws.receive_json()
            assert msg == {"type": "status", "content": "connected"}

    @patch("code.shukketsu.web.routers.chat._get_agent", return_value=_mock_agent())
    def test_rejects_empty_message(self, mock_get: MagicMock) -> None:
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": ""})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "empty" in msg["content"].lower()

    @patch("code.shukketsu.web.routers.chat._get_agent", return_value=_mock_agent())
    def test_rejects_unknown_type(self, mock_get: MagicMock) -> None:
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "bogus"})
            msg = ws.receive_json()
            assert msg["type"] == "error"

    @patch("code.shukketsu.web.routers.chat._get_agent", return_value=_mock_agent())
    def test_rejects_missing_type(self, mock_get: MagicMock) -> None:
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"content": "hello"})
            msg = ws.receive_json()
            assert msg["type"] == "error"


class TestAgentResponse:
    """Tests for the agent-based response path."""

    @patch("code.shukketsu.web.routers.chat._get_agent")
    def test_sends_thinking_status(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_agent("Answer")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hi"})
            msg = ws.receive_json()
            assert msg == {"type": "status", "content": "thinking..."}

    @patch("code.shukketsu.web.routers.chat._get_agent")
    def test_sends_done_with_answer(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_agent("The hit cap is 9%.")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "What is hit cap?"})
            ws.receive_json()  # thinking
            done = ws.receive_json()
            assert done == {"type": "done", "content": "The hit cap is 9%."}

    @patch("code.shukketsu.web.routers.chat._get_agent")
    def test_error_sends_error_message(self, mock_get: MagicMock) -> None:
        agent = MagicMock()
        agent.run = AsyncMock(side_effect=LLMUnavailableError("Server down"))
        mock_get.return_value = agent
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hello"})
            ws.receive_json()  # thinking
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "Server down" in err["content"]

    @patch("code.shukketsu.web.routers.chat._get_agent")
    def test_second_message_after_response(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_agent("Answer")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "First"})
            ws.receive_json()  # thinking
            ws.receive_json()  # done
            ws.send_json({"type": "message", "content": "Second"})
            ws.receive_json()  # thinking
            done = ws.receive_json()
            assert done["type"] == "done"

    @patch("code.shukketsu.web.routers.chat._get_agent")
    def test_rejects_message_during_agent_run(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_agent("Answer")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hello"})
            ws.receive_json()  # thinking
            done = ws.receive_json()
            assert done["type"] == "done"
