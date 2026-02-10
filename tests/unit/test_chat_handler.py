"""Tests for the WebSocket chat handler."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from code.shukketsu.resilience.errors import LLMUnavailableError
from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity


def _get_app():
    from code.shukketsu.web.app import app

    return app


def _mock_agent(answer: str = "Test answer") -> MagicMock:
    agent = MagicMock()
    agent.run = AsyncMock(return_value=answer)
    return agent


def _trivial_decision(answer: str = "Direct answer.") -> RoutingDecision:
    return RoutingDecision(
        complexity=TaskComplexity.TRIVIAL,
        category=TaskCategory.CONVERSATION,
        needs_tools=False,
        suggested_agent="general",
        direct_answer=answer,
    )


def _complex_decision() -> RoutingDecision:
    return RoutingDecision(
        complexity=TaskComplexity.COMPLEX,
        category=TaskCategory.ANALYSIS,
        needs_tools=True,
        suggested_agent="general",
        direct_answer=None,
    )


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

    @patch("code.shukketsu.web.routers.chat._get_agent", return_value=_mock_agent())
    def test_rejects_oversized_message(self, mock_get: MagicMock) -> None:
        """Messages exceeding CHAT_MAX_MESSAGE_LENGTH should be rejected."""
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            huge_message = "x" * 10_001
            ws.send_json({"type": "message", "content": huge_message})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "too long" in msg["content"].lower()


class TestAgentResponse:
    """Tests for the agent-based response path."""

    @patch("code.shukketsu.web.routers.chat._get_agent")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_sends_routing_then_thinking_status(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        mock_classify.return_value = _complex_decision()
        mock_get.return_value = _mock_agent("Answer")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hi"})
            routing = ws.receive_json()
            assert routing == {"type": "status", "content": "routing..."}
            thinking = ws.receive_json()
            assert thinking == {"type": "status", "content": "thinking..."}

    @patch("code.shukketsu.web.routers.chat._get_agent")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_sends_done_with_answer(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        mock_classify.return_value = _complex_decision()
        mock_get.return_value = _mock_agent("The hit cap is 9%.")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "What is hit cap?"})
            ws.receive_json()  # routing
            ws.receive_json()  # thinking
            done = ws.receive_json()
            assert done == {"type": "done", "content": "The hit cap is 9%."}

    @patch("code.shukketsu.web.routers.chat._get_agent")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_error_sends_error_message(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        mock_classify.return_value = _complex_decision()
        agent = MagicMock()
        agent.run = AsyncMock(side_effect=LLMUnavailableError("Server down"))
        mock_get.return_value = agent
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hello"})
            ws.receive_json()  # routing
            ws.receive_json()  # thinking
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "Server down" in err["content"]

    @patch("code.shukketsu.web.routers.chat._get_agent")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_second_message_after_response(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        mock_classify.return_value = _complex_decision()
        mock_get.return_value = _mock_agent("Answer")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "First"})
            ws.receive_json()  # routing
            ws.receive_json()  # thinking
            ws.receive_json()  # done
            ws.send_json({"type": "message", "content": "Second"})
            ws.receive_json()  # routing
            ws.receive_json()  # thinking
            done = ws.receive_json()
            assert done["type"] == "done"

    @patch("code.shukketsu.web.routers.chat._get_agent")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_rejects_message_during_agent_run(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        mock_classify.return_value = _complex_decision()
        mock_get.return_value = _mock_agent("Answer")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hello"})
            ws.receive_json()  # routing
            ws.receive_json()  # thinking
            done = ws.receive_json()
            assert done["type"] == "done"


class TestQueryRouting:
    """Tests for the multi-model routing integration."""

    @patch("code.shukketsu.web.routers.chat._get_agent")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_trivial_query_returns_fast_answer(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """Trivial queries should return Qwen's direct answer without calling the agent."""
        mock_classify.return_value = _trivial_decision("Sinister Strike costs 40 energy.")
        mock_get.return_value = _mock_agent()
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "How much energy does SS cost?"})
            routing = ws.receive_json()
            assert routing == {"type": "status", "content": "routing..."}
            done = ws.receive_json()
            assert done == {"type": "done", "content": "Sinister Strike costs 40 energy."}
            mock_get.return_value.run.assert_not_called()

    @patch("code.shukketsu.web.routers.chat._get_agent")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_complex_query_routes_to_agent(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """Complex queries should route to the agent after classification."""
        mock_classify.return_value = _complex_decision()
        mock_get.return_value = _mock_agent("Detailed analysis here.")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Compare combat vs assassination"})
            routing = ws.receive_json()
            assert routing == {"type": "status", "content": "routing..."}
            thinking = ws.receive_json()
            assert thinking == {"type": "status", "content": "thinking..."}
            done = ws.receive_json()
            assert done == {"type": "done", "content": "Detailed analysis here."}
            mock_get.return_value.run.assert_called_once()

    @patch("code.shukketsu.web.routers.chat._get_agent")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_routing_failure_falls_through_to_agent(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """If classify_query returns a fallback (COMPLEX), agent should handle the query."""
        mock_classify.return_value = _complex_decision()
        mock_get.return_value = _mock_agent("Agent handled it.")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hello"})
            ws.receive_json()  # routing
            ws.receive_json()  # thinking
            done = ws.receive_json()
            assert done == {"type": "done", "content": "Agent handled it."}

    @patch("code.shukketsu.web.routers.chat._get_agent")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_trivial_with_empty_answer_falls_through(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """Trivial classification with empty direct_answer should fall through to agent."""
        mock_classify.return_value = _trivial_decision("")
        mock_get.return_value = _mock_agent("Agent answer.")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "test"})
            ws.receive_json()  # routing
            ws.receive_json()  # thinking
            done = ws.receive_json()
            assert done == {"type": "done", "content": "Agent answer."}
            mock_get.return_value.run.assert_called_once()
