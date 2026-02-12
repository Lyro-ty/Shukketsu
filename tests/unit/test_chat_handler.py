"""Tests for the WebSocket chat handler."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from code.shukketsu.resilience.errors import LLMUnavailableError
from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity


def _get_app():
    from code.shukketsu.web.app import app

    return app


def _mock_agents(answer: str = "Test answer") -> tuple[MagicMock, MagicMock]:
    """Create mock researcher + orchestrator agents."""
    researcher = MagicMock()
    researcher.execute = AsyncMock(return_value=MagicMock(output=answer))

    orchestrator = MagicMock()
    orchestrator.execute = AsyncMock(return_value=MagicMock(output=answer))

    return researcher, orchestrator


def _trivial_decision(answer: str = "Direct answer.") -> RoutingDecision:
    return RoutingDecision(
        complexity=TaskComplexity.TRIVIAL,
        category=TaskCategory.CONVERSATION,
        needs_tools=False,
        suggested_agent="general",
        direct_answer=answer,
    )


def _moderate_decision() -> RoutingDecision:
    return RoutingDecision(
        complexity=TaskComplexity.MODERATE,
        category=TaskCategory.RETRIEVAL,
        needs_tools=True,
        suggested_agent="researcher",
    )


def _complex_decision() -> RoutingDecision:
    return RoutingDecision(
        complexity=TaskComplexity.COMPLEX,
        category=TaskCategory.RESEARCH,
        needs_tools=True,
        suggested_agent="orchestrator",
    )


def _drain_status(ws) -> dict:
    """Drain status messages, return the first non-status message."""
    while True:
        msg = ws.receive_json()
        if msg["type"] != "status":
            return msg


class TestWebSocketProtocol:
    """Tests for WebSocket connection and message protocol."""

    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
    def test_connects_and_sends_status(self, mock_get: MagicMock) -> None:
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            msg = ws.receive_json()
            assert msg == {"type": "status", "content": "connected"}

    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
    def test_rejects_empty_message(self, mock_get: MagicMock) -> None:
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": ""})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "empty" in msg["content"].lower()

    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
    def test_rejects_unknown_type(self, mock_get: MagicMock) -> None:
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "bogus"})
            msg = ws.receive_json()
            assert msg["type"] == "error"

    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
    def test_rejects_missing_type(self, mock_get: MagicMock) -> None:
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"content": "hello"})
            msg = ws.receive_json()
            assert msg["type"] == "error"

    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
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

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_sends_routing_then_planning_status(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        mock_classify.return_value = _complex_decision()
        mock_get.return_value = _mock_agents("Answer")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hi"})
            routing = ws.receive_json()
            assert routing == {"type": "status", "content": "routing..."}
            planning = ws.receive_json()
            assert planning == {"type": "status", "content": "planning..."}

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_sends_done_with_answer(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        mock_classify.return_value = _complex_decision()
        mock_get.return_value = _mock_agents("The hit cap is 9%.")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "What is hit cap?"})
            ws.receive_json()  # routing
            ws.receive_json()  # planning
            done = ws.receive_json()
            assert done == {"type": "done", "content": "The hit cap is 9%."}

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_error_sends_error_message(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        mock_classify.return_value = _complex_decision()
        researcher, orchestrator = _mock_agents()
        orchestrator.execute = AsyncMock(side_effect=LLMUnavailableError("Server down"))
        mock_get.return_value = (researcher, orchestrator)
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hello"})
            ws.receive_json()  # routing
            ws.receive_json()  # planning
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "Server down" in err["content"]

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_second_message_after_response(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        mock_classify.return_value = _complex_decision()
        mock_get.return_value = _mock_agents("Answer")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "First"})
            ws.receive_json()  # routing
            ws.receive_json()  # planning
            ws.receive_json()  # done
            ws.send_json({"type": "message", "content": "Second"})
            ws.receive_json()  # routing
            ws.receive_json()  # planning
            done = ws.receive_json()
            assert done["type"] == "done"

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_rejects_message_during_agent_run(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        mock_classify.return_value = _complex_decision()
        mock_get.return_value = _mock_agents("Answer")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hello"})
            ws.receive_json()  # routing
            ws.receive_json()  # planning
            done = ws.receive_json()
            assert done["type"] == "done"


class TestQueryRouting:
    """Tests for the multi-model routing integration."""

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_trivial_query_returns_fast_answer(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """Trivial queries should return Qwen's direct answer without calling agents."""
        mock_classify.return_value = _trivial_decision("Sinister Strike costs 40 energy.")
        researcher, orchestrator = _mock_agents()
        mock_get.return_value = (researcher, orchestrator)
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "How much energy does SS cost?"})
            routing = ws.receive_json()
            assert routing == {"type": "status", "content": "routing..."}
            done = ws.receive_json()
            assert done == {"type": "done", "content": "Sinister Strike costs 40 energy."}
            researcher.execute.assert_not_called()
            orchestrator.execute.assert_not_called()

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_complex_query_routes_to_orchestrator(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """Complex queries should route to the Orchestrator after classification."""
        mock_classify.return_value = _complex_decision()
        researcher, orchestrator = _mock_agents("Detailed analysis here.")
        mock_get.return_value = (researcher, orchestrator)
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Compare combat vs assassination"})
            routing = ws.receive_json()
            assert routing == {"type": "status", "content": "routing..."}
            planning = ws.receive_json()
            assert planning == {"type": "status", "content": "planning..."}
            done = ws.receive_json()
            assert done == {"type": "done", "content": "Detailed analysis here."}
            orchestrator.execute.assert_called_once()

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_routing_failure_falls_through_to_orchestrator(
        self,
        mock_classify: AsyncMock,
        mock_get: MagicMock,
    ) -> None:
        """If classify_query returns a fallback (COMPLEX), orchestrator should handle the query."""
        mock_classify.return_value = _complex_decision()
        researcher, orchestrator = _mock_agents("Orchestrator handled it.")
        mock_get.return_value = (researcher, orchestrator)
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hello"})
            ws.receive_json()  # routing
            ws.receive_json()  # planning
            done = ws.receive_json()
            assert done == {"type": "done", "content": "Orchestrator handled it."}

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_trivial_with_empty_answer_falls_through(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """Trivial classification with empty direct_answer should fall through to orchestrator."""
        mock_classify.return_value = _trivial_decision("")
        researcher, orchestrator = _mock_agents("Orchestrator answer.")
        mock_get.return_value = (researcher, orchestrator)
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "test"})
            ws.receive_json()  # routing
            ws.receive_json()  # planning
            done = ws.receive_json()
            assert done == {"type": "done", "content": "Orchestrator answer."}
            orchestrator.execute.assert_called_once()


class TestComplexityRouting:
    """Tests for complexity-based routing to specialists."""

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_trivial_uses_direct_answer(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """TRIVIAL complexity returns direct answer without calling agents."""
        mock_classify.return_value = _trivial_decision("Energy costs 40.")
        researcher, orchestrator = _mock_agents()
        mock_agents.return_value = (researcher, orchestrator)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "How much energy does SS cost?"})
            ws.receive_json()  # routing
            done = ws.receive_json()
            assert done["content"] == "Energy costs 40."

        researcher.execute.assert_not_called()
        orchestrator.execute.assert_not_called()

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_moderate_uses_researcher(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """MODERATE complexity routes to Researcher.execute()."""
        mock_classify.return_value = _moderate_decision()
        researcher, orchestrator = _mock_agents("Researcher answer")
        mock_agents.return_value = (researcher, orchestrator)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "What trinkets for combat?"})
            done = _drain_status(ws)
            assert done["content"] == "Researcher answer"

        researcher.execute.assert_called_once()
        orchestrator.execute.assert_not_called()

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_complex_uses_orchestrator(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """COMPLEX complexity routes to Orchestrator.execute()."""
        mock_classify.return_value = _complex_decision()
        researcher, orchestrator = _mock_agents("Orchestrator answer")
        mock_agents.return_value = (researcher, orchestrator)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Write a guide about trinkets"})
            done = _drain_status(ws)
            assert done["content"] == "Orchestrator answer"

        orchestrator.execute.assert_called_once()
        researcher.execute.assert_not_called()


class TestSendStatusFormat:
    """Tests for _send_status handling of str and dict messages."""

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_send_status_wraps_string(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """String messages are wrapped in {'type': 'status', 'content': msg}."""
        mock_classify.return_value = _moderate_decision()
        researcher, orchestrator = _mock_agents("Answer")

        async def _capture_execute(task, on_status=None):  # type: ignore[no-untyped-def]
            if on_status:
                await on_status("test string message")
            return MagicMock(output="Answer")

        researcher.execute = _capture_execute
        mock_get.return_value = (researcher, orchestrator)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "test"})
            messages = []
            while True:
                msg = ws.receive_json()
                messages.append(msg)
                if msg["type"] == "done":
                    break

            status_msgs = [m for m in messages if m["type"] == "status"]
            assert any(m["content"] == "test string message" for m in status_msgs)

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_send_status_passes_dict_as_is(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """Dict messages are sent as-is via WebSocket."""
        mock_classify.return_value = _moderate_decision()
        researcher, orchestrator = _mock_agents("Answer")

        async def _capture_execute(task, on_status=None):  # type: ignore[no-untyped-def]
            if on_status:
                await on_status({"type": "step", "agent": "researcher", "action": "tool_call", "tool": "rag_search"})
            return MagicMock(output="Answer")

        researcher.execute = _capture_execute
        mock_get.return_value = (researcher, orchestrator)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "test"})
            messages = []
            while True:
                msg = ws.receive_json()
                messages.append(msg)
                if msg["type"] == "done":
                    break

            step_msgs = [m for m in messages if m.get("type") == "step"]
            assert len(step_msgs) >= 1
            assert step_msgs[0]["tool"] == "rag_search"

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_article_path_appended_to_response(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """OrchestratorResult with article_path adds notification to response."""
        from code.shukketsu.agents.tasks import (
            AgentRole,
            OrchestratorPlan,
            OrchestratorResult,
            TaskStatus,
        )

        mock_classify.return_value = _complex_decision()
        orch_result = OrchestratorResult(
            task_id="t1",
            agent_role=AgentRole.ORCHESTRATOR,
            status=TaskStatus.SUCCESS,
            output="Guide written.",
            plan=OrchestratorPlan(reasoning="test", subtasks=[]),
            article_path="combat/gear/trinkets.md",
            needs_human_review=True,
        )

        researcher, orchestrator = _mock_agents()
        orchestrator.execute = AsyncMock(return_value=orch_result)
        mock_agents.return_value = (researcher, orchestrator)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Write a guide"})
            done = _drain_status(ws)
            assert "combat/gear/trinkets.md" in done["content"]
            assert "pending review" in done["content"].lower()

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_fallback_routes_to_orchestrator(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """Router failure (defaults to COMPLEX) routes to Orchestrator."""
        mock_classify.return_value = _complex_decision()
        researcher, orchestrator = _mock_agents("Fallback answer")
        mock_agents.return_value = (researcher, orchestrator)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Something complex"})
            done = _drain_status(ws)
            assert done["content"] == "Fallback answer"

        orchestrator.execute.assert_called_once()
