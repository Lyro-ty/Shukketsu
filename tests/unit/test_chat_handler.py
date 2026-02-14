"""Tests for the WebSocket chat handler."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from code.shukketsu.resilience.errors import LLMUnavailableError
from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity


def _get_app():
    from code.shukketsu.web.app import app

    return app


def _mock_agents(answer: str = "Test answer") -> tuple[MagicMock, MagicMock, MagicMock]:
    """Create mock researcher + orchestrator + analyst agents."""
    researcher = MagicMock()
    researcher.execute = AsyncMock(return_value=MagicMock(output=answer))

    orchestrator = MagicMock()
    orchestrator.execute = AsyncMock(return_value=MagicMock(output=answer))

    analyst = MagicMock()
    analyst.execute = AsyncMock(return_value=MagicMock(output=answer))

    return researcher, orchestrator, analyst


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


def _analysis_decision() -> RoutingDecision:
    return RoutingDecision(
        complexity=TaskComplexity.COMPLEX,
        category=TaskCategory.ANALYSIS,
        needs_tools=True,
        suggested_agent="analyst",
    )


def _drain_status(ws) -> dict:
    """Drain status messages, return the first non-status message."""
    while True:
        msg = ws.receive_json()
        if msg["type"] != "status":
            return msg


class TestChatSession:
    """Tests for ChatSession state management."""

    def test_add_message_appends(self) -> None:
        """add_message should append the message to history."""
        from code.shukketsu.web.routers.chat import ChatSession

        session = ChatSession()
        session.add_message("user", "Hello")
        assert session.history == [{"role": "user", "content": "Hello"}]

    def test_add_message_trims_to_max_pairs(self) -> None:
        """History should be trimmed to CHAT_MAX_HISTORY_PAIRS * 2 messages."""
        from code.shukketsu.web.routers.chat import ChatSession

        session = ChatSession()
        # config.CHAT_MAX_HISTORY_PAIRS is typically 10, meaning 20 messages max
        # Add more than the max
        with patch("code.shukketsu.web.routers.chat.config") as mock_config:
            mock_config.CHAT_MAX_HISTORY_PAIRS = 3  # 6 messages max
            mock_config.CHAT_MAX_MESSAGE_LENGTH = 10000
            for i in range(10):
                session.add_message("user", f"msg {i}")
                session.add_message("assistant", f"reply {i}")
            # Should keep only the last 6 messages
            assert len(session.history) == 6
            assert session.history[0]["content"] == "msg 7"

    def test_initial_state(self) -> None:
        """New session should have empty history and is_streaming=False."""
        from code.shukketsu.web.routers.chat import ChatSession

        session = ChatSession()
        assert session.history == []
        assert session.is_streaming is False


class TestFormatMemoryContext:
    """Tests for _format_memory_context truncation logic."""

    def test_empty_memories_returns_empty(self) -> None:
        """No memories should return empty string."""
        from code.shukketsu.web.routers.chat import _format_memory_context

        assert _format_memory_context([]) == ""

    def test_formats_summary_and_facts(self) -> None:
        """Memories should be formatted with summary and key facts."""
        from code.shukketsu.memory.models import SessionMemory
        from code.shukketsu.web.routers.chat import _format_memory_context

        mem = SessionMemory(
            id=1,
            query="hit cap",
            answer_summary="9% hit cap",
            key_facts=["142 rating", "melee only"],
            entities_mentioned=[],
            retrieval_quality=0.8,
            created_at="2026-02-12T00:00:00+00:00",
            score=0.9,
        )
        result = _format_memory_context([mem])
        assert "hit cap" in result
        assert "9% hit cap" in result
        assert "142 rating" in result
        assert "melee only" in result

    def test_truncates_at_max_context_chars(self) -> None:
        """Should stop adding memories when exceeding MEMORY_MAX_CONTEXT_CHARS."""
        from code.shukketsu.memory.models import SessionMemory
        from code.shukketsu.web.routers.chat import _format_memory_context

        memories = [
            SessionMemory(
                id=i,
                query=f"query {i}",
                answer_summary="A" * 500,
                key_facts=[],
                entities_mentioned=[],
                retrieval_quality=0.5,
                created_at="2026-02-12T00:00:00+00:00",
                score=0.5,
            )
            for i in range(20)
        ]
        with patch("code.shukketsu.web.routers.chat.config") as mock_config:
            mock_config.MEMORY_MAX_CONTEXT_CHARS = 1000
            result = _format_memory_context(memories)
            # Should have truncated — not all 20 memories should be present
            assert len(result) <= 1200  # Some margin for the entry that pushed us over
            assert result.count("- **query") < 20


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
            assert done["type"] == "done"
            assert done["content"] == "The hit cap is 9%."

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_error_sends_error_message(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        mock_classify.return_value = _complex_decision()
        researcher, orchestrator, analyst = _mock_agents()
        orchestrator.execute = AsyncMock(side_effect=LLMUnavailableError("Server down"))
        mock_get.return_value = (researcher, orchestrator, analyst)
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hello"})
            ws.receive_json()  # routing
            ws.receive_json()  # planning
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "temporarily unavailable" in err["content"]

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
        researcher, orchestrator, analyst = _mock_agents()
        mock_get.return_value = (researcher, orchestrator, analyst)
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "How much energy does SS cost?"})
            routing = ws.receive_json()
            assert routing == {"type": "status", "content": "routing..."}
            done = ws.receive_json()
            assert done["type"] == "done"
            assert done["content"] == "Sinister Strike costs 40 energy."
            researcher.execute.assert_not_called()
            orchestrator.execute.assert_not_called()

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_complex_query_routes_to_orchestrator(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """Complex queries should route to the Orchestrator after classification."""
        mock_classify.return_value = _complex_decision()
        researcher, orchestrator, analyst = _mock_agents("Detailed analysis here.")
        mock_get.return_value = (researcher, orchestrator, analyst)
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Compare combat vs assassination"})
            routing = ws.receive_json()
            assert routing == {"type": "status", "content": "routing..."}
            planning = ws.receive_json()
            assert planning == {"type": "status", "content": "planning..."}
            done = ws.receive_json()
            assert done["type"] == "done"
            assert done["content"] == "Detailed analysis here."
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
        researcher, orchestrator, analyst = _mock_agents("Orchestrator handled it.")
        mock_get.return_value = (researcher, orchestrator, analyst)
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hello"})
            ws.receive_json()  # routing
            ws.receive_json()  # planning
            done = ws.receive_json()
            assert done["type"] == "done"
            assert done["content"] == "Orchestrator handled it."

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_trivial_with_empty_answer_falls_through(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """Trivial classification with empty direct_answer should fall through to orchestrator."""
        mock_classify.return_value = _trivial_decision("")
        researcher, orchestrator, analyst = _mock_agents("Orchestrator answer.")
        mock_get.return_value = (researcher, orchestrator, analyst)
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "test"})
            ws.receive_json()  # routing
            ws.receive_json()  # planning
            done = ws.receive_json()
            assert done["type"] == "done"
            assert done["content"] == "Orchestrator answer."
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
        researcher, orchestrator, analyst = _mock_agents()
        mock_agents.return_value = (researcher, orchestrator, analyst)

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
        researcher, orchestrator, analyst = _mock_agents("Researcher answer")
        mock_agents.return_value = (researcher, orchestrator, analyst)

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
        researcher, orchestrator, analyst = _mock_agents("Orchestrator answer")
        mock_agents.return_value = (researcher, orchestrator, analyst)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Write a guide about trinkets"})
            done = _drain_status(ws)
            assert done["content"] == "Orchestrator answer"

        orchestrator.execute.assert_called_once()
        researcher.execute.assert_not_called()


class TestFastModelRouting:
    """Tests for fast model tier routing in moderate vs complex queries."""

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_moderate_passes_fast_model(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """MODERATE complexity passes config.FAST_MODEL to researcher.execute()."""
        from code.shukketsu import config

        mock_classify.return_value = _moderate_decision()
        researcher, orchestrator, analyst = _mock_agents("Researcher answer")
        mock_agents.return_value = (researcher, orchestrator, analyst)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "What is the hit cap?"})
            _drain_status(ws)

        researcher.execute.assert_called_once()
        call_kwargs = researcher.execute.call_args
        assert call_kwargs.kwargs.get("model_name") == config.FAST_MODEL

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_complex_no_model_override(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """COMPLEX complexity does not pass model_name to orchestrator.execute()."""
        mock_classify.return_value = _complex_decision()
        researcher, orchestrator, analyst = _mock_agents("Orchestrator answer")
        mock_agents.return_value = (researcher, orchestrator, analyst)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Write a full guide"})
            _drain_status(ws)

        orchestrator.execute.assert_called_once()
        call_kwargs = orchestrator.execute.call_args
        # No model_name kwarg should be passed (orchestrator always uses 70B)
        assert "model_name" not in call_kwargs.kwargs


class TestSendStatusFormat:
    """Tests for _send_status handling of str and dict messages."""

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_send_status_wraps_string(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """String messages are wrapped in {'type': 'status', 'content': msg}."""
        mock_classify.return_value = _moderate_decision()
        researcher, orchestrator, analyst = _mock_agents("Answer")

        async def _capture_execute(task, on_status=None, **kwargs):  # type: ignore[no-untyped-def]
            if on_status:
                await on_status("test string message")
            return MagicMock(output="Answer")

        researcher.execute = _capture_execute
        mock_get.return_value = (researcher, orchestrator, analyst)

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
        researcher, orchestrator, analyst = _mock_agents("Answer")

        async def _capture_execute(task, on_status=None, **kwargs):  # type: ignore[no-untyped-def]
            if on_status:
                await on_status({"type": "step", "agent": "researcher", "action": "tool_call", "tool": "rag_search"})
            return MagicMock(output="Answer")

        researcher.execute = _capture_execute
        mock_get.return_value = (researcher, orchestrator, analyst)

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

        researcher, orchestrator, analyst = _mock_agents()
        orchestrator.execute = AsyncMock(return_value=orch_result)
        mock_agents.return_value = (researcher, orchestrator, analyst)

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
        researcher, orchestrator, analyst = _mock_agents("Fallback answer")
        mock_agents.return_value = (researcher, orchestrator, analyst)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Something complex"})
            done = _drain_status(ws)
            assert done["content"] == "Fallback answer"

        orchestrator.execute.assert_called_once()


class TestMemoryIntegration:
    """Tests for session memory recall and extraction hooks in the chat handler."""

    @patch("code.shukketsu.web.routers.chat.config.MEMORY_ENABLED", True)
    @patch("code.shukketsu.web.routers.chat._get_memory_manager")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_memory_recall_injects_context(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
        mock_get_mm: MagicMock,
    ) -> None:
        """Recalled memories should cause memory_context to be set on the task."""
        from code.shukketsu.memory.models import SessionMemory

        mock_classify.return_value = _moderate_decision()
        researcher, orchestrator, analyst = _mock_agents("Researcher answer")
        mock_agents.return_value = (researcher, orchestrator, analyst)

        memory = SessionMemory(
            id=1,
            query="hit cap for rogues",
            answer_summary="9% hit cap in TBC",
            key_facts=["9% hit cap"],
            entities_mentioned=["Rogue"],
            retrieval_quality=0.8,
            created_at="2026-02-12T00:00:00+00:00",
            score=0.9,
        )
        mm = MagicMock()
        mm.recall_relevant = AsyncMock(return_value=[memory])
        mm.extract_session_memory = AsyncMock()
        mock_get_mm.return_value = mm

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "What trinkets for combat?"})
            done = _drain_status(ws)
            assert done["type"] == "done"

        # Verify researcher.execute was called with memory_context in context dict
        researcher.execute.assert_called_once()
        call_args = researcher.execute.call_args
        task = call_args[0][0] if call_args[0] else call_args.kwargs.get("task")
        assert task.context.get("memory_context") is not None
        assert "9% hit cap" in task.context["memory_context"]

    @patch("code.shukketsu.web.routers.chat.config.MEMORY_ENABLED", True)
    @patch("code.shukketsu.web.routers.chat._get_memory_manager")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_memory_extraction_fires_after_response(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
        mock_get_mm: MagicMock,
    ) -> None:
        """extract_session_memory should be called after a successful response."""
        mock_classify.return_value = _moderate_decision()
        researcher, orchestrator, analyst = _mock_agents("The answer is 42")
        mock_agents.return_value = (researcher, orchestrator, analyst)

        mm = MagicMock()
        mm.recall_relevant = AsyncMock(return_value=[])
        mm.extract_session_memory = AsyncMock()
        mock_get_mm.return_value = mm

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "What is the hit cap?"})
            done = _drain_status(ws)
            assert done["type"] == "done"

        mm.extract_session_memory.assert_called_once()
        call_kwargs = mm.extract_session_memory.call_args
        # First positional arg is query, second is answer
        assert call_kwargs[1]["query"] == "What is the hit cap?"
        assert call_kwargs[1]["answer"] == "The answer is 42"

    @patch("code.shukketsu.web.routers.chat.config.MEMORY_ENABLED", False)
    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_memory_disabled_skips_both(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """When MEMORY_ENABLED is False, no recall or extraction should happen."""
        mock_classify.return_value = _moderate_decision()
        researcher, orchestrator, analyst = _mock_agents("Answer")
        mock_agents.return_value = (researcher, orchestrator, analyst)

        # We do NOT patch _get_memory_manager — if it were called, the test would
        # fail with an error because no mock is set up.
        with patch("code.shukketsu.web.routers.chat._get_memory_manager") as mock_get_mm:
            client = TestClient(_get_app())
            with client.websocket_connect("/ws/chat") as ws:
                ws.receive_json()  # connected
                ws.send_json({"type": "message", "content": "Test query"})
                done = _drain_status(ws)
                assert done["type"] == "done"

            mock_get_mm.assert_not_called()

    @patch("code.shukketsu.web.routers.chat.config.MEMORY_ENABLED", True)
    @patch("code.shukketsu.web.routers.chat._get_memory_manager")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_memory_recall_empty_no_injection(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
        mock_get_mm: MagicMock,
    ) -> None:
        """Empty recall should not add memory_context to the task."""
        mock_classify.return_value = _moderate_decision()
        researcher, orchestrator, analyst = _mock_agents("Answer")
        mock_agents.return_value = (researcher, orchestrator, analyst)

        mm = MagicMock()
        mm.recall_relevant = AsyncMock(return_value=[])
        mm.extract_session_memory = AsyncMock()
        mock_get_mm.return_value = mm

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Test query"})
            done = _drain_status(ws)
            assert done["type"] == "done"

        researcher.execute.assert_called_once()
        call_args = researcher.execute.call_args
        task = call_args[0][0] if call_args[0] else call_args.kwargs.get("task")
        # No memory_context should be set when recall returns empty
        assert not task.context.get("memory_context")


class TestAnalysisRouting:
    """Tests for ANALYSIS category routing to the Analyst agent."""

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_analysis_routes_to_analyst(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """ANALYSIS category routes to Analyst.execute()."""
        mock_classify.return_value = _analysis_decision()
        researcher, orchestrator, analyst = _mock_agents("DPS: 1500")
        mock_agents.return_value = (researcher, orchestrator, analyst)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "What is my DPS?"})
            done = _drain_status(ws)
            assert done["type"] == "done"
            assert done["content"] == "DPS: 1500"

        analyst.execute.assert_called_once()
        researcher.execute.assert_not_called()
        orchestrator.execute.assert_not_called()

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_analysis_sends_analyzing_status(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """ANALYSIS path should send 'analyzing...' status."""
        mock_classify.return_value = _analysis_decision()
        mock_agents.return_value = _mock_agents("DPS result")

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Sim my character"})
            routing = ws.receive_json()
            assert routing == {"type": "status", "content": "routing..."}
            analyzing = ws.receive_json()
            assert analyzing == {"type": "status", "content": "analyzing..."}

    @patch("code.shukketsu.web.routers.chat.config.MEMORY_ENABLED", True)
    @patch("code.shukketsu.web.routers.chat._get_memory_manager")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_analysis_injects_memory_context(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
        mock_get_mm: MagicMock,
    ) -> None:
        """ANALYSIS path should inject memory_context when available."""
        from code.shukketsu.memory.models import SessionMemory

        mock_classify.return_value = _analysis_decision()
        researcher, orchestrator, analyst = _mock_agents("DPS: 1500")
        mock_agents.return_value = (researcher, orchestrator, analyst)

        memory = SessionMemory(
            id=1,
            query="my sim setup",
            answer_summary="Assassination with T6 gear",
            key_facts=["T6 gear"],
            entities_mentioned=["Assassination"],
            retrieval_quality=0.8,
            created_at="2026-02-12T00:00:00+00:00",
            score=0.9,
        )
        mm = MagicMock()
        mm.recall_relevant = AsyncMock(return_value=[memory])
        mm.extract_session_memory = AsyncMock()
        mm.record_strategy = AsyncMock()
        mock_get_mm.return_value = mm

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Sim my character"})
            done = _drain_status(ws)
            assert done["type"] == "done"

        analyst.execute.assert_called_once()
        call_args = analyst.execute.call_args
        task = call_args[0][0] if call_args[0] else call_args.kwargs.get("task")
        assert task.context.get("memory_context") is not None
        assert "T6 gear" in task.context["memory_context"]


class TestLangfuseResilience:
    """Tests that Langfuse failures never block answer delivery."""

    @patch("code.shukketsu.web.routers.chat.get_client")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_langfuse_trace_update_failure_still_delivers_answer(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
        mock_langfuse: MagicMock,
    ) -> None:
        """If langfuse.update_current_trace() raises, the answer should still be delivered."""
        mock_classify.return_value = _trivial_decision("Direct answer works.")

        # Make update_current_trace raise but get_current_trace_id work
        client_instance = MagicMock()
        client_instance.update_current_trace.side_effect = RuntimeError("Langfuse is down")
        client_instance.get_current_trace_id.return_value = None
        mock_langfuse.return_value = client_instance

        mock_agents.return_value = _mock_agents()
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "test"})
            done = _drain_status(ws)
            assert done["type"] == "done"
            assert done["content"] == "Direct answer works."

    @patch("code.shukketsu.web.routers.chat.get_client")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_langfuse_trace_id_failure_still_delivers_answer(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
        mock_langfuse: MagicMock,
    ) -> None:
        """If get_current_trace_id() raises, the answer should still be delivered with trace_id=None."""
        mock_classify.return_value = _trivial_decision("Answer despite trace failure.")

        client_instance = MagicMock()
        client_instance.update_current_trace.return_value = None
        client_instance.get_current_trace_id.side_effect = RuntimeError("Trace retrieval failed")
        mock_langfuse.return_value = client_instance

        mock_agents.return_value = _mock_agents()
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "test"})
            done = _drain_status(ws)
            assert done["type"] == "done"
            assert done["content"] == "Answer despite trace failure."
            assert done["trace_id"] is None
