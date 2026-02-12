"""Tier 1: Wiring smoke tests -- all LLM calls mocked, CI-safe."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from code.shukketsu.agents.factory import AgentFactory
from code.shukketsu.agents.tasks import (
    AgentRole,
    OrchestratorPlan,
    OrchestratorResult,
    ResearchResult,
    ResearchTask,
    TaskStatus,
)
from code.shukketsu.llm.schemas import ActionType, AgentStep, ToolCall
from code.shukketsu.tools.registry import ToolRegistry
from code.shukketsu.tools.schemas import Tool


class _DummyTool(Tool):
    name = "rag_search"
    description = "Search knowledge base."
    parameters_schema = {"query": {"type": "string", "description": "Query"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        return "Found result about rogues."


class _GraphTool(Tool):
    name = "graph_search"
    description = "Search knowledge graph."
    parameters_schema = {"entity": {"type": "string", "description": "Entity"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        return "Entity: combat swords -> has_stat -> expertise"


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_DummyTool())
    reg.register(_GraphTool())
    return reg


def _final(answer: str) -> AgentStep:
    return AgentStep(reasoning="Done.", action=ActionType.FINAL_ANSWER, answer=answer)


def _call(name: str, inp: dict[str, Any]) -> AgentStep:
    return AgentStep(
        reasoning="Need info.",
        action=ActionType.TOOL_CALL,
        tool_call=ToolCall(thought="go", tool_name=name, tool_input=inp),
    )


def _mock_langfuse() -> MagicMock:
    """Create a mock langfuse client with update_current_trace."""
    client = MagicMock()
    client.update_current_trace = MagicMock()
    return client


class TestFactoryCreatesAllRoles:
    """Verify AgentFactory instantiates each specialist class correctly."""

    def test_researcher(self) -> None:
        from code.shukketsu.agents.researcher import Researcher

        agent = AgentFactory().create(AgentRole.RESEARCHER)
        assert isinstance(agent, Researcher)

    def test_writer(self) -> None:
        from code.shukketsu.agents.writer import Writer

        agent = AgentFactory().create(AgentRole.WRITER, knowledge_manager=MagicMock())
        assert isinstance(agent, Writer)

    def test_editor(self) -> None:
        from code.shukketsu.agents.editor import Editor

        agent = AgentFactory().create(AgentRole.EDITOR, knowledge_manager=MagicMock())
        assert isinstance(agent, Editor)

    def test_orchestrator(self) -> None:
        from code.shukketsu.agents.orchestrator import Orchestrator

        factory = AgentFactory()
        agent = factory.create(AgentRole.ORCHESTRATOR, factory=factory)
        assert isinstance(agent, Orchestrator)


class TestFactoryToolConfigs:
    """Verify tool registries and kwargs are wired through correctly."""

    def test_researcher_accepts_tool_registry(self) -> None:
        reg = _registry()
        agent = AgentFactory().create(AgentRole.RESEARCHER, tool_registry=reg)
        assert agent.tool_registry is reg
        assert "rag_search" in agent.tool_registry

    def test_editor_accepts_tool_registry(self) -> None:
        reg = _registry()
        agent = AgentFactory().create(AgentRole.EDITOR, tool_registry=reg, knowledge_manager=MagicMock())
        assert agent.tool_registry is reg

    def test_writer_receives_knowledge_manager(self) -> None:
        km = MagicMock()
        agent = AgentFactory().create(AgentRole.WRITER, knowledge_manager=km)
        assert agent._km is km  # type: ignore[attr-defined]  # noqa: SLF001

    def test_orchestrator_receives_factory_and_km(self) -> None:
        factory = AgentFactory()
        km = MagicMock()
        agent = factory.create(
            AgentRole.ORCHESTRATOR,
            factory=factory,
            knowledge_manager=km,
        )
        assert agent._factory is factory  # type: ignore[attr-defined]  # noqa: SLF001
        assert agent._km is km  # type: ignore[attr-defined]  # noqa: SLF001


class TestChatHandlerWiring:
    """Verify the chat handler routes queries to the correct specialist."""

    @patch("code.shukketsu.web.routers.chat.get_client")
    @patch("code.shukketsu.web.routers.chat.classify_query")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    async def test_moderate_routes_to_researcher(
        self,
        mock_agents: MagicMock,
        mock_classify: AsyncMock,
        mock_langfuse_client: MagicMock,
    ) -> None:
        from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity
        from code.shukketsu.web.routers.chat import ChatSession, _agent_response

        mock_langfuse_client.return_value = _mock_langfuse()

        mock_classify.return_value = RoutingDecision(
            complexity=TaskComplexity.MODERATE,
            category=TaskCategory.RETRIEVAL,
            needs_tools=True,
            suggested_agent="researcher",
        )

        mock_researcher = AsyncMock()
        mock_researcher.execute.return_value = ResearchResult(
            task_id="t1",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="The hit cap is 142.",
        )
        mock_agents.return_value = (mock_researcher, AsyncMock())

        ws = AsyncMock()
        session = ChatSession()
        await _agent_response(ws, session, "What is the hit cap?")

        mock_researcher.execute.assert_called_once()
        done_calls = [c for c in ws.send_json.call_args_list if c.args[0].get("type") == "done"]
        assert len(done_calls) == 1
        assert "142" in done_calls[0].args[0]["content"]

    @patch("code.shukketsu.web.routers.chat.get_client")
    @patch("code.shukketsu.web.routers.chat.classify_query")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    async def test_complex_routes_to_orchestrator(
        self,
        mock_agents: MagicMock,
        mock_classify: AsyncMock,
        mock_langfuse_client: MagicMock,
    ) -> None:
        from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity
        from code.shukketsu.web.routers.chat import ChatSession, _agent_response

        mock_langfuse_client.return_value = _mock_langfuse()

        mock_classify.return_value = RoutingDecision(
            complexity=TaskComplexity.COMPLEX,
            category=TaskCategory.ANALYSIS,
            needs_tools=True,
            suggested_agent="orchestrator",
        )

        mock_orch = AsyncMock()
        mock_orch.execute.return_value = OrchestratorResult(
            task_id="t2",
            agent_role=AgentRole.ORCHESTRATOR,
            status=TaskStatus.SUCCESS,
            output="Comparison complete.",
            plan=OrchestratorPlan(reasoning="plan", subtasks=[]),
        )
        mock_agents.return_value = (AsyncMock(), mock_orch)

        ws = AsyncMock()
        session = ChatSession()
        await _agent_response(ws, session, "Compare combat vs mutilate")

        mock_orch.execute.assert_called_once()


class TestResearcherResultTrajectory:
    """Verify the Researcher populates trajectory from tool calls."""

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_researcher_result_includes_trajectory(self, mock_loop: AsyncMock, mock_struct: AsyncMock) -> None:
        from code.shukketsu.agents.researcher import Researcher, StructuredFindings
        from code.shukketsu.agents.tasks import Finding

        mock_loop.side_effect = [
            _call("rag_search", {"query": "hit cap"}),
            _final("142 rating"),
        ]
        mock_struct.return_value = StructuredFindings(
            findings=[Finding(claim="Hit cap is 142", confidence=0.9)],
            sufficient=True,
        )

        researcher = Researcher(tool_registry=_registry(), role=AgentRole.RESEARCHER)
        result = await researcher.execute(ResearchTask(query="hit cap"))

        assert len(result.trajectory) == 1
        assert result.trajectory[0].tool_name == "rag_search"


class TestErrorPropagation:
    """Verify errors from agents surface correctly over the WebSocket."""

    @patch("code.shukketsu.web.routers.chat.get_client")
    @patch("code.shukketsu.web.routers.chat.classify_query")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    async def test_shukketsu_error_surfaces_as_ws_error(
        self,
        mock_agents: MagicMock,
        mock_classify: AsyncMock,
        mock_langfuse_client: MagicMock,
    ) -> None:
        from code.shukketsu.resilience.errors import LLMUnavailableError
        from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity
        from code.shukketsu.web.routers.chat import ChatSession, _agent_response

        mock_langfuse_client.return_value = _mock_langfuse()

        mock_classify.return_value = RoutingDecision(
            complexity=TaskComplexity.MODERATE,
            category=TaskCategory.RETRIEVAL,
            needs_tools=True,
            suggested_agent="researcher",
        )
        mock_researcher = AsyncMock()
        mock_researcher.execute.side_effect = LLMUnavailableError("Server down")
        mock_agents.return_value = (mock_researcher, AsyncMock())

        ws = AsyncMock()
        session = ChatSession()
        await _agent_response(ws, session, "test")

        error_calls = [c for c in ws.send_json.call_args_list if c.args[0].get("type") == "error"]
        assert len(error_calls) == 1
        assert "Server down" in error_calls[0].args[0]["content"]

    @patch("code.shukketsu.web.routers.chat.get_client")
    @patch("code.shukketsu.web.routers.chat.classify_query")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    async def test_unexpected_error_surfaces_generic_message(
        self,
        mock_agents: MagicMock,
        mock_classify: AsyncMock,
        mock_langfuse_client: MagicMock,
    ) -> None:
        from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity
        from code.shukketsu.web.routers.chat import ChatSession, _agent_response

        mock_langfuse_client.return_value = _mock_langfuse()

        mock_classify.return_value = RoutingDecision(
            complexity=TaskComplexity.MODERATE,
            category=TaskCategory.RETRIEVAL,
            needs_tools=True,
            suggested_agent="researcher",
        )
        mock_researcher = AsyncMock()
        mock_researcher.execute.side_effect = RuntimeError("Unexpected")
        mock_agents.return_value = (mock_researcher, AsyncMock())

        ws = AsyncMock()
        session = ChatSession()
        await _agent_response(ws, session, "test")

        error_calls = [c for c in ws.send_json.call_args_list if c.args[0].get("type") == "error"]
        assert len(error_calls) == 1
        assert "unexpected error" in error_calls[0].args[0]["content"].lower()
