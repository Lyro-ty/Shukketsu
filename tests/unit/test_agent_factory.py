"""Tests for the AgentFactory."""

from typing import Any
from unittest.mock import MagicMock

from code.shukketsu.agents.base import BaseAgent
from code.shukketsu.agents.factory import AgentFactory
from code.shukketsu.agents.tasks import AgentRole
from code.shukketsu.knowledge.manager import KnowledgeManager
from code.shukketsu.tools.registry import ToolRegistry
from code.shukketsu.tools.schemas import Tool


class StubTool(Tool):
    name = "stub"
    description = "A stub tool for testing."
    parameters_schema = {"input": {"type": "string", "description": "test"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        return "stub result"


def _mock_km() -> KnowledgeManager:
    """Create a mock KnowledgeManager for factory tests that don't need real CRUD."""
    return MagicMock(spec=KnowledgeManager)


def _factory_create(factory: AgentFactory, role: AgentRole, **kwargs: Any) -> BaseAgent:
    """Create an agent, passing knowledge_manager for Writer role."""
    if role == AgentRole.WRITER and "knowledge_manager" not in kwargs:
        kwargs["knowledge_manager"] = _mock_km()
    return factory.create(role, **kwargs)


class TestAgentFactoryCreate:
    def test_creates_base_agent(self) -> None:
        factory = AgentFactory()
        agent = factory.create(AgentRole.RESEARCHER)
        assert isinstance(agent, BaseAgent)

    def test_sets_role(self) -> None:
        factory = AgentFactory()
        agent = _factory_create(factory, AgentRole.WRITER)
        assert agent.role == AgentRole.WRITER

    def test_each_role_gets_different_prompt(self) -> None:
        factory = AgentFactory()
        researcher = factory.create(AgentRole.RESEARCHER)
        writer = _factory_create(factory, AgentRole.WRITER)
        assert researcher._system_prompt != writer._system_prompt

    def test_researcher_prompt_mentions_search(self) -> None:
        factory = AgentFactory()
        agent = factory.create(AgentRole.RESEARCHER)
        assert "search" in agent._system_prompt.lower()

    def test_writer_prompt_mentions_article(self) -> None:
        factory = AgentFactory()
        agent = _factory_create(factory, AgentRole.WRITER)
        prompt = agent._system_prompt.lower()
        assert "article" in prompt or "markdown" in prompt

    def test_editor_prompt_mentions_verify(self) -> None:
        factory = AgentFactory()
        agent = factory.create(AgentRole.EDITOR)
        assert "verify" in agent._system_prompt.lower()

    def test_orchestrator_prompt_mentions_decompose(self) -> None:
        factory = AgentFactory()
        agent = factory.create(AgentRole.ORCHESTRATOR)
        assert "decompose" in agent._system_prompt.lower()

    def test_role_classes_registry_exists(self) -> None:
        from code.shukketsu.agents.factory import _ROLE_CLASSES
        from code.shukketsu.agents.tasks import AgentRole

        assert AgentRole.RESEARCHER in _ROLE_CLASSES


class TestAgentFactoryToolRegistry:
    def test_default_empty_registry(self) -> None:
        factory = AgentFactory()
        agent = factory.create(AgentRole.RESEARCHER)
        assert len(agent.tool_registry) == 0

    def test_custom_registry(self) -> None:
        registry = ToolRegistry()
        registry.register(StubTool())
        factory = AgentFactory()
        agent = factory.create(AgentRole.RESEARCHER, tool_registry=registry)
        assert "stub" in agent.tool_registry

    def test_agents_get_independent_registries(self) -> None:
        """Two agents created without explicit registry should not share state."""
        factory = AgentFactory()
        a1 = factory.create(AgentRole.RESEARCHER)
        a2 = _factory_create(factory, AgentRole.WRITER)
        assert a1.tool_registry is not a2.tool_registry


class TestAgentFactoryIterationLimits:
    def test_researcher_gets_higher_limit(self) -> None:
        from code.shukketsu import config

        factory = AgentFactory()
        agent = factory.create(AgentRole.RESEARCHER)
        assert agent.max_iterations == config.RESEARCHER_MAX_ITERATIONS

    def test_non_researcher_gets_default_limit(self) -> None:
        from code.shukketsu import config

        factory = AgentFactory()
        for role in [AgentRole.WRITER, AgentRole.EDITOR, AgentRole.ORCHESTRATOR]:
            agent = _factory_create(factory, role)
            assert agent.max_iterations == config.AGENT_MAX_ITERATIONS
