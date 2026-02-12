"""Tier 2: Live agent tests — real Ollama, real DB with seed data.

Run with: python3 -m pytest tests/integration/test_live_agents.py -m e2e -v
"""

import pytest

from code.shukketsu.agents.factory import AgentFactory
from code.shukketsu.agents.tasks import AgentRole, AgentTask, ResearchTask, TaskStatus
from code.shukketsu.ingest.embedder import get_embedder
from code.shukketsu.routing.router import classify_query
from code.shukketsu.tools.knowledge.graph_search import GraphSearchTool
from code.shukketsu.tools.knowledge.search import RagSearchTool
from code.shukketsu.tools.registry import ToolRegistry

pytestmark = pytest.mark.e2e


def _make_registry(conn):  # type: ignore[no-untyped-def]
    """Create a tool registry with RAG and graph search tools."""
    embedder = get_embedder()
    reg = ToolRegistry()
    reg.register(RagSearchTool(conn=conn, embed_fn=embedder.embed_query))
    reg.register(GraphSearchTool(conn=conn))
    return reg


class TestResearcherLive:
    async def test_answers_from_seeded_content(self, seeded_db) -> None:  # type: ignore[no-untyped-def]
        """Researcher finds facts from seed content, returns non-empty output."""
        reg = _make_registry(seeded_db)
        factory = AgentFactory()
        researcher = factory.create(AgentRole.RESEARCHER, tool_registry=reg)

        result = await researcher.execute(ResearchTask(query="What is the hit cap for a combat rogue?"))

        assert result.status in (TaskStatus.SUCCESS, TaskStatus.PARTIAL)
        assert len(result.output) > 0


class TestRouterLive:
    async def test_classifies_trivial_question(self) -> None:
        """Qwen 4B returns TRIVIAL for simple factual question."""
        decision = await classify_query("How much energy does Sinister Strike cost?")
        assert decision.complexity.value == "trivial"

    async def test_classifies_complex_question(self) -> None:
        """Qwen 4B returns COMPLEX for multi-part question."""
        decision = await classify_query(
            "Compare combat swords vs mutilate for Phase 1 raiding, "
            "including stat priorities, rotation differences, and gear requirements"
        )
        assert decision.complexity.value == "complex"


class TestOrchestratorLive:
    async def test_decomposes_and_dispatches(self, seeded_db) -> None:  # type: ignore[no-untyped-def]
        """Full orchestrator flow produces OrchestratorResult with specialist_results."""
        from code.shukketsu import config
        from code.shukketsu.agents.tasks import OrchestratorResult
        from code.shukketsu.knowledge.manager import KnowledgeManager

        reg = _make_registry(seeded_db)
        factory = AgentFactory()
        km = KnowledgeManager(seeded_db, config.WIKI_PATH)

        orch = factory.create(
            AgentRole.ORCHESTRATOR,
            tool_registry=reg,
            factory=factory,
            knowledge_manager=km,
        )
        result = await orch.execute(AgentTask(query="Compare combat swords vs mutilate for Phase 1 raiding"))

        assert isinstance(result, OrchestratorResult)
        assert len(result.output) > 0
