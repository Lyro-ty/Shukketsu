"""Integration smoke tests for the agent pipeline."""

import sqlite3
import struct
from unittest.mock import AsyncMock, patch

from code.shukketsu.agents.base import BaseAgent
from code.shukketsu.llm.schemas import ActionType, AgentStep, ToolCall
from code.shukketsu.tools.knowledge.search import RagSearchTool
from code.shukketsu.tools.registry import ToolRegistry


def _pack(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


async def _mock_embed(text: str) -> list[float]:
    return [0.1] * 768


def _seed(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO sources (url, title, source_type, trust_score) "
        "VALUES ('https://example.com', 'Test Guide', 'guide', 0.8)"
    )
    conn.execute(
        "INSERT INTO chunks (source_id, content, chunk_index) VALUES (1, 'The hit cap is 9% or 142 rating.', 0)"
    )
    conn.commit()
    conn.execute("INSERT INTO chunks_vec (rowid, embedding) VALUES (1, ?)", (_pack([0.1] * 768),))
    conn.commit()


class TestAgentEndToEnd:
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_answers_from_seeded_knowledge(self, mock_llm: AsyncMock, test_db: sqlite3.Connection) -> None:
        _seed(test_db)
        registry = ToolRegistry()
        registry.register(RagSearchTool(conn=test_db, embed_fn=_mock_embed))
        agent = BaseAgent(tool_registry=registry)

        mock_llm.side_effect = [
            AgentStep(
                reasoning="Search KB",
                action=ActionType.TOOL_CALL,
                tool_call=ToolCall(thought="search", tool_name="rag_search", tool_input={"query": "hit cap"}),
            ),
            AgentStep(reasoning="Found it", action=ActionType.FINAL_ANSWER, answer="The hit cap is 9%."),
        ]

        result = await agent.run("What is the hit cap?")
        assert result == "The hit cap is 9%."

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_handles_empty_results(self, mock_llm: AsyncMock, test_db: sqlite3.Connection) -> None:
        registry = ToolRegistry()
        registry.register(RagSearchTool(conn=test_db, embed_fn=_mock_embed))
        agent = BaseAgent(tool_registry=registry)

        mock_llm.side_effect = [
            AgentStep(
                reasoning="Search",
                action=ActionType.TOOL_CALL,
                tool_call=ToolCall(thought="search", tool_name="rag_search", tool_input={"query": "nothing"}),
            ),
            AgentStep(reasoning="No results", action=ActionType.FINAL_ANSWER, answer="I don't have that info."),
        ]

        result = await agent.run("Something obscure")
        assert "I don't have that info" in result

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_stops_at_max_iterations(self, mock_llm: AsyncMock, test_db: sqlite3.Connection) -> None:
        from code.shukketsu import config

        registry = ToolRegistry()
        registry.register(RagSearchTool(conn=test_db, embed_fn=_mock_embed))
        agent = BaseAgent(tool_registry=registry, max_iterations=2)

        mock_llm.return_value = AgentStep(
            reasoning="Keep searching",
            action=ActionType.TOOL_CALL,
            tool_call=ToolCall(thought="search", tool_name="rag_search", tool_input={"query": "loop"}),
        )

        result = await agent.run("Loop")
        assert result == config.AGENT_GRACEFUL_FAILURE
