"""Tests for ToolCallRecord and AgentResult trajectory field."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from code.shukketsu.agents.base import BaseAgent
from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    TaskStatus,
    ToolCallRecord,
)
from code.shukketsu.llm.schemas import ActionType, AgentStep, ToolCall
from code.shukketsu.tools.registry import ToolRegistry
from code.shukketsu.tools.schemas import Tool


class TestToolCallRecord:
    def test_valid_record(self) -> None:
        r = ToolCallRecord(tool_name="rag_search", tool_input={"query": "hit cap"})
        assert r.tool_name == "rag_search"
        assert r.tool_input == {"query": "hit cap"}

    def test_empty_tool_input(self) -> None:
        r = ToolCallRecord(tool_name="graph_search")
        assert r.tool_input == {}

    def test_requires_tool_name(self) -> None:
        with pytest.raises(ValidationError):
            ToolCallRecord(tool_input={"query": "test"})  # type: ignore[call-arg]


class TestAgentResultTrajectory:
    def test_default_empty_trajectory(self) -> None:
        r = AgentResult(
            task_id="t1",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="answer",
        )
        assert r.trajectory == []

    def test_trajectory_with_records(self) -> None:
        records = [
            ToolCallRecord(tool_name="rag_search", tool_input={"query": "q1"}),
            ToolCallRecord(tool_name="graph_search", tool_input={"entity": "e1"}),
        ]
        r = AgentResult(
            task_id="t1",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="answer",
            trajectory=records,
        )
        assert len(r.trajectory) == 2
        assert r.trajectory[0].tool_name == "rag_search"

    def test_trajectory_serializes_to_json(self) -> None:
        r = AgentResult(
            task_id="t1",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="answer",
            trajectory=[ToolCallRecord(tool_name="rag_search", tool_input={"query": "q"})],
        )
        data = r.model_dump()
        assert len(data["trajectory"]) == 1
        assert data["trajectory"][0]["tool_name"] == "rag_search"


class _EchoTool(Tool):
    name = "echo"
    description = "Echo."
    parameters_schema = {"text": {"type": "string", "description": "Text"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        return f"Echo: {tool_input.get('text', '')}"


def _registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _final(answer: str) -> AgentStep:
    return AgentStep(reasoning="Done.", action=ActionType.FINAL_ANSWER, answer=answer)


def _call(name: str, inp: dict[str, Any]) -> AgentStep:
    return AgentStep(
        reasoning="Need.",
        action=ActionType.TOOL_CALL,
        tool_call=ToolCall(thought="go", tool_name=name, tool_input=inp),
    )


class TestBaseAgentTrajectory:
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_execute_populates_trajectory(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = [
            _call("echo", {"text": "hello"}),
            _final("Done"),
        ]
        agent = BaseAgent(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await agent.execute(AgentTask(query="q"))
        assert len(result.trajectory) == 1
        assert result.trajectory[0].tool_name == "echo"
        assert result.trajectory[0].tool_input == {"text": "hello"}

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_empty_trajectory_on_immediate_answer(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _final("Answer")
        agent = BaseAgent(tool_registry=_registry(), role=AgentRole.RESEARCHER)
        result = await agent.execute(AgentTask(query="q"))
        assert result.trajectory == []

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_multiple_tool_calls_in_trajectory(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = [
            _call("echo", {"text": "a"}),
            _call("echo", {"text": "b"}),
            _final("Done"),
        ]
        agent = BaseAgent(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await agent.execute(AgentTask(query="q"))
        assert len(result.trajectory) == 2


class TestResearcherTrajectory:
    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_researcher_populates_trajectory(self, mock_loop: AsyncMock, mock_struct: AsyncMock) -> None:
        from code.shukketsu.agents.researcher import Researcher, StructuredFindings
        from code.shukketsu.agents.tasks import Finding

        mock_loop.side_effect = [
            _call("echo", {"text": "search"}),
            _final("Found it"),
        ]
        mock_struct.return_value = StructuredFindings(
            findings=[Finding(claim="test", confidence=0.9)],
            sufficient=True,
        )
        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await researcher.execute(AgentTask(query="q"))
        assert len(result.trajectory) == 1
        assert result.trajectory[0].tool_name == "echo"


class TestOrchestratorTrajectory:
    @patch("code.shukketsu.agents.orchestrator.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    @patch("code.shukketsu.agents.researcher.get_structured_output")
    async def test_orchestrator_dispatch_trajectory(
        self, mock_struct: AsyncMock, mock_base: AsyncMock, mock_orch: AsyncMock
    ) -> None:
        from code.shukketsu.agents.factory import AgentFactory
        from code.shukketsu.agents.orchestrator import Orchestrator, _SynthesisOutput
        from code.shukketsu.agents.researcher import StructuredFindings
        from code.shukketsu.agents.tasks import Finding, OrchestratorPlan, SubTask

        plan = OrchestratorPlan(
            reasoning="Research needed",
            subtasks=[SubTask(agent_role=AgentRole.RESEARCHER, description="Find hit cap info")],
        )

        # orchestrator.get_structured_output: decompose -> plan, synthesize -> response
        mock_orch.side_effect = [
            plan,
            _SynthesisOutput(response="Hit cap is 142 for TBC rogues."),
        ]

        # base.get_structured_output: researcher ReAct loop
        mock_base.side_effect = [
            _final("Hit cap is 142"),
        ]

        # researcher.get_structured_output: structuring pass
        mock_struct.return_value = StructuredFindings(
            findings=[Finding(claim="Hit cap is 142", confidence=0.9)],
            sufficient=True,
        )

        factory = AgentFactory()
        orch = Orchestrator(
            tool_registry=_registry(),
            role=AgentRole.ORCHESTRATOR,
            factory=factory,
        )
        result = await orch.execute(AgentTask(query="What is the hit cap?"))

        assert len(result.trajectory) >= 1
        assert result.trajectory[0].tool_name == "dispatch:researcher"
