"""Tests for ToolCallRecord and AgentResult trajectory field."""

import pytest
from pydantic import ValidationError

from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    TaskStatus,
    ToolCallRecord,
)


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
