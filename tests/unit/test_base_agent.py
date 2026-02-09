"""Tests for the BaseAgent ReAct loop."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from code.shukketsu.agents.base import BaseAgent
from code.shukketsu.llm.schemas import ActionType, AgentStep, ToolCall
from code.shukketsu.resilience.errors import LLMUnavailableError, StructuredOutputError
from code.shukketsu.tools.registry import ToolRegistry
from code.shukketsu.tools.schemas import Tool


class EchoTool(Tool):
    name = "echo"
    description = "Echoes the input back."
    parameters_schema = {"text": {"type": "string", "description": "Text to echo"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        return f"Echo: {tool_input.get('text', '')}"


def _registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    return registry


def _final_answer(answer: str) -> AgentStep:
    return AgentStep(reasoning="I have the answer.", action=ActionType.FINAL_ANSWER, answer=answer)


def _tool_call(tool_name: str, tool_input: dict[str, Any]) -> AgentStep:
    return AgentStep(
        reasoning="Need info.",
        action=ActionType.TOOL_CALL,
        tool_call=ToolCall(thought="Calling tool", tool_name=tool_name, tool_input=tool_input),
    )


class TestBaseAgentInit:
    def test_stores_registry(self) -> None:
        reg = _registry()
        agent = BaseAgent(tool_registry=reg)
        assert agent.tool_registry is reg

    def test_default_max_iterations(self) -> None:
        from code.shukketsu import config

        agent = BaseAgent(tool_registry=_registry())
        assert agent.max_iterations == config.AGENT_MAX_ITERATIONS

    def test_custom_max_iterations(self) -> None:
        agent = BaseAgent(tool_registry=_registry(), max_iterations=3)
        assert agent.max_iterations == 3


class TestBaseAgentRun:
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_immediate_final_answer(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _final_answer("The answer is 42.")
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        result = await agent.run("What?")
        assert result == "The answer is 42."
        assert mock_llm.call_count == 1

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_tool_call_then_answer(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = [
            _tool_call("echo", {"text": "hello"}),
            _final_answer("Based on echo: hello"),
        ]
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        result = await agent.run("Say hello")
        assert result == "Based on echo: hello"
        assert mock_llm.call_count == 2

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_scratchpad_grows(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = [
            _tool_call("echo", {"text": "first"}),
            _tool_call("echo", {"text": "second"}),
            _final_answer("Done"),
        ]
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        await agent.run("Do two things")
        first_msgs = mock_llm.call_args_list[0].kwargs["messages"]
        third_msgs = mock_llm.call_args_list[2].kwargs["messages"]
        assert len(third_msgs) > len(first_msgs)

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_max_iterations_graceful(self, mock_llm: AsyncMock) -> None:
        from code.shukketsu import config

        mock_llm.return_value = _tool_call("echo", {"text": "loop"})
        agent = BaseAgent(tool_registry=_registry(EchoTool()), max_iterations=2)
        result = await agent.run("Loop forever")
        assert result == config.AGENT_GRACEFUL_FAILURE
        assert mock_llm.call_count == 2

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_unknown_tool_continues(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = [
            _tool_call("nonexistent", {"q": "test"}),
            _final_answer("Couldn't find tool."),
        ]
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        result = await agent.run("Bad tool")
        assert result == "Couldn't find tool."
        assert mock_llm.call_count == 2

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_propagates_llm_unavailable(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = LLMUnavailableError("Server down")
        agent = BaseAgent(tool_registry=_registry())
        with pytest.raises(LLMUnavailableError, match="Server down"):
            await agent.run("test")

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_propagates_structured_output_error(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = StructuredOutputError("Validation failed")
        agent = BaseAgent(tool_registry=_registry())
        with pytest.raises(StructuredOutputError, match="Validation failed"):
            await agent.run("test")


class TestBuildMessages:
    def test_starts_with_system(self) -> None:
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        msgs = agent._build_messages("query", [])
        assert msgs[0]["role"] == "system"

    def test_includes_user_query(self) -> None:
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        msgs = agent._build_messages("What is hit cap?", [])
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert any("What is hit cap?" in m["content"] for m in user_msgs)

    def test_includes_scratchpad(self) -> None:
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        scratchpad = [
            {
                "reasoning": "Need to search",
                "tool_name": "echo",
                "tool_input": {"text": "test"},
                "observation": "Echo: test",
            }
        ]
        msgs = agent._build_messages("query", scratchpad)
        all_content = " ".join(m["content"] for m in msgs)
        assert "Echo: test" in all_content

    def test_system_contains_tool_descriptions(self) -> None:
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        msgs = agent._build_messages("query", [])
        system = msgs[0]["content"]
        assert "echo" in system.lower()
        assert "Echoes the input" in system
