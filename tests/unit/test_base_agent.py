"""Tests for the BaseAgent ReAct loop."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from code.shukketsu import config
from code.shukketsu.agents.base import BaseAgent
from code.shukketsu.agents.tasks import AgentResult, AgentRole, AgentTask, TaskStatus
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
    async def test_structured_output_error_returns_graceful_failure(self, mock_llm: AsyncMock) -> None:
        """StructuredOutputError should return graceful failure, not crash."""
        mock_llm.side_effect = StructuredOutputError("Validation failed")
        agent = BaseAgent(tool_registry=_registry())
        result = await agent.run("test")
        assert result == config.AGENT_GRACEFUL_FAILURE


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


class TestLoopDetection:
    """Tests for loop detection integration in BaseAgent."""

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_stops_on_consecutive_loop(self, mock_llm: AsyncMock) -> None:
        """Agent should stop and return graceful message when loop detected."""
        mock_llm.return_value = _tool_call("echo", {"text": "stuck"})
        agent = BaseAgent(tool_registry=_registry(EchoTool()), max_iterations=10)
        result = await agent.run("Loop test")
        # Default max_consecutive_same=3, so it should stop after 3 identical calls
        assert mock_llm.call_count <= 4
        assert isinstance(result, str)

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_loop_returns_partial_observations(self, mock_llm: AsyncMock) -> None:
        """Agent should include partial results when loop forces stop."""
        mock_llm.return_value = _tool_call("echo", {"text": "repeated"})
        agent = BaseAgent(tool_registry=_registry(EchoTool()), max_iterations=10)
        result = await agent.run("Loop test")
        assert len(result) > 0

    async def test_partial_answer_includes_all_observations(self) -> None:
        """When a loop is detected, all unique observations should be included."""
        agent = BaseAgent(tool_registry=_registry())
        scratchpad = [
            {
                "observation": "Rogues have a 9% hit cap.",
                "tool_name": "rag_search",
                "tool_input": {"query": "hit cap"},
                "reasoning": "look up hit cap",
            },
            {
                "observation": "Combat swords is the best spec.",
                "tool_name": "rag_search",
                "tool_input": {"query": "best spec"},
                "reasoning": "look up spec",
            },
            {
                "observation": "Rogues have a 9% hit cap.",
                "tool_name": "rag_search",
                "tool_input": {"query": "hit cap"},
                "reasoning": "look up hit cap",
            },
        ]
        result = agent._synthesize_partial_answer(scratchpad)
        assert "9% hit cap" in result
        assert "Combat swords" in result


class TestBaseAgentRole:
    def test_default_role_is_none(self) -> None:
        agent = BaseAgent(tool_registry=_registry())
        assert agent.role is None

    def test_explicit_role(self) -> None:
        agent = BaseAgent(tool_registry=_registry(), role=AgentRole.RESEARCHER)
        assert agent.role == AgentRole.RESEARCHER


class TestRunLoop:
    """Tests for _run_loop() returning _RunOutcome with status."""

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_success_outcome(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _final_answer("The answer.")
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        outcome = await agent._run_loop("query", on_status=None)
        assert outcome.output == "The answer."
        assert outcome.status == TaskStatus.SUCCESS

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_failed_outcome_on_max_iterations(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _tool_call("echo", {"text": "loop"})
        agent = BaseAgent(tool_registry=_registry(EchoTool()), max_iterations=1)
        outcome = await agent._run_loop("query", on_status=None)
        assert outcome.status == TaskStatus.FAILED

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_partial_outcome_on_loop_detection(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _tool_call("echo", {"text": "stuck"})
        agent = BaseAgent(tool_registry=_registry(EchoTool()), max_iterations=10)
        outcome = await agent._run_loop("query", on_status=None)
        assert outcome.status == TaskStatus.PARTIAL
        assert "partial" in outcome.output.lower()

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_run_outcome_has_scratchpad(self, mock_llm: AsyncMock) -> None:
        """_RunOutcome includes the scratchpad field."""
        mock_llm.return_value = _final_answer("answer")
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        outcome = await agent._run_loop("query", on_status=None)
        assert hasattr(outcome, "scratchpad")
        assert isinstance(outcome.scratchpad, list)

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_run_loop_returns_scratchpad_with_entries(self, mock_llm: AsyncMock) -> None:
        """_run_loop scratchpad contains tool call entries."""
        mock_llm.side_effect = [
            _tool_call("echo", {"text": "hello"}),
            _final_answer("Done"),
        ]
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        outcome = await agent._run_loop("query", on_status=None)
        assert len(outcome.scratchpad) == 1
        assert outcome.scratchpad[0]["tool_name"] == "echo"
        assert "Echo: hello" in outcome.scratchpad[0]["observation"]


class TestBaseAgentExecute:
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_returns_agent_result(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _final_answer("The answer.")
        agent = BaseAgent(tool_registry=_registry(), role=AgentRole.RESEARCHER)
        task = AgentTask(query="What is the hit cap?")
        result = await agent.execute(task)
        assert isinstance(result, AgentResult)
        assert result.task_id == task.task_id
        assert result.agent_role == AgentRole.RESEARCHER
        assert result.output == "The answer."

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_success_status(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _final_answer("Good answer.")
        agent = BaseAgent(tool_registry=_registry(), role=AgentRole.RESEARCHER)
        result = await agent.execute(AgentTask(query="q"))
        assert result.status == TaskStatus.SUCCESS

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_failed_status_on_graceful_failure(self, mock_llm: AsyncMock) -> None:
        """Max iterations reached -> AGENT_GRACEFUL_FAILURE -> FAILED status."""
        mock_llm.return_value = _tool_call("echo", {"text": "loop"})
        agent = BaseAgent(tool_registry=_registry(EchoTool()), role=AgentRole.RESEARCHER, max_iterations=1)
        result = await agent.execute(AgentTask(query="q"))
        assert result.status == TaskStatus.FAILED

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_partial_status_on_loop(self, mock_llm: AsyncMock) -> None:
        """Loop detected -> partial answer -> PARTIAL status."""
        mock_llm.return_value = _tool_call("echo", {"text": "stuck"})
        agent = BaseAgent(tool_registry=_registry(EchoTool()), role=AgentRole.RESEARCHER, max_iterations=10)
        result = await agent.execute(AgentTask(query="q"))
        assert result.status == TaskStatus.PARTIAL

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_execute_propagates_llm_errors(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = LLMUnavailableError("Server down")
        agent = BaseAgent(tool_registry=_registry(), role=AgentRole.RESEARCHER)
        with pytest.raises(LLMUnavailableError):
            await agent.execute(AgentTask(query="q"))

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_execute_requires_role(self, mock_llm: AsyncMock) -> None:
        """execute() requires a role to be set (needed for AgentResult)."""
        mock_llm.return_value = _final_answer("answer")
        agent = BaseAgent(tool_registry=_registry())  # no role
        with pytest.raises(ValueError, match="role"):
            await agent.execute(AgentTask(query="q"))

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_on_status_callback(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _final_answer("answer")
        agent = BaseAgent(tool_registry=_registry(), role=AgentRole.RESEARCHER)
        statuses: list[str] = []

        async def capture(msg: str) -> None:
            statuses.append(msg)

        await agent.execute(AgentTask(query="q"), on_status=capture)
        assert len(statuses) > 0


class TestBackwardCompat:
    """Verify that the run() interface is completely unchanged."""

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_run_still_returns_string(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _final_answer("42")
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        result = await agent.run("What?")
        assert isinstance(result, str)
        assert result == "42"

    def test_init_without_role(self) -> None:
        """Existing code that creates BaseAgent without role still works."""
        agent = BaseAgent(tool_registry=_registry())
        assert agent.role is None
        assert agent.max_iterations > 0


class TestStatusCallbackEvents:
    """Tests for structured dict events emitted via on_status."""

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_status_callback_receives_dict_for_tool_call(self, mock_llm: AsyncMock) -> None:
        """After a tool call, the callback receives a dict event with tool metadata."""
        mock_llm.side_effect = [
            _tool_call("echo", {"text": "hello"}),
            _final_answer("Done"),
        ]
        agent = BaseAgent(tool_registry=_registry(EchoTool()), role=AgentRole.RESEARCHER)
        events: list[str | dict] = []

        async def capture(msg: str | dict) -> None:
            events.append(msg)

        await agent.execute(AgentTask(query="q"), on_status=capture)

        dict_events = [e for e in events if isinstance(e, dict)]
        assert len(dict_events) >= 1
        step_event = dict_events[0]
        assert step_event["type"] == "step"
        assert step_event["action"] == "tool_call"
        assert step_event["tool"] == "echo"

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_status_callback_still_receives_strings(self, mock_llm: AsyncMock) -> None:
        """Existing string status messages (e.g. 'thinking...') still work."""
        mock_llm.return_value = _final_answer("Done")
        agent = BaseAgent(tool_registry=_registry(EchoTool()), role=AgentRole.RESEARCHER)
        events: list[str | dict] = []

        async def capture(msg: str | dict) -> None:
            events.append(msg)

        await agent.execute(AgentTask(query="q"), on_status=capture)

        str_events = [e for e in events if isinstance(e, str)]
        assert len(str_events) >= 1
        assert any("thinking" in s for s in str_events)

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_dict_event_includes_agent_role(self, mock_llm: AsyncMock) -> None:
        """Dict events include the agent's role."""
        mock_llm.side_effect = [
            _tool_call("echo", {"text": "hi"}),
            _final_answer("Done"),
        ]
        agent = BaseAgent(tool_registry=_registry(EchoTool()), role=AgentRole.RESEARCHER)
        events: list[str | dict] = []

        async def capture(msg: str | dict) -> None:
            events.append(msg)

        await agent.execute(AgentTask(query="q"), on_status=capture)

        dict_events = [e for e in events if isinstance(e, dict)]
        assert dict_events[0]["agent"] == "researcher"


class TestModelNameThreading:
    """Tests for model_name parameter threading through run/execute/_run_loop."""

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_run_passes_model_name_to_loop(self, mock_llm: AsyncMock) -> None:
        """run(model_name=...) passes model= to get_structured_output."""
        mock_llm.return_value = _final_answer("Answer.")
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        await agent.run("query", model_name="qwen2.5:7b")

        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["model"] == "qwen2.5:7b"

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_execute_passes_model_name_to_loop(self, mock_llm: AsyncMock) -> None:
        """execute(model_name=...) passes model= to get_structured_output."""
        mock_llm.return_value = _final_answer("Answer.")
        agent = BaseAgent(tool_registry=_registry(EchoTool()), role=AgentRole.RESEARCHER)
        await agent.execute(AgentTask(query="q"), model_name="qwen2.5:7b")

        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["model"] == "qwen2.5:7b"

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_default_model_name_is_none(self, mock_llm: AsyncMock) -> None:
        """Without model_name, get_structured_output receives model=None (defaults to 70B)."""
        mock_llm.return_value = _final_answer("Answer.")
        agent = BaseAgent(tool_registry=_registry(EchoTool()), role=AgentRole.RESEARCHER)
        await agent.execute(AgentTask(query="q"))

        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["model"] is None


class TestTokenEstimation:
    """Tests for _estimate_tokens helper."""

    def test_estimate_tokens_basic(self) -> None:
        """Simple sanity check: known string length maps to approximate token count."""
        agent = BaseAgent(tool_registry=_registry())
        messages = [{"role": "user", "content": "a" * 400}]
        tokens = agent._estimate_tokens(messages)
        assert tokens == 100  # 400 chars // 4

    def test_estimate_tokens_empty(self) -> None:
        """Empty messages list returns 0."""
        agent = BaseAgent(tool_registry=_registry())
        assert agent._estimate_tokens([]) == 0

    def test_estimate_tokens_multiple(self) -> None:
        """Multiple messages sum their content lengths."""
        agent = BaseAgent(tool_registry=_registry())
        messages = [
            {"role": "system", "content": "a" * 200},
            {"role": "user", "content": "b" * 800},
        ]
        tokens = agent._estimate_tokens(messages)
        assert tokens == 250  # (200 + 800) // 4


class TestContextCompaction:
    """Tests for _compact_scratchpad and its integration with _build_messages."""

    def test_compaction_not_triggered_below_threshold(self) -> None:
        """Small scratchpad -> all observations preserved verbatim."""
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        scratchpad = [
            {
                "reasoning": "step 1",
                "tool_name": "echo",
                "tool_input": {"text": "hello"},
                "observation": "Echo: hello",
            },
            {
                "reasoning": "step 2",
                "tool_name": "echo",
                "tool_input": {"text": "world"},
                "observation": "Echo: world",
            },
        ]
        messages = agent._build_messages("query", scratchpad)
        all_content = " ".join(m["content"] for m in messages)
        assert "Echo: hello" in all_content
        assert "Echo: world" in all_content
        assert "[Summarized]" not in all_content

    @patch("code.shukketsu.config.COMPACTION_THRESHOLD_TOKENS", 10)
    def test_compaction_triggered_above_threshold(self) -> None:
        """Large scratchpad -> older entries truncated, 2 most recent preserved verbatim."""
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        scratchpad = [
            {
                "reasoning": f"step {i}",
                "tool_name": "echo",
                "tool_input": {"text": f"call{i}"},
                "observation": f"Result {i}: " + "x" * 500,
            }
            for i in range(5)
        ]
        messages = agent._build_messages("query", scratchpad)
        all_content = " ".join(m["content"] for m in messages)

        # The last 2 entries should be verbatim
        assert "Result 3: " + "x" * 500 in all_content
        assert "Result 4: " + "x" * 500 in all_content

        # Older entries should be summarized
        assert "[Summarized]" in all_content

    @patch("code.shukketsu.config.COMPACTION_THRESHOLD_TOKENS", 10)
    def test_compaction_preserves_tool_name(self) -> None:
        """Truncated entries still mention the tool name."""
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        scratchpad = [
            {
                "reasoning": f"step {i}",
                "tool_name": "rag_search",
                "tool_input": {"query": f"q{i}"},
                "observation": "Result: " + "x" * 500,
            }
            for i in range(5)
        ]
        messages = agent._build_messages("query", scratchpad)

        # Find observation messages for older entries (should mention tool name)
        observation_msgs = [m for m in messages if m["role"] == "user" and "[Summarized]" in m["content"]]
        assert len(observation_msgs) >= 1
        for msg in observation_msgs:
            assert "rag_search" in msg["content"]

    def test_compaction_idempotent(self) -> None:
        """Calling _build_messages twice with same scratchpad produces same result."""
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        scratchpad = [
            {
                "reasoning": f"step {i}",
                "tool_name": "echo",
                "tool_input": {"text": f"call{i}"},
                "observation": f"Result {i}: " + "x" * 200,
            }
            for i in range(5)
        ]
        messages1 = agent._build_messages("query", scratchpad)
        messages2 = agent._build_messages("query", scratchpad)
        assert messages1 == messages2

    @patch("code.shukketsu.config.COMPACTION_THRESHOLD_TOKENS", 10)
    def test_compaction_truncation_format(self) -> None:
        """Truncated observations use the format: [Summarized] [tool_name] first_100...last_100."""
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        long_obs = "START" + "m" * 300 + "END"
        scratchpad = [
            {
                "reasoning": "step",
                "tool_name": "rag_search",
                "tool_input": {"query": "q"},
                "observation": long_obs,
            },
            {
                "reasoning": "step",
                "tool_name": "echo",
                "tool_input": {"text": "recent1"},
                "observation": "recent observation 1" + "x" * 500,
            },
            {
                "reasoning": "step",
                "tool_name": "echo",
                "tool_input": {"text": "recent2"},
                "observation": "recent observation 2" + "y" * 500,
            },
        ]
        messages = agent._build_messages("query", scratchpad)
        observation_msgs = [m for m in messages if m["role"] == "user" and "[Summarized]" in m["content"]]

        assert len(observation_msgs) >= 1
        summarized = observation_msgs[0]["content"]
        assert "[Summarized]" in summarized
        assert "[rag_search]" in summarized
        assert "START" in summarized  # from first 100 chars
        assert "END" in summarized  # from last 100 chars
        assert "[truncated]" in summarized
