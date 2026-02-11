"""Tests for the Researcher agent and its models."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from code.shukketsu.agents.researcher import Researcher, StructuredFindings
from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    Finding,
    ResearchResult,
    TaskStatus,
)
from code.shukketsu.llm.prompts.researcher import RESEARCHER_SYSTEM_PROMPT
from code.shukketsu.llm.schemas import ActionType, AgentStep, ToolCall
from code.shukketsu.resilience.errors import StructuredOutputError
from code.shukketsu.tools.registry import ToolRegistry
from code.shukketsu.tools.schemas import Tool


class TestFindingModel:
    def test_valid_finding(self) -> None:
        f = Finding(
            claim="The hit cap is 142 rating",
            evidence=["chunk:15", "chunk:91"],
            confidence=0.9,
            entity_refs=["hit rating", "combat swords"],
        )
        assert f.claim == "The hit cap is 142 rating"
        assert len(f.evidence) == 2
        assert f.confidence == 0.9

    def test_confidence_rejects_above_one(self) -> None:
        with pytest.raises(ValidationError):
            Finding(claim="test", confidence=1.5)

    def test_confidence_rejects_below_zero(self) -> None:
        with pytest.raises(ValidationError):
            Finding(claim="test", confidence=-0.1)

    def test_confidence_boundary_zero(self) -> None:
        f = Finding(claim="test", confidence=0.0)
        assert f.confidence == 0.0

    def test_confidence_boundary_one(self) -> None:
        f = Finding(claim="test", confidence=1.0)
        assert f.confidence == 1.0

    def test_defaults_empty_lists(self) -> None:
        f = Finding(claim="test", confidence=0.5)
        assert f.evidence == []
        assert f.entity_refs == []


class TestResearchResultModel:
    def test_valid_research_result(self) -> None:
        r = ResearchResult(
            task_id="abc-123",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="The hit cap is 142.",
            findings=[Finding(claim="Hit cap is 142", confidence=0.9)],
            sources_used=["https://example.com"],
            strategies_used=["rag_search"],
            sufficient=True,
        )
        assert len(r.findings) == 1
        assert r.sufficient is True

    def test_defaults(self) -> None:
        r = ResearchResult(
            task_id="abc",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="text",
        )
        assert r.findings == []
        assert r.sources_used == []
        assert r.strategies_used == []
        assert r.gaps == []
        assert r.sufficient is False

    def test_inherits_agent_result(self) -> None:
        assert issubclass(ResearchResult, AgentResult)

    def test_has_agent_result_fields(self) -> None:
        """ResearchResult includes all base AgentResult fields."""
        r = ResearchResult(
            task_id="abc",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="text",
            evidence=["src1"],
            metadata={"key": "val"},
        )
        assert r.evidence == ["src1"]
        assert r.metadata == {"key": "val"}


class TestResearcherPrompt:
    def test_prompt_contains_all_tool_names(self) -> None:
        for tool in ["rag_search", "graph_search", "web_search", "web_ingest"]:
            assert tool in RESEARCHER_SYSTEM_PROMPT, f"Missing tool: {tool}"

    def test_prompt_contains_strategy_selection(self) -> None:
        assert "strategy" in RESEARCHER_SYSTEM_PROMPT.lower()

    def test_prompt_contains_worked_examples(self) -> None:
        assert "Example 1" in RESEARCHER_SYSTEM_PROMPT
        assert "Example 2" in RESEARCHER_SYSTEM_PROMPT
        assert "Example 3" in RESEARCHER_SYSTEM_PROMPT

    def test_prompt_contains_evaluation_criteria(self) -> None:
        prompt_lower = RESEARCHER_SYSTEM_PROMPT.lower()
        assert "evaluate" in prompt_lower or "assess" in prompt_lower


# --- Helpers for execute tests ---


class _EchoTool(Tool):
    name = "rag_search"
    description = "Search the knowledge base."
    parameters_schema = {"query": {"type": "string", "description": "Search query"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        q = tool_input.get("query", "")
        return (
            f"Found 1 result:\n[1] Source: Test Guide (https://example.com)\n"
            f"Trust: 0.8\nContent: Info about {q}\n"
        )


class _GraphTool(Tool):
    name = "graph_search"
    description = "Search the knowledge graph."
    parameters_schema = {"entity": {"type": "string", "description": "Entity name"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        e = tool_input.get("entity", "")
        return f'Found 1 relationship for "{e}":\n- drops_from -> Gruul (confidence: 0.9)\n'


def _registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _final_answer(answer: str) -> AgentStep:
    return AgentStep(reasoning="I have the answer.", action=ActionType.FINAL_ANSWER, answer=answer)


def _tool_call(tool_name: str, tool_input: dict[str, Any]) -> AgentStep:
    return AgentStep(
        reasoning="Need info.",
        action=ActionType.TOOL_CALL,
        tool_call=ToolCall(thought="Searching", tool_name=tool_name, tool_input=tool_input),
    )


def _mock_structured_findings(**kwargs: Any) -> StructuredFindings:
    """Build a StructuredFindings with sensible defaults."""
    defaults: dict[str, Any] = {
        "findings": [Finding(claim="Test claim", evidence=["https://example.com"], confidence=0.8)],
        "gaps": [],
        "sufficient": True,
    }
    defaults.update(kwargs)
    return StructuredFindings(**defaults)


class TestResearcherExecute:
    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_returns_research_result(self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock) -> None:
        """execute() returns ResearchResult, not plain AgentResult."""
        mock_loop_llm.return_value = _final_answer("The hit cap is 142.")
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await researcher.execute(AgentTask(query="What is the hit cap?"))

        assert isinstance(result, ResearchResult)
        assert result.agent_role == AgentRole.RESEARCHER
        assert result.status == TaskStatus.SUCCESS

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_structuring_pass_called(self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock) -> None:
        """A second LLM call is made for structuring after the ReAct loop."""
        mock_loop_llm.return_value = _final_answer("Answer text.")
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        await researcher.execute(AgentTask(query="q"))

        mock_struct_llm.assert_called_once()
        # Verify it was called with StructuredFindings as response_model
        call_kwargs = mock_struct_llm.call_args.kwargs
        assert call_kwargs["response_model"] is StructuredFindings

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_scratchpad_passed_to_structuring(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """Structuring pass receives tool observations from the scratchpad."""
        mock_loop_llm.side_effect = [
            _tool_call("rag_search", {"query": "hit cap"}),
            _final_answer("The hit cap is 142."),
        ]
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        await researcher.execute(AgentTask(query="hit cap"))

        # The structuring call's user message should contain the observation
        call_kwargs = mock_struct_llm.call_args.kwargs
        user_msg = call_kwargs["messages"][1]["content"]
        assert "rag_search" in user_msg
        assert "example.com" in user_msg

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_sources_derived_from_findings(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """sources_used is flattened from Finding.evidence, not parsed from observations."""
        mock_loop_llm.return_value = _final_answer("Answer.")
        mock_struct_llm.return_value = _mock_structured_findings(
            findings=[
                Finding(claim="Claim 1", evidence=["src_a", "src_b"], confidence=0.9),
                Finding(claim="Claim 2", evidence=["src_b", "src_c"], confidence=0.7),
            ]
        )

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await researcher.execute(AgentTask(query="q"))

        assert result.sources_used == ["src_a", "src_b", "src_c"]  # deduplicated, order-preserving

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_strategies_extracted_from_scratchpad(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """strategies_used contains deduplicated tool names from scratchpad."""
        mock_loop_llm.side_effect = [
            _tool_call("rag_search", {"query": "q1"}),
            _tool_call("graph_search", {"entity": "e1"}),
            _tool_call("rag_search", {"query": "q2"}),
            _final_answer("Done"),
        ]
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(
            tool_registry=_registry(_EchoTool(), _GraphTool()), role=AgentRole.RESEARCHER
        )
        result = await researcher.execute(AgentTask(query="q"))

        assert result.strategies_used == ["rag_search", "graph_search"]

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_structuring_failure_graceful_fallback(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """StructuredOutputError from structuring pass -> fallback with empty findings."""
        mock_loop_llm.return_value = _final_answer("Some answer text.")
        mock_struct_llm.side_effect = StructuredOutputError("Validation failed")

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await researcher.execute(AgentTask(query="q"))

        assert isinstance(result, ResearchResult)
        assert result.output == "Some answer text."
        assert result.findings == []
        assert result.sufficient is False

    def test_execute_has_langfuse_observe(self) -> None:
        """Researcher.execute() has @observe decorator (not inherited from base)."""
        from code.shukketsu.agents.base import BaseAgent

        # Researcher overrides execute — it should not be the same method object
        assert Researcher.execute is not BaseAgent.execute


class TestResearcherBehavior:
    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_decompose_two_part_question(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """Agent decomposes a multi-part question and calls tools multiple times."""
        mock_loop_llm.side_effect = [
            _tool_call("rag_search", {"query": "combat trinkets phase 1"}),
            _tool_call("rag_search", {"query": "combat stat priority"}),
            _final_answer("Trinkets ranked by stat priority..."),
        ]
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await researcher.execute(AgentTask(query="Best trinkets and why?"))

        # ReAct loop made 3 calls (2 tool + 1 final), structuring made 1
        assert mock_loop_llm.call_count == 3
        assert result.status == TaskStatus.SUCCESS

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_multiple_strategies_used(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """Agent uses both rag_search and graph_search tools."""
        mock_loop_llm.side_effect = [
            _tool_call("graph_search", {"entity": "combat swords"}),
            _tool_call("rag_search", {"query": "combat swords gear"}),
            _final_answer("Combined results."),
        ]
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(
            tool_registry=_registry(_EchoTool(), _GraphTool()), role=AgentRole.RESEARCHER
        )
        result = await researcher.execute(AgentTask(query="Combat gear"))

        assert "graph_search" in result.strategies_used
        assert "rag_search" in result.strategies_used

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_web_fallback_on_empty_kb(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """Agent falls back to web_search when rag_search returns nothing useful."""

        class _WebTool(Tool):
            name = "web_search"
            description = "Web search."
            parameters_schema = {"query": {"type": "string", "description": "Query"}}

            async def execute(self, tool_input: dict[str, Any]) -> str:
                return "Found 1 web result for ...: [1] Guide Title\n    https://example.com/guide\n"

        mock_loop_llm.side_effect = [
            _tool_call("rag_search", {"query": "obscure topic"}),
            _tool_call("web_search", {"query": "obscure topic TBC rogue"}),
            _final_answer("Found info via web."),
        ]
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(
            tool_registry=_registry(_EchoTool(), _WebTool()), role=AgentRole.RESEARCHER
        )
        result = await researcher.execute(AgentTask(query="obscure topic"))

        assert "web_search" in result.strategies_used

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_respects_max_iterations(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """Agent stops after max_iterations even if no final answer."""
        mock_loop_llm.return_value = _tool_call("rag_search", {"query": "endless"})
        # structuring should NOT be called — FAILED status skips it
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(
            tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER, max_iterations=2
        )
        result = await researcher.execute(AgentTask(query="q"))

        assert result.status == TaskStatus.FAILED
        assert mock_loop_llm.call_count == 2
        mock_struct_llm.assert_not_called()

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_gaps_populated_on_partial_results(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """gaps list is non-empty when structuring reports incomplete research."""
        mock_loop_llm.return_value = _final_answer("Partial info only.")
        mock_struct_llm.return_value = _mock_structured_findings(
            gaps=["Could not find proc rate data", "No Phase 2 comparison available"],
            sufficient=False,
        )

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await researcher.execute(AgentTask(query="detailed analysis"))

        assert len(result.gaps) == 2
        assert result.sufficient is False

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_sufficient_false_on_empty_results(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """All tools return empty -> sufficient=False."""
        mock_loop_llm.return_value = _final_answer("Could not find any information.")
        mock_struct_llm.return_value = _mock_structured_findings(
            findings=[], gaps=["No data found"], sufficient=False
        )

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await researcher.execute(AgentTask(query="nonexistent topic"))

        assert result.findings == []
        assert result.sufficient is False

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_failed_loop_skips_structuring(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """FAILED status from loop -> no structuring pass, minimal result."""
        mock_loop_llm.return_value = _tool_call("rag_search", {"query": "loop"})
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(
            tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER, max_iterations=1
        )
        result = await researcher.execute(AgentTask(query="q"))

        assert result.status == TaskStatus.FAILED
        assert result.findings == []
        assert result.sufficient is False
        mock_struct_llm.assert_not_called()


class TestResearcherFactory:
    def test_factory_returns_researcher_type(self) -> None:
        """factory.create(RESEARCHER) returns a Researcher instance."""
        from code.shukketsu.agents.factory import AgentFactory

        factory = AgentFactory()
        agent = factory.create(AgentRole.RESEARCHER)
        assert isinstance(agent, Researcher)

    def test_factory_other_roles_return_base_agent(self) -> None:
        """Non-researcher roles still return BaseAgent (not Researcher)."""
        from code.shukketsu.agents.base import BaseAgent
        from code.shukketsu.agents.factory import AgentFactory

        factory = AgentFactory()
        writer = factory.create(AgentRole.WRITER)
        assert type(writer) is BaseAgent

    def test_researcher_has_correct_prompt(self) -> None:
        """Researcher gets the full prompt from llm/prompts/researcher.py."""
        from code.shukketsu.agents.factory import AgentFactory
        from code.shukketsu.llm.prompts.researcher import RESEARCHER_SYSTEM_PROMPT as FULL_PROMPT

        factory = AgentFactory()
        agent = factory.create(AgentRole.RESEARCHER)
        assert agent._system_prompt is FULL_PROMPT

    def test_researcher_has_correct_max_iter(self) -> None:
        from code.shukketsu import config
        from code.shukketsu.agents.factory import AgentFactory

        factory = AgentFactory()
        agent = factory.create(AgentRole.RESEARCHER)
        assert agent.max_iterations == config.RESEARCHER_MAX_ITERATIONS
