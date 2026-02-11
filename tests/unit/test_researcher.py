"""Tests for the Researcher agent and its models."""

import pytest
from pydantic import ValidationError

from code.shukketsu.llm.prompts.researcher import RESEARCHER_SYSTEM_PROMPT

from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    Finding,
    ResearchResult,
    TaskStatus,
)


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
