"""Tests for the Researcher agent and its models."""

import pytest
from pydantic import ValidationError

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
