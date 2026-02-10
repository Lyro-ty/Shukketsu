"""Tests for LLM structured output Pydantic schemas."""

import pytest
from pydantic import ValidationError


class TestToolCall:
    """Tests for the ToolCall schema."""

    def test_valid_tool_call(self) -> None:
        """ToolCall should accept valid thought, tool_name, tool_input."""
        from code.shukketsu.llm.schemas import ToolCall

        tc = ToolCall(
            thought="Need to search the knowledge base",
            tool_name="rag_search",
            tool_input={"query": "hit cap for combat rogues"},
        )
        assert tc.thought == "Need to search the knowledge base"
        assert tc.tool_name == "rag_search"
        assert tc.tool_input == {"query": "hit cap for combat rogues"}

    def test_missing_required_fields(self) -> None:
        """ToolCall should reject missing required fields."""
        from code.shukketsu.llm.schemas import ToolCall

        with pytest.raises(ValidationError):
            ToolCall(thought="test")  # type: ignore[call-arg]


class TestAgentStep:
    """Tests for the AgentStep schema with conditional validation."""

    def test_valid_tool_call_action(self) -> None:
        """AgentStep with action=tool_call and populated tool_call should pass."""
        from code.shukketsu.llm.schemas import ActionType, AgentStep, ToolCall

        step = AgentStep(
            reasoning="I need to look this up",
            action=ActionType.TOOL_CALL,
            tool_call=ToolCall(
                thought="Search for hit cap info",
                tool_name="rag_search",
                tool_input={"query": "hit cap"},
            ),
        )
        assert step.action == ActionType.TOOL_CALL
        assert step.tool_call is not None
        assert step.answer is None

    def test_valid_final_answer_action(self) -> None:
        """AgentStep with action=final_answer and populated answer should pass."""
        from code.shukketsu.llm.schemas import ActionType, AgentStep

        step = AgentStep(
            reasoning="I have enough information to answer",
            action=ActionType.FINAL_ANSWER,
            answer="The hit cap for combat rogues is 9% (142 hit rating).",
        )
        assert step.action == ActionType.FINAL_ANSWER
        assert step.answer is not None
        assert step.tool_call is None

    def test_tool_call_action_requires_tool_call(self) -> None:
        """AgentStep should reject action=tool_call when tool_call is None."""
        from code.shukketsu.llm.schemas import ActionType, AgentStep

        with pytest.raises(ValidationError, match="tool_call required"):
            AgentStep(
                reasoning="I need to search",
                action=ActionType.TOOL_CALL,
                tool_call=None,
            )

    def test_final_answer_action_requires_answer(self) -> None:
        """AgentStep should reject action=final_answer when answer is None."""
        from code.shukketsu.llm.schemas import ActionType, AgentStep

        with pytest.raises(ValidationError, match="answer required"):
            AgentStep(
                reasoning="I know the answer",
                action=ActionType.FINAL_ANSWER,
                answer=None,
            )

    def test_missing_required_fields(self) -> None:
        """AgentStep should reject missing reasoning or action."""
        from code.shukketsu.llm.schemas import AgentStep

        with pytest.raises(ValidationError):
            AgentStep()  # type: ignore[call-arg]


class TestActionTypeSerialization:
    """Tests for ActionType JSON round-trip behavior."""

    def test_serializes_to_string(self) -> None:
        """ActionType values should serialize as plain strings in model_dump."""
        from code.shukketsu.llm.schemas import ActionType, AgentStep

        step = AgentStep(
            reasoning="Done",
            action=ActionType.FINAL_ANSWER,
            answer="42",
        )
        dumped = step.model_dump()
        assert dumped["action"] == "final_answer"
        assert isinstance(dumped["action"], str)

    def test_round_trip_json(self) -> None:
        """AgentStep should survive JSON serialization and deserialization."""
        from code.shukketsu.llm.schemas import ActionType, AgentStep, ToolCall

        original = AgentStep(
            reasoning="Search needed",
            action=ActionType.TOOL_CALL,
            tool_call=ToolCall(
                thought="look up",
                tool_name="rag_search",
                tool_input={"q": "test"},
            ),
        )
        json_str = original.model_dump_json()
        restored = AgentStep.model_validate_json(json_str)
        assert restored == original


class TestRoutingDecisionSchema:
    """Tests for the RoutingDecision schema."""

    def test_direct_answer_defaults_to_none(self) -> None:
        """RoutingDecision.direct_answer should default to None when omitted."""
        from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity

        decision = RoutingDecision(
            complexity=TaskComplexity.MODERATE,
            category=TaskCategory.RETRIEVAL,
            needs_tools=True,
            suggested_agent="general",
        )
        assert decision.direct_answer is None

    def test_direct_answer_accepts_string(self) -> None:
        """RoutingDecision.direct_answer should accept a string value."""
        from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity

        decision = RoutingDecision(
            complexity=TaskComplexity.TRIVIAL,
            category=TaskCategory.CONVERSATION,
            needs_tools=False,
            suggested_agent="general",
            direct_answer="Sinister Strike costs 40 energy.",
        )
        assert decision.direct_answer == "Sinister Strike costs 40 energy."
