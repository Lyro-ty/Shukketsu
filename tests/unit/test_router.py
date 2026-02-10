"""Tests for the multi-model query router."""

from unittest.mock import AsyncMock, patch

import pytest

from code.shukketsu.resilience.errors import LLMUnavailableError, StructuredOutputError
from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity


def _make_decision(
    complexity: TaskComplexity = TaskComplexity.COMPLEX,
    category: TaskCategory = TaskCategory.CONVERSATION,
    needs_tools: bool = True,
    suggested_agent: str = "general",
    direct_answer: str | None = None,
) -> RoutingDecision:
    """Helper to build a RoutingDecision with defaults."""
    return RoutingDecision(
        complexity=complexity,
        category=category,
        needs_tools=needs_tools,
        suggested_agent=suggested_agent,
        direct_answer=direct_answer,
    )


class TestClassifyQuery:
    """Tests for the classify_query router function."""

    @pytest.mark.asyncio
    async def test_classify_trivial_returns_direct_answer(self) -> None:
        """Trivial queries should return a RoutingDecision with direct_answer filled."""
        trivial = _make_decision(
            complexity=TaskComplexity.TRIVIAL,
            category=TaskCategory.CONVERSATION,
            needs_tools=False,
            direct_answer="Sinister Strike costs 40 energy.",
        )

        with patch(
            "code.shukketsu.routing.router.get_structured_output",
            new_callable=AsyncMock,
            return_value=trivial,
        ):
            from code.shukketsu.routing.router import classify_query

            result = await classify_query("How much energy does Sinister Strike cost?")

        assert result.complexity == TaskComplexity.TRIVIAL
        assert result.direct_answer == "Sinister Strike costs 40 energy."
        assert result.needs_tools is False

    @pytest.mark.asyncio
    async def test_classify_complex_returns_no_direct_answer(self) -> None:
        """Complex queries should return a RoutingDecision without direct_answer."""
        complex_decision = _make_decision(
            complexity=TaskComplexity.COMPLEX,
            category=TaskCategory.ANALYSIS,
            needs_tools=True,
        )

        with patch(
            "code.shukketsu.routing.router.get_structured_output",
            new_callable=AsyncMock,
            return_value=complex_decision,
        ):
            from code.shukketsu.routing.router import classify_query

            result = await classify_query("Compare combat vs assassination DPS in P3 BiS gear")

        assert result.complexity == TaskComplexity.COMPLEX
        assert result.direct_answer is None
        assert result.needs_tools is True

    @pytest.mark.asyncio
    async def test_classify_fallback_on_ollama_unavailable(self) -> None:
        """When Ollama is down, classify_query should return a safe fallback."""
        with patch(
            "code.shukketsu.routing.router.get_structured_output",
            new_callable=AsyncMock,
            side_effect=LLMUnavailableError("Cannot connect to ollama server."),
        ):
            from code.shukketsu.routing.router import classify_query

            result = await classify_query("What is the hit cap?")

        assert result.complexity == TaskComplexity.COMPLEX
        assert result.direct_answer is None
        assert result.needs_tools is True
        assert result.suggested_agent == "general"

    @pytest.mark.asyncio
    async def test_classify_fallback_on_structured_output_error(self) -> None:
        """When Qwen returns invalid output, classify_query should return a safe fallback."""
        with patch(
            "code.shukketsu.routing.router.get_structured_output",
            new_callable=AsyncMock,
            side_effect=StructuredOutputError("Failed to get valid RoutingDecision after 3 retries."),
        ):
            from code.shukketsu.routing.router import classify_query

            result = await classify_query("Tell me about sword spec")

        assert result.complexity == TaskComplexity.COMPLEX
        assert result.direct_answer is None
        assert result.needs_tools is True

    @pytest.mark.asyncio
    async def test_classify_passes_correct_backend_and_params(self) -> None:
        """classify_query should call get_structured_output with Ollama backend, temp=0.0, max_tokens=512."""
        mock_output = AsyncMock(return_value=_make_decision())

        with patch(
            "code.shukketsu.routing.router.get_structured_output",
            mock_output,
        ):
            from code.shukketsu.routing.router import classify_query

            await classify_query("test query")

        mock_output.assert_called_once()
        call_kwargs = mock_output.call_args
        assert call_kwargs.kwargs["backend"].value == "ollama"
        assert call_kwargs.kwargs["temperature"] == 0.0
        assert call_kwargs.kwargs["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_classify_fallback_on_circuit_open(self) -> None:
        """When Ollama circuit breaker is open, classify_query should return safe fallback."""
        from code.shukketsu.resilience.errors import CircuitOpenError

        with patch(
            "code.shukketsu.routing.router.ollama_router_breaker",
        ) as mock_breaker:
            mock_breaker.call = AsyncMock(side_effect=CircuitOpenError("ollama_router"))

            from code.shukketsu.routing.router import classify_query

            result = await classify_query("What is the hit cap?")

        assert result.complexity == TaskComplexity.COMPLEX
        assert result.direct_answer is None
        assert result.needs_tools is True

    @pytest.mark.asyncio
    async def test_classify_prompt_includes_domain_context(self) -> None:
        """The system prompt sent to Qwen should include WoW TBC Rogue domain context."""
        mock_output = AsyncMock(return_value=_make_decision())

        with patch(
            "code.shukketsu.routing.router.get_structured_output",
            mock_output,
        ):
            from code.shukketsu.routing.router import classify_query

            await classify_query("What is the hit cap?")

        call_args = mock_output.call_args
        messages = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("messages", [])
        system_msg = next((m for m in messages if m["role"] == "system"), None)
        assert system_msg is not None
        content = system_msg["content"].lower()
        assert "rogue" in content
        assert "tbc" in content or "burning crusade" in content
