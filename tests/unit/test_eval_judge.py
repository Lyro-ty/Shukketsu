"""Tests for eval judge module — all LLM calls mocked."""

from unittest.mock import AsyncMock, patch

import pytest

from code.shukketsu.evals.judge import (
    _AccuracyScore,
    _ClaimExtraction,
    _FaithfulnessVerdict,
    _SingleVerdict,
    extract_claims,
    judge_domain_accuracy,
    judge_faithfulness,
)
from code.shukketsu.evals.metrics import ClaimFaithfulness, FaithfulnessJudgment


class TestExtractClaims:
    @patch("code.shukketsu.evals.judge.get_structured_output")
    async def test_returns_claim_list(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _ClaimExtraction(claims=["Claim A", "Claim B"])
        result = await extract_claims("Some answer text.")
        assert result == ["Claim A", "Claim B"]

    @patch("code.shukketsu.evals.judge.get_structured_output")
    async def test_error_returns_empty_list(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = Exception("LLM timeout")
        result = await extract_claims("Some answer.")
        assert result == []

    @patch("code.shukketsu.evals.judge.get_structured_output")
    async def test_calls_with_reasoning_backend(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _ClaimExtraction(claims=["X"])
        await extract_claims("answer")
        call_kwargs = mock_llm.call_args.kwargs
        from code.shukketsu.llm.structured import ModelBackend

        assert call_kwargs["backend"] == ModelBackend.REASONING


class TestJudgeFaithfulness:
    @patch("code.shukketsu.evals.judge.get_structured_output")
    async def test_returns_claim_faithfulness_list(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _FaithfulnessVerdict(
            judgments=[
                _SingleVerdict(
                    claim="Claim A",
                    judgment=FaithfulnessJudgment.SUPPORTED,
                    evidence_snippet="evidence here",
                ),
            ]
        )
        result = await judge_faithfulness(["Claim A"], ["evidence here"])
        assert len(result) == 1
        assert isinstance(result[0], ClaimFaithfulness)
        assert result[0].judgment == FaithfulnessJudgment.SUPPORTED

    @patch("code.shukketsu.evals.judge.get_structured_output")
    async def test_error_returns_all_unclear(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = Exception("Timeout")
        result = await judge_faithfulness(["Claim A", "Claim B"], ["evidence"])
        assert len(result) == 2
        assert all(c.judgment == FaithfulnessJudgment.UNCLEAR for c in result)


class TestJudgeDomainAccuracy:
    @patch("code.shukketsu.evals.judge.get_structured_output")
    async def test_returns_score(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _AccuracyScore(score=0.85, reasoning="Good")
        result = await judge_domain_accuracy("answer", "truth", ["fact1"])
        assert result == pytest.approx(0.85)

    @patch("code.shukketsu.evals.judge.get_structured_output")
    async def test_error_returns_zero(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = Exception("Crash")
        result = await judge_domain_accuracy("answer", "truth", ["fact1"])
        assert result == 0.0
