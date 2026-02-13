"""Tests for eval judge module — all LLM calls mocked."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code.shukketsu.evals.judge import (
    _AccuracyScore,
    _ClaimExtraction,
    _FaithfulnessVerdict,
    _SingleVerdict,
    extract_claims,
    extract_dps_from_answer,
    judge_answer_relevancy,
    judge_domain_accuracy,
    judge_faithfulness,
    judge_sim_accuracy,
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


class TestJudgeAnswerRelevancy:
    @patch("code.shukketsu.evals.judge.get_structured_output", new_callable=AsyncMock)
    async def test_returns_score(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = MagicMock(score=0.85, reasoning="Good")
        score = await judge_answer_relevancy("What is hit cap?", "Hit cap is 9%.")
        assert score == 0.85

    @patch("code.shukketsu.evals.judge.get_structured_output", new_callable=AsyncMock)
    async def test_returns_zero_on_failure(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = Exception("LLM down")
        score = await judge_answer_relevancy("What?", "Answer")
        assert score == 0.0


class TestExtractDpsFromAnswer:
    def test_integer_dps(self) -> None:
        assert extract_dps_from_answer("The result is 1234 DPS on Patchwerk.") == 1234.0

    def test_comma_separated_dps(self) -> None:
        assert extract_dps_from_answer("Expected output: 1,234.5 DPS") == 1234.5

    def test_lowercase_dps(self) -> None:
        assert extract_dps_from_answer("about 950 dps") == 950.0

    def test_no_match_returns_none(self) -> None:
        assert extract_dps_from_answer("The hit cap is 9%.") is None


class TestJudgeSimAccuracy:
    def test_within_range(self) -> None:
        assert judge_sim_accuracy(1150.0, 1100.0, 1200.0) == 1.0

    def test_at_boundary(self) -> None:
        assert judge_sim_accuracy(1100.0, 1100.0, 1200.0) == 1.0

    def test_outside_range_falloff(self) -> None:
        # Range is 100, tolerance = 100 * 5% = 5. At distance 2.5, score = 0.5
        score = judge_sim_accuracy(1097.5, 1100.0, 1200.0)
        assert score == pytest.approx(0.5)

    def test_far_outside_returns_zero(self) -> None:
        score = judge_sim_accuracy(900.0, 1100.0, 1200.0)
        assert score == 0.0
