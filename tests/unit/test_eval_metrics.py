"""Tests for eval metrics — pure scoring functions."""

import pytest

from code.shukketsu.evals.metrics import (
    ACCURACY_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    TRAJECTORY_THRESHOLD,
    ClaimFaithfulness,
    EvalQuestionResult,
    FaithfulnessJudgment,
    PhaseGateReport,
    TrajectoryScore,
    compute_faithfulness,
    compute_phase_gate,
    compute_trajectory_precision,
)


class TestComputeFaithfulness:
    def test_all_supported(self) -> None:
        claims = [
            ClaimFaithfulness(claim="A", judgment=FaithfulnessJudgment.SUPPORTED),
            ClaimFaithfulness(claim="B", judgment=FaithfulnessJudgment.SUPPORTED),
        ]
        assert compute_faithfulness(claims) == 1.0

    def test_all_not_supported(self) -> None:
        claims = [
            ClaimFaithfulness(claim="A", judgment=FaithfulnessJudgment.NOT_SUPPORTED),
            ClaimFaithfulness(claim="B", judgment=FaithfulnessJudgment.NOT_SUPPORTED),
        ]
        assert compute_faithfulness(claims) == 0.0

    def test_mixed_claims(self) -> None:
        claims = [
            ClaimFaithfulness(claim="A", judgment=FaithfulnessJudgment.SUPPORTED),
            ClaimFaithfulness(claim="B", judgment=FaithfulnessJudgment.NOT_SUPPORTED),
            ClaimFaithfulness(claim="C", judgment=FaithfulnessJudgment.SUPPORTED),
        ]
        assert compute_faithfulness(claims) == pytest.approx(2 / 3)

    def test_all_unclear(self) -> None:
        claims = [
            ClaimFaithfulness(claim="A", judgment=FaithfulnessJudgment.UNCLEAR),
            ClaimFaithfulness(claim="B", judgment=FaithfulnessJudgment.UNCLEAR),
        ]
        assert compute_faithfulness(claims) == 1.0

    def test_unclear_excluded_from_denominator(self) -> None:
        claims = [
            ClaimFaithfulness(claim="A", judgment=FaithfulnessJudgment.SUPPORTED),
            ClaimFaithfulness(claim="B", judgment=FaithfulnessJudgment.UNCLEAR),
            ClaimFaithfulness(claim="C", judgment=FaithfulnessJudgment.NOT_SUPPORTED),
        ]
        assert compute_faithfulness(claims) == pytest.approx(0.5)

    def test_empty_claims(self) -> None:
        assert compute_faithfulness([]) == 1.0

    def test_evidence_snippet_optional(self) -> None:
        c = ClaimFaithfulness(
            claim="X",
            judgment=FaithfulnessJudgment.SUPPORTED,
            evidence_snippet="some evidence",
        )
        assert c.evidence_snippet == "some evidence"


class TestComputeTrajectoryPrecision:
    def test_exact_match(self) -> None:
        actual = ["rag_search", "graph_search"]
        expected = ["rag_search", "graph_search"]
        assert compute_trajectory_precision(actual, expected) == 1.0

    def test_no_expected_tools(self) -> None:
        assert compute_trajectory_precision([], []) == 1.0

    def test_wasted_consecutive_calls(self) -> None:
        actual = ["rag_search", "rag_search", "rag_search"]
        expected = ["rag_search"]
        assert compute_trajectory_precision(actual, expected) == pytest.approx(1 / 3)

    def test_unexpected_tool_lowers_precision(self) -> None:
        actual = ["rag_search", "web_search"]
        expected = ["rag_search"]
        assert compute_trajectory_precision(actual, expected) == pytest.approx(0.5)

    def test_empty_actual_returns_one(self) -> None:
        assert compute_trajectory_precision([], ["rag_search"]) == 1.0

    def test_non_consecutive_repeats_not_penalized(self) -> None:
        actual = ["rag_search", "graph_search", "rag_search"]
        expected = ["rag_search", "graph_search"]
        assert compute_trajectory_precision(actual, expected) == 1.0


class TestComputePhaseGate:
    def _make_result(self, *, faith: float = 0.9, traj: float = 0.8, acc: float = 0.8) -> EvalQuestionResult:
        return EvalQuestionResult(
            question_id="q01",
            faithfulness=faith,
            trajectory_precision=traj,
            domain_accuracy=acc,
            claims=[],
            tool_calls=[],
        )

    def test_all_passing(self) -> None:
        results = [self._make_result() for _ in range(5)]
        report = compute_phase_gate(results)
        assert report.passed is True
        assert report.avg_faithfulness == pytest.approx(0.9)
        assert report.avg_trajectory_precision == pytest.approx(0.8)
        assert report.avg_domain_accuracy == pytest.approx(0.8)

    def test_one_metric_below_threshold(self) -> None:
        results = [self._make_result(faith=0.5) for _ in range(5)]
        report = compute_phase_gate(results)
        assert report.passed is False
        assert report.avg_faithfulness == pytest.approx(0.5)

    def test_empty_results(self) -> None:
        report = compute_phase_gate([])
        assert report.passed is False
        assert report.avg_faithfulness == 0.0
        assert report.avg_trajectory_precision == 0.0
        assert report.avg_domain_accuracy == 0.0

    def test_report_includes_question_results(self) -> None:
        results = [self._make_result()]
        report = compute_phase_gate(results)
        assert len(report.question_results) == 1

    def test_borderline_passes(self) -> None:
        results = [
            self._make_result(
                faith=FAITHFULNESS_THRESHOLD,
                traj=TRAJECTORY_THRESHOLD,
                acc=ACCURACY_THRESHOLD,
            )
        ]
        report = compute_phase_gate(results)
        assert report.passed is True


class TestModels:
    def test_trajectory_score_fields(self) -> None:
        ts = TrajectoryScore(precision=0.8, expected_tools_hit=3, wasted_calls=1)
        assert ts.precision == 0.8
        assert ts.expected_tools_hit == 3
        assert ts.wasted_calls == 1

    def test_phase_gate_report_wiki_coverage(self) -> None:
        report = PhaseGateReport(
            avg_faithfulness=0.9,
            avg_trajectory_precision=0.8,
            avg_domain_accuracy=0.8,
            passed=True,
            wiki_coverage={"combat": 3, "assassination": 1},
            question_results=[],
        )
        assert report.wiki_coverage["combat"] == 3
