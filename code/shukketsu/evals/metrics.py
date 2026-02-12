"""Pure scoring functions for phase gate evaluation.

All functions are deterministic and unit-testable — no LLM calls.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

# --- Thresholds ---

FAITHFULNESS_THRESHOLD = 0.8
TRAJECTORY_THRESHOLD = 0.7
ACCURACY_THRESHOLD = 0.7


# --- Models ---


class FaithfulnessJudgment(StrEnum):
    """Judgment of whether a claim is supported by evidence."""

    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    UNCLEAR = "unclear"


class ClaimFaithfulness(BaseModel):
    """Result of judging one claim against evidence."""

    claim: str
    judgment: FaithfulnessJudgment
    evidence_snippet: str | None = None


class TrajectoryScore(BaseModel):
    """Detailed trajectory analysis."""

    precision: float
    expected_tools_hit: int
    wasted_calls: int


class EvalQuestionResult(BaseModel):
    """Per-question evaluation scores."""

    question_id: str
    faithfulness: float
    trajectory_precision: float
    domain_accuracy: float
    claims: list[ClaimFaithfulness]
    tool_calls: list[str]


class PhaseGateReport(BaseModel):
    """Aggregate evaluation report for the phase gate."""

    avg_faithfulness: float
    avg_trajectory_precision: float
    avg_domain_accuracy: float
    passed: bool
    wiki_coverage: dict[str, int] = Field(default_factory=dict)
    question_results: list[EvalQuestionResult] = Field(default_factory=list)


# --- Scoring Functions ---


def compute_faithfulness(claims: list[ClaimFaithfulness]) -> float:
    """Compute faithfulness score from claim judgments.

    SUPPORTED / (SUPPORTED + NOT_SUPPORTED). UNCLEAR claims excluded
    from the denominator. Returns 1.0 if no scoreable claims.
    """
    supported = sum(1 for c in claims if c.judgment == FaithfulnessJudgment.SUPPORTED)
    not_supported = sum(1 for c in claims if c.judgment == FaithfulnessJudgment.NOT_SUPPORTED)

    denominator = supported + not_supported
    if denominator == 0:
        return 1.0

    return supported / denominator


def compute_trajectory_precision(
    actual_calls: list[str],
    expected_tools: list[str],
) -> float:
    """Compute trajectory precision.

    Fraction of actual calls that match an expected tool. Repeated
    identical consecutive calls after the first count as wasted
    (precision penalty). Returns 1.0 if no tool calls were made.
    """
    if not actual_calls:
        return 1.0

    expected_set = set(expected_tools)
    useful = 0
    total = len(actual_calls)

    for i, call in enumerate(actual_calls):
        # Consecutive duplicate = wasted
        if i > 0 and call == actual_calls[i - 1]:
            continue  # wasted — not counted as useful
        if call in expected_set:
            useful += 1

    return useful / total


def compute_phase_gate(
    results: list[EvalQuestionResult],
) -> PhaseGateReport:
    """Aggregate per-question scores into a PhaseGateReport.

    Averages each metric across all questions. Sets passed=True only
    if all three averages meet or exceed their thresholds.
    """
    if not results:
        return PhaseGateReport(
            avg_faithfulness=0.0,
            avg_trajectory_precision=0.0,
            avg_domain_accuracy=0.0,
            passed=False,
            question_results=results,
        )

    n = len(results)
    avg_faith = sum(r.faithfulness for r in results) / n
    avg_traj = sum(r.trajectory_precision for r in results) / n
    avg_acc = sum(r.domain_accuracy for r in results) / n

    passed = avg_faith >= FAITHFULNESS_THRESHOLD and avg_traj >= TRAJECTORY_THRESHOLD and avg_acc >= ACCURACY_THRESHOLD

    return PhaseGateReport(
        avg_faithfulness=avg_faith,
        avg_trajectory_precision=avg_traj,
        avg_domain_accuracy=avg_acc,
        passed=passed,
        question_results=results,
    )
