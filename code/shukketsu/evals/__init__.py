"""Evaluation pipeline for Shukketsu.

Provides metrics, LLM-as-judge scoring, Langfuse dataset management,
eval runner, and fine-tuning export.
"""

from code.shukketsu.evals.dataset import EvalDatasetManager
from code.shukketsu.evals.export import TrainingExporter
from code.shukketsu.evals.judge import (
    extract_claims,
    extract_dps_from_answer,
    judge_answer_relevancy,
    judge_domain_accuracy,
    judge_faithfulness,
    judge_sim_accuracy,
)
from code.shukketsu.evals.metrics import (
    EvalQuestionResult,
    PhaseGateReport,
    compute_answer_relevancy,
    compute_faithfulness,
    compute_phase_gate,
    compute_trajectory_precision,
)
from code.shukketsu.evals.runner import EvalRunner

__all__ = [
    "EvalDatasetManager",
    "EvalRunner",
    "EvalQuestionResult",
    "PhaseGateReport",
    "TrainingExporter",
    "compute_answer_relevancy",
    "compute_faithfulness",
    "compute_phase_gate",
    "compute_trajectory_precision",
    "extract_claims",
    "extract_dps_from_answer",
    "judge_answer_relevancy",
    "judge_domain_accuracy",
    "judge_faithfulness",
    "judge_sim_accuracy",
]
