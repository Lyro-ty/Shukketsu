"""Phase 2 gate evaluation runner.

Loads the eval dataset, runs each question through the live system,
judges results via LLM, and produces a PhaseGateReport.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from code.shukketsu.agents.factory import AgentFactory
from code.shukketsu.agents.tasks import AgentRole, AgentTask, ResearchTask
from code.shukketsu.db.connection import get_connection, init_db
from code.shukketsu.evals.judge import extract_claims, judge_domain_accuracy, judge_faithfulness
from code.shukketsu.evals.metrics import (
    EvalQuestionResult,
    PhaseGateReport,
    compute_faithfulness,
    compute_phase_gate,
    compute_trajectory_precision,
)
from code.shukketsu.ingest.embedder import get_embedder
from code.shukketsu.routing.router import classify_query
from code.shukketsu.tools.knowledge.search import RagSearchTool
from code.shukketsu.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_DATASET_PATH = Path(__file__).parent / "datasets" / "phase2_questions.json"


def _load_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the eval question dataset."""
    p = path or _DATASET_PATH
    with p.open() as f:
        data: list[dict[str, Any]] = json.load(f)
    return data


async def run_phase_gate(
    db_path: Path | None = None,
    dataset_path: Path | None = None,
) -> PhaseGateReport:
    """Run the full phase gate evaluation.

    Loads the dataset, creates agents, runs each question through
    the live system, judges results, and produces a PhaseGateReport.
    """
    from code.shukketsu import config
    from code.shukketsu.knowledge.manager import KnowledgeManager
    from code.shukketsu.tools.knowledge.graph_search import GraphSearchTool

    dataset = _load_dataset(dataset_path)

    conn = get_connection(db_path) if db_path else get_connection()
    init_db(conn)
    embedder = get_embedder()

    registry = ToolRegistry()
    registry.register(RagSearchTool(conn=conn, embed_fn=embedder.embed_query))
    registry.register(GraphSearchTool(conn=conn))

    factory = AgentFactory()
    km = KnowledgeManager(conn, config.WIKI_PATH)

    researcher = factory.create(AgentRole.RESEARCHER, tool_registry=registry)
    orchestrator = factory.create(
        AgentRole.ORCHESTRATOR,
        tool_registry=registry,
        factory=factory,
        knowledge_manager=km,
    )

    results: list[EvalQuestionResult] = []

    for q in dataset:
        try:
            result = await _eval_question(q, researcher, orchestrator)
            results.append(result)
            logger.info(
                "  %s  faith=%.2f  traj=%.2f  acc=%.2f  [%s]",
                q["id"],
                result.faithfulness,
                result.trajectory_precision,
                result.domain_accuracy,
                q["complexity"],
            )
        except Exception as exc:
            logger.warning("Question %s failed: %s", q["id"], exc)
            results.append(
                EvalQuestionResult(
                    question_id=q["id"],
                    faithfulness=0.0,
                    trajectory_precision=0.0,
                    domain_accuracy=0.0,
                    claims=[],
                    tool_calls=[],
                )
            )

    report = compute_phase_gate(results)

    # Soft wiki coverage check
    try:
        articles = km.list_articles()
        coverage: dict[str, int] = {}
        for a in articles:
            coverage[a.spec] = coverage.get(a.spec, 0) + 1
        report = report.model_copy(update={"wiki_coverage": coverage})
    except Exception:
        logger.warning("Failed to compute wiki coverage", exc_info=True)

    return report


async def _eval_question(
    q: dict[str, Any],
    researcher: Any,
    orchestrator: Any,
) -> EvalQuestionResult:
    """Evaluate a single question."""
    complexity = q["complexity"]
    question = q["question"]

    # Route and execute — dataset uses lowercase complexity values
    if complexity == "trivial":
        decision = await classify_query(question)
        if decision.direct_answer:
            answer = decision.direct_answer
            tool_calls: list[str] = []
            evidence: list[str] = []
        else:
            result = await researcher.execute(ResearchTask(query=question))
            answer = result.output
            tool_calls = [t.tool_name for t in result.trajectory]
            evidence = result.evidence
    elif complexity == "moderate":
        result = await researcher.execute(ResearchTask(query=question))
        answer = result.output
        tool_calls = [t.tool_name for t in result.trajectory]
        evidence = result.evidence
    else:
        result = await orchestrator.execute(AgentTask(query=question))
        answer = result.output
        tool_calls = [t.tool_name for t in result.trajectory]
        evidence = result.evidence

    # Judge
    claims_text = await extract_claims(answer)
    claim_judgments = await judge_faithfulness(claims_text, evidence) if claims_text else []
    accuracy = await judge_domain_accuracy(answer, q["ground_truth"], q["key_facts"])

    # Score
    faithfulness = compute_faithfulness(claim_judgments)
    traj_precision = compute_trajectory_precision(tool_calls, q["expected_tools"])

    return EvalQuestionResult(
        question_id=q["id"],
        faithfulness=faithfulness,
        trajectory_precision=traj_precision,
        domain_accuracy=accuracy,
        claims=claim_judgments,
        tool_calls=tool_calls,
    )


def _print_report(report: PhaseGateReport) -> None:
    """Print a human-readable report to stdout."""
    from code.shukketsu.evals.metrics import (
        ACCURACY_THRESHOLD,
        FAITHFULNESS_THRESHOLD,
        TRAJECTORY_THRESHOLD,
    )

    print("\nPhase 2 Gate Evaluation")
    print("=" * 50)
    print(f"Questions evaluated: {len(report.question_results)}")
    print()

    def _status(val: float, threshold: float) -> str:
        return "PASS" if val >= threshold else "FAIL"

    print(
        f"RAG Faithfulness:        {report.avg_faithfulness:.2f}"
        f"  (threshold: {FAITHFULNESS_THRESHOLD})"
        f"  {_status(report.avg_faithfulness, FAITHFULNESS_THRESHOLD)}"
    )
    print(
        f"Trajectory Precision:    {report.avg_trajectory_precision:.2f}"
        f"  (threshold: {TRAJECTORY_THRESHOLD})"
        f"  {_status(report.avg_trajectory_precision, TRAJECTORY_THRESHOLD)}"
    )
    print(
        f"Domain Accuracy:         {report.avg_domain_accuracy:.2f}"
        f"  (threshold: {ACCURACY_THRESHOLD})"
        f"  {_status(report.avg_domain_accuracy, ACCURACY_THRESHOLD)}"
    )
    print()

    if report.wiki_coverage:
        print("Wiki Coverage (informational):")
        for spec, count in sorted(report.wiki_coverage.items()):
            print(f"  {spec:15s} {count} article(s)")
        print()

    print("Per-question breakdown:")
    for qr in report.question_results:
        print(
            f"  {qr.question_id}  faith={qr.faithfulness:.2f}"
            f"  traj={qr.trajectory_precision:.2f}"
            f"  acc={qr.domain_accuracy:.2f}"
        )

    print()
    status = "PASSED" if report.passed else "FAILED"
    print(f"Result: {status}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = asyncio.run(run_phase_gate())
    _print_report(report)
    sys.exit(0 if report.passed else 1)
