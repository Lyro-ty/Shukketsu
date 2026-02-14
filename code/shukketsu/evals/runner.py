"""Eval runner — executes dataset questions and records scores to Langfuse.

Runs each question through the same routing path as real chat:
Qwen 4B classifies complexity → appropriate agent handles the query →
LLM judges score the result → scores recorded to Langfuse.
"""

import logging
import sqlite3
from collections.abc import Awaitable, Callable
from typing import Any

from langfuse import Langfuse
from langfuse.model import CreateDatasetRunItemRequest

from code.shukketsu import config
from code.shukketsu.agents.base import BaseAgent
from code.shukketsu.agents.tasks import AgentTask, ResearchTask
from code.shukketsu.evals.dataset import EvalDatasetManager
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
from code.shukketsu.routing.models import TaskCategory, TaskComplexity
from code.shukketsu.routing.router import classify_query

logger = logging.getLogger(__name__)


class EvalRunner:
    """Executes eval dataset against the live agent system."""

    def __init__(
        self,
        dataset_manager: EvalDatasetManager,
        langfuse_client: Langfuse,
    ) -> None:
        self._dm = dataset_manager
        self._client = langfuse_client
        self._agents: tuple[BaseAgent, BaseAgent, BaseAgent] | None = None
        self._conn: sqlite3.Connection | None = None

    def _get_agents(self) -> tuple[BaseAgent, BaseAgent, BaseAgent]:
        """Lazy-init agents (same pattern as chat handler)."""
        if self._agents is None:
            from code.shukketsu.agents.factory import AgentFactory
            from code.shukketsu.agents.tasks import AgentRole
            from code.shukketsu.db.connection import get_connection, init_db
            from code.shukketsu.ingest.embedder import get_embedder
            from code.shukketsu.ingest.pipeline import IngestPipeline
            from code.shukketsu.knowledge.manager import KnowledgeManager
            from code.shukketsu.scraping.fetcher import WebFetcher
            from code.shukketsu.scraping.rate_limiter import RateLimiter
            from code.shukketsu.scraping.robots import RobotsChecker
            from code.shukketsu.tools.knowledge.graph_search import GraphSearchTool
            from code.shukketsu.tools.knowledge.search import RagSearchTool
            from code.shukketsu.tools.registry import ToolRegistry
            from code.shukketsu.tools.research.web_ingest import WebIngestTool
            from code.shukketsu.tools.research.web_search import WebSearchTool

            conn = get_connection()
            self._conn = conn
            init_db(conn)
            embedder = get_embedder()

            registry = ToolRegistry()
            registry.register(RagSearchTool(conn=conn, embed_fn=embedder.embed_query))
            registry.register(GraphSearchTool(conn=conn))
            fetcher = WebFetcher(rate_limiter=RateLimiter(), robots_checker=RobotsChecker())
            pipeline = IngestPipeline(conn=conn, embedder=embedder)
            registry.register(WebSearchTool())
            registry.register(WebIngestTool(fetcher=fetcher, pipeline=pipeline))

            factory = AgentFactory()
            km = KnowledgeManager(conn, config.WIKI_PATH)
            researcher = factory.create(AgentRole.RESEARCHER, tool_registry=registry)
            orchestrator = factory.create(
                AgentRole.ORCHESTRATOR,
                tool_registry=registry,
                factory=factory,
                knowledge_manager=km,
            )

            # Analyst with sim tools only (matching chat handler pattern)
            analyst_registry = ToolRegistry()
            try:
                from code.shukketsu.tools.analysis.sim_compare import SimCompareTool
                from code.shukketsu.tools.analysis.sim_optimize import SimOptimizeTool
                from code.shukketsu.tools.analysis.sim_run import SimRunTool

                analyst_registry.register(SimRunTool())
                analyst_registry.register(SimCompareTool())
                analyst_registry.register(SimOptimizeTool())
            except ImportError:
                logger.warning("Sim tools not available")

            analyst = factory.create(AgentRole.ANALYST, tool_registry=analyst_registry)
            self._agents = (researcher, orchestrator, analyst)
        return self._agents

    def close(self) -> None:
        """Close the DB connection created by _get_agents."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def run(
        self,
        tier: str | None = None,
        run_name: str | None = None,
        on_progress: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> PhaseGateReport:
        """Execute eval suite. Returns PhaseGateReport."""
        run_name = self._dm.create_run(run_name=run_name)
        items = self._dm.get_dataset_items()

        if tier:
            items = [it for it in items if it["metadata"].get("tier") == tier]

        total = len(items)
        results: list[EvalQuestionResult] = []

        for i, item in enumerate(items):
            result = await self._run_single(item, run_name)
            results.append(result)
            if on_progress:
                await on_progress(i + 1, total)

        report = compute_phase_gate(results)

        # Store report summary as a score on the run's last trace
        last_trace_id = results[-1].trace_id if results else None
        if last_trace_id:
            try:
                self._client.create_score(
                    trace_id=last_trace_id,
                    name="eval_run_summary",
                    value=1.0 if report.passed else 0.0,
                    comment=(
                        f"faith={report.avg_faithfulness:.2f} "
                        f"rel={report.avg_answer_relevancy:.2f} "
                        f"traj={report.avg_trajectory_precision:.2f} "
                        f"acc={report.avg_domain_accuracy:.2f}"
                    ),
                    metadata={
                        "run_name": run_name,
                        "avg_faithfulness": report.avg_faithfulness,
                        "avg_answer_relevancy": report.avg_answer_relevancy,
                        "avg_trajectory_precision": report.avg_trajectory_precision,
                        "avg_domain_accuracy": report.avg_domain_accuracy,
                        "passed": report.passed,
                        "tier_breakdown": report.tier_breakdown,
                    },
                )
            except Exception:
                logger.warning("Failed to record eval run summary score", exc_info=True)

        logger.info(
            "Eval run '%s' complete: %d questions, passed=%s",
            run_name,
            total,
            report.passed,
        )
        return report

    async def _run_single(self, item: dict[str, Any], run_name: str) -> EvalQuestionResult:
        """Execute and judge a single eval question."""
        metadata = item["metadata"]
        question = item["input"]["question"]
        qid = metadata["id"]
        item_tier = metadata.get("tier", "")
        expected_tools = metadata.get("expected_tools", [])
        ground_truth = item.get("expected_output", "")
        key_facts = metadata.get("key_facts", [])
        sim_validation = metadata.get("sim_validation")

        # Create a trace for this eval question via span API
        span = self._client.start_span(
            name="eval_question",
            metadata={"question_id": qid, "tier": item_tier},
            input={"question": question},
        )
        trace_id = span.trace_id

        evidence: list[str] = []
        try:
            # Route through same path as chat
            decision = await classify_query(question)
            researcher, orchestrator, analyst = self._get_agents()

            if (
                decision.complexity == TaskComplexity.TRIVIAL
                and decision.direct_answer
                and decision.direct_answer.strip()
            ):
                answer = decision.direct_answer
                tool_calls: list[str] = []
            elif decision.category == TaskCategory.ANALYSIS:
                from code.shukketsu.agents.tasks import AnalysisTask

                result = await analyst.execute(AnalysisTask(query=question, context={}))
                answer = result.output
                tool_calls = [t.tool_name for t in result.trajectory]
                evidence = result.evidence
            elif decision.complexity == TaskComplexity.MODERATE:
                result = await researcher.execute(
                    ResearchTask(query=question, context={}),
                    model_name=config.FAST_MODEL,
                )
                answer = result.output
                tool_calls = [t.tool_name for t in result.trajectory]
                evidence = result.evidence
            else:
                result = await orchestrator.execute(AgentTask(query=question, context={}))
                answer = result.output
                tool_calls = [t.tool_name for t in result.trajectory]
                evidence = result.evidence

        except Exception as exc:
            logger.warning("Eval question %s failed: %s", qid, exc)
            answer = f"ERROR: {exc}"
            tool_calls = []

        # --- Judging ---
        # Faithfulness judges claims against retrieved evidence (not ground truth).
        # Falls back to ground_truth if no evidence was retrieved (e.g. trivial queries).
        claims_text = await extract_claims(answer)
        faithfulness_evidence = evidence if evidence else [ground_truth]
        claim_results = await judge_faithfulness(claims_text, faithfulness_evidence)
        faithfulness = compute_faithfulness(claim_results)
        relevancy_raw = await judge_answer_relevancy(question, answer)
        relevancy = compute_answer_relevancy(relevancy_raw)
        accuracy = await judge_domain_accuracy(answer, ground_truth, key_facts)
        trajectory = compute_trajectory_precision(tool_calls, expected_tools)

        # Sim accuracy override for tier 3
        if sim_validation and item_tier == "simulation":
            dps = extract_dps_from_answer(answer)
            if dps is not None:
                sim_acc = judge_sim_accuracy(
                    dps,
                    sim_validation["expected_dps_min"],
                    sim_validation["expected_dps_max"],
                )
                # Blend sim accuracy into domain accuracy (50/50)
                accuracy = (accuracy + sim_acc) / 2.0

        # Close the span with output
        span.update(output={"answer": answer})
        span.end()

        # Record scores to Langfuse
        if trace_id:
            for score_name, score_value in [
                ("faithfulness", faithfulness),
                ("answer_relevancy", relevancy),
                ("trajectory_precision", trajectory),
                ("domain_accuracy", accuracy),
            ]:
                self._client.create_score(trace_id=trace_id, name=score_name, value=score_value)

            # Link trace to dataset run
            self._client.api.dataset_run_items.create(
                request=CreateDatasetRunItemRequest(
                    runName=run_name,
                    datasetItemId=item["id"],
                    traceId=trace_id,
                )
            )

        return EvalQuestionResult(
            question_id=qid,
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            trajectory_precision=trajectory,
            domain_accuracy=accuracy,
            claims=claim_results,
            tool_calls=tool_calls,
            tier=item_tier,
            trace_id=trace_id,
        )
