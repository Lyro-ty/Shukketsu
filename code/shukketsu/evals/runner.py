"""Eval runner — executes dataset questions and records scores to Langfuse.

Runs each question through the same routing path as real chat:
Qwen 4B classifies complexity → appropriate agent handles the query →
LLM judges score the result → scores recorded to Langfuse.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langfuse import Langfuse

from code.shukketsu import config
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
from code.shukketsu.routing.models import TaskComplexity
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
        self._agents: tuple | None = None

    def _get_agents(self):  # type: ignore[no-untyped-def]
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
            init_db(conn)
            embedder = get_embedder()

            registry = ToolRegistry()
            registry.register(RagSearchTool(conn=conn, embed_fn=embedder.embed_query))
            registry.register(GraphSearchTool(conn=conn))
            fetcher = WebFetcher(rate_limiter=RateLimiter(), robots_checker=RobotsChecker())
            pipeline = IngestPipeline(conn=conn, embedder=embedder)
            registry.register(WebSearchTool())
            registry.register(WebIngestTool(fetcher=fetcher, pipeline=pipeline))

            # Register sim tools if available
            try:
                from code.shukketsu.tools.analysis.sim_compare import SimCompareTool
                from code.shukketsu.tools.analysis.sim_optimize import SimOptimizeTool
                from code.shukketsu.tools.analysis.sim_run import SimRunTool

                registry.register(SimRunTool())
                registry.register(SimCompareTool())
                registry.register(SimOptimizeTool())
            except ImportError:
                logger.warning("Sim tools not available")

            factory = AgentFactory()
            km = KnowledgeManager(conn, config.WIKI_PATH)
            researcher = factory.create(AgentRole.RESEARCHER, tool_registry=registry)
            orchestrator = factory.create(
                AgentRole.ORCHESTRATOR,
                tool_registry=registry,
                factory=factory,
                knowledge_manager=km,
            )
            self._agents = (researcher, orchestrator)
        return self._agents

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
        # Store report metadata on the run
        self._client.update_dataset_run(
            dataset_name=config.EVAL_DATASET_NAME,
            run_name=run_name,
            metadata={
                "avg_faithfulness": report.avg_faithfulness,
                "avg_answer_relevancy": report.avg_answer_relevancy,
                "avg_trajectory_precision": report.avg_trajectory_precision,
                "avg_domain_accuracy": report.avg_domain_accuracy,
                "passed": report.passed,
                "tier_breakdown": report.tier_breakdown,
            },
        )

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

        # Create a trace for this eval question
        trace = self._client.trace(name="eval_question", metadata={"question_id": qid, "tier": item_tier})
        trace_id = trace.id

        try:
            # Route through same path as chat
            decision = await classify_query(question)
            researcher, orchestrator = self._get_agents()

            if (
                decision.complexity == TaskComplexity.TRIVIAL
                and decision.direct_answer
                and decision.direct_answer.strip()
            ):
                answer = decision.direct_answer
                tool_calls: list[str] = []
            elif decision.complexity == TaskComplexity.MODERATE:
                result = await researcher.execute(
                    ResearchTask(query=question, context={}),
                    model_name=config.FAST_MODEL,
                )
                answer = result.output
                tool_calls = [t.tool_name for t in result.trajectory]
            else:
                result = await orchestrator.execute(AgentTask(query=question, context={}))
                answer = result.output
                tool_calls = [t.tool_name for t in result.trajectory]

        except Exception as exc:
            logger.warning("Eval question %s failed: %s", qid, exc)
            answer = f"ERROR: {exc}"
            tool_calls = []

        # --- Judging ---
        claims_text = await extract_claims(answer)
        claim_results = await judge_faithfulness(claims_text, [ground_truth])
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

        # Record scores to Langfuse
        if trace_id:
            for name, value in [
                ("faithfulness", faithfulness),
                ("answer_relevancy", relevancy),
                ("trajectory_precision", trajectory),
                ("domain_accuracy", accuracy),
            ]:
                self._client.score(trace_id=trace_id, name=name, value=value)

            # Link trace to dataset run
            self._client.create_dataset_run_item(
                dataset_name=config.EVAL_DATASET_NAME,
                run_name=run_name,
                dataset_item_id=item["id"],
                trace_id=trace_id,
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
        )
