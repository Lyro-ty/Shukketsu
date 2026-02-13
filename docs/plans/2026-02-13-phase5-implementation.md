# Phase 5: Evaluation + Observability — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Langfuse-primary eval pipeline with 60-question 3-tier dataset, user feedback, lean dashboard, and fine-tuning export.

**Architecture:** All eval data flows through Langfuse — no new SQLite tables. Extends existing `evals/metrics.py` and `evals/judge.py` with answer relevancy + sim accuracy. New modules: dataset manager, runner, export CLI. New web routes: `/evals/` dashboard with Chart.js history. Chat feedback via WebSocket thumbs up/down.

**Tech Stack:** Langfuse Python SDK (datasets, scores, traces), FastAPI + Jinja2 + HTMX, Chart.js, Pydantic v2, pytest.

**Dependency graph:**
```
Task 1 (Config + Metrics)
├── Task 2 (Judge Extensions)
├── Task 3 (Dataset Content + Manager)
├── Task 5 (Chat Feedback)
│
Tasks 2+3 ──► Task 4 (Eval Runner)
│
Task 4 ──► Task 6 (Dashboard UI)
Task 4 ──► Task 7 (Export CLI)
│
Tasks 6+7 ──► Task 8 (Integration Wiring)
```

**Batches:** 1→[1] 2→[2,3,5] 3→[4] 4→[6,7] 5→[8]

---

### Task 1: Config + Metrics Foundation

Extend the config and metrics models to support answer relevancy, tier breakdown, and the new eval pipeline constants.

**Files:**
- Modify: `code/shukketsu/config.py`
- Modify: `code/shukketsu/evals/metrics.py`
- Modify: `tests/unit/test_eval_metrics.py`

**Config additions** (`code/shukketsu/config.py`): Add these constants after the existing `# Backup` section (line ~168):

```python
# Eval
EVAL_DATASET_NAME = "shukketsu-eval-v1"
EVAL_DATASET_PATH = Path(os.getenv("EVAL_DATASET_PATH", "/project/datasets/eval_questions.json"))
EVAL_CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "1"))
EVAL_SIM_TOLERANCE_PCT = float(os.getenv("EVAL_SIM_TOLERANCE_PCT", "5.0"))
EXPORT_OUTPUT_DIR = Path(os.getenv("EXPORT_OUTPUT_DIR", "/project/data/scratch/exports/"))
EXPORT_MIN_FEEDBACK_SCORE = float(os.getenv("EXPORT_MIN_FEEDBACK_SCORE", "1.0"))
RELEVANCY_THRESHOLD = float(os.getenv("RELEVANCY_THRESHOLD", "0.7"))
```

**Metrics changes** (`code/shukketsu/evals/metrics.py`):

1. Add `RELEVANCY_THRESHOLD` import from config (lazy, to avoid circular):

```python
# At module top, after existing threshold constants:
RELEVANCY_THRESHOLD = 0.7
```

2. Add `tier` and `answer_relevancy` to `EvalQuestionResult`:

```python
class EvalQuestionResult(BaseModel):
    """Per-question evaluation scores."""

    question_id: str
    faithfulness: float
    answer_relevancy: float = 0.0
    trajectory_precision: float
    domain_accuracy: float
    claims: list[ClaimFaithfulness]
    tool_calls: list[str]
    tier: str = ""
```

3. Add `avg_answer_relevancy` and `tier_breakdown` to `PhaseGateReport`:

```python
class PhaseGateReport(BaseModel):
    """Aggregate evaluation report for the phase gate."""

    avg_faithfulness: float
    avg_answer_relevancy: float = 0.0
    avg_trajectory_precision: float
    avg_domain_accuracy: float
    passed: bool
    wiki_coverage: dict[str, int] = Field(default_factory=dict)
    question_results: list[EvalQuestionResult] = Field(default_factory=list)
    tier_breakdown: dict[str, dict[str, float]] = Field(default_factory=dict)
```

4. Add `compute_answer_relevancy()`:

```python
def compute_answer_relevancy(relevancy_score: float) -> float:
    """Clamp a raw relevancy score to [0.0, 1.0]."""
    return max(0.0, min(1.0, relevancy_score))
```

5. Update `compute_phase_gate()` to include relevancy in the pass/fail check and compute tier breakdown:

```python
def compute_phase_gate(
    results: list[EvalQuestionResult],
) -> PhaseGateReport:
    """Aggregate per-question scores into a PhaseGateReport."""
    if not results:
        return PhaseGateReport(
            avg_faithfulness=0.0,
            avg_answer_relevancy=0.0,
            avg_trajectory_precision=0.0,
            avg_domain_accuracy=0.0,
            passed=False,
            question_results=results,
        )

    n = len(results)
    avg_faith = sum(r.faithfulness for r in results) / n
    avg_rel = sum(r.answer_relevancy for r in results) / n
    avg_traj = sum(r.trajectory_precision for r in results) / n
    avg_acc = sum(r.domain_accuracy for r in results) / n

    passed = (
        avg_faith >= FAITHFULNESS_THRESHOLD
        and avg_rel >= RELEVANCY_THRESHOLD
        and avg_traj >= TRAJECTORY_THRESHOLD
        and avg_acc >= ACCURACY_THRESHOLD
    )

    # Tier breakdown
    tier_groups: dict[str, list[EvalQuestionResult]] = {}
    for r in results:
        tier_groups.setdefault(r.tier or "unknown", []).append(r)

    tier_breakdown: dict[str, dict[str, float]] = {}
    for tier_name, tier_results in tier_groups.items():
        tn = len(tier_results)
        tier_breakdown[tier_name] = {
            "faithfulness": sum(r.faithfulness for r in tier_results) / tn,
            "answer_relevancy": sum(r.answer_relevancy for r in tier_results) / tn,
            "trajectory_precision": sum(r.trajectory_precision for r in tier_results) / tn,
            "domain_accuracy": sum(r.domain_accuracy for r in tier_results) / tn,
            "count": float(tn),
        }

    return PhaseGateReport(
        avg_faithfulness=avg_faith,
        avg_answer_relevancy=avg_rel,
        avg_trajectory_precision=avg_traj,
        avg_domain_accuracy=avg_acc,
        passed=passed,
        question_results=results,
        tier_breakdown=tier_breakdown,
    )
```

**Tests** (`tests/unit/test_eval_metrics.py`): Add to existing file:

```python
class TestAnswerRelevancy:
    def test_clamp_normal(self) -> None:
        assert compute_answer_relevancy(0.75) == 0.75

    def test_clamp_above_one(self) -> None:
        assert compute_answer_relevancy(1.5) == 1.0

    def test_clamp_below_zero(self) -> None:
        assert compute_answer_relevancy(-0.3) == 0.0


class TestTierBreakdown:
    def _make_result(
        self, *, qid: str = "q01", tier: str = "retrieval",
        faith: float = 0.9, rel: float = 0.8, traj: float = 0.8, acc: float = 0.8,
    ) -> EvalQuestionResult:
        return EvalQuestionResult(
            question_id=qid, faithfulness=faith, answer_relevancy=rel,
            trajectory_precision=traj, domain_accuracy=acc,
            claims=[], tool_calls=[], tier=tier,
        )

    def test_tier_breakdown_groups_correctly(self) -> None:
        results = [
            self._make_result(qid="q1", tier="retrieval", faith=1.0),
            self._make_result(qid="q2", tier="retrieval", faith=0.8),
            self._make_result(qid="q3", tier="simulation", faith=0.9),
        ]
        report = compute_phase_gate(results)
        assert "retrieval" in report.tier_breakdown
        assert "simulation" in report.tier_breakdown
        assert report.tier_breakdown["retrieval"]["count"] == 2.0
        assert report.tier_breakdown["simulation"]["count"] == 1.0

    def test_relevancy_included_in_pass_check(self) -> None:
        results = [self._make_result(rel=0.3)]
        report = compute_phase_gate(results)
        assert report.passed is False

    def test_relevancy_above_threshold_passes(self) -> None:
        results = [self._make_result()]
        report = compute_phase_gate(results)
        assert report.passed is True
```

**Verification:**
```bash
python3 -m pytest tests/unit/test_eval_metrics.py -v
```
Expected: All existing tests + 6 new tests pass. ~170 lines total.

---

### Task 2: Judge Extensions

Add `judge_answer_relevancy()`, `judge_sim_accuracy()`, and `extract_dps_from_answer()` to the existing judge module.

**Files:**
- Modify: `code/shukketsu/evals/judge.py`
- Modify: `tests/unit/test_eval_judge.py`

**Judge additions** (`code/shukketsu/evals/judge.py`):

1. Add new internal schema after existing `_AccuracyScore`:

```python
class _RelevancyScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
```

2. Add new system prompt:

```python
_RELEVANCY_PROMPT = """\
Score how well the answer addresses the specific question asked.
1.0: fully addresses the question with appropriate detail.
0.5: partially relevant — addresses part of the question or goes off-topic.
0.0: completely off-topic or does not answer what was asked.
Consider: Does it answer the question? Does it stay on topic? Is the detail level appropriate?"""
```

3. Add new public functions:

```python
import re

from code.shukketsu import config


async def judge_answer_relevancy(question: str, answer: str) -> float:
    """Score 0.0-1.0: does the answer address the question asked?

    Returns 0.0 on failure (conservative — flags for review).
    """
    try:
        result: _RelevancyScore = await get_structured_output(
            response_model=_RelevancyScore,
            messages=[
                {"role": "system", "content": _RELEVANCY_PROMPT},
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nAnswer:\n{answer}",
                },
            ],
            backend=ModelBackend.REASONING,
        )
        return result.score
    except Exception as exc:
        logger.warning("Answer relevancy judgment failed: %s", exc)
        return 0.0


_DPS_PATTERN = re.compile(r"(\d[\d,]*\.?\d*)\s*(?:dps|DPS)")


def extract_dps_from_answer(answer: str) -> float | None:
    """Extract a DPS number from an agent answer.

    Looks for patterns like '1,234.5 DPS', '1234 dps'.
    Returns None if no match found.
    """
    match = _DPS_PATTERN.search(answer)
    if not match:
        return None
    raw = match.group(1).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def judge_sim_accuracy(
    actual_dps: float,
    expected_min: float,
    expected_max: float,
) -> float:
    """Score 0.0-1.0: is the DPS value within the expected range?

    Returns 1.0 if within range, linear falloff outside range.
    Pure numeric — no LLM call.
    """
    if expected_min <= actual_dps <= expected_max:
        return 1.0

    distance = min(abs(actual_dps - expected_min), abs(actual_dps - expected_max))
    range_size = expected_max - expected_min
    if range_size <= 0:
        range_size = 1.0  # Avoid division by zero
    tolerance = range_size * (config.EVAL_SIM_TOLERANCE_PCT / 100.0)
    if tolerance <= 0:
        return 0.0

    return max(0.0, 1.0 - (distance / tolerance))
```

**Tests** (`tests/unit/test_eval_judge.py`): Add to existing file:

```python
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
```

**Verification:**
```bash
python3 -m pytest tests/unit/test_eval_judge.py -v
```
Expected: All existing 8 tests + 8 new tests pass.

---

### Task 3: Eval Dataset Content + Manager

Create the 60-question dataset JSON and a Langfuse dataset manager to sync it.

**Files:**
- Create: `datasets/eval_questions.json`
- Create: `code/shukketsu/evals/dataset.py`
- Create: `tests/unit/test_eval_dataset.py`

**Dataset** (`datasets/eval_questions.json`): Create a JSON file with 60 questions across three tiers. Structure per question:

```json
{
    "id": "t1_q01",
    "tier": "retrieval",
    "complexity": "trivial",
    "category": "retrieval",
    "spec": "general",
    "question": "What is the hit cap for a dual-wielding Rogue in TBC?",
    "ground_truth": "The special attack hit cap is 9% (142 hit rating). The white hit cap is 28% but not worth reaching.",
    "key_facts": ["9%", "142 hit rating", "28% white hit cap"],
    "expected_tools": ["rag_search"],
    "sim_validation": null
}
```

**Tier breakdown:**
- **Tier 1 (Retrieval)**: 20 questions — `t1_q01` to `t1_q20`. Mix of trivial (8) and moderate (12). Topics: hit cap, crit cap, expertise, poison mechanics, weapon speed, energy regen, combo points, stealth, armor pen, haste rating, meta gem, enchants, consumables, spell hit, white vs yellow hits, weapon skill, expose armor, poisons, rupture ticks, slice and dice uptime.
- **Tier 2 (Reasoning)**: 20 questions — `t2_q01` to `t2_q20`. Mix of moderate (8) and complex (12). Topics: spec comparisons (combat vs assassination), gear optimization per boss, talent choices for specific scenarios, rotation for multi-target, stat priority analysis, phase progression, gem choices, enchant tradeoffs, consumable planning, trinket comparison.
- **Tier 3 (Simulation)**: 20 questions — `t3_q01` to `t3_q20`. All complex. Topics: P1 BiS DPS, stat weights, item swap comparisons, talent spec DPS differences, buff impact analysis, weapon comparison, poison DPS contribution, cooldown timing impact. Each has `sim_validation` with `expected_dps_min` and `expected_dps_max`.

For the `sim_validation` DPS ranges on Tier 3: use ranges from Phase 4 validation profiles where available, and reasonable ranges (150-200 DPS spread) for others.

**Dataset Manager** (`code/shukketsu/evals/dataset.py`):

```python
"""Langfuse dataset manager for eval pipeline.

Syncs questions from a local JSON file into a Langfuse Dataset,
creates runs, and fetches run summaries for the dashboard.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langfuse import Langfuse

from code.shukketsu import config

logger = logging.getLogger(__name__)


class EvalDatasetManager:
    """Manages the Langfuse Dataset lifecycle for evaluations."""

    def __init__(self, langfuse_client: Langfuse) -> None:
        self._client = langfuse_client
        self._dataset_name = config.EVAL_DATASET_NAME

    def sync_dataset(self, dataset_path: Path | None = None) -> int:
        """Load questions from JSON, upsert into Langfuse Dataset.

        Returns the number of items synced.
        """
        path = dataset_path or config.EVAL_DATASET_PATH
        with open(path) as f:
            questions: list[dict[str, Any]] = json.load(f)

        # Create dataset if it doesn't exist (idempotent)
        self._client.create_dataset(name=self._dataset_name)

        for q in questions:
            self._client.create_dataset_item(
                dataset_name=self._dataset_name,
                input={"question": q["question"]},
                expected_output=q.get("ground_truth", ""),
                metadata={
                    "id": q["id"],
                    "tier": q["tier"],
                    "complexity": q["complexity"],
                    "category": q.get("category", ""),
                    "spec": q.get("spec", ""),
                    "expected_tools": q.get("expected_tools", []),
                    "key_facts": q.get("key_facts", []),
                    "sim_validation": q.get("sim_validation"),
                },
            )

        logger.info("Synced %d questions to Langfuse dataset '%s'", len(questions), self._dataset_name)
        return len(questions)

    def create_run(self, run_name: str | None = None) -> str:
        """Create a new DatasetRun. Auto-generates name if not provided.

        Returns the run name.
        """
        if run_name is None:
            run_name = f"run-{datetime.now(tz=timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        return run_name

    def get_dataset_items(self) -> list[dict[str, Any]]:
        """Fetch all dataset items from Langfuse.

        Returns list of dicts with id, input, expected_output, metadata.
        """
        dataset = self._client.get_dataset(name=self._dataset_name)
        items: list[dict[str, Any]] = []
        for item in dataset.items:
            items.append({
                "id": item.id,
                "input": item.input,
                "expected_output": item.expected_output,
                "metadata": item.metadata or {},
            })
        return items

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent dataset runs.

        Returns list of {run_name, created_at, metadata}.
        """
        runs_response = self._client.get_dataset_runs(
            dataset_name=self._dataset_name,
            page=1,
            limit=limit,
        )
        runs: list[dict[str, Any]] = []
        for run in runs_response.data:
            runs.append({
                "run_name": run.name,
                "created_at": str(run.created_at),
                "metadata": run.metadata or {},
            })
        return runs

    def get_run_detail(self, run_name: str) -> dict[str, Any]:
        """Get detailed info for a specific run."""
        run = self._client.get_dataset_run(
            dataset_name=self._dataset_name,
            run_name=run_name,
        )
        return {
            "run_name": run.name,
            "created_at": str(run.created_at),
            "metadata": run.metadata or {},
        }
```

**Tests** (`tests/unit/test_eval_dataset.py`):

```python
"""Tests for eval dataset manager."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from code.shukketsu.evals.dataset import EvalDatasetManager


def _sample_questions() -> list[dict]:
    return [
        {
            "id": "t1_q01",
            "tier": "retrieval",
            "complexity": "trivial",
            "category": "retrieval",
            "spec": "general",
            "question": "What is the hit cap?",
            "ground_truth": "9% for specials",
            "key_facts": ["9%"],
            "expected_tools": ["rag_search"],
            "sim_validation": None,
        },
        {
            "id": "t3_q01",
            "tier": "simulation",
            "complexity": "complex",
            "category": "analysis",
            "spec": "combat",
            "question": "What DPS does P1 BiS combat do?",
            "ground_truth": "1100-1250 DPS",
            "key_facts": ["1100-1250"],
            "expected_tools": ["sim_run"],
            "sim_validation": {"expected_dps_min": 1100, "expected_dps_max": 1250},
        },
    ]


class TestSyncDataset:
    def test_sync_creates_dataset_and_items(self, tmp_path: Path) -> None:
        questions = _sample_questions()
        qpath = tmp_path / "questions.json"
        qpath.write_text(json.dumps(questions))

        client = MagicMock()
        manager = EvalDatasetManager(langfuse_client=client)
        count = manager.sync_dataset(dataset_path=qpath)

        assert count == 2
        client.create_dataset.assert_called_once()
        assert client.create_dataset_item.call_count == 2

    def test_sync_upserts_are_idempotent(self, tmp_path: Path) -> None:
        questions = _sample_questions()[:1]
        qpath = tmp_path / "questions.json"
        qpath.write_text(json.dumps(questions))

        client = MagicMock()
        manager = EvalDatasetManager(langfuse_client=client)
        manager.sync_dataset(dataset_path=qpath)
        manager.sync_dataset(dataset_path=qpath)

        assert client.create_dataset.call_count == 2  # Idempotent call
        assert client.create_dataset_item.call_count == 2

    def test_sync_passes_metadata(self, tmp_path: Path) -> None:
        questions = _sample_questions()[:1]
        qpath = tmp_path / "questions.json"
        qpath.write_text(json.dumps(questions))

        client = MagicMock()
        manager = EvalDatasetManager(langfuse_client=client)
        manager.sync_dataset(dataset_path=qpath)

        call_kwargs = client.create_dataset_item.call_args
        assert call_kwargs.kwargs["metadata"]["tier"] == "retrieval"


class TestCreateRun:
    def test_auto_generates_name(self) -> None:
        manager = EvalDatasetManager(langfuse_client=MagicMock())
        name = manager.create_run()
        assert name.startswith("run-")

    def test_uses_custom_name(self) -> None:
        manager = EvalDatasetManager(langfuse_client=MagicMock())
        name = manager.create_run(run_name="my-run")
        assert name == "my-run"


class TestGetDatasetItems:
    def test_returns_items(self) -> None:
        client = MagicMock()
        mock_item = MagicMock()
        mock_item.id = "item-1"
        mock_item.input = {"question": "What?"}
        mock_item.expected_output = "Answer"
        mock_item.metadata = {"tier": "retrieval"}
        mock_dataset = MagicMock()
        mock_dataset.items = [mock_item]
        client.get_dataset.return_value = mock_dataset

        manager = EvalDatasetManager(langfuse_client=client)
        items = manager.get_dataset_items()
        assert len(items) == 1
        assert items[0]["input"] == {"question": "What?"}


class TestListRuns:
    def test_returns_runs(self) -> None:
        client = MagicMock()
        mock_run = MagicMock()
        mock_run.name = "run-20260213"
        mock_run.created_at = "2026-02-13T00:00:00Z"
        mock_run.metadata = {}
        mock_response = MagicMock()
        mock_response.data = [mock_run]
        client.get_dataset_runs.return_value = mock_response

        manager = EvalDatasetManager(langfuse_client=client)
        runs = manager.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_name"] == "run-20260213"

    def test_empty_runs(self) -> None:
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = []
        client.get_dataset_runs.return_value = mock_response

        manager = EvalDatasetManager(langfuse_client=client)
        runs = manager.list_runs()
        assert runs == []


class TestGetRunDetail:
    def test_returns_run_detail(self) -> None:
        client = MagicMock()
        mock_run = MagicMock()
        mock_run.name = "run-test"
        mock_run.created_at = "2026-02-13T00:00:00Z"
        mock_run.metadata = {"total": 60}
        client.get_dataset_run.return_value = mock_run

        manager = EvalDatasetManager(langfuse_client=client)
        detail = manager.get_run_detail("run-test")
        assert detail["run_name"] == "run-test"
        assert detail["metadata"]["total"] == 60
```

**Verification:**
```bash
python3 -m pytest tests/unit/test_eval_dataset.py -v
```
Expected: 10 tests pass.

---

### Task 4: Eval Runner

Execute eval questions against the live agent system and record scores to Langfuse.

**Files:**
- Create: `code/shukketsu/evals/runner.py`
- Create: `tests/unit/test_eval_runner.py`

**Implementation** (`code/shukketsu/evals/runner.py`):

```python
"""Eval runner — executes dataset questions and records scores to Langfuse.

Runs each question through the same routing path as real chat:
Qwen 4B classifies complexity → appropriate agent handles the query →
LLM judges score the result → scores recorded to Langfuse.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe

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

    def _get_agents(self):
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
        """Execute eval suite. Returns PhaseGateReport.

        Args:
            tier: Filter to "retrieval", "reasoning", or "simulation". None = all.
            run_name: Custom run name. Auto-generated if None.
            on_progress: Callback(completed, total) for progress reporting.
        """
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
            run_name, total, report.passed,
        )
        return report

    @observe(name="eval_question")
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

        trace_id = langfuse_context.get_current_trace_id()

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
```

**Tests** (`tests/unit/test_eval_runner.py`): All mocked (no external deps). Test the runner's routing logic, scoring integration, tier filtering, error handling, and progress callback. Key tests:

```python
"""Tests for eval runner."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code.shukketsu.evals.runner import EvalRunner


def _make_item(
    qid: str = "t1_q01",
    tier: str = "retrieval",
    complexity: str = "moderate",
    question: str = "What is the hit cap?",
    ground_truth: str = "9%",
    expected_tools: list[str] | None = None,
    sim_validation: dict | None = None,
) -> dict:
    return {
        "id": f"item-{qid}",
        "input": {"question": question},
        "expected_output": ground_truth,
        "metadata": {
            "id": qid,
            "tier": tier,
            "complexity": complexity,
            "expected_tools": expected_tools or ["rag_search"],
            "key_facts": ["9%"],
            "sim_validation": sim_validation,
        },
    }


@pytest.fixture
def runner() -> EvalRunner:
    dm = MagicMock()
    dm.create_run.return_value = "test-run"
    client = MagicMock()
    return EvalRunner(dataset_manager=dm, langfuse_client=client)


class TestRun:
    @patch("code.shukketsu.evals.runner.classify_query", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_answer_relevancy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_domain_accuracy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_faithfulness", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.extract_claims", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.langfuse_context")
    async def test_runs_all_items(
        self, mock_ctx, mock_extract, mock_faith, mock_acc, mock_rel, mock_classify, runner
    ) -> None:
        from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity

        mock_ctx.get_current_trace_id.return_value = None
        mock_extract.return_value = []
        mock_faith.return_value = []
        mock_rel.return_value = 0.9
        mock_acc.return_value = 0.8
        mock_classify.return_value = RoutingDecision(
            complexity=TaskComplexity.TRIVIAL,
            category=TaskCategory.CONVERSATION,
            needs_tools=False,
            suggested_agent="general",
            direct_answer="9%",
        )

        runner._dm.get_dataset_items.return_value = [_make_item(), _make_item(qid="t1_q02")]

        report = await runner.run()
        assert len(report.question_results) == 2

    @patch("code.shukketsu.evals.runner.classify_query", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_answer_relevancy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_domain_accuracy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_faithfulness", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.extract_claims", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.langfuse_context")
    async def test_tier_filter(
        self, mock_ctx, mock_extract, mock_faith, mock_acc, mock_rel, mock_classify, runner
    ) -> None:
        from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity

        mock_ctx.get_current_trace_id.return_value = None
        mock_extract.return_value = []
        mock_faith.return_value = []
        mock_rel.return_value = 0.9
        mock_acc.return_value = 0.8
        mock_classify.return_value = RoutingDecision(
            complexity=TaskComplexity.TRIVIAL,
            category=TaskCategory.CONVERSATION,
            needs_tools=False,
            suggested_agent="general",
            direct_answer="answer",
        )

        runner._dm.get_dataset_items.return_value = [
            _make_item(qid="t1_q01", tier="retrieval"),
            _make_item(qid="t3_q01", tier="simulation"),
        ]

        report = await runner.run(tier="retrieval")
        assert len(report.question_results) == 1
        assert report.question_results[0].tier == "retrieval"

    @patch("code.shukketsu.evals.runner.classify_query", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_answer_relevancy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_domain_accuracy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_faithfulness", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.extract_claims", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.langfuse_context")
    async def test_progress_callback(
        self, mock_ctx, mock_extract, mock_faith, mock_acc, mock_rel, mock_classify, runner
    ) -> None:
        from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity

        mock_ctx.get_current_trace_id.return_value = None
        mock_extract.return_value = []
        mock_faith.return_value = []
        mock_rel.return_value = 0.9
        mock_acc.return_value = 0.8
        mock_classify.return_value = RoutingDecision(
            complexity=TaskComplexity.TRIVIAL,
            category=TaskCategory.CONVERSATION,
            needs_tools=False,
            suggested_agent="general",
            direct_answer="9%",
        )

        runner._dm.get_dataset_items.return_value = [_make_item()]
        progress = AsyncMock()

        await runner.run(on_progress=progress)
        progress.assert_called_once_with(1, 1)

    @patch("code.shukketsu.evals.runner.classify_query", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_answer_relevancy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_domain_accuracy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_faithfulness", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.extract_claims", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.langfuse_context")
    async def test_agent_error_records_zero(
        self, mock_ctx, mock_extract, mock_faith, mock_acc, mock_rel, mock_classify, runner
    ) -> None:
        mock_ctx.get_current_trace_id.return_value = None
        mock_extract.return_value = []
        mock_faith.return_value = []
        mock_rel.return_value = 0.0
        mock_acc.return_value = 0.0
        mock_classify.side_effect = Exception("Router down")

        runner._dm.get_dataset_items.return_value = [_make_item()]

        report = await runner.run()
        assert len(report.question_results) == 1
        # Answer will be "ERROR: ..." which judges score as 0.0

    @patch("code.shukketsu.evals.runner.classify_query", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_answer_relevancy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_domain_accuracy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_faithfulness", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.extract_claims", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.langfuse_context")
    async def test_scores_recorded_to_langfuse(
        self, mock_ctx, mock_extract, mock_faith, mock_acc, mock_rel, mock_classify, runner
    ) -> None:
        from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity

        mock_ctx.get_current_trace_id.return_value = "trace-123"
        mock_extract.return_value = []
        mock_faith.return_value = []
        mock_rel.return_value = 0.9
        mock_acc.return_value = 0.8
        mock_classify.return_value = RoutingDecision(
            complexity=TaskComplexity.TRIVIAL,
            category=TaskCategory.CONVERSATION,
            needs_tools=False,
            suggested_agent="general",
            direct_answer="9%",
        )

        runner._dm.get_dataset_items.return_value = [_make_item()]

        await runner.run()
        # Should have 4 scores (one per metric) + 1 run item link
        assert runner._client.score.call_count == 4
        runner._client.create_dataset_run_item.assert_called_once()

    @patch("code.shukketsu.evals.runner.classify_query", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_answer_relevancy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_domain_accuracy", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.judge_faithfulness", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.extract_claims", new_callable=AsyncMock)
    @patch("code.shukketsu.evals.runner.extract_dps_from_answer")
    @patch("code.shukketsu.evals.runner.langfuse_context")
    async def test_sim_tier_blends_accuracy(
        self, mock_ctx, mock_dps, mock_extract, mock_faith, mock_acc, mock_rel, mock_classify, runner
    ) -> None:
        from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity

        mock_ctx.get_current_trace_id.return_value = None
        mock_extract.return_value = []
        mock_faith.return_value = []
        mock_rel.return_value = 0.9
        mock_acc.return_value = 0.8
        mock_dps.return_value = 1150.0
        mock_classify.return_value = RoutingDecision(
            complexity=TaskComplexity.TRIVIAL,
            category=TaskCategory.CONVERSATION,
            needs_tools=False,
            suggested_agent="general",
            direct_answer="1150 DPS",
        )

        runner._dm.get_dataset_items.return_value = [
            _make_item(
                qid="t3_q01",
                tier="simulation",
                sim_validation={"expected_dps_min": 1100, "expected_dps_max": 1200},
            )
        ]

        report = await runner.run()
        # sim_accuracy = 1.0 (within range), domain = 0.8, blend = 0.9
        assert report.question_results[0].domain_accuracy == pytest.approx(0.9)
```

**Verification:**
```bash
python3 -m pytest tests/unit/test_eval_runner.py -v
```
Expected: 7 tests pass.

---

### Task 5: Chat Feedback

Add thumbs up/down feedback to the chat WebSocket. Include `trace_id` in "done" messages so the frontend can link feedback to traces.

**Files:**
- Modify: `code/shukketsu/web/routers/chat.py`
- Modify: `code/shukketsu/web/static/js/chat.js`
- Modify: `code/shukketsu/web/templates/chat.html`
- Create: `tests/unit/test_feedback.py`

**Chat handler changes** (`code/shukketsu/web/routers/chat.py`):

1. Add import at top:

```python
from langfuse.decorators import langfuse_context
```

2. In `_handle_message()`, add feedback handling before the existing `msg_type != "message"` check (around line 174):

```python
    if msg_type == "feedback":
        trace_id = data.get("trace_id")
        score = data.get("score")
        if trace_id is not None and score is not None:
            try:
                langfuse = get_client()
                langfuse.score(
                    trace_id=trace_id,
                    name="user_feedback",
                    value=float(score),
                    comment=f"thumbs {'up' if score else 'down'}",
                )
            except Exception:
                logger.warning("Failed to record feedback", exc_info=True)
        return
```

3. In `_agent_response()`, change the "done" message to include `trace_id` (around line 305):

```python
        trace_id = langfuse_context.get_current_trace_id()
        session.add_message("assistant", answer)
        await websocket.send_json({"type": "done", "content": answer, "trace_id": trace_id})
```

**Chat JS changes** (`code/shukketsu/web/static/js/chat.js`):

1. Add a `lastTraceId` state variable at top:

```javascript
let lastTraceId = null;
```

2. In the `case "done":` handler, capture the trace_id and add feedback buttons:

```javascript
        case "done":
            if (currentAssistantEl) {
                fullResponse = msg.content;
                lastTraceId = msg.trace_id || null;
                if (fullResponse) {
                    currentAssistantEl.innerHTML =
                        DOMPurify.sanitize(marked.parse(fullResponse));
                    currentAssistantEl.classList.add("markdown-body");
                }
                // Add feedback buttons
                if (lastTraceId) {
                    const feedbackRow = document.createElement("div");
                    feedbackRow.className = "flex gap-2 mt-2 feedback-row";
                    feedbackRow.innerHTML = `
                        <button class="thumb-btn text-parchment-dim hover:text-green-400 transition-colors text-sm px-2 py-1 rounded border border-white/10 hover:border-green-400/50"
                                onclick="sendFeedback(1, '${lastTraceId}', this.parentElement)">👍</button>
                        <button class="thumb-btn text-parchment-dim hover:text-red-400 transition-colors text-sm px-2 py-1 rounded border border-white/10 hover:border-red-400/50"
                                onclick="sendFeedback(0, '${lastTraceId}', this.parentElement)">👎</button>
                    `;
                    currentAssistantEl.parentElement.appendChild(feedbackRow);
                }
            }
            endStreaming();
            break;
```

3. Add `sendFeedback` function:

```javascript
function sendFeedback(score, traceId, row) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "feedback", score: score, trace_id: traceId }));
    }
    // Disable buttons after click
    row.querySelectorAll(".thumb-btn").forEach(btn => {
        btn.disabled = true;
        btn.classList.add("opacity-30", "cursor-not-allowed");
    });
    // Highlight selected
    const btns = row.querySelectorAll(".thumb-btn");
    if (score === 1) btns[0].classList.add("text-green-400", "border-green-400/50");
    else btns[1].classList.add("text-red-400", "border-red-400/50");
}
```

**Tests** (`tests/unit/test_feedback.py`):

```python
"""Tests for chat feedback handling."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity


def _get_app():
    from code.shukketsu.web.app import app
    return app


def _trivial_decision(answer: str = "Direct answer.") -> RoutingDecision:
    return RoutingDecision(
        complexity=TaskComplexity.TRIVIAL,
        category=TaskCategory.CONVERSATION,
        needs_tools=False,
        suggested_agent="general",
        direct_answer=answer,
    )


def _mock_agents(answer: str = "Test answer") -> tuple[MagicMock, MagicMock]:
    researcher = MagicMock()
    researcher.execute = AsyncMock(return_value=MagicMock(output=answer))
    orchestrator = MagicMock()
    orchestrator.execute = AsyncMock(return_value=MagicMock(output=answer))
    return researcher, orchestrator


class TestFeedbackMessage:
    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
    def test_feedback_accepted_silently(self, mock_get: MagicMock) -> None:
        """Feedback message should be accepted without error response."""
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "feedback", "score": 1, "trace_id": "trace-123"})
            # Send a real message to verify connection still works
            ws.send_json({"type": "message", "content": ""})
            msg = ws.receive_json()
            assert msg["type"] == "error"  # Empty message error, not feedback error

    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
    def test_feedback_without_trace_id_ignored(self, mock_get: MagicMock) -> None:
        """Feedback without trace_id should be silently ignored."""
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "feedback", "score": 1})
            ws.send_json({"type": "message", "content": ""})
            msg = ws.receive_json()
            assert msg["type"] == "error"

    @patch("code.shukketsu.web.routers.chat.langfuse_context")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_done_includes_trace_id(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
        mock_ctx: MagicMock,
    ) -> None:
        """Done messages should include trace_id from Langfuse context."""
        mock_classify.return_value = _trivial_decision("Answer")
        mock_agents.return_value = _mock_agents()
        mock_ctx.get_current_trace_id.return_value = "trace-abc"

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hi"})
            ws.receive_json()  # routing
            done = ws.receive_json()
            assert done["type"] == "done"
            assert "trace_id" in done

    @patch("code.shukketsu.web.routers.chat.langfuse_context")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_done_trace_id_none_when_disabled(
        self,
        mock_classify: AsyncMock,
        mock_agents: MagicMock,
        mock_ctx: MagicMock,
    ) -> None:
        """trace_id should be None when Langfuse tracing is disabled."""
        mock_classify.return_value = _trivial_decision("Answer")
        mock_agents.return_value = _mock_agents()
        mock_ctx.get_current_trace_id.return_value = None

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hi"})
            ws.receive_json()  # routing
            done = ws.receive_json()
            assert done["trace_id"] is None

    @patch("code.shukketsu.web.routers.chat.get_client")
    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
    def test_feedback_records_to_langfuse(
        self, mock_agents: MagicMock, mock_langfuse: MagicMock,
    ) -> None:
        """Thumbs up should call langfuse.score()."""
        mock_client = MagicMock()
        mock_langfuse.return_value = mock_client

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "feedback", "score": 1, "trace_id": "trace-xyz"})
            # Verify by sending another message
            ws.send_json({"type": "message", "content": ""})
            ws.receive_json()  # error (empty)

        mock_client.score.assert_called_once()
        call_kwargs = mock_client.score.call_args.kwargs
        assert call_kwargs["trace_id"] == "trace-xyz"
        assert call_kwargs["value"] == 1.0

    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
    def test_thumbs_down_value_zero(self, mock_get: MagicMock) -> None:
        """Thumbs down should send score=0."""
        with patch("code.shukketsu.web.routers.chat.get_client") as mock_langfuse:
            mock_client = MagicMock()
            mock_langfuse.return_value = mock_client

            client = TestClient(_get_app())
            with client.websocket_connect("/ws/chat") as ws:
                ws.receive_json()  # connected
                ws.send_json({"type": "feedback", "score": 0, "trace_id": "trace-xyz"})
                ws.send_json({"type": "message", "content": ""})
                ws.receive_json()

            call_kwargs = mock_client.score.call_args.kwargs
            assert call_kwargs["value"] == 0.0
```

**Verification:**
```bash
python3 -m pytest tests/unit/test_feedback.py tests/unit/test_chat_handler.py -v
```
Expected: 7 new feedback tests + 27 existing chat tests pass.

---

### Task 6: Dashboard UI

Lean eval dashboard with summary cards, history chart, and run trigger.

**Files:**
- Create: `code/shukketsu/web/routers/evals.py`
- Create: `code/shukketsu/web/templates/evals/index.html`
- Create: `code/shukketsu/web/templates/evals/partials/run_status.html`
- Create: `code/shukketsu/web/templates/evals/partials/summary_card.html`
- Create: `code/shukketsu/web/static/js/eval-charts.js`
- Modify: `code/shukketsu/web/app.py`
- Modify: `code/shukketsu/web/templates/base.html`
- Create: `tests/unit/test_eval_routes.py`

**Routes** (`code/shukketsu/web/routers/evals.py`):

```python
"""Evaluation dashboard routes.

Provides a lean in-app dashboard for triggering eval runs, viewing
latest results, and tracking metric trends over time. Detailed
trace drill-down links to Langfuse.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langfuse import Langfuse

from code.shukketsu import config
from code.shukketsu.evals.dataset import EvalDatasetManager

logger = logging.getLogger(__name__)

_WEB_DIR = Path(__file__).parent.parent
_templates = Jinja2Templates(directory=_WEB_DIR / "templates")

router = APIRouter(prefix="/evals", tags=["evals"])

# In-memory run tracking (simple — one run at a time)
_current_run: dict[str, Any] | None = None


def _get_langfuse() -> Langfuse:
    from code.shukketsu.observability.tracer import get_client
    return get_client()


def _get_dataset_manager() -> EvalDatasetManager:
    return EvalDatasetManager(langfuse_client=_get_langfuse())


@router.get("/")
async def evals_dashboard(request: Request) -> HTMLResponse:
    """Render the eval dashboard page."""
    dm = _get_dataset_manager()
    try:
        runs = dm.list_runs(limit=1)
        latest = runs[0] if runs else None
    except Exception:
        latest = None

    return _templates.TemplateResponse(
        request,
        "evals/index.html",
        {"latest_run": latest, "langfuse_host": config.LANGFUSE_HOST, "is_running": _current_run is not None},
    )


@router.post("/run")
async def trigger_eval_run(
    request: Request,
    tier: str | None = Query(default=None),
) -> HTMLResponse:
    """Trigger an eval run. Returns HTMX fragment with initial status."""
    global _current_run  # noqa: PLW0603

    if _current_run is not None:
        return _templates.TemplateResponse(
            request,
            "evals/partials/run_status.html",
            {"status": "already_running", "progress": 0, "total": 0},
        )

    _current_run = {"status": "starting", "progress": 0, "total": 0, "run_name": None}

    # Launch eval in background
    asyncio.create_task(_run_eval_background(tier))

    return _templates.TemplateResponse(
        request,
        "evals/partials/run_status.html",
        {"status": "starting", "progress": 0, "total": 0},
    )


async def _run_eval_background(tier: str | None) -> None:
    """Background task that runs the eval suite."""
    global _current_run  # noqa: PLW0603
    try:
        from code.shukketsu.evals.runner import EvalRunner

        dm = _get_dataset_manager()
        dm.sync_dataset()

        runner = EvalRunner(dataset_manager=dm, langfuse_client=_get_langfuse())

        async def _on_progress(completed: int, total: int) -> None:
            if _current_run is not None:
                _current_run["progress"] = completed
                _current_run["total"] = total

        report = await runner.run(tier=tier, on_progress=_on_progress)

        if _current_run is not None:
            _current_run["status"] = "complete"
            _current_run["report"] = {
                "passed": report.passed,
                "avg_faithfulness": report.avg_faithfulness,
                "avg_answer_relevancy": report.avg_answer_relevancy,
                "avg_trajectory_precision": report.avg_trajectory_precision,
                "avg_domain_accuracy": report.avg_domain_accuracy,
                "tier_breakdown": report.tier_breakdown,
                "question_count": len(report.question_results),
            }
    except Exception:
        logger.exception("Eval run failed")
        if _current_run is not None:
            _current_run["status"] = "failed"


@router.get("/run/status")
async def run_status(request: Request) -> HTMLResponse:
    """HTMX polling endpoint for run progress."""
    if _current_run is None:
        return _templates.TemplateResponse(
            request,
            "evals/partials/run_status.html",
            {"status": "idle", "progress": 0, "total": 0},
        )

    status = _current_run["status"]
    ctx: dict[str, Any] = {
        "status": status,
        "progress": _current_run.get("progress", 0),
        "total": _current_run.get("total", 0),
    }

    if status in ("complete", "failed"):
        ctx["report"] = _current_run.get("report")
        # Reset for next run
        global _current_run  # noqa: PLW0603
        _current_run = None

    return _templates.TemplateResponse(request, "evals/partials/run_status.html", ctx)


@router.get("/history")
async def eval_history() -> JSONResponse:
    """Return recent runs as JSON for Chart.js."""
    dm = _get_dataset_manager()
    try:
        runs = dm.list_runs(limit=20)
    except Exception:
        runs = []

    return JSONResponse(content={"runs": runs})


@router.get("/latest")
async def latest_summary(request: Request) -> HTMLResponse:
    """HTMX fragment: summary card for latest run."""
    dm = _get_dataset_manager()
    try:
        runs = dm.list_runs(limit=1)
        latest = runs[0] if runs else None
    except Exception:
        latest = None

    return _templates.TemplateResponse(
        request,
        "evals/partials/summary_card.html",
        {"latest_run": latest, "langfuse_host": config.LANGFUSE_HOST},
    )
```

**Templates**: Create `evals/index.html` extending `base.html` with three sections (run controls, latest summary, history chart). Create partials for `run_status.html` (progress bar + status text) and `summary_card.html` (metric gauges + pass/fail badge + Langfuse link). Follow the same Tailwind + HTMX patterns used in `sim/index.html` and `wiki/` templates.

**Chart JS** (`eval-charts.js`): Line chart with 4 metric lines (faithfulness=green, relevancy=blue, trajectory=yellow, accuracy=purple). Threshold lines as dashed horizontals. X-axis: run timestamps. Follow the pattern in `sim-charts.js`.

**App registration** (`code/shukketsu/web/app.py`): Add:

```python
from code.shukketsu.web.routers.evals import router as evals_router
# ...
app.include_router(evals_router)
```

**Nav link** (`code/shukketsu/web/templates/base.html`): Add after the Sim link:

```html
<a href="/evals/"
   class="text-sm transition-colors {% if request.url.path.startswith('/evals') %}text-wow-gold{% else %}text-parchment-dim hover:text-parchment{% endif %}">
    Evals
</a>
```

**Tests** (`tests/unit/test_eval_routes.py`): Test routes return 200, trigger run returns status, history returns JSON, latest returns fragment, empty state handling. Mock `_get_langfuse` and `_get_dataset_manager`. Pattern from `test_sim_routes.py` and `test_wiki_routes.py`:

```python
"""Tests for eval dashboard routes."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def _get_app():
    from code.shukketsu.web.app import app
    return app


def _mock_dm(runs=None):
    dm = MagicMock()
    dm.list_runs.return_value = runs or []
    dm.sync_dataset.return_value = 60
    return dm


class TestDashboard:
    @patch("code.shukketsu.web.routers.evals._get_dataset_manager")
    def test_returns_200(self, mock_dm_fn: MagicMock) -> None:
        mock_dm_fn.return_value = _mock_dm()
        client = TestClient(_get_app())
        resp = client.get("/evals/")
        assert resp.status_code == 200

    @patch("code.shukketsu.web.routers.evals._get_dataset_manager")
    def test_empty_state(self, mock_dm_fn: MagicMock) -> None:
        mock_dm_fn.return_value = _mock_dm()
        client = TestClient(_get_app())
        resp = client.get("/evals/")
        assert resp.status_code == 200


class TestHistory:
    @patch("code.shukketsu.web.routers.evals._get_dataset_manager")
    def test_returns_json(self, mock_dm_fn: MagicMock) -> None:
        mock_dm_fn.return_value = _mock_dm([
            {"run_name": "run-1", "created_at": "2026-02-13", "metadata": {}},
        ])
        client = TestClient(_get_app())
        resp = client.get("/evals/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "runs" in data
        assert len(data["runs"]) == 1

    @patch("code.shukketsu.web.routers.evals._get_dataset_manager")
    def test_empty_history(self, mock_dm_fn: MagicMock) -> None:
        mock_dm_fn.return_value = _mock_dm()
        client = TestClient(_get_app())
        resp = client.get("/evals/history")
        assert resp.json()["runs"] == []


class TestLatest:
    @patch("code.shukketsu.web.routers.evals._get_dataset_manager")
    def test_returns_fragment(self, mock_dm_fn: MagicMock) -> None:
        mock_dm_fn.return_value = _mock_dm()
        client = TestClient(_get_app())
        resp = client.get("/evals/latest")
        assert resp.status_code == 200


class TestRunStatus:
    @patch("code.shukketsu.web.routers.evals._current_run", None)
    def test_idle_status(self) -> None:
        client = TestClient(_get_app())
        resp = client.get("/evals/run/status")
        assert resp.status_code == 200


class TestTriggerRun:
    @patch("code.shukketsu.web.routers.evals._current_run", None)
    @patch("code.shukketsu.web.routers.evals._run_eval_background")
    @patch("code.shukketsu.web.routers.evals._get_dataset_manager")
    def test_trigger_returns_status(self, mock_dm_fn, mock_bg, mock_run=None) -> None:
        mock_dm_fn.return_value = _mock_dm()
        client = TestClient(_get_app())
        resp = client.post("/evals/run")
        assert resp.status_code == 200

    @patch("code.shukketsu.web.routers.evals._get_dataset_manager")
    def test_tier_filter_accepted(self, mock_dm_fn) -> None:
        mock_dm_fn.return_value = _mock_dm()
        client = TestClient(_get_app())
        resp = client.post("/evals/run?tier=retrieval")
        assert resp.status_code == 200
```

**Verification:**
```bash
python3 -m pytest tests/unit/test_eval_routes.py -v
```
Expected: 8 tests pass.

---

### Task 7: Fine-Tuning Export CLI

CLI to export positively-scored traces as JSONL for fine-tuning datasets.

**Files:**
- Create: `code/shukketsu/evals/export.py`
- Create: `tests/unit/test_eval_export.py`

**Implementation** (`code/shukketsu/evals/export.py`):

```python
"""Fine-tuning dataset export from Langfuse traces.

Exports positively-scored conversation traces as JSONL files in
ShareGPT or function-calling format for fine-tuning.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langfuse import Langfuse

from code.shukketsu import config

logger = logging.getLogger(__name__)


class TrainingExporter:
    """Exports Langfuse traces as JSONL training data."""

    def __init__(self, langfuse_client: Langfuse) -> None:
        self._client = langfuse_client

    def export(
        self,
        format: str = "sharegpt",
        min_feedback_score: float = 1.0,
        min_trajectory_precision: float = 0.7,
        output_path: Path | None = None,
    ) -> Path:
        """Export positively-scored traces as JSONL.

        Args:
            format: "sharegpt" or "function_calling".
            min_feedback_score: Minimum user feedback score to include.
            min_trajectory_precision: Minimum trajectory precision to include.
            output_path: Where to write the file. Auto-generated if None.

        Returns:
            Path to the written JSONL file.
        """
        if output_path is None:
            config.EXPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
            output_path = config.EXPORT_OUTPUT_DIR / f"training-{format}-{timestamp}.jsonl"

        traces = self._fetch_scored_traces(min_feedback_score, min_trajectory_precision)

        count = 0
        with open(output_path, "w") as f:
            for trace in traces:
                if format == "sharegpt":
                    record = self._to_sharegpt(trace)
                else:
                    record = self._to_function_calling(trace)
                if record:
                    f.write(json.dumps(record) + "\n")
                    count += 1

        logger.info("Exported %d traces to %s (format=%s)", count, output_path, format)
        return output_path

    def _fetch_scored_traces(
        self,
        min_feedback: float,
        min_trajectory: float,
    ) -> list[dict[str, Any]]:
        """Fetch traces with positive feedback from Langfuse.

        Returns list of trace dicts with input, output, scores.
        """
        try:
            traces_response = self._client.fetch_traces(
                tags=["chat"],
                limit=500,
            )
            scored: list[dict[str, Any]] = []
            for trace in traces_response.data:
                scores = {s.name: s.value for s in (trace.scores or [])}
                feedback = scores.get("user_feedback", -1)
                trajectory = scores.get("trajectory_precision", -1)

                if feedback >= min_feedback and trajectory >= min_trajectory:
                    scored.append({
                        "input": trace.input,
                        "output": trace.output,
                        "scores": scores,
                        "observations": trace.observations or [],
                    })
            return scored
        except Exception:
            logger.warning("Failed to fetch traces from Langfuse", exc_info=True)
            return []

    def _to_sharegpt(self, trace: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a trace to ShareGPT format."""
        input_text = trace.get("input")
        output_text = trace.get("output")
        if not input_text or not output_text:
            return None

        return {
            "conversations": [
                {"from": "system", "value": config.SYSTEM_PROMPT},
                {"from": "human", "value": str(input_text)},
                {"from": "gpt", "value": str(output_text)},
            ]
        }

    def _to_function_calling(self, trace: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a trace to function-calling format."""
        input_text = trace.get("input")
        output_text = trace.get("output")
        if not input_text or not output_text:
            return None

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": str(input_text)},
        ]

        # Extract tool calls from observations
        for obs in trace.get("observations", []):
            if hasattr(obs, "type") and obs.type == "tool":
                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "function": {
                            "name": obs.name or "unknown",
                            "arguments": json.dumps(obs.input or {}),
                        }
                    }],
                })
                messages.append({
                    "role": "tool",
                    "content": str(obs.output or ""),
                })

        messages.append({"role": "assistant", "content": str(output_text)})
        return {"messages": messages}


def main() -> None:
    """CLI entry point for fine-tuning export."""
    import argparse

    from code.shukketsu.observability.tracer import get_client, init_langfuse

    parser = argparse.ArgumentParser(description="Export Langfuse traces as training data")
    parser.add_argument("--format", choices=["sharegpt", "function_calling"], default="sharegpt")
    parser.add_argument("--min-score", type=float, default=1.0, help="Minimum feedback score")
    parser.add_argument("--min-trajectory", type=float, default=0.7)
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    init_langfuse()
    client = get_client()
    exporter = TrainingExporter(langfuse_client=client)
    output = args.output and Path(args.output)
    path = exporter.export(
        format=args.format,
        min_feedback_score=args.min_score,
        min_trajectory_precision=args.min_trajectory,
        output_path=output,
    )
    print(f"Exported to: {path}")


if __name__ == "__main__":
    main()
```

**Tests** (`tests/unit/test_eval_export.py`):

```python
"""Tests for fine-tuning export CLI."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from code.shukketsu.evals.export import TrainingExporter


def _mock_trace(input_text="What is hit cap?", output_text="9%", feedback=1.0, trajectory=0.8):
    trace = MagicMock()
    trace.input = input_text
    trace.output = output_text
    score_fb = MagicMock()
    score_fb.name = "user_feedback"
    score_fb.value = feedback
    score_traj = MagicMock()
    score_traj.name = "trajectory_precision"
    score_traj.value = trajectory
    trace.scores = [score_fb, score_traj]
    trace.observations = []
    return trace


class TestShareGPTExport:
    def test_produces_valid_jsonl(self, tmp_path: Path) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.data = [_mock_trace()]
        client.fetch_traces.return_value = resp

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(format="sharegpt", output_path=tmp_path / "out.jsonl")

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "conversations" in record
        assert record["conversations"][1]["from"] == "human"

    def test_skips_empty_output(self, tmp_path: Path) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.data = [_mock_trace(output_text=None)]
        client.fetch_traces.return_value = resp

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(format="sharegpt", output_path=tmp_path / "out.jsonl")

        assert path.read_text().strip() == ""


class TestFunctionCallingExport:
    def test_produces_valid_format(self, tmp_path: Path) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.data = [_mock_trace()]
        client.fetch_traces.return_value = resp

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(format="function_calling", output_path=tmp_path / "out.jsonl")

        lines = path.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert "messages" in record
        assert record["messages"][0]["role"] == "system"


class TestFiltering:
    def test_filters_by_feedback_score(self, tmp_path: Path) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.data = [_mock_trace(feedback=0.0), _mock_trace(feedback=1.0)]
        client.fetch_traces.return_value = resp

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(
            format="sharegpt",
            min_feedback_score=1.0,
            output_path=tmp_path / "out.jsonl",
        )

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_filters_by_trajectory(self, tmp_path: Path) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.data = [_mock_trace(trajectory=0.3), _mock_trace(trajectory=0.9)]
        client.fetch_traces.return_value = resp

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(
            format="sharegpt",
            min_trajectory_precision=0.7,
            output_path=tmp_path / "out.jsonl",
        )

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_empty_result_set(self, tmp_path: Path) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.data = []
        client.fetch_traces.return_value = resp

        exporter = TrainingExporter(langfuse_client=client)
        path = exporter.export(format="sharegpt", output_path=tmp_path / "out.jsonl")

        assert path.read_text() == ""

    def test_output_path_created(self, tmp_path: Path) -> None:
        client = MagicMock()
        resp = MagicMock()
        resp.data = [_mock_trace()]
        client.fetch_traces.return_value = resp

        nested = tmp_path / "subdir" / "out.jsonl"
        exporter = TrainingExporter(langfuse_client=client)
        # Should not fail even though subdir doesn't exist — export creates parents
        path = exporter.export(format="sharegpt", output_path=nested)
        assert path.exists()
```

Note: Fix the `export` method to create parent dirs for explicit paths too:

```python
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
```

**Verification:**
```bash
python3 -m pytest tests/unit/test_eval_export.py -v
```
Expected: 7 tests pass.

---

### Task 8: Integration Wiring

Wire everything together: update `__init__.py` exports, ensure templates exist, run full verification.

**Files:**
- Modify: `code/shukketsu/evals/__init__.py`
- Create: `code/shukketsu/web/templates/evals/` directory + templates (if not already created in Task 6)
- Verify: All imports resolve, all routes register, full test suite passes

**Evals `__init__.py`** — add public exports:

```python
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
```

**Verification steps:**

1. Lint and format:
```bash
ruff check code/shukketsu/evals/ code/shukketsu/web/routers/evals.py tests/unit/test_eval_*.py tests/unit/test_feedback.py --fix
ruff format code/shukketsu/evals/ code/shukketsu/web/routers/evals.py tests/unit/test_eval_*.py tests/unit/test_feedback.py
```

2. Type check:
```bash
python3 -m mypy code/shukketsu/evals/ code/shukketsu/web/routers/evals.py
```

3. Run all new tests:
```bash
python3 -m pytest tests/unit/test_eval_metrics.py tests/unit/test_eval_judge.py tests/unit/test_eval_dataset.py tests/unit/test_eval_runner.py tests/unit/test_feedback.py tests/unit/test_eval_routes.py tests/unit/test_eval_export.py -v
```

4. Run full suite to verify no regressions:
```bash
python3 -m pytest tests/unit/ -v
```

Expected: All ~1478 tests pass (1408 existing + ~70 new).

**Update CLAUDE.md**: Mark Phase 5 steps as COMPLETE, update test count.
