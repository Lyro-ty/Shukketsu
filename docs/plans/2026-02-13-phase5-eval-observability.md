# Phase 5: Evaluation + Observability Polish

> Design document for Phase 5. Validated through brainstorming session 2026-02-13.

## Overview

Phase 5 adds a production-grade evaluation pipeline, user feedback collection, and a fine-tuning
data export workflow. All evaluation data flows through Langfuse — we build no custom trace
storage or trace viewer, relying instead on Langfuse's native dashboard for drill-down and the
Langfuse Datasets API for structured eval runs.

The existing eval infrastructure from Phase 2 Step 10 (metrics.py, judge.py, phase2_gate.py)
provides a solid foundation. This phase extends it with: a three-tier 60-question dataset that
covers retrieval, reasoning, and simulation; a new answer-relevancy metric; a Langfuse-integrated
runner that records per-question scores as dataset run items; a thumbs-up/down feedback widget in
the chat UI; a lean in-app dashboard showing aggregate scores and history charts; and a CLI
exporter that produces JSONL training files from positively-scored traces.

The phase gate is: the full 60-question eval suite runs end-to-end, scores are visible in both
the in-app dashboard and Langfuse, user feedback attaches to traces, and the export CLI produces
valid JSONL files filtered by feedback score.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Eval data storage | Langfuse-primary (no new SQLite tables) | Already deployed and tracing; native Dataset/Run/Score APIs; avoids duplicating trace storage |
| Trace viewer | Langfuse dashboard at :3000 | Building our own would take weeks; Langfuse's is production-quality |
| Eval dataset | 3 tiers, ~60 questions, local JSON synced to Langfuse | Tiers map to system capabilities (retrieval, reasoning, sim); local JSON is version-controlled |
| Feedback mechanism | Thumbs up/down only | Minimal UI, maximum signal; free-text comments deferred to Phase 6 |
| Answer relevancy | Custom LLM judge (not Ragas library) | Ragas may not work on ARM64; we already have judge infrastructure; same LLM-as-judge pattern |
| Dashboard scope | Summary card + history chart + run trigger | Lean; links to Langfuse for detail drill-down |
| Fine-tuning export | CLI-only, JSONL output | Not fine-tuning until Phase 6; CLI is sufficient for manual review |
| Sim accuracy judging | Numeric range comparison (no LLM) | DPS values are deterministic; range check is faster and more reliable |
| Backward compatibility | Keep phase2_gate.py as-is | CLI still works for quick spot-checks; new runner is the primary path |

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `code/shukketsu/evals/dataset.py` | Langfuse dataset manager — sync questions from JSON, create/list runs |
| `code/shukketsu/evals/runner.py` | Execute eval questions against live system, record scores to Langfuse |
| `code/shukketsu/evals/export.py` | CLI to export positively-scored traces as JSONL for fine-tuning |
| `code/shukketsu/web/routers/evals.py` | Dashboard routes — summary, history, run trigger |
| `code/shukketsu/web/templates/evals/index.html` | Dashboard page with summary cards and Chart.js history |
| `code/shukketsu/web/templates/evals/partials/run_status.html` | HTMX fragment for run progress polling |
| `code/shukketsu/web/templates/evals/partials/summary_card.html` | HTMX fragment for latest run summary |
| `code/shukketsu/web/static/js/eval-charts.js` | Chart.js rendering for eval history |
| `datasets/eval_questions.json` | 60-question three-tier eval dataset |
| `tests/unit/test_eval_dataset.py` | Tests for dataset manager |
| `tests/unit/test_eval_runner.py` | Tests for eval runner |
| `tests/unit/test_eval_export.py` | Tests for fine-tuning export |
| `tests/unit/test_eval_routes.py` | Tests for dashboard routes |
| `tests/unit/test_feedback.py` | Tests for chat feedback handling |

### Modified Files

| File | Change |
|------|--------|
| `code/shukketsu/evals/judge.py` | Add `judge_answer_relevancy()` and `judge_sim_accuracy()` |
| `code/shukketsu/evals/metrics.py` | Add `answer_relevancy` to `EvalQuestionResult` and `PhaseGateReport`; add `compute_answer_relevancy()` |
| `code/shukketsu/web/routers/chat.py` | Handle `{"type": "feedback"}` messages; include `trace_id` in "done" response |
| `code/shukketsu/web/app.py` | Register evals router |
| `code/shukketsu/web/templates/base.html` | Add "Evals" nav link |
| `code/shukketsu/web/templates/chat.html` | Add thumbs up/down buttons after agent response |
| `code/shukketsu/config.py` | Add eval config constants |
| `code/shukketsu/evals/__init__.py` | Public exports for new modules |

### Unchanged Files

These files are related but will NOT be touched (prevents scope creep):

- `code/shukketsu/observability/tracer.py` — Already correctly configured
- `code/shukketsu/evals/phase2_gate.py` — Kept for backward compatibility
- `code/shukketsu/agents/` — No agent changes needed
- `code/shukketsu/tools/` — No tool changes needed
- `code/shukketsu/rag/` — No RAG changes needed
- `code/shukketsu/sim/` — No sim changes needed (runner invoked through existing public API)
- `code/shukketsu/db/schema.sql` — No new SQLite tables (Langfuse stores everything)

## Component 1: Config Extensions

New constants in `config.py`:

```python
# Eval
EVAL_DATASET_NAME = "shukketsu-eval-v1"
EVAL_DATASET_PATH = Path(os.getenv("EVAL_DATASET_PATH", "/project/datasets/eval_questions.json"))
EVAL_CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "1"))  # Sequential on GB10
EVAL_SIM_TOLERANCE_PCT = float(os.getenv("EVAL_SIM_TOLERANCE_PCT", "5.0"))  # 5% range for sim accuracy
EXPORT_OUTPUT_DIR = Path(os.getenv("EXPORT_OUTPUT_DIR", "/project/data/scratch/exports/"))
EXPORT_MIN_FEEDBACK_SCORE = float(os.getenv("EXPORT_MIN_FEEDBACK_SCORE", "1.0"))  # Only thumbs-up
RELEVANCY_THRESHOLD = float(os.getenv("RELEVANCY_THRESHOLD", "0.7"))
```

No new SQLite tables.

## Component 2: Metrics Extensions

Extend `evals/metrics.py` with answer relevancy support.

**New model field** on `EvalQuestionResult`:

```python
@dataclass
class EvalQuestionResult:
    question_id: str
    faithfulness: float
    answer_relevancy: float        # NEW
    trajectory_precision: float
    domain_accuracy: float
    claims: list[ClaimFaithfulness]
    tool_calls: list[str]
    tier: str                      # NEW: "retrieval", "reasoning", "simulation"
```

**New field** on `PhaseGateReport`:

```python
@dataclass
class PhaseGateReport:
    avg_faithfulness: float
    avg_answer_relevancy: float    # NEW
    avg_trajectory_precision: float
    avg_domain_accuracy: float
    passed: bool
    wiki_coverage: dict[str, int]
    question_results: list[EvalQuestionResult]
    tier_breakdown: dict[str, dict[str, float]]  # NEW: per-tier averages
```

**New function**:

```python
def compute_answer_relevancy(relevancy_score: float) -> float:
    """Pass-through — the LLM judge returns a 0.0-1.0 score directly."""
    return max(0.0, min(1.0, relevancy_score))
```

**Updated `compute_phase_gate()`**: Includes `avg_answer_relevancy` in the pass/fail
check (must meet `RELEVANCY_THRESHOLD`). Adds `tier_breakdown` grouping results by tier.

## Component 3: Judge Extensions

Extend `evals/judge.py` with two new functions.

### `judge_answer_relevancy(question: str, answer: str) -> float`

Uses Llama 70B to score how well the answer addresses the specific question asked.

- **Input**: The original question and the agent's answer
- **Output**: 0.0–1.0 float (1.0 = perfectly relevant)
- **Internal schema**: `_RelevancyScore(score: float, reasoning: str)`
- **System prompt**: "You are evaluating whether an AI assistant's answer directly addresses
  the user's question. Score 1.0 if the answer fully addresses the question, 0.5 if partially
  relevant, 0.0 if completely off-topic. Consider: Does it answer what was asked? Does it stay
  on topic? Is the level of detail appropriate?"
- **Failure mode**: Returns 0.0 (conservative, flags for review)

### `judge_sim_accuracy(actual_dps: float, expected_min: float, expected_max: float) -> float`

Pure numeric comparison for Tier 3 sim questions. No LLM call.

- **Input**: The DPS number extracted from the answer, plus the expected range from the dataset
- **Output**: 1.0 if within range, linear falloff outside range (0.0 at 2x tolerance)
- **Logic**:
  ```
  if expected_min <= actual_dps <= expected_max: return 1.0
  distance = min(abs(actual_dps - expected_min), abs(actual_dps - expected_max))
  range_size = expected_max - expected_min
  tolerance = range_size * (EVAL_SIM_TOLERANCE_PCT / 100)
  return max(0.0, 1.0 - (distance / tolerance))
  ```
- **Failure mode**: If DPS can't be extracted from answer, returns 0.0

### Helper: `extract_dps_from_answer(answer: str) -> float | None`

Regex extraction of DPS numbers from agent answers. Looks for patterns like
"1,234.5 DPS", "1234 dps", "DPS: 1234". Returns `None` if no match found.

## Component 4: Eval Dataset Manager (`evals/dataset.py`)

Manages the Langfuse Dataset lifecycle.

### Class: `EvalDatasetManager`

```python
class EvalDatasetManager:
    def __init__(self, langfuse_client: Langfuse) -> None: ...

    def sync_dataset(self, dataset_path: Path) -> int:
        """Load questions from JSON, upsert into Langfuse Dataset.

        Returns the number of items synced.
        """

    def create_run(self, run_name: str | None = None) -> str:
        """Create a new DatasetRun. Auto-generates name if not provided.

        Returns the run name (used as identifier).
        """

    def list_runs(self, limit: int = 20) -> list[dict]:
        """List recent dataset runs with aggregate scores.

        Returns list of {run_name, created_at, avg_faithfulness,
        avg_relevancy, avg_trajectory, avg_accuracy, item_count, passed}.
        """

    def get_run_summary(self, run_name: str) -> dict:
        """Get detailed summary for a specific run including per-tier breakdown."""
```

**Sync flow**:
1. Read `datasets/eval_questions.json`
2. Call `langfuse.create_dataset(name=EVAL_DATASET_NAME)` (idempotent)
3. For each question: `langfuse.create_dataset_item(dataset_name, input=question, expected_output=ground_truth, metadata={tier, complexity, spec, expected_tools, key_facts, sim_validation})`
4. Items are keyed by `question_id` — re-syncing updates existing items

**Run name format**: `run-YYYY-MM-DD-HHMMSS` (auto-generated) or user-provided.

## Component 5: Eval Runner (`evals/runner.py`)

Executes eval questions against the live agent system and records results to Langfuse.

### Class: `EvalRunner`

```python
class EvalRunner:
    def __init__(
        self,
        dataset_manager: EvalDatasetManager,
        langfuse_client: Langfuse,
    ) -> None: ...

    async def run(
        self,
        tier: str | None = None,
        run_name: str | None = None,
        on_progress: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> str:
        """Execute eval suite. Returns run_name.

        Args:
            tier: Filter to specific tier ("retrieval", "reasoning", "simulation").
                  None runs all tiers.
            run_name: Custom run name. Auto-generated if None.
            on_progress: Callback(completed, total) for progress reporting.
        """
```

**Execution flow** (per question):

1. Create agent infrastructure (researcher + orchestrator, same as chat handler)
2. Route query through `classify_query()` (same path as real chat)
3. Execute via appropriate agent (trivial → direct, moderate → researcher, complex → orchestrator)
4. The `@observe()` decorator creates a Langfuse trace automatically
5. Extract answer, trajectory, tool calls from agent result
6. Run judges:
   - `extract_claims()` → `judge_faithfulness()` → `compute_faithfulness()`
   - `judge_answer_relevancy(question, answer)`
   - `judge_domain_accuracy(answer, ground_truth, key_facts)`
   - For Tier 3: `extract_dps_from_answer()` → `judge_sim_accuracy()`
   - `compute_trajectory_precision(actual_tools, expected_tools)`
7. Record scores to Langfuse: `langfuse.score(trace_id, name="faithfulness", value=...)` for each metric
8. Link trace to dataset run: `dataset_item.link(trace, run_name)`
9. Call `on_progress(i, total)`

**Error handling**: If a question fails (agent error, judge error), record score of 0.0
for all metrics and log the error. Don't abort the run — continue to next question.

**Concurrency**: Sequential by default (`EVAL_CONCURRENCY=1`) since GB10 can only
serve one Llama 70B request at a time. Config allows bumping if hardware changes.

## Component 6: Chat Feedback (`web/routers/chat.py`)

### Trace ID in responses

Modify `_agent_response()` to include the Langfuse trace ID in the "done" message:

```python
# After computing answer:
from langfuse.decorators import langfuse_context

trace_id = langfuse_context.get_current_trace_id()
await websocket.send_json({"type": "done", "content": answer, "trace_id": trace_id})
```

The `trace_id` may be `None` if Langfuse tracing is disabled — the frontend handles this
gracefully by hiding the feedback buttons.

### Feedback message handler

Add handling for `{"type": "feedback"}` in `_handle_message()`:

```python
if msg_type == "feedback":
    trace_id = data.get("trace_id")
    score = data.get("score")  # 1 or 0
    if trace_id and score is not None:
        langfuse = get_client()
        langfuse.score(
            trace_id=trace_id,
            name="user_feedback",
            value=float(score),
            comment=f"thumbs {'up' if score else 'down'}",
        )
    return
```

No response is sent back to the client — fire-and-forget.

### Chat UI changes (`chat.html`)

After each "done" message, render a thumbs row below the agent's response:

```html
<div class="feedback-row" data-trace-id="{{ trace_id }}">
    <button class="thumb thumb-up" onclick="sendFeedback(1)">👍</button>
    <button class="thumb thumb-down" onclick="sendFeedback(0)">👎</button>
</div>
```

On click: send `{"type": "feedback", "score": 1, "trace_id": "..."}` through the WebSocket.
Disable both buttons after click (one-shot). Hide the row if `trace_id` is null.

## Component 7: Dashboard (`web/routers/evals.py`)

### Routes

```python
router = APIRouter(prefix="/evals", tags=["evals"])

@router.get("/")
async def evals_dashboard(request: Request) -> HTMLResponse:
    """Render the eval dashboard page."""

@router.post("/run")
async def trigger_eval_run(request: Request, tier: str | None = None) -> HTMLResponse:
    """Trigger an eval run. Returns HTMX fragment with run status."""

@router.get("/run/{run_name}/status")
async def run_status(run_name: str) -> HTMLResponse:
    """HTMX polling endpoint for run progress."""

@router.get("/history")
async def eval_history() -> JSONResponse:
    """Return last 20 runs as JSON for Chart.js."""

@router.get("/latest")
async def latest_summary() -> HTMLResponse:
    """HTMX fragment: summary card for latest run."""
```

### Dashboard page (`evals/index.html`)

Three sections:

1. **Run Controls** — "Run Full Suite" button + tier filter dropdown (All / Retrieval /
   Reasoning / Simulation). Triggers `POST /evals/run` via HTMX. Shows progress bar
   during execution via polling `/evals/run/{name}/status`.

2. **Latest Run Summary** — Card showing:
   - Run name + timestamp
   - Four metric gauges: faithfulness, relevancy, trajectory, accuracy
   - Pass/fail badge (green/red)
   - Per-tier breakdown table
   - Link to Langfuse: `{LANGFUSE_HOST}/datasets/{EVAL_DATASET_NAME}/runs/{run_name}`

3. **History Chart** — Chart.js line chart (reusing pattern from sim-charts.js).
   X-axis: run timestamps. Y-axis: 0.0–1.0. Four lines (one per metric).
   Horizontal dashed lines at threshold values. Fetches data from `GET /evals/history`.

### Langfuse deep links

Every run summary includes a direct link to Langfuse's dataset run view. The URL
pattern is `{LANGFUSE_HOST}/project/{project_id}/datasets/{dataset_name}/runs/{run_id}`.
The project ID is obtained once from `langfuse.auth_check()` or configured in env.

## Component 8: Fine-Tuning Export (`evals/export.py`)

### CLI interface

```bash
python3 -m code.shukketsu.evals.export \
    --format sharegpt \
    --min-score 1.0 \
    --output data/scratch/exports/training-2026-02-13.jsonl
```

### Class: `TrainingExporter`

```python
class TrainingExporter:
    def __init__(self, langfuse_client: Langfuse) -> None: ...

    def export(
        self,
        format: str = "sharegpt",
        min_feedback_score: float = 1.0,
        min_trajectory_precision: float = 0.7,
        output_path: Path | None = None,
    ) -> Path:
        """Export positively-scored traces as JSONL.

        Filters:
        - Only traces with user_feedback score >= min_feedback_score
        - Only traces with trajectory_precision >= min_trajectory_precision
        - Excludes trivial (direct-answer) traces (no learning signal)

        Returns path to written file.
        """
```

**ShareGPT format** (for chat fine-tuning):
```json
{"conversations": [
    {"from": "system", "value": "You are Shukketsu..."},
    {"from": "human", "value": "What is the hit cap for rogues?"},
    {"from": "gpt", "value": "The hit cap for..."}
]}
```

**Function-calling format** (for tool-use fine-tuning):
```json
{"messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "What is the hit cap?"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "rag_search", "arguments": "{\"query\": \"hit cap rogue\"}"}}]},
    {"role": "tool", "content": "..."},
    {"role": "assistant", "content": "The hit cap is 9%..."}
]}
```

The exporter fetches traces from Langfuse, filters by score, extracts the conversation
and tool-call structure, and writes one JSON object per line.

## Component 9: Eval Dataset Content

`datasets/eval_questions.json` — 60 questions organized in three tiers.

### Structure per question

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

For Tier 3 (simulation), the `sim_validation` field:

```json
{
    "id": "t3_q01",
    "tier": "simulation",
    "complexity": "complex",
    "category": "analysis",
    "spec": "combat",
    "question": "What DPS does a P1 BiS Combat Swords Rogue do on a Patchwerk-style fight?",
    "ground_truth": "A P1 BiS Combat Swords Rogue does approximately 1100-1250 DPS on a Patchwerk-style fight.",
    "key_facts": ["P1 BiS", "Combat Swords", "1100-1250 DPS"],
    "expected_tools": ["sim_run"],
    "sim_validation": {
        "expected_dps_min": 1100,
        "expected_dps_max": 1250
    }
}
```

### Tier breakdown

| Tier | Count | Complexity | Tests |
|------|-------|------------|-------|
| Retrieval | 20 | Trivial (8) + Moderate (12) | RAG faithfulness, answer relevancy |
| Reasoning | 20 | Moderate (8) + Complex (12) | Multi-agent coordination, trajectory, accuracy |
| Simulation | 20 | Complex (20) | Sim tool invocation, DPS accuracy, answer relevancy |

### Coverage

- All three specs (Combat, Assassination, Subtlety) + general
- Topics: hit/crit/haste caps, talents, gear (pre-raid, P1, P2), rotation, consumables,
  enchants, gems, boss-specific strategy, spec comparisons, stat weights
- Backward compatible: includes the 30 Phase 2 questions (remapped to tier/ID scheme)

## Risks and Unknowns

| Risk | Impact | Mitigation |
|------|--------|------------|
| Langfuse Dataset API rate limits | Eval runs could be throttled | Sequential execution, single concurrent run |
| Langfuse Python SDK ARM64 compatibility | SDK might have native deps | Pure Python SDK, verified working in Phase 2 |
| 60-question eval takes >2 hours on GB10 | Slow feedback loop | Tier filtering allows running 20-question subsets; trivial questions are fast |
| `langfuse_context.get_current_trace_id()` availability | Feedback can't link to trace | Graceful degradation: hide feedback buttons if trace_id is None |
| Sim DPS ranges may drift as sim engine evolves | Tier 3 accuracy scores become unreliable | Validation profiles (Phase 4) pin expected values; update dataset when sim changes |
| Langfuse is down during eval run | Run fails partway through | Catch Langfuse errors per-question, log warning, continue with 0.0 scores |
| Chart.js bundle size | Page load time | Already in project from sim UI; single shared bundle |

## What This Does NOT Include

- **Automated/scheduled eval runs** — Manual trigger only. Cron/CI deferred to Phase 6.
- **A/B testing between model versions** — Would need model version tagging in Langfuse. Not needed yet.
- **Regression detection / alerting** — Dashboard shows trends; automated alerts are Phase 6.
- **Custom trace viewer** — Langfuse's built-in viewer handles this.
- **Free-text feedback comments** — Thumbs only for now. Comments in Phase 6.
- **Fine-tuning execution** — Export only. LoRA training is Phase 6.
- **Eval on sim engine accuracy** — Covered by sim validation profiles (Phase 4 Step 12).
- **Ragas library integration** — Our custom judges implement the same concepts without the dependency.
- **New SQLite tables** — Everything stored in Langfuse.

## Dependencies

What must exist before implementation:

- **Phase 4 complete** — Sim engine, analyst agent, sim tools (all done)
- **Langfuse running** — Docker Compose stack at :3000 with configured keys
- **Existing eval modules** — `evals/metrics.py`, `evals/judge.py` (Phase 2 Step 10)
- **Existing tracing** — `@observe()` decorators, `observability/tracer.py` (Phase 2 Step 10)
- **Chart.js** — Already in `web/static/js/` from sim UI (Phase 4 Step 11)
- **HTMX** — Already in use for wiki and sim pages

## Test Plan

Starting count: **1408 tests**

| Test File | Tests | What's Covered |
|-----------|-------|----------------|
| `tests/unit/test_eval_metrics.py` | +6 | answer_relevancy field, compute_answer_relevancy, tier_breakdown, updated phase gate |
| `tests/unit/test_eval_judge.py` | +8 | judge_answer_relevancy (success, failure, edge cases), judge_sim_accuracy (in range, out of range, linear falloff, no extraction), extract_dps_from_answer (various formats) |
| `tests/unit/test_eval_dataset.py` | +10 | sync_dataset (create, upsert, idempotent), create_run (auto name, custom name), list_runs (empty, populated, ordering), get_run_summary (with tier breakdown) |
| `tests/unit/test_eval_runner.py` | +14 | run full suite, run single tier, per-question scoring, error handling (agent fail, judge fail), progress callback, score recording to Langfuse mock, trivial/moderate/complex routing, sim tier DPS extraction |
| `tests/unit/test_feedback.py` | +8 | feedback message accepted, score recorded to Langfuse, trace_id in done message, feedback without trace_id ignored, invalid score ignored, feedback buttons disabled after click, thumbs up value=1, thumbs down value=0 |
| `tests/unit/test_eval_routes.py` | +12 | dashboard page renders, trigger run returns status, run status polling, history JSON endpoint, latest summary fragment, tier filter parameter, Langfuse deep link generation, empty state handling |
| `tests/unit/test_eval_export.py` | +10 | sharegpt format output, function-calling format output, min score filtering, min trajectory filtering, trivial traces excluded, empty result set, output path creation, CLI argument parsing, JSONL validity, idempotent export |
| `tests/unit/test_chat_handler.py` | +2 | feedback handling in existing test file (feedback accepted, feedback ignored without trace_id) |

**Estimated total: 1408 + 70 = ~1478 tests**

### Step Breakdown (estimated)

1. **Config + Metrics Foundation** — Config constants, extend metrics models, scoring functions (~8 tests)
2. **Judge Extensions** — answer_relevancy + sim_accuracy + DPS extraction (~8 tests)
3. **Eval Dataset Content + Manager** — 60 questions JSON + Langfuse dataset sync (~10 tests)
4. **Eval Runner** — Execute questions, record scores (~14 tests)
5. **Chat Feedback** — Thumbs UI, feedback WS handler, trace_id in responses (~10 tests)
6. **Dashboard UI** — Routes, templates, Chart.js history (~12 tests)
7. **Fine-Tuning Export** — CLI, JSONL output, filtering (~10 tests)
8. **Integration Wiring** — Register routes, update nav, smoke test (~varies)
