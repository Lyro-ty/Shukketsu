# Phase 2 Step 10: Integration + Phase Gate Evaluation

> Design document for Phase 2, Step 10. Validated through brainstorming session 2026-02-12.

## Overview

Step 10 is the final step of Phase 2. It produces no new features — it's about making
the pieces work as a whole and proving it against measurable quality thresholds.

The chat handler is already wired (Step 7), all four specialists are registered in the
factory, and the wiki UI, freshness, and backup systems are operational (Steps 8-9). What
remains is building an evaluation harness that measures the three phase gate metrics
(RAG faithfulness, agent trajectory precision, domain accuracy) against a curated dataset
of WoW TBC Rogue questions, plus integration tests that verify the composed system works
end-to-end.

The architecture is two-layered: pure scoring functions in `evals/metrics.py` (unit-testable,
no LLM needed) and an eval runner in `evals/phase2_gate.py` that calls Llama 70B as judge
and feeds results into the scoring functions. A small `evals/judge.py` module handles the
LLM-as-judge calls (claim extraction, faithfulness judgment, accuracy scoring).

Integration tests are organized in three tiers: wiring smoke tests (mocked LLM, CI-safe),
live agent tests (real Ollama), and the formal phase gate eval run.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Eval architecture | Two-layer (metrics + judge/runner) | Scoring functions are pure and unit-testable; LLM calls isolated to judge module |
| Judge model | Llama 70B (REASONING) | Qwen 4B too weak for nuanced faithfulness/accuracy judgments; self-preference bias acceptable for phase gate |
| Eval dataset size | 30 questions | Sufficient for Phase 2 coverage; 50+ would over-engineer given small KB |
| Faithfulness method | Extracted claims, then per-claim judgment | More granular diagnostics than whole-answer scoring; matches RAGAS methodology |
| Trajectory capture | `trajectory` field on AgentResult | Clean, always available; avoids monkey-patching agent internals |
| Wiki coverage check | Soft report (informational) | Hard gate on article counts would block iterative eval during KB population |
| Seed data for integration tests | Ingest-based (real pipeline + embedder) | Tier 2/3 tests already require Ollama; real embeddings matter for RAG faithfulness |
| Integration test tiers | 3 tiers (wiring/live/gate) | Wiring tests are CI-safe; live tests need Ollama; gate tests are formal evaluation |
| UNCLEAR claim handling | Excluded from faithfulness denominator | Neither penalize nor reward — conservative approach |

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `code/shukketsu/evals/metrics.py` | Pure scoring functions for all three phase gate metrics |
| `code/shukketsu/evals/judge.py` | LLM-as-judge calls: claim extraction, faithfulness, accuracy |
| `code/shukketsu/evals/phase2_gate.py` | Eval runner: loads dataset, runs questions, produces PhaseGateReport |
| `code/shukketsu/evals/datasets/phase2_questions.json` | 30 curated eval questions with ground truth |
| `code/shukketsu/evals/datasets/seed_content/combat-basics.md` | Seed content: Combat Rogue spec overview |
| `code/shukketsu/evals/datasets/seed_content/assassination-basics.md` | Seed content: Assassination/Mutilate spec overview |
| `code/shukketsu/evals/datasets/seed_content/subtlety-basics.md` | Seed content: Subtlety Rogue spec overview |
| `tests/unit/test_eval_metrics.py` | Unit tests for scoring functions |
| `tests/unit/test_eval_judge.py` | Unit tests for judge module (mocked LLM) |
| `tests/integration/test_wiring.py` | Tier 1: composition smoke tests (mocked LLM) |
| `tests/integration/test_live_agents.py` | Tier 2: live agent tests (real Ollama) |
| `tests/integration/test_phase2_gate.py` | Tier 3: phase gate threshold tests |
| `tests/integration/conftest.py` | Integration test fixtures (seeded_db, etc.) |

### Modified Files

| File | Change |
|------|--------|
| `code/shukketsu/agents/tasks.py` | Add `ToolCallRecord` model and `trajectory` field to `AgentResult` |
| `code/shukketsu/agents/base.py` | Populate `trajectory` from scratchpad in `execute()` |
| `code/shukketsu/agents/researcher.py` | Populate `trajectory` in overridden `execute()` |
| `code/shukketsu/agents/writer.py` | Populate `trajectory` in overridden `execute()` |
| `code/shukketsu/agents/editor.py` | Populate `trajectory` in overridden `execute()` |
| `code/shukketsu/agents/orchestrator.py` | Populate `trajectory` with dispatch records in `execute()` |

### Unchanged Files

| File | Why |
|------|-----|
| `web/routers/chat.py` | Already fully wired in Step 7 — no changes needed |
| `agents/factory.py` | Already creates all four specialists — no changes needed |
| `routing/router.py` | Already classifies TRIVIAL/MODERATE/COMPLEX — no changes needed |
| `llm/structured.py` | Judge module reuses existing `get_structured_output` |
| `tools/registry.py` | No changes |
| `rag/search.py` | No changes |
| `web/app.py` | No changes |

## Component 1: Eval Metrics (`evals/metrics.py`)

Pure scoring functions — no LLM calls, fully unit-testable.

### Models

```python
class FaithfulnessJudgment(StrEnum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    UNCLEAR = "unclear"

class ClaimFaithfulness(BaseModel):
    claim: str
    judgment: FaithfulnessJudgment
    evidence_snippet: str | None = None

class TrajectoryScore(BaseModel):
    precision: float          # useful_calls / total_calls
    expected_tools_hit: int   # how many expected tools were used
    wasted_calls: int         # repeated or irrelevant calls

class EvalQuestionResult(BaseModel):
    question_id: str
    faithfulness: float       # 0.0 - 1.0
    trajectory_precision: float
    domain_accuracy: float    # 0.0 - 1.0 from judge
    claims: list[ClaimFaithfulness]
    tool_calls: list[str]

class PhaseGateReport(BaseModel):
    avg_faithfulness: float
    avg_trajectory_precision: float
    avg_domain_accuracy: float
    passed: bool              # all three above threshold
    wiki_coverage: dict[str, int]  # spec -> article count (soft report)
    question_results: list[EvalQuestionResult]
```

### Scoring Functions

```python
FAITHFULNESS_THRESHOLD = 0.8
TRAJECTORY_THRESHOLD = 0.7
ACCURACY_THRESHOLD = 0.7

def compute_faithfulness(claims: list[ClaimFaithfulness]) -> float:
    """Compute faithfulness score from claim judgments.

    SUPPORTED / (SUPPORTED + NOT_SUPPORTED). UNCLEAR claims excluded
    from the denominator (they don't penalize but don't help).
    Returns 1.0 if no scoreable claims.
    """

def compute_trajectory_precision(
    actual_calls: list[str],
    expected_tools: list[str],
) -> float:
    """Compute trajectory precision.

    Fraction of actual calls that match an expected tool.
    Repeated identical consecutive calls after the first count as wasted
    (precision penalty). Returns 1.0 if no tool calls were made
    (trivial queries).
    """

def compute_phase_gate(
    results: list[EvalQuestionResult],
) -> PhaseGateReport:
    """Aggregate per-question scores into a PhaseGateReport.

    Averages each metric across all questions. Sets passed=True only
    if all three averages exceed their thresholds.
    """
```

### Edge Cases

- Zero claims → faithfulness = 1.0 (nothing to contradict)
- Zero tool calls → trajectory = 1.0 (trivial query answered directly)
- All claims UNCLEAR → faithfulness = 1.0 (no scoreable claims)
- Empty results list → all metrics 0.0, passed=False

## Component 2: Eval Judge (`evals/judge.py`)

LLM-as-judge calls using Llama 70B via `get_structured_output`.

### Functions

```python
async def extract_claims(answer: str) -> list[str]:
    """Extract verifiable factual claims from an answer.

    Instructs the LLM to extract specific, verifiable statements only.
    Skips opinions, hedging, meta-commentary, and structural text.
    """

async def judge_faithfulness(
    claims: list[str],
    evidence: list[str],
) -> list[ClaimFaithfulness]:
    """Judge whether each claim is supported by the evidence.

    All claims for one answer are batched into a single LLM call.
    Returns SUPPORTED, NOT_SUPPORTED, or UNCLEAR for each claim
    with the most relevant evidence snippet.
    """

async def judge_domain_accuracy(
    answer: str,
    ground_truth: str,
    key_facts: list[str],
) -> float:
    """Score 0.0-1.0: does the answer contain the key facts?

    Checks for presence of key facts, absence of contradictions,
    and overall alignment with ground truth. Returns the score.
    """
```

### Internal Schemas

```python
class _ClaimExtraction(BaseModel):
    claims: list[str]

class _SingleVerdict(BaseModel):
    claim: str
    judgment: FaithfulnessJudgment
    evidence_snippet: str | None = None

class _FaithfulnessVerdict(BaseModel):
    judgments: list[_SingleVerdict]

class _AccuracyScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
```

### System Prompts

Each judge function has a focused system prompt:

- **Claim extraction**: "Extract only verifiable factual statements. Skip opinions,
  hedging language ('might', 'could'), structural text ('In this section...'), and
  meta-commentary. Each claim should be a standalone statement that can be checked
  against evidence."

- **Faithfulness**: "For each claim, determine if the provided evidence supports it.
  SUPPORTED: evidence directly states or strongly implies the claim. NOT_SUPPORTED:
  evidence contradicts or does not mention the claim. UNCLEAR: evidence is ambiguous
  or tangentially related."

- **Domain accuracy**: "You are a WoW TBC Rogue expert. Score how well the answer
  matches the ground truth. Check each key fact: is it present and correct? Are there
  contradictions? A score of 1.0 means all key facts present and no errors. A score
  of 0.0 means completely wrong or missing all key facts."

### Error Handling

If a judge call fails (LLM timeout, structured output validation error), the function
logs the error and returns a conservative default:
- `extract_claims` → empty list (faithfulness = 1.0, no penalty)
- `judge_faithfulness` → all UNCLEAR (faithfulness = 1.0, no penalty)
- `judge_domain_accuracy` → 0.0 (assumes wrong on failure)

This ensures the eval harness completes even with intermittent LLM issues.

## Component 3: Eval Runner (`evals/phase2_gate.py`)

### Entry Point

```python
async def run_phase_gate(
    db_path: Path | None = None,
    dataset_path: Path | None = None,
) -> PhaseGateReport:
    """Run the full phase gate evaluation.

    Loads the dataset, creates agents, runs each question through
    the live system, judges results, and produces a PhaseGateReport.
    """
```

### Flow Per Question

1. Load question from dataset
2. Route through `classify_query()` — log whether complexity matches expected tier
3. Create the appropriate agent via `AgentFactory` and call `execute()`
4. Extract trajectory from `result.trajectory`
5. Call `extract_claims(result.output)` → list of claims
6. Call `judge_faithfulness(claims, result.evidence)` for each claim
7. Call `judge_domain_accuracy(result.output, ground_truth, key_facts)`
8. Compute per-question scores via `metrics.py` functions
9. Aggregate into `PhaseGateReport`

### Wiki Coverage

Checked separately via `KnowledgeManager.list_articles(status=PUBLISHED)`,
grouped by spec. Reported in `PhaseGateReport.wiki_coverage` as informational.

### CLI Interface

```python
if __name__ == "__main__":
    # python3 -m code.shukketsu.evals.phase2_gate
    report = asyncio.run(run_phase_gate())
    _print_report(report)
    sys.exit(0 if report.passed else 1)
```

Output format:

```
Phase 2 Gate Evaluation
=======================
Questions evaluated: 30

RAG Faithfulness:        0.83  (threshold: 0.80)  PASS
Trajectory Precision:    0.75  (threshold: 0.70)  PASS
Domain Accuracy:         0.72  (threshold: 0.70)  PASS

Wiki Coverage (informational):
  combat:        3 articles
  assassination: 1 article
  subtlety:      0 articles

Per-question breakdown:
  q01  faith=0.90  traj=1.00  acc=0.85  [TRIVIAL]  How much energy does SS cost?
  q02  faith=0.80  traj=0.75  acc=0.70  [MODERATE]  What is the hit cap for...
  ...

Result: PASSED (3/3 metrics above threshold)
```

### Error Handling

If a question fails entirely (agent crashes, LLM timeout), it gets scored as
`faithfulness=0, trajectory=0, accuracy=0` with the error logged. The eval
doesn't abort — it reports partial results.

## Component 4: AgentResult Trajectory Field

### New Model in `agents/tasks.py`

```python
class ToolCallRecord(BaseModel):
    """Record of a single tool call made during agent execution."""

    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
```

### Modified AgentResult

```python
class AgentResult(BaseModel):
    task_id: str
    agent_role: AgentRole
    status: TaskStatus
    output: str
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    trajectory: list[ToolCallRecord] = Field(default_factory=list)  # NEW
```

### Population Points

**BaseAgent.execute()** — extract from scratchpad:
```python
trajectory = [
    ToolCallRecord(tool_name=entry["tool_name"], tool_input=entry["tool_input"])
    for entry in outcome.scratchpad
]
```

**Specialist agents** (Researcher, Writer, Editor) — same extraction in their
overridden `execute()` methods, which already have access to the scratchpad via
`_run_loop`.

**Orchestrator** — trajectory represents dispatched subtasks, not tool calls:
```python
ToolCallRecord(
    tool_name=f"dispatch:{subtask.agent_role}",
    tool_input={"description": subtask.description},
)
```

## Component 5: Eval Dataset

### Format

```json
{
  "id": "q01",
  "question": "How much energy does Sinister Strike cost?",
  "complexity": "TRIVIAL",
  "category": "RETRIEVAL",
  "ground_truth": "Sinister Strike costs 45 energy at max rank. With 2/2 Improved Sinister Strike it costs 40 energy, but most PvE builds skip this talent.",
  "expected_tools": [],
  "key_facts": ["45 energy", "40 with Improved Sinister Strike"],
  "spec": "combat"
}
```

### Questions (30 total)

**TRIVIAL (8 questions)** — direct factual answers, no tool calls expected:

| ID | Question | Key Facts | Spec |
|----|----------|-----------|------|
| q01 | How much energy does Sinister Strike cost? | 45 energy; 40 with Improved SS | combat |
| q02 | How many combo points does Mutilate generate? | 2 base; up to 4 with Seal Fate double crit | assassination |
| q03 | What weapon type does Mutilate require? | Daggers in both main hand and off hand | assassination |
| q04 | What is the cooldown of Blade Flurry? | 2 minutes; 15 second duration; 25 energy cost | combat |
| q05 | What does Slice and Dice do? | Increases attack speed by 30% | general |
| q06 | What is the base energy regeneration rate for rogues? | 20 energy per 2 seconds (10 per second) | general |
| q07 | How much energy does Hemorrhage cost? | 35 energy | subtlety |
| q08 | What is Cold Blood's cooldown? | 3 minutes; guarantees next offensive ability crits | assassination |

**MODERATE (12 questions)** — need KB search, single-strategy retrieval:

| ID | Question | Key Facts | Expected Tools | Spec |
|----|----------|-----------|----------------|------|
| q09 | What is the hit cap for a dual-wield combat rogue? | 9% total (142 rating); 4% from gear with Precision (64 rating); DW penalty only affects white hits | rag_search | combat |
| q10 | What trinkets are best in slot for a combat rogue in Phase 1? | Dragonspine Trophy from Gruul; Bloodlust Brooch from badges | rag_search | combat |
| q11 | What is the stat priority for combat swords? | Expertise (to 26) > hit > agility > haste > AP > crit | rag_search | combat |
| q12 | Why is off-hand weapon speed important for combat rogues? | Faster OH = more Combat Potency procs (20% chance per OH hit for 15 energy); 1.4 speed preferred | rag_search | combat |
| q13 | How does Expose Armor compare to Sunder Armor? | Improved EA (2/2) = 3075 armor reduction vs Sunder 5 stacks = 2600; mutually exclusive | rag_search | general |
| q14 | What does the Mongoose enchant proc? | +120 agility and +30 haste rating for 15 seconds; ~1 PPM; dual mongoose stacks | rag_search | general |
| q15 | What poisons should a Mutilate rogue use in PvE? | Instant Poison MH, Deadly Poison OH; or DP OH with Windfury Totem on MH | rag_search | assassination |
| q16 | When does Mutilate become competitive with Combat in TBC? | Phase 3+ with T6 gear; consistently trails combat but gap narrows | rag_search | assassination |
| q17 | What is the expertise soft cap for rogues? | 26 expertise skill (~103 rating); removes 6.5% boss dodge; Weapon Expertise talent gives 10 free | rag_search | general |
| q18 | What does Combat Potency do and why is it important? | 20% chance on successful OH melee to gain 15 energy; why fast OH matters | rag_search | combat |
| q19 | How does Envenom work in TBC? | Consumes Deadly Poison stacks (1 per CP); Nature damage ignores armor; AP scaling | rag_search | assassination |
| q20 | What are the most important raid buffs for rogues? | Windfury Totem (20% extra attack), Blessing of Might (+220 AP), Leader of the Pack (+5% crit) | rag_search | general |

**COMPLEX (10 questions)** — multi-step analysis or article generation:

| ID | Question | Key Facts | Expected Tools | Spec |
|----|----------|-----------|----------------|------|
| q21 | Compare combat swords vs mutilate for Phase 1 raiding | Combat superior in P1; weapon availability; stat priority differences; Combat Potency vs Seal Fate | rag_search, graph_search | general |
| q22 | Explain the combat rogue rotation priority system | SnD #1 priority (100% uptime); Rupture at 5 CP; SS filler; cooldown stacking (BF + AR + Haste Pot) | rag_search | combat |
| q23 | Explain the dual-wield hit table and how it affects rogue gearing | Yellow vs white hit tables; 9% vs 28% miss; DW penalty on autos only; Precision talent; hit valuable beyond cap for combat | rag_search | general |
| q24 | Compare all three rogue specs for PvE in TBC | Combat dominant; Mutilate viable P3+; Subtlety weakest PvE (Hemo debuff utility); structural DPS reasons | rag_search, graph_search | general |
| q25 | What is the optimal consumable setup for a combat rogue in a 25-man raid? | Haste Potion; Flask of Relentless Assault; Warp Burger; Adamantite Weightstone; Mongoose enchant | rag_search | combat |
| q26 | How should a rogue gear for expertise and hit rating, and what are the caps? | Expertise cap 26 (103 rating); special hit cap 64 rating with Precision; hit valuable beyond cap for combat; less for mutilate | rag_search, graph_search | general |
| q27 | How does Combat Potency interact with weapon speed and why does it matter for gearing? | OH speed affects proc rate; 1.4s > 1.8s for procs; Latro's Shifting Sword preferred; energy economy | rag_search | combat |
| q28 | What makes Dragonspine Trophy the best trinket and how long does it stay relevant? | +325 haste proc (~20.6% haste); ~1 PPM; drops from Gruul; remains competitive deep into TBC | rag_search, graph_search | combat |
| q29 | Explain Mutilate rogue poison mechanics and how they affect DPS | DP stacking; Envenom consumption; Vile Poisons +20%; IP vs DP setup; Nature damage bypasses armor | rag_search | assassination |
| q30 | How does Hemorrhage compare to Sinister Strike for PvE? | Hemo 35 energy / 110% weapon damage vs SS 45 energy / weapon + 98; Hemo debuff +42 physical x10 charges; lower personal DPS | rag_search | subtlety |

### Spec Distribution

- Combat: 10 questions (q01, q04, q09-q12, q18, q22, q25, q27-q28)
- Assassination: 7 questions (q02, q03, q08, q15, q16, q19, q29)
- Subtlety: 3 questions (q07, q30, partial q24)
- General: 10 questions (q05, q06, q13, q14, q17, q20, q21, q23, q24, q26)

This distribution reflects the real-world importance of each spec in TBC PvE (combat
dominant, assassination viable, subtlety niche).

## Component 6: Seed Content

Three Markdown files in `evals/datasets/seed_content/`, ingested by the integration
test fixture via the real ingest pipeline.

### `combat-basics.md` (~400 words)

Covers: Combat Swords overview, stat priority (expertise > hit > agi), hit cap mechanics
(9% / 142 rating, 4% / 64 with Precision), rotation (SnD uptime > Rupture > SS), key
talents (Combat Potency, Surprise Attacks, Blade Flurry, Adrenaline Rush), Phase 1 BiS
trinkets (DST from Gruul, Bloodlust Brooch from badges), weapon speed importance
(fast OH for Combat Potency), Mongoose enchant stats, consumables (Haste Potion,
Flask of Relentless Assault, Warp Burger).

### `assassination-basics.md` (~400 words)

Covers: Mutilate spec overview, weapon requirement (daggers both hands), energy cost
(60), combo points (2 base, 4 with double crit + Seal Fate), poison setup (IP MH / DP OH),
Deadly Poison mechanics (30% proc, 5 stacks, 180 Nature over 12s), Envenom (consumes
DP stacks, Nature damage, AP scaling), key talents (Find Weakness, Seal Fate, Cold Blood,
Vile Poisons), competitive timing (Phase 3+ with T6), stat priority differences from combat
(hit less valuable beyond cap, no Combat Potency).

### `subtlety-basics.md` (~300 words)

Covers: Subtlety PvE niche, Hemorrhage (35 energy, 110% weapon damage, +42 physical
debuff x10 charges), why lower DPS (no AR/BF, Hemo < SS damage), Shadowstep (25 yd
teleport, +20% damage buff), Preparation (resets CDs), key talents (Serrated Blades,
Sinister Calling, Deadliness), Hemo rotation (SnD > Rupture > Hemo > Evis), when viable
(T4/T5 where DPS checks are lenient).

### Seeding Fixture

```python
@pytest.fixture
async def seeded_db(test_db):
    """Ingest seed content files into a test database.

    Requires Ollama running (for nomic-embed-text embeddings).
    """
    from code.shukketsu.ingest.embedder import get_embedder
    from code.shukketsu.ingest.pipeline import IngestPipeline

    embedder = get_embedder()
    pipeline = IngestPipeline(conn=test_db, embedder=embedder)

    seed_dir = Path(__file__).parent.parent / "code" / "shukketsu" / "evals" / "datasets" / "seed_content"
    for md_file in sorted(seed_dir.glob("*.md")):
        content = md_file.read_text()
        await pipeline.ingest(content, f"file://{md_file.name}", md_file.stem)

    return test_db
```

Lives in `tests/integration/conftest.py`. Entity extraction during ingest will
populate the knowledge graph with items, spells, stats, and spec entities from
the seed content.

## Component 7: Integration Tests

### Tier 1: Wiring Smoke Tests (`tests/integration/test_wiring.py`)

No `@pytest.mark.integration` — runs with the unit suite. All LLM calls mocked.

| Test | Verifies |
|------|----------|
| `test_factory_creates_all_roles` | AgentFactory produces a valid agent for each AgentRole |
| `test_factory_researcher_has_correct_tools` | Researcher gets rag_search, graph_search, web_search, web_ingest |
| `test_factory_editor_has_search_tools_only` | Editor gets rag_search + graph_search, no web tools |
| `test_factory_writer_receives_knowledge_manager` | Writer constructor accepts KM without error |
| `test_factory_orchestrator_receives_factory_and_km` | Orchestrator gets both factory and KM |
| `test_chat_handler_wires_researcher_for_moderate` | Full chat.py path with mocked classify + mocked agent |
| `test_chat_handler_wires_orchestrator_for_complex` | Same but COMPLEX routing |
| `test_orchestrator_dispatches_to_real_factory` | Orchestrator._dispatch creates agents via factory (mocked LLM, real factory) |
| `test_researcher_result_includes_trajectory` | execute() returns AgentResult with populated trajectory |
| `test_orchestrator_result_includes_dispatch_trajectory` | Orchestrator trajectory shows dispatch records |
| `test_error_propagation_agent_to_chat` | ShukketsuError from agent surfaces as WebSocket error message |
| `test_error_propagation_unexpected_to_chat` | Non-Shukketsu exception surfaces as generic error |

~12 tests.

### Tier 2: Live Agent Tests (`tests/integration/test_live_agents.py`)

Marked `@pytest.mark.e2e`. Real Ollama, real DB with seed data.

| Test | Verifies |
|------|----------|
| `test_researcher_answers_from_seeded_content` | Researcher finds facts from seed content, returns non-empty output |
| `test_router_classifies_trivial_question` | Qwen 4B returns TRIVIAL for simple factual question |
| `test_router_classifies_complex_question` | Qwen 4B returns COMPLEX for multi-part question |
| `test_orchestrator_decomposes_and_dispatches` | Full orchestrator flow produces OrchestratorResult with specialist_results |
| `test_article_workflow_produces_draft` | Research -> Write -> Edit produces a draft article file |

~5 tests.

### Tier 3: Phase Gate (`tests/integration/test_phase2_gate.py`)

Marked `@pytest.mark.e2e`. Runs the full eval harness.

| Test | Verifies |
|------|----------|
| `test_faithfulness_above_threshold` | avg_faithfulness >= 0.8 |
| `test_trajectory_precision_above_threshold` | avg_trajectory_precision >= 0.7 |
| `test_domain_accuracy_above_threshold` | avg_domain_accuracy >= 0.7 |

~3 tests.

## Risks and Unknowns

| Risk | Impact | Mitigation |
|------|--------|------------|
| Self-preference bias in Llama 70B judge | Inflated faithfulness/accuracy scores | Acceptable for Phase 2 gate; revisit with external judge model in Phase 4 |
| Seed content too small for meaningful RAG | Low faithfulness scores (not enough evidence retrieved) | Seed files crafted to contain exact facts the eval questions reference |
| Eval questions reference facts not in seed content | Accuracy scores unfairly penalized | All key_facts in questions map to content in seed files |
| Ollama latency on 30 questions + judge calls | Eval run takes 30+ minutes | Acceptable for a phase gate check; not a CI job |
| Entity extraction flaky during seed ingest | Knowledge graph empty, graph_search returns nothing | Graph search failures fall through to rag_search; scores degraded but eval still runs |
| Wiring tests overlap with existing unit tests | Redundant test maintenance | Small overlap is intentional — wiring tests focus on composition, unit tests on isolation |

## What This Does NOT Include

- **External judge model** — Using a different model (e.g., GPT-4) as judge to avoid self-preference bias. Deferred to Phase 4 eval polish.
- **Automated eval CI pipeline** — The eval harness is manual/on-demand. Deferred to Phase 4.
- **Eval dashboard / visualization** — CLI text output only. Deferred to Phase 4.
- **A/B comparison tooling** — No before/after metric comparison. Deferred to Phase 4.
- **Per-question regression tracking** — No historical tracking of scores across runs. Deferred to Phase 4.
- **Scheduler for freshness/backup** — APScheduler integration mentioned in original plan. The API endpoints exist; periodic scheduling is a deployment concern, not a code concern.
- **Chat handler changes** — Already fully wired in Step 7. No modifications needed.
- **Factory changes** — Already creates all four specialists. No modifications needed.

## Dependencies

- **All Phase 2 Steps 1-9 complete** (690 tests passing)
- **Ollama running** with Llama 3.3 70B, Qwen3 4B (qwen3-router), nomic-embed-text
- **Existing modules**: `get_structured_output`, `AgentFactory`, `KnowledgeManager`, `IngestPipeline`, `classify_query`
- **Existing test infrastructure**: `test_db` fixture, `conftest.py` circuit breaker reset

## Test Plan

### Unit Tests (new)

| File | Tests | Count |
|------|-------|-------|
| `tests/unit/test_eval_metrics.py` | compute_faithfulness (all SUPPORTED, mixed, all NOT_SUPPORTED, all UNCLEAR, empty), compute_trajectory_precision (exact match, wasted calls, no expected tools, empty), compute_phase_gate (all passing, one failing, empty) | ~15 |
| `tests/unit/test_eval_judge.py` | extract_claims (mocked LLM returns claims), judge_faithfulness (mocked verdicts), judge_domain_accuracy (mocked score), error fallbacks for each | ~8 |
| `tests/unit/test_trajectory.py` | ToolCallRecord model validation, AgentResult with trajectory, BaseAgent populates trajectory, specialist agents populate trajectory, Orchestrator dispatch trajectory | ~8 |

### Integration Tests (new)

| File | Tests | Count |
|------|-------|-------|
| `tests/integration/test_wiring.py` | Tier 1 wiring smoke tests | ~12 |
| `tests/integration/test_live_agents.py` | Tier 2 live agent tests (@pytest.mark.e2e) | ~5 |
| `tests/integration/test_phase2_gate.py` | Tier 3 phase gate threshold (@pytest.mark.e2e) | ~3 |

### Summary

Starting count: **690 tests**

New unit tests: ~31 (metrics + judge + trajectory)
New integration tests (tier 1, CI-safe): ~12

Estimated after this step: **~733 tests** (unit + tier 1 integration)

Tier 2 and Tier 3 tests (~8) are `@pytest.mark.e2e` and won't run in the default
`python3 -m pytest tests/unit/` invocation. They run separately with
`python3 -m pytest tests/integration/ -m e2e`.
