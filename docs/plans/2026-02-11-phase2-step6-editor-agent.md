# Phase 2, Step 6: Editor Agent

> Design document for Phase 2, Step 6. Validated through brainstorming session 2026-02-11.

## Overview

The Editor is the third specialist agent. Like the Writer, it uses **direct verification** — it
overrides `execute()` with deterministic Python control flow rather than a ReAct loop. The Editor
receives an `EditTask` containing an article path and a list of claims to verify, then
systematically checks each claim against the knowledge base (via `rag_search`) and knowledge graph
(via `graph_search`).

For each claim, the Editor gathers evidence from both search tools, then uses a single Llama 70B
structured output call to judge verification status (VERIFIED, UNCERTAIN, CONTRADICTED, or
UNSUPPORTED). Entity names for graph search are identified by matching the claim text against the
article's existing `entity_refs` from frontmatter — zero extra LLM calls for entity extraction
since the Writer already did that work.

After verifying all claims, the Editor computes an overall article confidence score (weighted
average with contradiction penalty), updates the article frontmatter with verification results,
and conditionally transitions the article from `draft` to `review` status if confidence meets the
threshold (>= 0.6). The Editor is full-service: verification, frontmatter update, and status
transition are all part of completing the job.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Agent architecture | Direct verification (override `execute()`) | Deterministic claim-by-claim loop in Python. No ReAct loop needed — the Editor's workflow is fixed: search evidence per claim, judge, aggregate. Testable and predictable. |
| Evidence gathering | Call `tool.execute()` on RagSearchTool/GraphSearchTool | Tools already return formatted strings suitable for LLM input. No new search code needed. Consistent with the tool architecture. |
| LLM judgment | One `get_structured_output()` call per claim (Llama 70B) | Isolated judgment per claim. Each call gets the claim + its specific evidence. Slower (~3s per claim) but accurate — fact-checking needs reasoning-grade judgment. |
| Entity identification for graph search | Match claim text against article `entity_refs` (case-insensitive substring) | Writer's extraction pass already populated `entity_refs`. Reusing that data avoids extra LLM calls. Simple substring match is sufficient — entity names are normalized. |
| Side effects | Update frontmatter + conditionally transition `draft → review` | Verification IS the Editor's job. Updating claims and transitioning status completes that job. Deferring to a non-existent Orchestrator would leave state management in limbo. |
| Internal consistency | Skip for now (field defaults to `True`) | Per-claim KB verification already catches contradictions with external sources. Self-contradictions within articles are rare when claims originate from the same `ResearchResult`. |
| Claim mapping | Match by text equality | `task.claims` came from the Writer's extraction pass, which populated `meta.claims[].text`. Exact string match is deterministic and reliable. |
| Confidence threshold | 0.6 for `approved_for_review` | Matches the phase plan. Low enough to allow articles with some UNCERTAIN claims through to human review, high enough to block articles with contradictions. |
| KnowledgeManager dependency | Injected via `**kwargs` from factory | Same pattern as Writer. Factory forwards `knowledge_manager=km` to the constructor. |
| `corrections` / `needs_more_research` | Auto-derived from CONTRADICTED / UNSUPPORTED claims | No extra LLM calls. CONTRADICTED claims become corrections ("Claim X contradicted — evidence says Y"). UNSUPPORTED claims become research gaps. |

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `code/shukketsu/agents/editor.py` | `Editor` subclass with direct verification `execute()` override, `compute_article_confidence()` function |
| `code/shukketsu/llm/prompts/editor.py` | `EDITOR_SYSTEM_PROMPT` and `VERIFICATION_PROMPT` constants |
| `tests/unit/test_editor.py` | All Editor-specific tests |

### Modified Files

| File | Change |
|------|--------|
| `code/shukketsu/agents/tasks.py` | Add `VerificationStatus` enum, `ClaimVerification` model, `ClaimJudgment` model, `EditResult` class |
| `code/shukketsu/agents/factory.py` | Add `Editor` to `_ROLE_CLASSES` dict |
| `code/shukketsu/agents/__init__.py` | Export `Editor`, `EditResult`, `ClaimVerification`, `VerificationStatus` |
| `code/shukketsu/config.py` | Replace stub `EDITOR_SYSTEM_PROMPT` with import from `llm.prompts.editor`; add `EDITOR_CONFIDENCE_THRESHOLD` |

### Unchanged Files

These related files will NOT be touched (prevents scope creep):

- `agents/writer.py` — Writer is complete, produces claims consumed by Editor
- `agents/researcher.py` — Researcher is complete, no interaction with Editor
- `agents/base.py` — No changes needed; Editor overrides `execute()` but doesn't touch the ReAct loop
- `tools/knowledge/search.py` — RagSearchTool used as-is via `tool.execute()`
- `tools/knowledge/graph_search.py` — GraphSearchTool used as-is via `tool.execute()`
- `knowledge/manager.py` — KnowledgeManager API is complete (`read_article`, `update_draft`, `set_status`)
- `db/schema.sql` — No schema changes needed; articles table v3 already has all required columns
- `db/connection.py` — No migration needed
- `routing/router.py` — Routing changes deferred to Step 7 (Orchestrator)
- `web/routers/chat.py` — Chat handler changes deferred to Step 10 (Integration)
- `rag/reranker.py` — Reranker is used internally by rag_search, not directly by Editor

## Component 1: New Models in `tasks.py`

### `VerificationStatus` Enum

```python
class VerificationStatus(StrEnum):
    """Outcome of verifying a single claim against the knowledge base."""

    VERIFIED = "verified"         # 2+ independent sources agree
    UNCERTAIN = "uncertain"       # Some evidence but not conclusive
    CONTRADICTED = "contradicted" # Evidence directly conflicts
    UNSUPPORTED = "unsupported"   # No evidence found in KB
```

### `ClaimVerification` Model

```python
class ClaimVerification(BaseModel):
    """Verification result for a single claim."""

    claim: str
    status: VerificationStatus
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    note: str = ""
```

### `ClaimJudgment` Model

Used as the structured output schema for the per-claim LLM call:

```python
class ClaimJudgment(BaseModel):
    """LLM's assessment of a single claim against gathered evidence."""

    status: VerificationStatus
    confidence: float = Field(ge=0.0, le=1.0)
    supporting: list[str] = Field(default_factory=list)
    contradicting: list[str] = Field(default_factory=list)
    note: str
```

### `EditResult` Model

```python
class EditResult(AgentResult):
    """Structured output from the Editor agent."""

    article_path: str
    claim_results: list[ClaimVerification] = Field(default_factory=list)
    overall_confidence: float = 0.0
    approved_for_review: bool = False
    internal_consistency: bool = True           # Always True for now (YAGNI)
    corrections: list[str] = Field(default_factory=list)
    needs_more_research: list[str] = Field(default_factory=list)
```

## Component 2: Confidence Scoring

### Location: `code/shukketsu/agents/editor.py` (module-level function)

```python
def compute_article_confidence(claims: list[ClaimVerification]) -> float:
    """Compute overall article confidence from individual claim verifications.

    Returns the average claim confidence with a penalty for contradicted claims.
    Each contradicted claim reduces the score by 0.15.
    """
    if not claims:
        return 0.0
    base = sum(c.confidence for c in claims) / len(claims)
    contradicted = sum(1 for c in claims if c.status == VerificationStatus.CONTRADICTED)
    penalty = contradicted * 0.15
    return max(0.0, min(1.0, base - penalty))
```

This is a module-level function (not a method) for independent testability.

## Component 3: Editor System Prompt

### Location: `code/shukketsu/llm/prompts/editor.py`

Same pattern as `researcher.py` and `writer.py` — pure string constants with zero imports.

### `EDITOR_SYSTEM_PROMPT`

Full system prompt teaching the Editor how to assess evidence:

```
You are a Fact-Checking Editor for WoW: The Burning Crusade (TBC) Rogue content.
Your job is to verify claims in draft articles against evidence from the knowledge
base and knowledge graph.

## Verification Rules

For each claim, you will receive:
- The claim text
- Evidence from the knowledge base (text search results)
- Evidence from the knowledge graph (entity relationships)

Assign a verification status:
- VERIFIED: 2+ independent pieces of evidence agree with the claim
- UNCERTAIN: Some evidence supports the claim but not conclusive (1 source, or weak match)
- CONTRADICTED: Evidence directly conflicts with the claim — flag the contradiction clearly
- UNSUPPORTED: No relevant evidence found in the knowledge base

## What Counts as Evidence

NUMBERS: Stat caps, DPS values, proc rates, item stats — must match exactly
RELATIONSHIPS: "X drops from Y", "X is BiS for Z" — verify against graph data
RANKINGS: "A is better than B" — look for comparative evidence
MECHANICS: Combat formulas, hit tables, proc behavior — verify against mechanic descriptions
PHASE ACCURACY: "Available in Phase X" — verify against phase data in graph

## Confidence Scoring

- VERIFIED with strong evidence: 0.8-1.0
- VERIFIED with moderate evidence: 0.6-0.8
- UNCERTAIN: 0.3-0.6
- CONTRADICTED: 0.0-0.2 (low confidence that the claim is correct)
- UNSUPPORTED: 0.2-0.4 (absence of evidence is not evidence of absence)

## You Do NOT

- Rewrite the article (that's the Writer's job)
- Research new topics (that's the Researcher's job)
- Make judgment calls on contradictions (that's the human's job)
- Invent evidence you weren't given
```

### `VERIFICATION_PROMPT`

Template for the per-claim LLM call (used as user message):

```
Verify the following claim against the provided evidence.

## Claim
{claim}

## Knowledge Base Evidence
{rag_evidence}

## Knowledge Graph Evidence
{graph_evidence}

Assess whether the evidence supports, contradicts, or is silent on this claim.
```

## Component 4: Editor Agent

### Location: `code/shukketsu/agents/editor.py`

### Class Signature

```python
class Editor(BaseAgent):
    """Fact-checking editor that verifies article claims against the knowledge base."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        role: AgentRole | None = None,
        max_iterations: int = config.AGENT_MAX_ITERATIONS,
        system_prompt: str = config.SYSTEM_PROMPT,
        knowledge_manager: KnowledgeManager,
    ) -> None:
        super().__init__(
            tool_registry=tool_registry,
            role=role,
            max_iterations=max_iterations,
            system_prompt=system_prompt,
        )
        self._km = knowledge_manager
```

Same constructor pattern as Writer — receives `KnowledgeManager` via `**kwargs` from factory.

### `execute()` Flow

```python
async def execute(self, task: AgentTask, *, on_status=None) -> EditResult | AgentResult:
```

**Step 1: Validate input**
- Verify `self.role` is set (raise `ValueError` if not)
- Verify `task` is an `EditTask` (isinstance check; if not, fall back to `super().execute()`)

**Step 2: Guard — empty claims**
- If `task.claims` is empty, return `EditResult` with `status=FAILED`, `overall_confidence=0.0`, `approved_for_review=False`

**Step 3: Read article**
- `meta, content = self._km.read_article(task.article_path)`
- Extract `entity_refs` from `meta.entity_refs` for graph search matching

**Step 4: Verify each claim**
- For each claim in `task.claims`:
  1. **RAG search**: Get `rag_search` tool from `self._tool_registry`, call `tool.execute({"query": claim})` → evidence text
  2. **Entity matching**: `_match_entities(claim, meta.entity_refs)` → list of entity names found in claim
  3. **Graph search**: For each matched entity, get `graph_search` tool, call `tool.execute({"entity": entity})` → relationship evidence. Concatenate all graph results.
  4. **LLM judgment**: Build messages with `EDITOR_SYSTEM_PROMPT` (system) and `VERIFICATION_PROMPT.format(claim=claim, rag_evidence=rag_text, graph_evidence=graph_text)` (user). Call `get_structured_output()` with `ClaimJudgment` schema → judgment result.
  5. **Map to ClaimVerification**: Convert `ClaimJudgment` fields into `ClaimVerification`.

**Step 5: Compute confidence**
- `overall_confidence = compute_article_confidence(claim_results)`

**Step 6: Build corrections and research gaps**
- `corrections`: For each CONTRADICTED claim, format as `"Claim '{claim}' contradicted: {note}"`
- `needs_more_research`: For each UNSUPPORTED claim, the claim text itself

**Step 7: Update article frontmatter**
- Match `ClaimVerification` results back to `meta.claims` by text equality
- For each matched `ClaimRef`: set `verified = (status == VERIFIED)`, merge evidence
- Update `meta.confidence` with `overall_confidence`
- Update `meta.updated_at` to now
- Call `self._km.update_draft(task.article_path, meta, content)` (content unchanged)

**Step 8: Conditional status transition**
- If `overall_confidence >= config.EDITOR_CONFIDENCE_THRESHOLD`:
  - Call `self._km.set_status(task.article_path, ArticleStatus.REVIEW)`
  - Set `approved_for_review = True`

**Step 9: Return EditResult**
```python
return EditResult(
    task_id=task.task_id,
    agent_role=self.role,
    status=TaskStatus.SUCCESS,
    output=f"Verified {len(claim_results)} claims. Confidence: {overall_confidence:.2f}",
    article_path=task.article_path,
    claim_results=claim_results,
    overall_confidence=overall_confidence,
    approved_for_review=approved_for_review,
    corrections=corrections,
    needs_more_research=needs_more_research,
)
```

### Entity Matching Helper

```python
def _match_entities(claim: str, entity_refs: list[str]) -> list[str]:
    """Find entity_refs that appear in the claim text (case-insensitive)."""
    claim_lower = claim.lower()
    return [e for e in entity_refs if e.lower() in claim_lower]
```

Module-level function for testability.

### Error Handling

- **Empty claims list** → return `EditResult` with `status=FAILED` (step 2 guard)
- **Article not found** → propagate `FileNotFoundError` (caller error, not recoverable)
- **Individual claim judgment fails** (LLM error) → mark that claim as `UNSUPPORTED` with `note="Verification failed: {error}"`, continue to next claim
- **All judgments fail** → return `EditResult` with `status=PARTIAL`, whatever confidence was computed
- **Tool not found** (rag_search or graph_search missing from registry) → skip that search source, proceed with whatever evidence is available. If neither tool is available, all claims become `UNSUPPORTED`.
- **KnowledgeManager update fails** → propagate exception (filesystem/DB error is not recoverable by the Editor)
- **Status transition fails** (article not in draft) → catch `ValueError`, set `approved_for_review=False`, include note in output

## Component 5: Frontmatter Update Logic

```python
def _update_claims(meta: ArticleMeta, results: list[ClaimVerification]) -> None:
    """Update article ClaimRefs based on verification results."""
    result_map = {r.claim: r for r in results}
    for claim_ref in meta.claims:
        if claim_ref.text in result_map:
            verification = result_map[claim_ref.text]
            claim_ref.verified = verification.status == VerificationStatus.VERIFIED
            # Merge evidence: keep Writer's original + add Editor's findings
            new_evidence = verification.supporting_evidence + verification.contradicting_evidence
            claim_ref.evidence = list(dict.fromkeys(claim_ref.evidence + new_evidence))
```

Module-level function in `editor.py`. Called from `execute()` step 7. Uses `dict.fromkeys` for
order-preserving deduplication of evidence lists.

## Component 6: Factory + Config Changes

### `agents/factory.py`

Add `Editor` to imports and `_ROLE_CLASSES`:

```python
from code.shukketsu.agents.editor import Editor

_ROLE_CLASSES: dict[AgentRole, type[BaseAgent]] = {
    AgentRole.RESEARCHER: Researcher,
    AgentRole.WRITER: Writer,
    AgentRole.EDITOR: Editor,
}
```

Creating an Editor:
```python
factory.create(AgentRole.EDITOR, knowledge_manager=km)
```

If `knowledge_manager` is omitted, `Editor.__init__` raises `TypeError` — clear and immediate.

### `config.py`

Replace the stub `EDITOR_SYSTEM_PROMPT` with an import:

```python
from code.shukketsu.llm.prompts.editor import (  # noqa: E402
    EDITOR_SYSTEM_PROMPT as EDITOR_SYSTEM_PROMPT,
)
```

Add confidence threshold constant:

```python
# Editor
EDITOR_CONFIDENCE_THRESHOLD = float(os.getenv("EDITOR_CONFIDENCE_THRESHOLD", "0.6"))
```

### `agents/__init__.py`

Add exports:

```python
from code.shukketsu.agents.editor import Editor
from code.shukketsu.agents.tasks import (
    ...
    ClaimJudgment,
    ClaimVerification,
    EditResult,
    VerificationStatus,
)
```

## Risks and Unknowns

| Risk | Impact | Mitigation |
|------|--------|------------|
| Claims with no entity_refs match → graph search skipped | Some verifiable relationships not checked | RAG search still runs. Graph search is supplementary. If verification quality is poor, can add LLM entity extraction later. |
| Many claims (10+) → slow verification (~30s+) | User waits a long time | Editor is a background task dispatched by Orchestrator, not interactive. Status callbacks keep the user informed. Acceptable for Phase 2. |
| LLM judges VERIFIED with weak evidence | False confidence in incorrect claims | Prompt instructs 2+ independent sources for VERIFIED. Confidence scoring (0.6-0.8 for moderate evidence) adds nuance. Human review is the final gate. |
| `tool.execute()` returns formatted strings, not structured data | Can't programmatically separate evidence items | The formatted strings are fed directly to the LLM judgment call. The LLM parses them. If structured evidence is needed later, can refactor tools to expose raw results. |
| Article frontmatter claim text doesn't exactly match `task.claims` | Claim mapping fails silently | Both come from the same Writer extraction pass. If the Orchestrator passes `WriteResult.claims` into `EditTask.claims`, they're identical strings. |
| `update_draft` fails because article is already in `review` status | Editor can't write verification results | Check status before calling `update_draft`. If not `draft`, skip frontmatter update and just return results. |

## What This Does NOT Include

- **Article rewriting** — The Editor verifies and flags, it does not fix. The Writer handles rewrites (deferred to Orchestrator coordination in Step 7).
- **Internal consistency checking** — Cross-claim contradiction detection within the same article. `internal_consistency` field defaults to `True`. Add later if needed.
- **Web search for verification** — Editor only uses `rag_search` + `graph_search`. No `web_search` tool. Verification is against the existing KB, not the open web. If KB is insufficient, the Editor flags UNSUPPORTED and lets the Researcher fill the gap.
- **Batch LLM judgment** — Each claim gets its own LLM call. Batching all claims into one call risks cross-contamination of evidence. Can optimize later if latency becomes a problem.
- **Verification caching** — Claims are re-verified every time the Editor runs. No caching of previous verification results. The KB may have changed since the last run.
- **Confidence decay** — Article confidence doesn't decrease over time. Content freshness (Step 9) handles staleness separately.
- **Editor tools** — The Editor doesn't have tools in the agent sense (no ReAct loop). It calls `tool.execute()` directly in Python. The `ToolRegistry` it receives is only used to look up `rag_search` and `graph_search` by name.

## Dependencies

What must exist before this can be implemented:

- **Step 5 complete** — `KnowledgeManager` (`read_article`, `update_draft`, `set_status`), `WriteResult` (provides claims to verify), `ArticleMeta`/`ClaimRef` models (all done)
- **Step 4 complete** — `ResearchResult`, `Finding` models consumed by Writer which produces claims for Editor (all done)
- **Step 1 complete** — `AgentTask`/`AgentResult`, `EditTask`, `AgentFactory` with `**kwargs` forwarding (all done)
- **`get_structured_output()`** — Used for per-claim LLM judgment (exists in `llm/structured.py`)
- **`RagSearchTool`** — Used for KB evidence gathering (exists in `tools/knowledge/search.py`)
- **`GraphSearchTool`** — Used for graph evidence gathering (exists in `tools/knowledge/graph_search.py`)
- **`test_db` fixture** — Creates DB with full schema, used for KnowledgeManager in tests (exists in `tests/conftest.py`)

## Test Plan

Starting count: **527 tests** → Estimated after this step: **~570-580 tests**

### `tests/unit/test_editor.py` (~30-35 tests)

**Confidence scoring:**
- `test_compute_confidence_basic` — average of claim confidences
- `test_compute_confidence_contradiction_penalty` — each contradicted claim reduces by 0.15
- `test_compute_confidence_empty_claims` — returns 0.0
- `test_compute_confidence_clamps_to_zero` — many contradictions don't go below 0.0
- `test_compute_confidence_clamps_to_one` — high values don't exceed 1.0
- `test_compute_confidence_single_claim` — works with one claim

**Entity matching:**
- `test_match_entities_exact` — "Dragonspine Trophy" matches claim containing "Dragonspine Trophy"
- `test_match_entities_case_insensitive` — "dragonspine trophy" matches "Dragonspine Trophy drops from Gruul"
- `test_match_entities_no_match` — returns empty list when no entity appears in claim
- `test_match_entities_multiple` — returns all matching entities

**Frontmatter update:**
- `test_update_claims_marks_verified` — VERIFIED claim sets `claim_ref.verified = True`
- `test_update_claims_marks_unverified` — CONTRADICTED claim sets `claim_ref.verified = False`
- `test_update_claims_merges_evidence` — evidence lists combined and deduplicated
- `test_update_claims_unmatched_ignored` — claims not in frontmatter are skipped gracefully

**Input validation:**
- `test_execute_requires_role` — `ValueError` when role is None
- `test_execute_requires_edit_task` — non-EditTask falls back to `super().execute()`

**Empty claims guard:**
- `test_execute_empty_claims_returns_failed` — returns FAILED when claims list is empty

**Verification flow (mocked LLM + tools):**
- `test_execute_calls_rag_search_per_claim` — rag_search called once per claim
- `test_execute_calls_graph_search_for_matched_entities` — graph_search called for each entity found in claim
- `test_execute_skips_graph_search_no_entity_match` — no graph_search call when claim has no matching entities
- `test_execute_returns_edit_result` — returns EditResult with correct fields
- `test_execute_verified_claim` — VERIFIED claim with supporting evidence
- `test_execute_contradicted_claim` — CONTRADICTED claim populates corrections
- `test_execute_unsupported_claim` — UNSUPPORTED claim populates needs_more_research
- `test_execute_mixed_claims` — mix of VERIFIED, UNCERTAIN, CONTRADICTED, UNSUPPORTED

**Confidence and approval:**
- `test_execute_above_threshold_approves` — confidence >= 0.6 sets `approved_for_review=True`
- `test_execute_below_threshold_rejects` — confidence < 0.6 sets `approved_for_review=False`
- `test_execute_updates_frontmatter_confidence` — article meta.confidence updated
- `test_execute_transitions_to_review` — article status changes from draft to review when approved

**Error handling:**
- `test_execute_llm_failure_marks_unsupported` — LLM error on one claim → UNSUPPORTED, continues
- `test_execute_all_failures_returns_partial` — all LLM calls fail → status=PARTIAL
- `test_execute_missing_tool_skips_search` — missing rag_search tool → still verifies with available evidence
- `test_execute_non_draft_skips_update` — article in review/published → skip frontmatter update, still return results

**Factory integration:**
- `test_factory_creates_editor` — `AgentFactory.create(EDITOR, knowledge_manager=km)` returns Editor
- `test_factory_editor_requires_knowledge_manager` — omitting kwarg raises TypeError
- `test_factory_editor_in_role_classes` — Editor in `_ROLE_CLASSES` dict

### `tests/unit/test_tasks.py` (modify existing, ~5-7 new tests)

**VerificationStatus:**
- `test_verification_status_values` — all four values exist
- `test_verification_status_is_str_enum` — string serialization works

**ClaimVerification:**
- `test_claim_verification_valid` — construct with all fields
- `test_claim_verification_confidence_bounds` — rejects < 0.0 and > 1.0

**ClaimJudgment:**
- `test_claim_judgment_valid` — construct with all fields

**EditResult:**
- `test_edit_result_valid` — construct with all fields
- `test_edit_result_defaults` — default values correct (internal_consistency=True, etc.)
