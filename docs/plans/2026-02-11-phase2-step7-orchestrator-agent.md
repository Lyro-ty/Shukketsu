# Phase 2 Step 7: Orchestrator Agent

> Design document for Phase 2, Step 7. Validated through brainstorming session 2026-02-11.

## Overview

The Orchestrator is the coordination agent that ties all specialists together. It receives
complex queries from the chat handler, decomposes them into typed sub-tasks for the Researcher,
Writer, and Editor, executes those tasks in dependency order, and synthesizes results into a
final response.

Unlike the Researcher (ReAct loop + structuring pass), the Orchestrator overrides `execute()`
with a deterministic three-phase Python flow: **decompose → dispatch → synthesize**. The
decomposition step uses a single Llama 70B structured output call to produce an
`OrchestratorPlan`. The dispatch step walks the dependency graph in topological order, building
typed tasks from `SubTask` descriptions via deterministic Python wiring (the LLM plans *what*
to do; Python handles *how* to connect task inputs/outputs). The synthesis step either calls
Llama 70B for natural language answers (research-only plans) or uses a template for article
workflows.

This step also wires the chat handler to route queries by complexity: TRIVIAL gets a direct
answer from Qwen 4B (existing), MODERATE goes to the Researcher solo (new), and COMPLEX goes
to the Orchestrator (new).

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Decomposition strategy | Single LLM call, retry only on structural invalidity | Simpler, predictable. Validation catches cycles, too many subtasks, invalid roles. No "quality" retries. |
| Task wiring | Deterministic Python (`_build_task()`) + `task_params` for simple metadata | Python wires complex dependencies (ResearchResult → WriteTask.research). LLM provides simple metadata (spec, category, article_type) via `task_params`. Best of both: reliable wiring + reliable metadata. |
| Concurrent execution | Sequential with topological ordering | Ollama serializes inference on single GB10. Topological sort respects dependencies. `asyncio.gather()` can be added later trivially. |
| Synthesis strategy | Conditional: LLM for research answers, template for article workflows | Research queries need natural language; article workflows just need confirmation of what happened. |
| Error handling | No retries at Orchestrator level | Specialists have their own retry/circuit-breaker logic. Orchestrator manages outcomes: failed deps → skip dependents, partial → proceed, all failed → FAILED. |
| Router integration | Wire chat handler in Step 7 | Change is ~15 lines. Gives immediate end-to-end browser visibility. Step 10 can focus on polish. |
| Execute override | Override `execute()` completely (like Writer/Editor) | Three-phase flow is fundamentally different from ReAct. No benefit from inheriting the loop. |

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `agents/orchestrator.py` | Orchestrator agent: decompose, dispatch, synthesize |
| `llm/prompts/orchestrator.py` | System prompt + decomposition prompt for structured output |

### Modified Files

| File | Change |
|------|--------|
| `agents/tasks.py` | Add `OrchestratorResult` model |
| `agents/factory.py` | Register Orchestrator class in `_ROLE_CLASSES` |
| `config.py` | Replace stub `ORCHESTRATOR_SYSTEM_PROMPT` with import from prompts module |
| `web/routers/chat.py` | Route MODERATE → Researcher, COMPLEX → Orchestrator |

### Unchanged Files

- `agents/base.py` — No changes to BaseAgent or ReAct loop
- `agents/researcher.py` — Researcher execution unchanged
- `agents/writer.py` — Writer execution unchanged
- `agents/editor.py` — Editor execution unchanged
- `routing/router.py` — `classify_query()` already returns complexity levels correctly
- `routing/models.py` — TaskComplexity/RoutingDecision already have what we need
- `tools/` — No new tools; Orchestrator dispatches to agents, not tools directly
- `db/schema.sql` — No schema changes

## Component 1: OrchestratorResult Model (`tasks.py`)

New model added to the existing task protocol:

```python
class OrchestratorResult(AgentResult):
    """Structured output from the Orchestrator agent."""

    plan: OrchestratorPlan
    specialist_results: list[AgentResult] = Field(default_factory=list)
    article_path: str | None = None
    needs_human_review: bool = False
    skipped_tasks: list[str] = Field(default_factory=list)
```

Fields:
- `plan` — The decomposition plan (for tracing/debugging)
- `specialist_results` — All sub-task results (including PARTIAL/FAILED ones)
- `article_path` — Set if a Writer produced an article
- `needs_human_review` — Set if an Editor approved an article for review
- `skipped_tasks` — Descriptions of tasks skipped due to failed dependencies

The inherited `output` holds the synthesized response text. The inherited `status` is
SUCCESS/PARTIAL/FAILED based on overall outcome. The inherited `evidence` aggregates
evidence from all specialist results.

## Component 2: Orchestrator Prompt (`llm/prompts/orchestrator.py`)

Two prompts in this module:

### System Prompt (`ORCHESTRATOR_SYSTEM_PROMPT`)

Used for the decomposition call. Teaches the LLM about available specialists and
decomposition rules:

```python
ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the Orchestrator for a WoW TBC Rogue knowledge system. Your job is to
decompose complex queries into sub-tasks for specialist agents.

## Available Specialists

- RESEARCHER: Searches the knowledge base (hybrid vector + keyword search),
  knowledge graph (entity relationships), and web. Returns structured findings
  with evidence and confidence scores. Use for any information gathering.

- WRITER: Produces wiki articles from research findings. Requires research
  results as input — always schedule a RESEARCHER task first. Use when the user
  asks for a guide, article, or comprehensive write-up.

- EDITOR: Fact-checks a draft article against the knowledge base. Requires a
  written article — always schedule after WRITER. Use when an article has been
  produced and needs verification.

## Planning Rules

1. DECOMPOSE into the smallest useful sub-tasks. Each sub-task should have a
   clear, specific description.
2. Use depends_on to express ordering. A WRITER task must depend on the
   RESEARCHER task(s) that feed it. An EDITOR task must depend on WRITER.
3. DO NOT over-decompose. A simple factual question needs one RESEARCHER, not
   three. Reserve multi-step plans for genuinely multi-part questions.
4. If you can answer the question directly without specialists, set
   can_answer_directly=True and provide direct_answer.
5. Maximum 6 sub-tasks. If the question needs more, simplify.

## task_params

For WRITER sub-tasks, you MUST include these fields in task_params:
- "spec": one of "combat", "assassination", "subtlety", "general"
- "category": a short topic category like "gear", "rotation", "talents", "consumables", "general"
- "article_type": one of "guide", "reference", "analysis"

For RESEARCHER and EDITOR sub-tasks, task_params can be empty.

## Examples

Query: "What are the best trinkets for combat rogues in Phase 1?"
→ 1 RESEARCHER task (straightforward retrieval)

Query: "Write a guide about Phase 1 BiS trinkets for combat rogues"
→ RESEARCHER (find trinkets + stat priorities) → WRITER (draft guide, task_params: {"spec": "combat", "category": "gear", "article_type": "guide"}) → EDITOR (verify)

Query: "Compare combat swords vs mutilate for Gruul"
→ RESEARCHER (combat swords gear + rotation for Gruul)
→ RESEARCHER (mutilate gear + rotation for Gruul) [parallel, no dependency]
→ These feed into the synthesis step

Query: "Hello, what can you do?"
→ can_answer_directly=True
"""
```

### Decomposition Prompt (`DECOMPOSITION_PROMPT`)

The user-message template for the structured output call:

```python
DECOMPOSITION_PROMPT = "Decompose this query into a plan:\n\n{query}"
```

### Synthesis Prompt (`SYNTHESIS_PROMPT`)

For LLM-based synthesis of research results:

```python
SYNTHESIS_PROMPT = """\
You are synthesizing research results into a coherent answer about WoW TBC Rogue content.

Combine the following specialist findings into a clear, well-structured response.
Include specific numbers, item names, and evidence where available.
If findings conflict, note the disagreement.
If there are gaps, acknowledge what couldn't be determined.

User's original question: {query}

Research findings:
{findings}
"""
```

## Component 3: Orchestrator Agent (`agents/orchestrator.py`)

### Class Signature

```python
class Orchestrator(BaseAgent):
    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        role: AgentRole | None = None,
        max_iterations: int | None = None,
        system_prompt: str | None = None,
        factory: AgentFactory,
    ) -> None: ...
```

The Orchestrator takes an `AgentFactory` as a constructor dependency (same pattern as Writer
taking `KnowledgeManager` and Editor taking `KnowledgeManager`). This allows it to create
specialist agents on demand.

### Execute Override

```python
async def execute(self, task: AgentTask, *, on_status: StatusCallback | None = None) -> OrchestratorResult:
```

Three-phase flow:

#### Phase 1: Decompose (`_decompose`)

1. Call Llama 70B structured output with `OrchestratorPlan` as response model
2. Validate the plan:
   - No cycles in dependency graph (topological sort succeeds)
   - Subtask count <= `ORCHESTRATOR_MAX_SUBTASKS` (6)
   - All `agent_role` values are valid `AgentRole` members
   - All `depends_on` indices are within range
   - Writer tasks depend on at least one Researcher
   - Editor tasks depend on at least one Writer
3. If validation fails: retry once with validation error as feedback
4. If second attempt also fails: return FAILED result
5. If `can_answer_directly=True`: return SUCCESS with `direct_answer` as output

```python
async def _decompose(self, query: str, on_status: StatusCallback | None = None) -> OrchestratorPlan: ...
def _validate_plan(self, plan: OrchestratorPlan) -> list[str]: ...
```

#### Phase 2: Dispatch (`_dispatch`)

1. Compute topological order via `_topological_sort(plan.subtasks)`
2. For each subtask index in topological order:
   - Check if all dependencies succeeded or are partial (at least something to work with)
   - If any dependency FAILED: skip this task, add description to `skipped_tasks`
   - Build the typed task via `_build_task(subtask, results)`:
     - RESEARCHER → `ResearchTask(query=subtask.description)`
     - WRITER → `WriteTask(research=results[dep_idx], ...)` pulling from completed Researcher
     - EDITOR → `EditTask(article_path=results[dep_idx].article_path, claims=results[dep_idx].claims)` pulling from completed Writer
   - Create the specialist agent via `self._factory.create(subtask.agent_role, ...)` with role-conditional kwargs:
     - RESEARCHER: `factory.create(RESEARCHER, tool_registry=self.tool_registry)`
     - WRITER: `factory.create(WRITER, tool_registry=self.tool_registry, knowledge_manager=self._km)`
     - EDITOR: `factory.create(EDITOR, tool_registry=self.tool_registry, knowledge_manager=self._km)`
     - **Important**: Writer/Editor constructors require `knowledge_manager` but Researcher does not accept it — passing it would raise `TypeError`. The Orchestrator must only pass `knowledge_manager` to roles that accept it.
   - Execute: `result = await agent.execute(typed_task, on_status=on_status)`
   - Store result at the subtask's index

```python
async def _dispatch(
    self, plan: OrchestratorPlan, task: AgentTask, on_status: StatusCallback | None = None,
) -> tuple[list[AgentResult | None], list[str]]: ...

def _topological_sort(self, subtasks: list[SubTask]) -> list[int]: ...
def _build_task(self, subtask: SubTask, results: list[AgentResult | None], trace_id: str) -> AgentTask: ...
```

#### Phase 3: Synthesize (`_synthesize`)

1. Collect all non-None results
2. Determine if this was an article workflow (any WriteResult in results)
3. If article workflow:
   - Template-based: report article path, claim count, Editor confidence/approval
   - Extract `article_path` and `needs_human_review` from WriteResult/EditResult
4. If research-only:
   - Gather all ResearchResult outputs
   - Call Llama 70B with `SYNTHESIS_PROMPT` for natural language answer
   - Fall back to concatenated outputs if synthesis LLM call fails
5. Determine overall status:
   - All succeeded → SUCCESS
   - Mix of success/failure → PARTIAL
   - All failed or all skipped → FAILED
6. Aggregate evidence from all specialist results

```python
async def _synthesize(
    self, task: AgentTask, plan: OrchestratorPlan,
    results: list[AgentResult | None], skipped: list[str],
) -> OrchestratorResult: ...
```

### Task Building Details

The `_build_task()` method handles the deterministic wiring:

**ResearchTask**: Always straightforward — `description` becomes `query`.

**WriteTask**: Needs a `ResearchResult`, `article_type`, `spec`, and `category`.
- Finds the first successful ResearchResult in dependencies
- If multiple Researcher dependencies: merge findings into a single synthetic ResearchResult:
  - `findings` = concatenated lists, deduped by `claim` text
  - `output` = joined with `"\n\n---\n\n"` separator
  - `sources_used` = union of both lists
  - `strategies_used` = union of both lists
  - `gaps` = union of both lists
  - `sufficient` = `all(r.sufficient for r in research_results)` (conservative)
- Reads `spec`, `category`, and `article_type` from `subtask.task_params` (populated by LLM per prompt instructions)
- Falls back to `spec="general"`, `category="general"`, `article_type=GUIDE` if task_params are missing

**EditTask**: Needs `article_path` and `claims`.
- Finds the WriteResult in dependencies
- Extracts `article_path` and `claims` directly

### Tool Registry

The Orchestrator does **not** use tools directly — it dispatches to specialists. However,
the factory still expects a `tool_registry` parameter. The Orchestrator receives an empty
registry (or a shared one that it passes to specialists).

For specialist tool registries: the chat handler constructs the shared tools (rag_search,
web_search, web_ingest, graph_search) and passes the registry through the factory. The
Orchestrator's factory reference must produce agents with working tool registries.

This means the factory needs to accept a default `tool_registry` that specialists inherit,
OR the Orchestrator passes its own registry to `factory.create()` calls. The simpler
approach: the Orchestrator stores a `tool_registry` and passes it when creating specialists.

### Validation Details

Cycle detection via Kahn's algorithm (standard topological sort):
1. Build adjacency list from `depends_on`
2. Compute in-degrees
3. Process nodes with in-degree 0, decrementing neighbors
4. If not all nodes processed → cycle exists

Additional validations:
- `depends_on` indices must be `< len(subtasks)` and `>= 0`
- No self-dependencies (`i not in subtask[i].depends_on`)
- `agent_role` must be in {RESEARCHER, WRITER, EDITOR} (not ORCHESTRATOR — no recursion)

### Error Handling

- Decomposition LLM call fails → return FAILED with explanation
- Validation fails twice → return FAILED with validation errors
- Individual specialist fails → skip dependents, continue with others
- All specialists fail → return FAILED
- Synthesis LLM call fails → fall back to concatenated specialist outputs
- Unexpected exception → catch at execute() level, return FAILED

## Component 4: Factory Registration (`factory.py`)

Minimal changes:

```python
from code.shukketsu.agents.orchestrator import Orchestrator

_ROLE_CLASSES: dict[AgentRole, type[BaseAgent]] = {
    AgentRole.RESEARCHER: Researcher,
    AgentRole.WRITER: Writer,
    AgentRole.EDITOR: Editor,
    AgentRole.ORCHESTRATOR: Orchestrator,
}
```

The Orchestrator constructor requires `factory: AgentFactory`. The chat handler creates the
factory first, then creates the Orchestrator with that factory:

```python
factory = AgentFactory()
orchestrator = factory.create(AgentRole.ORCHESTRATOR, tool_registry=registry, factory=factory)
```

This circular-looking pattern is fine because the factory is stateless and the Orchestrator
only calls `factory.create()` during `execute()`, not during construction.

## Component 5: Config Update (`config.py`)

Replace the stub prompt with an import:

```python
# Before:
ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are the Orchestrator for a WoW TBC Rogue knowledge system. "
    "You decompose complex queries into sub-tasks for specialist agents."
)

# After:
from code.shukketsu.llm.prompts.orchestrator import (
    ORCHESTRATOR_SYSTEM_PROMPT as ORCHESTRATOR_SYSTEM_PROMPT,
)
```

This follows the same pattern used for RESEARCHER/WRITER/EDITOR prompts.

## Component 6: Chat Handler Update (`web/routers/chat.py`)

### Agent Initialization

Replace the single `_get_agent()` singleton with a setup that creates the factory and
agents by role. The chat handler needs three things:

1. A Researcher agent (for MODERATE queries)
2. An Orchestrator agent (for COMPLEX queries)
3. A shared tool registry (both agents use the same tools)

```python
_researcher: BaseAgent | None = None
_orchestrator: BaseAgent | None = None

def _get_agents() -> tuple[BaseAgent, BaseAgent]:
    """Get or create agent singletons (researcher + orchestrator)."""
    global _researcher, _orchestrator

    if _researcher is None:
        # Build shared tools
        registry = _build_tool_registry()
        factory = AgentFactory()
        km = KnowledgeManager(...)

        _researcher = factory.create(
            AgentRole.RESEARCHER,
            tool_registry=registry,
        )
        _orchestrator = factory.create(
            AgentRole.ORCHESTRATOR,
            tool_registry=registry,
            factory=factory,
            knowledge_manager=km,
        )

    return _researcher, _orchestrator
```

### Routing Logic

```python
if decision.complexity == TaskComplexity.TRIVIAL and decision.direct_answer:
    # Existing: direct answer from Qwen 4B
    ...
elif decision.complexity == TaskComplexity.MODERATE:
    researcher, _ = _get_agents()
    result = await researcher.execute(ResearchTask(query=content), on_status=_send_status)
    answer = result.output
elif decision.complexity == TaskComplexity.COMPLEX:
    _, orchestrator = _get_agents()
    result = await orchestrator.execute(AgentTask(query=content), on_status=_send_status)
    answer = result.output
    # Append article notification if applicable
    if isinstance(result, OrchestratorResult) and result.article_path:
        answer += f"\n\n---\n*Draft article created: {result.article_path}*"
    if isinstance(result, OrchestratorResult) and result.needs_human_review:
        answer += "\n*Article pending review in Wiki*"
```

### Knowledge Manager for Orchestrator

The Orchestrator needs to pass a `KnowledgeManager` to Writer and Editor agents it creates.
Two options:
- Store it on the Orchestrator and pass it during dispatch
- Store it on the factory as a default kwarg

Simpler: the Orchestrator stores a `knowledge_manager` as an instance attribute (same as
Editor) and passes it to `factory.create()` when creating Writer/Editor agents.

Updated constructor:

```python
class Orchestrator(BaseAgent):
    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        role: AgentRole | None = None,
        max_iterations: int | None = None,
        system_prompt: str | None = None,
        factory: AgentFactory,
        knowledge_manager: KnowledgeManager | None = None,
    ) -> None: ...
```

`knowledge_manager` is optional because not every Orchestrator plan involves writing. If a
Writer/Editor subtask appears but no `knowledge_manager` was provided, the task is skipped
with an error message in `skipped_tasks`.

## Risks and Unknowns

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM produces invalid dependency indices | Dispatch crashes with IndexError | Validation catches this; retry with feedback |
| LLM can't reliably produce OrchestratorPlan | Decomposition always fails | Structured output via Instructor handles schema enforcement; prompt includes examples |
| LLM omits task_params for Writer subtask | Writer gets default spec/category | Prompt explicitly instructs LLM; validation could warn but safe fallbacks (general/general/GUIDE) prevent crashes |
| Merging multiple ResearchResults for Writer is lossy | Writer gets incomplete info | Concatenate findings; dedup by claim; conservative `sufficient` (all must agree) |
| Chat handler refactor breaks existing functionality | TRIVIAL/MODERATE queries regress | Existing trivial path unchanged; MODERATE path tests agent.run → agent.execute migration |
| Factory circular reference (Orchestrator holds factory that can create Orchestrators) | Recursive orchestration | Validation rejects ORCHESTRATOR role in subtasks; MAX_DEPTH=1 in config |
| Passing knowledge_manager to Researcher via factory.create() | TypeError crash | Orchestrator uses role-conditional kwargs — only passes knowledge_manager to Writer/Editor |

## What This Does NOT Include

- **Concurrent subtask execution** — Sequential only. Add `asyncio.gather()` when multi-GPU warrants it.
- **Retry at orchestration level** — Specialists handle their own resilience.
- **Recursive orchestration** — MAX_DEPTH=1, no Orchestrator-creates-Orchestrator.
- **Validation of task_params values** — If the LLM puts an invalid spec in task_params (e.g., "rogue"), we fall back to "general" rather than validating against the Spec enum. Strict validation is unnecessary overhead for three known specs.
- **Status callback forwarding** — The Orchestrator passes `on_status` to specialists. No aggregation or prefixing (e.g., "Researcher: searching...") — that's Step 10 polish.
- **Timeout enforcement** — `ORCHESTRATOR_TIMEOUT_SECONDS` exists in config but is not enforced in this step. Specialists have their own iteration limits. Full timeout is Step 10.
- **Orchestrator-initiated article writing** — The plan mentions "when research reveals comprehensive coverage, Writer auto-drafts." This step only writes articles when the LLM plan includes a WRITER subtask. Autonomous article triggering is a future enhancement.

## Dependencies

- **Step 1** (tasks.py): `AgentTask`, `AgentResult`, `SubTask`, `OrchestratorPlan`, `TaskStatus`
- **Step 1** (factory.py): `AgentFactory.create()`
- **Step 1** (base.py): `BaseAgent.execute()`, `StatusCallback` type
- **Step 4** (researcher.py): `Researcher` class, `ResearchResult`, `ResearchTask`
- **Step 5** (writer.py): `Writer` class, `WriteResult`, `WriteTask`, `KnowledgeManager`
- **Step 6** (editor.py): `Editor` class, `EditResult`, `EditTask`
- **Phase 1** (router.py): `classify_query()`, `RoutingDecision`
- **Phase 1** (chat.py): `_agent_response()`, `_get_agent()`

## Test Plan

All tests in `tests/unit/`. No integration tests in this step (those are Step 10).

### `tests/unit/test_orchestrator_tasks.py` (~4 tests)
- `test_orchestrator_result_validates` — OrchestratorResult round-trips correctly
- `test_orchestrator_result_defaults` — Default values (empty lists, None article_path)
- `test_orchestrator_result_with_skipped_tasks` — skipped_tasks populated
- `test_orchestrator_result_inherits_agent_result` — Has task_id, status, output, evidence

### `tests/unit/test_orchestrator_prompt.py` (~3 tests)
- `test_system_prompt_mentions_all_roles` — Contains RESEARCHER, WRITER, EDITOR
- `test_system_prompt_mentions_planning_rules` — Contains dependency, max subtask guidance
- `test_decomposition_prompt_includes_query` — Template fills correctly

### `tests/unit/test_orchestrator_validate.py` (~7 tests)
- `test_valid_plan_passes` — No errors for well-formed plan
- `test_cycle_detection` — Circular dependency caught
- `test_self_dependency_rejected` — subtask[0].depends_on=[0] caught
- `test_out_of_range_dependency` — depends_on=[99] caught
- `test_too_many_subtasks` — 7 subtasks rejected (max 6)
- `test_orchestrator_role_rejected` — ORCHESTRATOR in subtask roles rejected (no recursion)
- `test_writer_without_researcher_dependency` — Writer not depending on Researcher caught

### `tests/unit/test_orchestrator_topo.py` (~5 tests)
- `test_linear_chain` — A→B→C returns [0, 1, 2]
- `test_independent_tasks` — No deps returns all (order stable)
- `test_diamond_dependency` — A→C, B→C, C→D returns valid order
- `test_empty_plan` — Returns empty list
- `test_single_task` — Returns [0]

### `tests/unit/test_orchestrator_build_task.py` (~6 tests)
- `test_build_research_task` — SubTask with RESEARCHER → ResearchTask
- `test_build_write_task_from_research` — SubTask with WRITER pulls ResearchResult from deps
- `test_build_edit_task_from_write` — SubTask with EDITOR pulls article_path + claims
- `test_build_write_task_merges_multiple_research` — Two Researcher deps → merged findings
- `test_build_task_propagates_trace_id` — trace_id flows from parent task
- `test_build_write_task_reads_task_params` — task_params={"spec": "combat", "category": "gear"} → WriteTask fields
- `test_build_write_task_defaults_on_missing_params` — empty task_params → spec="general", category="general"

### `tests/unit/test_orchestrator_execute.py` (~10 tests)
- `test_direct_answer_skips_dispatch` — can_answer_directly returns immediately
- `test_single_researcher_plan` — One subtask dispatches and returns
- `test_writer_receives_knowledge_manager_researcher_does_not` — Role-conditional factory kwargs
- `test_research_write_edit_chain` — Full pipeline chains correctly
- `test_failed_dependency_skips_dependents` — Failed Researcher → Writer skipped
- `test_partial_dependency_proceeds` — Partial Researcher → Writer still runs
- `test_all_failed_returns_failed` — All specialists fail → FAILED status
- `test_decomposition_failure_returns_failed` — LLM can't produce valid plan → FAILED
- `test_validation_retry_on_structural_error` — First plan invalid, second valid → succeeds
- `test_synthesis_llm_failure_falls_back` — Synthesis LLM fails → concatenated output

### `tests/unit/test_orchestrator_factory.py` (~3 tests)
- `test_factory_creates_orchestrator` — `factory.create(AgentRole.ORCHESTRATOR, factory=factory)` works
- `test_orchestrator_has_correct_prompt` — System prompt loaded from prompts module
- `test_orchestrator_requires_factory_kwarg` — Missing factory raises TypeError

### `tests/unit/test_chat_routing.py` (~5 tests)
- `test_trivial_uses_direct_answer` — Existing behavior preserved
- `test_moderate_uses_researcher` — MODERATE → Researcher.execute()
- `test_complex_uses_orchestrator` — COMPLEX → Orchestrator.execute()
- `test_article_path_appended_to_response` — OrchestratorResult with article_path → message
- `test_fallback_routes_to_orchestrator` — Router failure → COMPLEX → Orchestrator

**Starting count: 570 tests → Estimated after this step: ~615 tests**
