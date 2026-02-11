# Phase 2, Step 4: Researcher Agent

> Design document for Phase 2, Step 4. Validated through brainstorming session 2026-02-11.

## Overview

The Researcher is the first specialist agent in the multi-agent system. It uses the
existing BaseAgent's ReAct loop with a research-focused system prompt, all four search
tools (rag_search, graph_search, web_search, web_ingest), and a structured result type
that captures findings, evidence, gaps, and a sufficiency flag.

The key architectural choice is a **light subclass** (`Researcher(BaseAgent)`) that
overrides `execute()` to call the base ReAct loop and then run a **Llama 70B structuring
pass** that converts the agent's free-text output + scratchpad into a typed
`ResearchResult`. This gives the Orchestrator (Step 7) and Writer (Step 5) structured
data to work with — claims with confidence scores, source evidence, identified gaps —
without changing how the ReAct loop itself works.

The system prompt lives in a dedicated `llm/prompts/researcher.py` module, establishing
the pattern for Writer/Editor/Orchestrator prompts in Steps 5-7. It includes 2-3 worked
examples demonstrating multi-strategy decomposition, KB-insufficient-to-web fallback, and
graph-first relationship queries.

No new infrastructure is needed. All tools, the reranker, circuit breakers, and the
task protocol are already in place from Steps 1-3. Step 4 is about specialization through
prompting, structured result types, and proving the agent decomposes queries effectively.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Subclass vs configuration-only | Light subclass `Researcher(BaseAgent)` | Need to override `execute()` for the structuring pass. Sets clean pattern for Writer/Editor. |
| Structuring mechanism | Second Llama 70B call via `get_structured_output()` | Structuring findings with confidence requires reasoning capability. Qwen 4B too weak for this. One extra call per task is acceptable latency. |
| Scratchpad access | Add `scratchpad` field to `_RunOutcome` NamedTuple | Structuring pass needs tool observations for evidence tracking. Backward-compatible — `run()` and base `execute()` just ignore it. |
| Prompt location | Dedicated `llm/prompts/researcher.py` module | Prompt is ~80-100 lines with 3 worked examples. Too large for `config.py`. Sets pattern for Steps 5-7. Config.py imports and re-exports. |
| Few-shot examples | 3 worked examples in prompt | Investment in prompt quality. Covers: multi-strategy decomposition, web fallback, graph-first queries. Fits within 150K token budget. |
| Factory class registry | `_ROLE_CLASSES` dict mapping `AgentRole` → subclass | Same pattern as `_ROLE_PROMPTS` and `_ROLE_MAX_ITERATIONS`. Scales for Writer/Editor/Orchestrator. Fallback to `BaseAgent` for unmapped roles. |
| Structuring input | Final answer text + full scratchpad (tool calls + observations) | Scratchpad has source URLs, trust scores, chunk content — everything needed to build `Finding.evidence` lists. Final answer alone would lose evidence provenance. |

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `code/shukketsu/agents/researcher.py` | Researcher subclass with `execute()` override and structuring pass |
| `code/shukketsu/llm/prompts/__init__.py` | Package init for prompts module |
| `code/shukketsu/llm/prompts/researcher.py` | Full Researcher system prompt with 3 worked examples |
| `tests/unit/test_researcher.py` | All Researcher-specific tests |

### Modified Files

| File | Change |
|------|--------|
| `code/shukketsu/agents/base.py` | Add `scratchpad` field to `_RunOutcome` NamedTuple |
| `code/shukketsu/agents/tasks.py` | Add `Finding` and `ResearchResult` models |
| `code/shukketsu/agents/factory.py` | Add `_ROLE_CLASSES` registry, return `Researcher` for `AgentRole.RESEARCHER` |
| `code/shukketsu/agents/__init__.py` | Export `Researcher`, `Finding`, `ResearchResult` |
| `code/shukketsu/config.py` | Import `RESEARCHER_SYSTEM_PROMPT` from `llm.prompts.researcher` instead of inline placeholder |
| `tests/unit/test_base_agent.py` | Minor: verify `_RunOutcome` scratchpad field exists, existing tests unaffected |
| `tests/unit/test_factory.py` | Add tests for `_ROLE_CLASSES` registry, verify Researcher type returned |

### Unchanged Files

These related files will NOT be touched (prevents scope creep):

- `tools/knowledge/search.py` — rag_search tool is complete
- `tools/knowledge/graph_search.py` — graph_search tool is complete
- `tools/research/web_search.py` — web_search tool is complete
- `tools/research/web_ingest.py` — web_ingest tool is complete
- `rag/reranker.py` — reranker is complete and integrated into rag_search
- `llm/structured.py` — `get_structured_output()` already supports any Pydantic model
- `llm/schemas.py` — `AgentStep`/`ActionType` unchanged
- `routing/router.py` — routing changes deferred to Step 7 (Orchestrator)
- `web/routers/chat.py` — chat handler changes deferred to Step 10 (Integration)

## Component 1: _RunOutcome Scratchpad Extension

### Change to `agents/base.py`

The `_RunOutcome` NamedTuple gains a `scratchpad` field:

```python
class _RunOutcome(NamedTuple):
    output: str
    status: TaskStatus
    scratchpad: list[dict[str, Any]]
```

The `_run_loop()` method already builds the scratchpad locally. The only change is
including it in the return value:

```python
# Every return statement in _run_loop adds scratchpad:
return _RunOutcome(output=step.answer, status=TaskStatus.SUCCESS, scratchpad=scratchpad)
return _RunOutcome(output=self._synthesize_partial_answer(scratchpad), status=TaskStatus.PARTIAL, scratchpad=scratchpad)
return _RunOutcome(output=config.AGENT_GRACEFUL_FAILURE, status=TaskStatus.FAILED, scratchpad=scratchpad)
```

**Backward compatibility**: `run()` and base `execute()` already destructure `_RunOutcome`
by field name (`outcome.output`, `outcome.status`), so the new field is invisible to them.

## Component 2: Finding and ResearchResult Models

### Addition to `agents/tasks.py`

```python
class Finding(BaseModel):
    """A single research finding with evidence and confidence."""

    claim: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    entity_refs: list[str] = Field(default_factory=list)


class ResearchResult(AgentResult):
    """Structured output from the Researcher agent.

    Extends AgentResult with research-specific fields that the Writer
    and Orchestrator consume.
    """

    findings: list[Finding] = Field(default_factory=list)
    sources_used: list[str] = Field(default_factory=list)
    strategies_used: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    sufficient: bool = False
```

### Structuring Schema (for the LLM call)

The structuring pass uses a separate Pydantic model (not `ResearchResult` directly)
to avoid coupling the LLM's output format to the inter-agent contract:

```python
class StructuredFindings(BaseModel):
    """Schema for the Llama 70B structuring pass.

    Parsed from the agent's output + scratchpad, then mapped
    to ResearchResult fields.
    """

    findings: list[Finding]
    gaps: list[str] = Field(default_factory=list)
    sufficient: bool = False
```

The Researcher's `execute()` maps this to `ResearchResult`, adding `task_id`,
`agent_role`, `status`, `sources_used`, and `strategies_used` from the scratchpad.

### Validation Rules

- `Finding.confidence` constrained to `[0.0, 1.0]` via `Field(ge=0.0, le=1.0)`
- `ResearchResult.sufficient` defaults to `False` (safe default — Orchestrator will
  request more research if needed)
- Empty `findings` list is valid (agent found nothing) — `sufficient` should be `False`
  in that case
- `sources_used` extracted from scratchpad tool observations (source URLs, chunk IDs)
- `strategies_used` extracted from scratchpad tool names (deduplicated)

## Component 3: Researcher Subclass

### `agents/researcher.py`

```python
class Researcher(BaseAgent):
    """Research specialist agent.

    Overrides execute() to run the standard ReAct loop and then
    structure the output into a typed ResearchResult via a second
    Llama 70B call.
    """

    @observe(as_type="agent")  # Must re-apply — decorators don't carry to overrides
    async def execute(
        self, task: AgentTask, *, on_status: StatusCallback | None = None
    ) -> ResearchResult:
        # 1. Validate role
        if self.role is None:
            raise ValueError("Researcher requires a role.")

        # 2. Run the standard ReAct loop
        outcome = await self._run_loop(task.query, on_status=on_status)

        # 3. Structure the output via Llama 70B
        if on_status:
            await on_status("structuring findings...")
        structured = await self._structure_findings(
            query=task.query,
            output=outcome.output,
            scratchpad=outcome.scratchpad,
        )

        # 4. Derive metadata — sources from findings, strategies from scratchpad
        sources_used = list(dict.fromkeys(e for f in structured.findings for e in f.evidence))
        strategies_used = self._extract_strategies(outcome.scratchpad)

        # 5. Build ResearchResult
        return ResearchResult(
            task_id=task.task_id,
            agent_role=self.role,
            status=outcome.status,
            output=outcome.output,
            evidence=sources_used,
            findings=structured.findings,
            sources_used=sources_used,
            strategies_used=strategies_used,
            gaps=structured.gaps,
            sufficient=structured.sufficient,
        )
```

### Helper Methods

```python
async def _structure_findings(
    self,
    query: str,
    output: str,
    scratchpad: list[dict[str, Any]],
) -> StructuredFindings:
    """Run the Llama 70B structuring pass."""
    observations_text = self._format_scratchpad(scratchpad)

    messages = [
        {"role": "system", "content": STRUCTURING_PROMPT},
        {"role": "user", "content": f"Query: {query}\n\nAgent output:\n{output}\n\nTool observations:\n{observations_text}"},
    ]

    return await get_structured_output(
        response_model=StructuredFindings,
        messages=messages,
    )

def _extract_strategies(self, scratchpad: list[dict[str, Any]]) -> list[str]:
    """Extract unique tool names used during research."""
    return list(dict.fromkeys(e["tool_name"] for e in scratchpad))

def _format_scratchpad(self, scratchpad: list[dict[str, Any]]) -> str:
    """Format scratchpad entries for the structuring prompt."""
    parts = []
    for i, entry in enumerate(scratchpad, 1):
        parts.append(f"[{i}] Tool: {entry['tool_name']}")
        parts.append(f"    Input: {entry['tool_input']}")
        parts.append(f"    Result: {entry['observation'][:500]}")  # Truncate long observations
    return "\n".join(parts)
```

**Note: `sources_used` is derived from structuring output, not parsed from observations.**
Each tool formats observations differently (rag_search uses `"Source: Title (url)"`,
web_search uses `"[1] Title\n    url"`, graph_search is completely different). Parsing
these with regex is fragile. Instead, `sources_used` is flattened from
`Finding.evidence` — the LLM already extracts source references during structuring:

```python
sources_used = list(dict.fromkeys(e for f in structured.findings for e in f.evidence))
```

### Structuring Prompt

A short, focused prompt for the structuring pass (NOT the main system prompt):

```
You are a research structuring assistant. Given a research query, the agent's
final answer, and the tool observations collected during research, extract:

1. FINDINGS: Each distinct factual claim with:
   - The claim text
   - Evidence references (source URLs, chunk IDs from observations)
   - Confidence (0.0-1.0): 1.0 = multiple sources agree, 0.5 = single source, 0.0 = no evidence
   - Entity references (game entities mentioned: items, spells, bosses, etc.)

2. GAPS: Topics the research could not adequately cover (empty results, low confidence)

3. SUFFICIENT: True if the research comprehensively answers the original query,
   False if significant gaps remain.

Be precise. Only extract claims that have evidence in the observations.
Do not invent findings. If the agent found nothing, return empty findings
with sufficient=False.
```

### Error Handling

- If the structuring pass fails (`StructuredOutputError`), fall back to a basic
  `ResearchResult` with just the output string, empty findings, and `sufficient=False`.
  Research is still usable as free text even without structured findings.
- If `_run_loop` returns FAILED status, skip the structuring pass entirely — return
  a minimal `ResearchResult` with the graceful failure message.

### Constructor

No special constructor args. Inherits everything from `BaseAgent`:

```python
def __init__(
    self,
    tool_registry: ToolRegistry,
    *,
    role: AgentRole | None = None,
    max_iterations: int = config.AGENT_MAX_ITERATIONS,
    system_prompt: str = config.SYSTEM_PROMPT,
) -> None:
    super().__init__(
        tool_registry=tool_registry,
        role=role,
        max_iterations=max_iterations,
        system_prompt=system_prompt,
    )
```

## Component 4: Researcher System Prompt

### `llm/prompts/researcher.py`

The prompt teaches six agentic retrieval patterns through instructions and worked examples.

### Structure

```python
RESEARCHER_SYSTEM_PROMPT = """You are a Research Specialist for WoW TBC Rogue content...

## Your Tools
- rag_search: Hybrid vector + keyword search over the knowledge base...
- graph_search: Knowledge graph traversal for entity relationships...
- web_search: Brave Search API for information not in the KB...
- web_ingest: Fetch and store a web page permanently in the KB...

## Search Strategy Selection
[When to use each tool — rag for factual, graph for relationships, web for gaps]

## Research Patterns

### Decompose Complex Questions
[Instructions + Example 1: multi-strategy decomposition]

### Evaluate Results Before Answering
[Self-evaluation criteria: relevance, trust, agreement, sufficiency]

### Iterate When Needed
[Refinement strategies: narrow, broaden, switch strategy]

### Gather Evidence
[How to note sources, cite trust scores, flag disagreements]

### Identify Gaps
[When to report gaps vs when to try another strategy]

### Know When to Stop
[Sufficiency criteria — when is enough research enough?]

## Example Research Traces

### Example 1: Multi-Strategy Decomposition
Query: "What are the best trinkets for a combat rogue in Phase 1 and why?"
Step 1: graph_search(entity="combat swords", target_type="item") → find related items
Step 2: rag_search("combat rogue trinket phase 1 BiS") → get detailed analysis
Step 3: Evaluate — graph gave 3 trinkets, rag confirmed 2 with stat details
Step 4: rag_search("dragonspine trophy proc rate haste") → fill detail gap
Step 5: Final answer with 4 findings, evidence from 3 sources, 0 gaps

### Example 2: KB Insufficient → Web Fallback
Query: "What is the optimal poison setup for mutilate rogues on Illidan?"
Step 1: rag_search("mutilate poison setup Illidan") → 1 low-trust result
Step 2: graph_search(entity="illidan", relation_types=["has_mechanic"]) → boss mechanics
Step 3: Evaluate — know the boss mechanics but no poison-specific advice
Step 4: web_search("TBC mutilate rogue poison Illidan fight") → found guide URL
Step 5: web_ingest(url=...) → stored guide in KB
Step 6: rag_search("mutilate poison setup Illidan") → now 3 results with details
Step 7: Final answer with findings, note gap on specific poison proc math

### Example 3: Graph-First Relationship Query
Query: "What items drop from Gruul that rogues can use?"
Step 1: graph_search(entity="gruul", relation_types=["drops_from"]) → reverse lookup
Step 2: Evaluate — found 5 items, filter by rogue-relevant stats
Step 3: rag_search("Gruul loot table rogue") → confirm and add context
Step 4: Final answer with item list, stat details, evidence from graph + search

## Important Rules
- NEVER guess. If you can't find evidence, say what's missing.
- ALWAYS cite your sources when making claims.
- If multiple sources disagree, note the disagreement.
- Prefer knowledge base over web search when both have relevant results.
- Trust scores matter — weight higher-trust sources more heavily.
"""
```

### Config Integration

`config.py` changes from:

```python
RESEARCHER_SYSTEM_PROMPT = "You are a Research Specialist..."  # placeholder
```

to:

```python
from code.shukketsu.llm.prompts.researcher import RESEARCHER_SYSTEM_PROMPT  # noqa: E402
```

This import happens at the top of config.py with the other imports. The factory's
`_ROLE_PROMPTS[AgentRole.RESEARCHER]` already reads `config.RESEARCHER_SYSTEM_PROMPT`,
so no factory prompt changes needed.

**Circular import risk**: `llm/prompts/researcher.py` must NOT import from `config.py`.
The prompt is a pure string constant with no dependencies. `config.py` imports from it.
One-way dependency, no cycle.

## Component 5: Factory Class Registry

### Changes to `agents/factory.py`

```python
from code.shukketsu.agents.researcher import Researcher

_ROLE_CLASSES: dict[AgentRole, type[BaseAgent]] = {
    AgentRole.RESEARCHER: Researcher,
}

class AgentFactory:
    def create(self, role: AgentRole, *, tool_registry: ToolRegistry | None = None) -> BaseAgent:
        registry = tool_registry if tool_registry is not None else ToolRegistry()
        prompt = _ROLE_PROMPTS.get(role, config.SYSTEM_PROMPT)
        max_iter = _ROLE_MAX_ITERATIONS.get(role, config.AGENT_MAX_ITERATIONS)
        cls = _ROLE_CLASSES.get(role, BaseAgent)

        agent = cls(
            tool_registry=registry,
            role=role,
            max_iterations=max_iter,
            system_prompt=prompt,
        )

        logger.info("Created %s agent (%s, max_iter=%d)", role.value, cls.__name__, max_iter)
        return agent
```

The only change is the `cls` lookup and using `cls(...)` instead of `BaseAgent(...)`.

## Risks and Unknowns

| Risk | Impact | Mitigation |
|------|--------|------------|
| Structuring pass fails to extract findings reliably | ResearchResult has empty findings, downstream agents get less structure | Graceful fallback to basic AgentResult with free-text output. `sufficient=False` triggers Orchestrator to use raw text. |
| System prompt too long, eats into context budget | Fewer iterations before hitting 150K token limit | Keep examples concise (5-7 steps each). Monitor actual token usage in first integration tests. |
| Circular import: `config.py` ← `llm/prompts/researcher.py` | Import error at startup | Strict rule: prompt modules are pure string constants, zero imports from config/agents/tools. |
| Scratchpad observation text too long for structuring pass | Exceeds `STRUCTURED_MAX_TOKENS` (4096) in structuring call | Truncate each observation to ~500 chars in `_format_scratchpad()`. Total scratchpad fits within budget even with 10 iterations. |
| `_RunOutcome` change breaks existing tests | Existing tests fail if they construct `_RunOutcome` directly | Low risk — `_RunOutcome` is only constructed inside `_run_loop()` with keyword args. `run()` and `execute()` use attribute access (`.output`, `.status`). NamedTuple field addition is safe. |
| Missing `@observe` on Researcher.execute() | Langfuse span not created for Researcher executions | Apply `@observe(as_type="agent")` to the override. Python decorators don't carry to subclass methods. Caught in review. |
| `RESEARCHER_MAX_TOKENS` is dead config | Unused — ReAct loop doesn't check token budget | Not a risk, just awareness. Don't add token budget checking (YAGNI). Value exists for future use. |

## What This Does NOT Include

- **Routing changes** — Routing MODERATE queries to Researcher happens in Step 7 (Orchestrator). The Researcher is invoked directly via `execute()` for now.
- **Chat handler integration** — Chat handler changes deferred to Step 10. Researcher is tested via direct `execute()` calls.
- **ResearchTask.search_strategy enforcement** — The `search_strategy` field on `ResearchTask` is advisory. The prompt teaches the agent to choose strategies. Forcing a specific strategy via code is a future optimization.
- **Parallel research** — Multiple Researchers running concurrently is an Orchestrator concern (Step 7). This step builds a single Researcher.
- **Tool auto-population** — The factory does NOT auto-register tools. Callers provide a pre-built `ToolRegistry`. A convenience method for building a Researcher's registry could come later.
- **Researcher-specific iteration limits in the prompt** — The agent doesn't know its iteration budget. It relies on the ReAct loop's natural termination. Teaching the agent about its budget could help but adds prompt complexity.

## Dependencies

### Must exist before implementation

- **BaseAgent** with `execute()` and `_run_loop()` — `agents/base.py` (Step 1, complete)
- **Task protocol** — `AgentTask`, `AgentResult`, `ResearchTask`, `TaskStatus` in `agents/tasks.py` (Step 1, complete)
- **AgentFactory** — `agents/factory.py` (Step 1, complete)
- **All four tools** — `rag_search`, `graph_search`, `web_search`, `web_ingest` (Steps 1-3, complete)
- **Qwen 4B reranker** — integrated into `rag_search` transparently (Step 3, complete)
- **`get_structured_output()`** — `llm/structured.py` (Phase 1, complete)
- **Circuit breakers** — all tool-level breakers in place (Phase 1, complete)

### External services (for integration tests only)

- Ollama running with Llama 3.3 70B and nomic-embed-text-v2
- SQLite database initialized with Phase 2 schema (entity tables)

## Test Plan

Starting count: **438 tests**. Estimated after this step: **~465-470 tests**.

### `tests/unit/test_researcher.py` (new file)

#### `TestResearchResultModels` (~7 tests)

| Test | What it verifies |
|------|-----------------|
| `test_finding_valid` | Finding with all fields validates |
| `test_finding_confidence_bounds` | confidence < 0.0 or > 1.0 rejected |
| `test_finding_defaults` | Empty evidence/entity_refs default to empty lists |
| `test_research_result_valid` | ResearchResult with findings validates |
| `test_research_result_defaults` | Empty findings/gaps/sources default correctly, sufficient=False |
| `test_research_result_inherits_agent_result` | ResearchResult is subclass of AgentResult |
| `test_structured_findings_valid` | StructuredFindings schema validates (structuring pass input) |

#### `TestResearcherExecute` (~7 tests)

| Test | What it verifies |
|------|-----------------|
| `test_returns_research_result` | execute() returns ResearchResult (not plain AgentResult) |
| `test_structuring_pass_called` | Second LLM call made after ReAct loop for structuring |
| `test_scratchpad_passed_to_structuring` | Structuring pass receives tool observations |
| `test_sources_derived_from_findings` | `sources_used` flattened from Finding.evidence, not parsed from observations |
| `test_strategies_extracted_from_scratchpad` | `strategies_used` = deduplicated tool names |
| `test_structuring_failure_graceful_fallback` | StructuredOutputError → basic result with empty findings |
| `test_execute_has_langfuse_observe` | `@observe` decorator applied to Researcher.execute() (not inherited from base) |

#### `TestResearcherBehavior` (~7 tests)

| Test | What it verifies |
|------|-----------------|
| `test_decompose_two_part_question` | Mock LLM decomposes and calls tools twice |
| `test_multiple_strategies_used` | Mock LLM uses both rag_search and graph_search |
| `test_web_fallback_on_empty_kb` | Mock LLM tries web_search after empty rag_search |
| `test_respects_max_iterations` | Stops at RESEARCHER_MAX_ITERATIONS |
| `test_gaps_populated_on_partial_results` | gaps list non-empty when research incomplete |
| `test_sufficient_false_on_empty_results` | All tools return empty → sufficient=False |
| `test_failed_loop_skips_structuring` | FAILED status → no structuring pass, minimal result |

#### `TestResearcherPrompt` (~4 tests)

| Test | What it verifies |
|------|-----------------|
| `test_prompt_contains_all_tool_names` | "rag_search", "graph_search", "web_search", "web_ingest" in prompt |
| `test_prompt_contains_strategy_selection` | Search strategy guidance present |
| `test_prompt_contains_worked_examples` | All 3 example traces present |
| `test_prompt_contains_evaluation_criteria` | Self-evaluation instructions present |

#### `TestResearcherFactory` (~4 tests)

| Test | What it verifies |
|------|-----------------|
| `test_factory_returns_researcher_type` | `factory.create(RESEARCHER)` returns `Researcher` instance |
| `test_factory_other_roles_return_base` | `factory.create(WRITER)` still returns `BaseAgent` |
| `test_researcher_has_correct_prompt` | Researcher's prompt is the full prompt from `llm/prompts/researcher.py` |
| `test_researcher_has_correct_max_iter` | max_iterations == RESEARCHER_MAX_ITERATIONS (10) |

### Modifications to existing test files

#### `tests/unit/test_base_agent.py` (~2 tests added)

| Test | What it verifies |
|------|-----------------|
| `test_run_outcome_has_scratchpad` | `_RunOutcome` has `scratchpad` field |
| `test_run_loop_returns_scratchpad` | `_run_loop()` includes scratchpad in outcome |

#### `tests/unit/test_factory.py` (~2 tests modified)

| Test | What it verifies |
|------|-----------------|
| `test_role_classes_registry_exists` | `_ROLE_CLASSES` dict is populated |
| (existing factory tests) | Verify they still pass with class registry change |

**Total new tests: ~31** (438 → ~469)
