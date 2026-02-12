# Phase 3: Memory, Reflection, and Performance

> Implementation plan. Each step produces a working system that does something
> visibly better than the previous one. Steps are ordered by dependency — each
> builds on what came before.
>
> **For each step**: Read the full step description first. Propose your
> implementation approach (files to create/modify, key design decisions, test
> plan) and get approval before writing code. After implementing, run the full
> test suite and update CLAUDE.md.

## Prerequisites

- Phase 2 complete (741 tests passing, all 10 steps done)
- Ollama running on port 11434 (Llama 3.3 70B, Qwen3 4B, nomic-embed-text-v2)
- Langfuse stack running (Docker Compose)
- SQLite database initialized with schema v3
- `sentence-transformers` added to `requirements.txt` (Step 1 will install it)

## Phase Gate

Phase 3 is complete when: the system demonstrates cross-session memory (ask a
question, disconnect, reconnect, ask a related question — the agent recalls
relevant context from the first session), reflection catches unsupported claims
before they reach the user, parallel research subtasks complete faster than
sequential execution, and all existing 741 tests still pass alongside the new
tests.

**Metrics:**
- Existing eval metrics maintained (faithfulness >= 0.8, trajectory >= 0.7, accuracy >= 0.7)
- Reranker latency reduced >= 50% vs. current Qwen 4B generative reranker
- Reflection catches >= 1 unsupported claim in the eval dataset
- Memory recall returns relevant facts for follow-up queries within the same topic

## Design Decisions

Settled during brainstorming (2026-02-12):

| Decision | Choice | Rationale |
|----------|--------|-----------|
| SearchResult rerank_score | Add optional field to existing frozen dataclass | `dataclasses.replace()` is idiomatic; avoids wrapper type |
| StatusCallback signature | Widen to `str \| dict[str, Any]` | Existing string calls unchanged; dicts for structured events |
| Context compaction | Mechanical truncation, no LLM | Zero latency, no failure mode; agent already processed older observations |
| Reflection trigger | Complex queries only (Orchestrator-routed) | Highest hallucination risk; users expect longer wait on complex queries |
| Memory extraction model | Qwen 4B (fire-and-forget) | Fast, leaves Llama 70B free for user-facing work |
| Trust scoring transition | Full replacement, no time decay | TBC is solved; freshness checker handles dead URLs; clean break |
| Phase structure | Single phase (9 steps) | Steps already dependency-ordered; phase gate requires both halves |

## Schema Migration (v3 -> v4)

Applied in Step 6. Adds memory tables + trust_events table together:

```sql
-- Facts extracted from conversations
CREATE TABLE session_memories (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    answer_summary TEXT NOT NULL,
    key_facts_json TEXT NOT NULL DEFAULT '[]',
    entities_mentioned TEXT NOT NULL DEFAULT '[]',
    user_feedback TEXT,
    retrieval_quality REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    session_id TEXT
);

CREATE INDEX idx_session_memories_created ON session_memories(created_at);

CREATE VIRTUAL TABLE session_memories_vec USING vec0(
    embedding float[768] distance_metric=cosine
);

-- What retrieval strategies work for what query patterns
CREATE TABLE strategy_memories (
    id INTEGER PRIMARY KEY,
    query_pattern TEXT NOT NULL,
    strategy_type TEXT NOT NULL DEFAULT 'routing',  -- 'routing' or 'retrieval'
    successful_tools TEXT NOT NULL DEFAULT '[]',
    failed_tools TEXT NOT NULL DEFAULT '[]',
    best_sources TEXT NOT NULL DEFAULT '[]',
    times_reinforced INTEGER NOT NULL DEFAULT 1,
    avg_quality REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_used TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_strategy_memories_pattern ON strategy_memories(query_pattern);
CREATE INDEX idx_strategy_memories_type ON strategy_memories(strategy_type);

-- Evidence-based trust events (append-only audit trail)
CREATE TABLE trust_events (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,  -- 'contradiction', 'correction', 'dead_url', 'confirmation', 'corroboration'
    delta REAL NOT NULL,
    details TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_trust_events_source ON trust_events(source_id);
```

---

## Step 1: Cross-Encoder Reranker (Replace Qwen 4B Generative Reranker)

**Goal**: Replace the generative Qwen 4B reranker with a cross-encoder model for
faster, more accurate relevance scoring with continuous scores instead of
categorical buckets.

**Why**: The current `rag/reranker.py` sends all 20 search results to Qwen 4B and
asks it to categorize each as HIGH/MEDIUM/LOW. This is a full autoregressive
generation for what should be a scoring operation. Cross-encoder models like
`cross-encoder/ms-marco-MiniLM-L-6-v2` score query-document pairs directly —
they're 20-40x faster and produce continuous relevance scores.

**Files to modify:**
- `requirements.txt` — add `sentence-transformers>=3.0`
- `code/shukketsu/rag/reranker.py` — replace generative reranker with cross-encoder
- `code/shukketsu/rag/search.py` — add `rerank_score: float | None = None` to `SearchResult`
- `code/shukketsu/config.py` — add `RERANKER_MODEL` config, remove Qwen reranker config
- `tests/unit/test_reranker.py` — update tests for new interface
- `tests/unit/test_search_reranking.py` — update integration tests

**Design:**
- Keep the same public interface: `async def rerank(query, results, top_k) -> list[SearchResult]`
- Use `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, ~5ms per pair on CPU)
- Load model lazily on first call and cache (same pattern as `_get_client()`)
- Sort by continuous cross-encoder score (descending), take top_k
- Populate `rerank_score` on returned `SearchResult` via `dataclasses.replace()`
- Keep circuit breaker pattern — fallback to unreranked results truncated to `top_k`
- Log reranking latency to Langfuse metadata
- Runs on CPU via PyTorch (sm_121 kernels not needed — model is tiny)

**Tests**: Mock the cross-encoder model. Test: empty results, fewer results than
top_k, normal reranking, fallback on model failure, verify sort order matches
score order.

---

## Step 2: Parallel Subtask Execution in Orchestrator

**Goal**: Execute independent subtasks concurrently via `asyncio.gather()` for
faster multi-research queries.

**Why**: The Orchestrator's `_dispatch()` processes subtasks sequentially. When two
Researcher subtasks have no dependencies, they still run one after another.

**Files to modify:**
- `code/shukketsu/agents/orchestrator.py` — modify `_dispatch()` to group and parallelize
- `tests/unit/test_orchestrator_execute.py` — add tests for parallel execution

**Design:**
- Add `_group_by_level()` method: takes topological order + subtask dependency info, groups indices by depth
- Extract per-subtask logic into `_execute_single()` method
- Within each level, `asyncio.gather(*coros, return_exceptions=True)` for concurrent execution
- Single-subtask levels run normally (no gather overhead)
- Failed subtasks converted to `FAILED` AgentResult (same as existing try/except)
- `on_status` callback fires per subtask (concurrent interleaved messages are fine)
- Cross-level execution remains sequential

**Tests**: Create a plan with 2 independent Researcher subtasks, mock agent
execution with `asyncio.sleep()` delays, verify both complete in ~max(delays)
not sum(delays). Test dependent subtasks still wait.

---

## Step 3: WebSocket Agent Step Streaming

**Goal**: Stream intermediate agent steps to the user in real-time instead of
only the final answer.

**Why**: Multi-agent coordination means 20-60 seconds of silence. Streaming
"Researcher: searching knowledge graph..." transforms perceived quality.

**Files to modify:**
- `code/shukketsu/agents/base.py` — update `StatusCallback` type, emit structured events from `_run_loop()`
- `code/shukketsu/agents/orchestrator.py` — emit dispatch/phase events
- `code/shukketsu/web/routers/chat.py` — enhance `_send_status()` to handle `str | dict`

**Design:**
- Widen `StatusCallback = Callable[[str | dict[str, Any]], Awaitable[None]]`
- All existing `on_status("string")` calls remain unchanged (backward compatible)
- New structured events alongside existing `{"type": "status"}`:
  - `{"type": "step", "agent": "researcher", "action": "tool_call", "tool": "rag_search", "input": {...}}`
  - `{"type": "step", "agent": "orchestrator", "action": "dispatch", "subtask": "...", "target": "researcher"}`
  - `{"type": "step", "agent": "editor", "action": "verify_claim", "claim_index": 3, "total": 5}`
- `_send_status()` in chat.py: if arg is dict, send as-is; if str, wrap in `{"type": "status", "content": str}`
- Don't change `{"type": "done"}` response format
- In `BaseAgent._run_loop()`, emit tool call details after execution (~line 158)
- In `Orchestrator.execute()`, emit dispatch events

**Tests**: Unit test callback receives expected arguments. Test both str and dict
paths through the handler.

---

## Step 4: Context Compaction in ReAct Loop

**Goal**: Prevent context window overflow by truncating older observations when
the scratchpad grows too large.

**Why**: With up to 10 iterations and large tool observations, the context grows
linearly. The LoopDetector catches budget overruns at 100K but by then context
is degraded.

**Files to modify:**
- `code/shukketsu/agents/base.py` — add compaction to `_build_messages()`
- `code/shukketsu/config.py` — add `COMPACTION_THRESHOLD_TOKENS` (default: 60000)
- `tests/unit/test_base_agent.py` — add compaction tests

**Design (mechanical truncation, no LLM):**
- Trigger in `_build_messages()` when estimated tokens exceed `COMPACTION_THRESHOLD_TOKENS`
- Keep the 2 most recent scratchpad entries verbatim
- For older entries: truncate each observation to ~200 chars, preserving tool name + first/last lines
- Format: `"[rag_search] Found 5 results about hit cap... [truncated] ...363 rating for 9%"`
- Prefix compacted observations with `"[Summarized]"` so the agent knows context is truncated
- Compaction is idempotent — mark compacted entries to avoid re-processing
- Token estimation: `len(text) // 4` (rough heuristic, same as existing patterns)

**Compaction flow:**
```
Scratchpad: [obs1, obs2, obs3, obs4, obs5, obs6]
                                        ^ threshold exceeded
After compaction:
  messages = [
    system prompt,
    user query,
    obs1 (truncated to 200 chars),
    obs2 (truncated to 200 chars),
    obs3 (truncated to 200 chars),
    obs4 (truncated to 200 chars),
    obs5 (verbatim),              <- recent
    obs6 (verbatim),              <- recent
  ]
```

**Tests**: Scratchpad below threshold (no compaction). Scratchpad above threshold
(compaction fires, older entries truncated). Verify recent entries preserved
verbatim. Verify idempotency.

---

## Step 5: Reflection Before Final Answer (Complex Queries Only)

**Goal**: Add a generate-critique-revise cycle to the Researcher, but only for
complex queries routed through the Orchestrator.

**Why**: Chat responses go straight to the user unreviewed. One Llama 70B call
checking "are your claims supported by your observations?" catches unsupported
claims and hallucinations.

**Files to create/modify:**
- `code/shukketsu/agents/researcher.py` — add reflection step after structuring pass
- `code/shukketsu/llm/prompts/researcher.py` — add `REFLECTION_PROMPT`
- `code/shukketsu/llm/schemas.py` — add `ReflectionResult` model
- `code/shukketsu/config.py` — add `REFLECTION_ENABLED = True`, `REFLECTION_TEMPERATURE = 0.1`
- `tests/unit/test_researcher.py` — add reflection tests

**Design:**
- Reflection runs after structuring pass succeeds, before `ResearchResult` is returned
- **Only fires when `task.metadata.get("complexity") == "complex"`** — the Orchestrator
  sets this when it creates typed tasks for specialist agents
- Uses `ReflectionResult` Pydantic model: `supported: bool`, `issues: list[str]`, `revised_answer: str`
- Same model (Llama 70B) at temperature 0.1
- Prompt receives: agent's final answer + all tool observations from scratchpad
- If `supported=True`: return original answer unchanged
- If `supported=False`: use `revised_answer`, log diff to Langfuse
- If reflection fails (circuit breaker, LLM error): return original answer (non-blocking)
- Skip for FAILED/PARTIAL outcomes
- Log metrics to Langfuse: `reflection_fired`, `claims_revised`, `latency_ms`

**Tests**: Well-supported answer (unchanged). Unsupported claim (revised). Reflection
failure (original returned). Skip for non-complex queries. Skip for FAILED/PARTIAL.

---

## Step 6: Memory Foundation — Schema + MemoryManager

**Goal**: Add persistent cross-session memory with schema v4 migration and a
`MemoryManager` class.

**Why**: Every session starts from zero. The agent can't recall previous answers,
user corrections, or which retrieval strategies work best.

**Files to create/modify:**
- `code/shukketsu/db/schema.sql` — add memory + trust_events tables (schema v4)
- `code/shukketsu/db/connection.py` — add `_migrate_v3_to_v4()` migration
- `code/shukketsu/memory/__init__.py` — new package
- `code/shukketsu/memory/manager.py` — `MemoryManager` class
- `code/shukketsu/memory/models.py` — Pydantic models for memory records
- `code/shukketsu/config.py` — add memory config constants
- `tests/unit/test_memory_manager.py` — comprehensive tests

**MemoryManager design:**
```python
class MemoryManager:
    def __init__(self, conn: sqlite3.Connection, embed_fn: Callable) -> None: ...

    # --- Extraction (fire-and-forget, uses Qwen 4B) ---
    async def extract_session_memory(
        self, query: str, answer: str, trajectory: list[ToolCallRecord],
        session_id: str | None = None,
    ) -> None: ...

    # --- Recall (called before agent runs) ---
    async def recall_relevant(self, query: str, top_k: int = 5) -> list[SessionMemory]:
        """Composite scoring: 0.6 * cosine_sim + 0.2 * recency + 0.2 * quality"""

    # --- Strategy tracking ---
    async def record_strategy(self, query: str, tools_used: list[str], quality: float) -> None: ...
    async def recall_strategies(self, query: str, top_k: int = 3) -> list[StrategyMemory]: ...
```

**Design constraints:**
- SQLite + sqlite-vec for embedding search (reuse existing `Embedder.embed_query()`)
- Recency score: `exp(-days_since_creation / 30)` (30-day half-life)
- Strategy memory uses exact pattern matching (categorical), not embedding search
- `strategy_type` field: `'routing'` (for Orchestrator) or `'retrieval'` (for Researcher)
- All writes fire-and-forget via `asyncio.create_task()`, errors logged, never block
- Extraction uses Qwen 4B via `get_structured_output` with `MemoryExtraction` schema

**Tests**: Extract stores to DB. Recall returns scored results. Composite scoring
formula verified. Strategy record/recall. Empty DB returns empty lists.
Fire-and-forget doesn't raise.

---

## Step 7: Memory Integration — Recall + Extraction Hooks

**Goal**: Wire MemoryManager into the chat pipeline.

**Files to modify:**
- `code/shukketsu/web/routers/chat.py` — hook recall and extraction into `_agent_response()`
- `code/shukketsu/agents/base.py` — accept optional memory context in system prompt
- `code/shukketsu/config.py` — add `MEMORY_ENABLED = True`, `MEMORY_RECALL_TOP_K = 5`
- `tests/unit/test_chat_handler.py` — memory integration tests

**Design:**
- **Recall hook** (before routing): query MemoryManager, format as system prompt addition
- **Extraction hook** (after response, fire-and-forget): extract facts via Qwen 4B, store
- **Strategy recording** (after response, fire-and-forget): record tools used + quality
- Memory injection goes into system prompt (not user message), max ~500 tokens
- MemoryManager created once alongside other singletons in `_get_agents()`
- `MEMORY_ENABLED` toggle disables both recall and extraction

**Tests**: Query triggers extraction. Recall injects context into system prompt.
Toggle off disables both.

---

## Step 8: Strategy Memory for Orchestrator Routing

**Goal**: Feed strategy memory into the Orchestrator's decomposition prompt.

**Files to modify:**
- `code/shukketsu/agents/orchestrator.py` — accept strategy hints in decomposition
- `code/shukketsu/llm/prompts/orchestrator.py` — modify `DECOMPOSITION_PROMPT`
- `code/shukketsu/web/routers/chat.py` — pass strategy hints to orchestrator
- `tests/unit/test_orchestrator_build_task.py` — update tests

**Design:**
- Strategy hints injected into decomposition prompt as advisory context
- Format: "Based on previous queries: [pattern] works best with [tools]"
- Max 3 hints to avoid prompt bloat
- `strategy_type='routing'` hints go to Orchestrator; `'retrieval'` go to Researcher
- Orchestrator is free to ignore hints — they're advisory

**Tests**: Strategy hints appear in prompt. Orchestrator works with empty hints.

---

## Step 9: Evidence-Based Trust Scoring

**Goal**: Replace time-based trust decay with evidence-based trust updates.

**Why**: WoW TBC is a solved game — old sources aren't less reliable. Trust should
change only when evidence warrants it.

**Files to modify:**
- `code/shukketsu/trust/scoring.py` — replace with evidence-based trust
- `code/shukketsu/agents/editor.py` — emit trust events on contradiction
- `code/shukketsu/memory/manager.py` — emit trust events on user feedback
- `code/shukketsu/freshness/checker.py` — emit trust events on dead URLs
- `tests/unit/test_trust_scoring.py` — update tests

**Design:**
- New formula: `effective_trust = base_trust + sum(event_deltas)`, clamped [0.1, 1.0]
- Full replacement — no time decay. Old function preserved as `_effective_trust_legacy()`
- Trust events are append-only (audit trail)
- Event types and deltas:
  - `contradiction` (Editor): -0.1
  - `correction` (user feedback): -0.15
  - `dead_url` (freshness checker): trust = 0.1
  - `confirmation` (user feedback): +0.05
  - `corroboration` (multi-source): +0.05
- User feedback via new WebSocket message: `{"type": "feedback", "rating": "positive"|"negative"|"correction"}`

**Tests**: Trust increases/decreases. Clamping. Editor emits on contradiction.
Legacy function preserved.

---

## Implementation Order

Steps 1-5 have no dependencies on each other. Steps 6-9 are sequential.

Recommended execution order:
1. Step 1 — Cross-encoder reranker (fastest win)
2. Step 2 — Parallel execution (visible speedup)
3. Step 3 — WebSocket streaming (UX transformation)
4. Step 4 — Context compaction (reliability)
5. Step 5 — Reflection (quality, complex queries only)
6. Step 6 — Memory foundation (schema v4 + MemoryManager)
7. Step 7 — Memory hooks (wire into pipeline)
8. Step 8 — Strategy routing (memory-informed orchestration)
9. Step 9 — Evidence-based trust (completes the learning loop)

## Test Requirements

- All 741 existing tests must continue to pass after each step
- Each step adds its own tests
- No test may require external services — mock everything
- Use `python3 -m pytest` (not bare `pytest`)
- Run `ruff check . --fix && ruff format . && python3 -m mypy . && python3 -m pytest` after each step
