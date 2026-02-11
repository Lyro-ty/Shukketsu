# Phase 2 Step 3: Graph Traversal Tool + Qwen 4B Reranking

> Design document. Validated through brainstorming session 2026-02-10.

## Overview

Step 3 adds two capabilities to the agentic retrieval system:

1. **Graph search tool** — Agents can query the knowledge graph for entity
   relationships (built in Step 2). Entity-centric, bidirectional, single-hop.
2. **Qwen 4B reranker** — The `rag_search` tool over-retrieves and uses Qwen 4B
   to filter results by relevance before returning them to the agent.

These are independent components. The graph search tool is a new agent tool. The
reranker is a shared utility wired into the existing `rag_search` tool. The agent
decides which tool to use in its ReAct loop — no unified search function needed.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| How graph + hybrid merge | Separate tools, agent chooses | Preserves agent agency; Researcher (Step 4) is designed to pick strategies |
| Graph result format | Entity-centric (not path-centric) | Simpler, matches GraphStore API, agent can chain single-hop queries |
| When reranking happens | Tool-level (transparent to agent) | Agent gets better results without learning a "rerank" tool |
| Reranker scoring | Categorical (HIGH/MEDIUM/LOW) | Robust to LLM scoring inconsistency; tiebreak by original RRF score |
| Reranker scope | `rag_search` only, not `graph_search` | Graph results are structural (correct or not); confidence score suffices |
| Multi-hop traversal | Not implemented (YAGNI) | Agent chains single-hop queries; add later if Researcher needs it |

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `tools/knowledge/graph_search.py` | Graph traversal tool for agent |
| `rag/reranker.py` | Qwen 4B categorical relevance scoring |
| `tests/unit/test_graph_search_tool.py` | Graph search tool tests |
| `tests/unit/test_reranker.py` | Reranker tests |
| `tests/unit/test_search_reranking.py` | Reranker integration with rag_search |

### Modified Files

| File | Change |
|------|--------|
| `rag/graph.py` | Add `find_entities_by_name()` method; extend `get_relationships()` to JOIN entity_types |
| `config.py` | Add `GRAPH_SEARCH_DEFAULT_TOP_K`, `RERANKER_FETCH_MULTIPLIER`, `RERANKER_TOP_K`, CB constants |
| `resilience/errors.py` | Add `RerankerError`, `GraphTraversalError` |
| `resilience/circuit_breaker.py` | Add `qwen_reranker_breaker` named instance; update `reset_all_breakers()` |
| `tools/knowledge/search.py` | Wire reranker into `RagSearchTool.execute()` |
| `tools/knowledge/__init__.py` | Export `GraphSearchTool` |

### Unchanged Files

`agents/base.py`, `agents/tasks.py`, `agents/factory.py`,
`rag/entities.py`, `rag/fusion.py`, `rag/search.py`, `tools/registry.py`,
`tools/schemas.py`.

---

## Component 1: Graph Search Tool

`tools/knowledge/graph_search.py`

### Tool Definition

```python
class GraphSearchTool(Tool):
    name = "graph_search"
    description = """Search the WoW TBC knowledge graph for entity relationships.
    Use when you need connections between game concepts:
    - "What items drop from [boss]?"
    - "What stats does [item] have?"
    - "What's BiS for [spec] in [phase]?"
    - "What talents synergize with [spell]?"
    Returns entity relationships and linked chunk IDs for follow-up with rag_search."""

    parameters_schema = {
        "entity": {
            "type": "str",
            "description": "Entity name to search for (e.g., 'Dragonspine Trophy', 'Gruul', 'DST')",
        },
        "relation_types": {
            "type": "list[str]",
            "description": "Filter by relationship type (e.g., ['drops_from', 'has_stat'])",
            "optional": True,
        },
        "target_type": {
            "type": "str",
            "description": "Filter target entity type (e.g., 'boss', 'stat')",
            "optional": True,
        },
    }
```

### Constructor

```python
def __init__(self, conn: sqlite3.Connection) -> None:
    self._graph = GraphStore(conn)
```

No embedding function needed (unlike `rag_search`). Pure SQLite queries.

### Execute Flow

1. Extract `entity`, `relation_types`, `target_type` from `tool_input`
2. Resolve entity name: `canonical = resolve_canonical(entity)`
3. Look up entity: `GraphStore.find_entities_by_name(canonical)` (new method, returns all matches)
4. **Ambiguity check**: if multiple matches returned, return disambiguation message
   listing matches with their types. If exactly one, proceed. If none, return not found.
5. If not found: return helpful message suggesting alternative names or rag_search
6. Fetch outgoing: `get_relationships(entity_id, direction="outgoing", relation_types=...)`
7. Fetch incoming: `get_relationships(entity_id, direction="incoming", relation_types=...)`
8. Apply `target_type` filter (post-query filter on joined entity type name)
9. Sort by confidence descending
10. Truncate to `GRAPH_SEARCH_DEFAULT_TOP_K` (20)
11. Collect `source_chunk_id` values (filter NULLs) from entities and relationships
12. Format as observation string

### Output Format

```
Entity: Dragonspine Trophy (item, confidence: 0.90)

Outgoing relationships (3):
- drops_from → Gruul the Dragonkiller (boss, confidence: 0.90)
- available_in → Phase 1 (phase, confidence: 0.95)
- has_stat → haste proc (stat, confidence: 0.85)

Incoming relationships (1):
- best_in_slot ← Combat Swords (spec, confidence: 0.70)

Related chunks: 42, 78, 103
```

- `→` for outgoing, `←` for incoming (clear direction for LLM)
- Related chunks from `source_chunk_id` on entities + relationships (NULLs filtered)
- Agent can use chunk IDs in follow-up `rag_search` for full text

### Ambiguity Handling

If `resolve_canonical("combat")` matches multiple entities across types:

```
Multiple entities match "combat":
- combat (talent_tree)
- combat swords (spec)
- combat potency (talent)

Please specify with the target_type parameter or use a more specific name.
```

Implementation: `GraphStore.find_entities_by_name(name)` returns all rows matching
the canonical name. The tool checks `len(matches)` and branches accordingly.

### Error Handling

- Entire `execute()` wrapped in try/except
- DB errors → `"Graph search failed: {error}. Try rag_search instead."`
- No circuit breaker needed (local SQLite, no network)
- `GraphTraversalError` raised only for genuinely unexpected failures

---

## Component 2: Qwen 4B Reranker

`rag/reranker.py`

### Pydantic Schemas

```python
class RankedItem(BaseModel):
    """A single result's relevance assessment."""
    index: int
    relevance: Literal["HIGH", "MEDIUM", "LOW"]
    reason: str

class RerankerResponse(BaseModel):
    """Batch relevance assessment from Qwen 4B."""
    rankings: list[RankedItem]
```

### Core Function

```python
async def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int = 5,
) -> list[SearchResult]:
```

### Flow

1. **Short-circuit**: if `len(results) <= top_k`, return as-is (nothing to filter)
2. **Call Qwen 4B** via circuit breaker:
   ```python
   try:
       return await qwen_reranker_breaker.call(_rerank_impl, query, results, top_k)
   except CircuitOpenError:
       return results[:top_k]
   ```
3. **Build prompt**: system message + user message with query and numbered results
   (truncated to ~200 chars each)
4. **Structured output**: `get_structured_output(RerankerResponse, messages, backend=ModelBackend.ROUTER)`
5. **Validate indices**: ignore out-of-range, deduplicate, treat unranked as LOW
6. **Partition** into HIGH / MEDIUM / LOW buckets
7. **Merge**: all HIGHs + MEDIUMs up to `top_k`, preserving original RRF score
   order within each bucket
8. Return filtered list

### Prompt Design

System message:
```
You are a relevance scorer for WoW TBC Rogue search results.
Given a query and numbered search results, rate each result's
relevance as HIGH, MEDIUM, or LOW.

HIGH: Directly answers or is essential to answering the query.
MEDIUM: Related and potentially useful but not a direct answer.
LOW: Irrelevant or only tangentially related.

Be strict — only mark HIGH if the result clearly helps answer the query.
```

User message:
```
Query: {query}

Results:
[0] {title} — {content[:200]}
[1] {title} — {content[:200]}
...
```

Single call, all results scored at once. 10-15 snippets at ~200 chars each fits
comfortably in Qwen 4B context. Expected latency ~200ms.

### Circuit Breaker

```python
qwen_reranker_breaker = CircuitBreaker(
    "qwen_reranker",
    failure_threshold=3,
    recovery_timeout=30.0,
)
```

Same profile as `reasoning_breaker`. On failure: return `results[:top_k]`
(unreranked, truncated). Degraded but functional. The caller never sees an
exception from `rerank()`.

### Defensive Validation

Qwen 4B output may be imperfect. Handle:

- **Out-of-range indices** (index >= len(results) or < 0): silently skip
- **Duplicate indices**: keep first occurrence only
- **Missing results** (not all indices ranked): treat unranked as LOW
- **Empty rankings list**: fall back to `results[:top_k]`

---

## Component 3: Wiring Changes

### config.py

```python
# Graph search
GRAPH_SEARCH_DEFAULT_TOP_K = int(os.getenv("GRAPH_SEARCH_DEFAULT_TOP_K", "20"))

# Reranker
RERANKER_FETCH_MULTIPLIER = int(os.getenv("RERANKER_FETCH_MULTIPLIER", "3"))
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))
```

### resilience/errors.py

```python
class RerankerError(ShukketsuError):
    """Raised when the reranker LLM call fails."""
    def __init__(self, message: str):
        super().__init__(message, FailureMode.MODEL_UNAVAILABLE)

class GraphTraversalError(ShukketsuError):
    """Raised when a knowledge graph query fails."""
    def __init__(self, message: str):
        super().__init__(message, FailureMode.DB_ERROR)
```

### tools/knowledge/search.py (RagSearchTool changes)

Three changes to `execute()`:

1. Import `rerank` from `rag.reranker`
2. Over-retrieve: `hybrid_search(..., top_k=top_k * RERANKER_FETCH_MULTIPLIER, fetch_k=max(RAG_SEARCH_FETCH_K, top_k * RERANKER_FETCH_MULTIPLIER))`
3. Rerank: `results = await rerank(query, raw_results, top_k)`

The FTS-only fallback path does NOT get reranking (already in degraded mode).

### Gotcha: fetch_k scaling

When `top_k * RERANKER_FETCH_MULTIPLIER` exceeds the default `fetch_k` (20), the
per-source candidate pool would be too small. Fix:

```python
rerank_fetch = top_k * RERANKER_FETCH_MULTIPLIER
effective_fetch_k = max(RAG_SEARCH_FETCH_K, rerank_fetch)
raw_results = await hybrid_search(conn, query, embedding, top_k=rerank_fetch, fetch_k=effective_fetch_k)
```

---

## Test Plan

### tests/unit/test_graph_search_tool.py (~14 tests)

**Entity lookup:**
- `test_search_existing_entity_returns_relationships`
- `test_search_with_alias_resolves_canonical` (DST → dragonspine trophy)
- `test_search_nonexistent_entity_returns_helpful_message`
- `test_search_ambiguous_entity_returns_disambiguation`

**Relationship filtering:**
- `test_filter_by_relation_type_returns_only_matching`
- `test_filter_by_target_type_returns_only_matching`
- `test_both_filters_combined`

**Bidirectional queries:**
- `test_outgoing_relationships_shown_with_arrow`
- `test_incoming_relationships_shown_with_reverse_arrow`
- `test_entity_with_both_directions_shows_both_sections`

**Output format:**
- `test_related_chunks_collected_from_entities_and_relationships`
- `test_null_source_chunk_ids_filtered_from_output`
- `test_results_sorted_by_confidence_descending`
- `test_results_truncated_to_top_k`

### tests/unit/test_reranker.py (~12 tests)

**Happy path:**
- `test_rerank_filters_low_keeps_high`
- `test_rerank_fills_medium_up_to_top_k`
- `test_rerank_preserves_original_order_within_category`
- `test_rerank_all_high_returns_top_k_best`

**Short-circuit:**
- `test_rerank_skips_when_results_lte_top_k`
- `test_rerank_empty_results_returns_empty`

**Circuit breaker:**
- `test_rerank_falls_back_to_truncated_on_circuit_open`
- `test_rerank_falls_back_on_llm_error`

**Defensive validation:**
- `test_rerank_ignores_out_of_range_indices`
- `test_rerank_deduplicates_indices`
- `test_rerank_treats_unranked_as_low`
- `test_reranker_response_model_validates`

### tests/unit/test_search_reranking.py (~6 tests)

- `test_rag_search_over_retrieves_with_multiplier`
- `test_rag_search_passes_results_through_reranker`
- `test_rag_search_fetch_k_scales_with_multiplier`
- `test_rag_search_fts_fallback_skips_reranking`
- `test_rag_search_reranker_failure_returns_unreranked`
- `test_rag_search_top_k_param_respected_after_reranking`

### Error & config tests (~3 tests)

- `test_reranker_error_inherits_shukketsu_error`
- `test_graph_traversal_error_inherits_shukketsu_error`
- `test_qwen_reranker_breaker_exists_with_correct_config`

### Total: ~35 new tests (381 → ~416)

Test patterns:
- Graph search tests use `test_db` fixture with seeded entities/relationships
- Reranker tests mock `get_structured_output` to return controlled responses
- Search reranking tests mock both `hybrid_search` and `rerank`

---

## What This Does NOT Include

- **Multi-hop traversal** — Agent chains single-hop queries (YAGNI)
- **Graph search reranking** — Graph results are structural; confidence suffices
- **Unified search function** — Agent picks tools in ReAct loop
- **Researcher agent wiring** — That's Step 4
- **Search strategy classification** — Agent decides, not a classifier function
- **`rag/search.py` changes** — `hybrid_search` itself is unchanged; changes are in the tool layer

## Dependencies

- Phase 2 Step 2 complete (GraphStore, entity types, extraction pipeline)
- Existing: `tools/schemas.py` (Tool ABC), `tools/registry.py`, `resilience/circuit_breaker.py`
- Existing: `llm/structured.py` (get_structured_output with ModelBackend.ROUTER)
- Existing: `rag/search.py` (hybrid_search), `rag/entities.py` (resolve_canonical)

---

## GraphStore Changes (rag/graph.py)

Two small additions needed, discovered during design review:

### 1. `find_entities_by_name()` — returns all matches

```python
def find_entities_by_name(
    self,
    name: str,
    entity_type: EntityType | None = None,
) -> list[sqlite3.Row]:
    """Look up all entities matching a name (resolves aliases).

    Unlike get_entity_by_name() which returns one row, this returns all
    matches — needed for ambiguity detection in graph_search tool.
    """
    canonical = resolve_canonical(name)
    if entity_type is not None:
        type_id = self.get_entity_type_id(entity_type)
        return self._conn.execute(
            "SELECT e.*, et.name AS entity_type_name "
            "FROM entities e JOIN entity_types et ON et.id = e.entity_type_id "
            "WHERE e.canonical_name = ? AND e.entity_type_id = ?",
            (canonical, type_id),
        ).fetchall()
    return self._conn.execute(
        "SELECT e.*, et.name AS entity_type_name "
        "FROM entities e JOIN entity_types et ON et.id = e.entity_type_id "
        "WHERE e.canonical_name = ?",
        (canonical,),
    ).fetchall()
```

### 2. Extend `get_relationships()` — include entity type name

Current query only joins `entities` for name/canonical_name. Add a second JOIN
to `entity_types` so the tool can display type names in output:

```python
query = (
    f"SELECT r.*, e.name AS related_name, e.canonical_name AS related_canonical, "
    f"et.name AS related_type "
    f"FROM relationships r "
    f"JOIN entities e ON e.id = r.{join_col} "
    f"JOIN entity_types et ON et.id = e.entity_type_id "
    f"WHERE r.{col} = ?"
)
```

This adds `related_type` (e.g., "boss", "stat", "phase") to each relationship row.

### Impact on existing tests

The existing `test_graph.py` tests call `get_relationships()` and check returned
columns. The new `related_type` column is additive — existing assertions on
`related_name` and `related_canonical` still pass. One or two existing tests
may need updating if they assert on the exact set of columns returned.

---

## Additional Implementation Notes

### relation_types string → enum conversion

The tool parameter `relation_types` is `list[str]` but `GraphStore.get_relationships()`
takes `list[RelationType]`. The tool converts with graceful handling:

```python
def _parse_relation_types(raw: list[str] | None) -> list[RelationType] | None:
    if not raw:
        return None
    valid = []
    for rt_str in raw:
        try:
            valid.append(RelationType(rt_str))
        except ValueError:
            logger.warning("Unknown relation type '%s', skipping", rt_str)
    return valid or None
```

### Entity type name for queried entity

The output header shows `Entity: Dragonspine Trophy (item, confidence: 0.90)`.
Since `find_entities_by_name()` joins entity_types, the returned row includes
`entity_type_name` directly. No extra lookup needed.

### config.py circuit breaker constants

```python
CB_QWEN_RERANKER_FAILURE_THRESHOLD = int(os.getenv("CB_QWEN_RERANKER_FAILURE_THRESHOLD", "3"))
CB_QWEN_RERANKER_RECOVERY_TIMEOUT = float(os.getenv("CB_QWEN_RERANKER_RECOVERY_TIMEOUT", "30"))
```

### reset_all_breakers() update

```python
def reset_all_breakers() -> None:
    for breaker in (reasoning_breaker, ollama_router_breaker, ollama_embed_breaker,
                    brave_breaker, qwen_reranker_breaker):
        breaker.reset()
```
