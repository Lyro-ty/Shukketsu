# Step 7: Multi-Model Router — Design

> Qwen 4B classifies queries; trivial ones get a fast direct answer, everything
> else goes to the Llama 70B agent with tools.

## Design Decisions

1. **Single-call classify + answer** — Qwen classifies AND answers trivial
   queries in one structured output call (~200ms). No second call needed.
2. **Full taxonomy, simple branching** — Keep the existing `RoutingDecision`
   model with all complexity/category values. Router logic only branches on
   `TRIVIAL` vs everything else. Extra classification data is logged for
   Phase 2.
3. **Same WebSocket protocol** — No new message types. Trivial answers use
   the existing `{"type": "done"}` format. Frontend unchanged.
4. **Route in chat handler** — Routing is orchestration, not agent logic.
   `_agent_response` calls the router first, then branches. `BaseAgent`
   stays a pure ReAct loop.

## Model Change

```python
# routing/models.py — add one field
class RoutingDecision(BaseModel):
    complexity: TaskComplexity
    category: TaskCategory
    needs_tools: bool
    suggested_agent: str
    direct_answer: str | None = None  # NEW: filled when TRIVIAL
```

## Router Module (`routing/router.py`)

Single async function:

```python
async def classify_query(query: str) -> RoutingDecision
```

- Builds a system prompt with WoW TBC Rogue domain context
- Calls `get_structured_output(backend=ModelBackend.OLLAMA, response_model=RoutingDecision,
  temperature=0.0, max_tokens=512)` — deterministic classification, low token budget
- On `LLMUnavailableError` or `StructuredOutputError`: returns fallback
  `RoutingDecision(complexity=COMPLEX, category=CONVERSATION, needs_tools=True,
  suggested_agent="general", direct_answer=None)` — routes to Llama 70B agent
- Logs classification result at INFO, fallback at WARNING

### Classification Prompt

Tells Qwen:
- **TRIVIAL**: Simple factual WoW TBC Rogue questions answerable in 1-2
  sentences (energy costs, ability names, basic stats, talent locations).
  Fill `direct_answer` with a concise answer.
- **MODERATE**: Questions needing knowledge base search (gear comparisons,
  rotation details, specific encounter advice).
- **COMPLEX**: Multi-step analysis, comparisons across specs/phases,
  simulation questions, anything needing multiple tools.

## Chat Handler Changes (`web/routers/chat.py`)

Updated `_agent_response` flow:

```
1. session.add_message("user", content)
2. Send {"type": "status", "content": "routing..."}
3. decision = await classify_query(content)
4. Log routing decision
5. If decision.complexity == TRIVIAL
      AND decision.direct_answer is not None
      AND decision.direct_answer.strip() != "":
     → session.add_message("assistant", decision.direct_answer)
     → Send {"type": "done", "content": decision.direct_answer}
6. Else:
     → Send {"type": "status", "content": "thinking..."}
     → answer = await agent.run(content)
     → session.add_message("assistant", answer)
     → Send {"type": "done", "content": answer}
```

Both paths explicitly record to chat history. Error handling unchanged.

## Error Handling & Edge Cases

| Scenario | Behavior |
|----------|----------|
| Ollama down | Fallback → agent (logged WARNING) |
| Qwen returns TRIVIAL but empty `direct_answer` | Fall through → agent |
| Qwen returns COMPLEX with `direct_answer` | Ignore answer → agent |
| `StructuredOutputError` from Qwen | Same as Ollama down → fallback |

## Tests (~8 new)

### `test_router.py` (new, ~4 tests)
- `test_classify_trivial_returns_direct_answer`
- `test_classify_complex_returns_no_direct_answer`
- `test_classify_fallback_on_ollama_unavailable`
- `test_classify_prompt_includes_domain_context`

### `test_chat_handler.py` (add ~3 tests)
- `test_trivial_query_returns_fast_answer`
- `test_complex_query_routes_to_agent`
- `test_routing_failure_falls_through_to_agent`

### `test_structured_schemas.py` (add ~1 test)
- `test_routing_decision_direct_answer_optional`

## Files Touched

| File | Change |
|------|--------|
| `routing/models.py` | Add `direct_answer` field |
| `routing/router.py` | New — `classify_query()` |
| `web/routers/chat.py` | Update `_agent_response` with routing branch |
| `tests/unit/test_router.py` | New — router unit tests |
| `tests/unit/test_chat_handler.py` | Add routing integration tests |
| `tests/unit/test_structured_schemas.py` | Add `direct_answer` field test |

## No Changes To

- `BaseAgent` / `agents/base.py`
- `llm/structured.py` / `llm/clients.py`
- `config.py` (all constants already exist)
- Frontend templates / JS / CSS
- Database schema
