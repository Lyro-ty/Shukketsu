# Step 10: Observability — Langfuse Tracing

> Design spec for Phase 1, Step 10. Adds LLM-specific observability using
> Langfuse v3 self-hosted, with `@observe` decorators on existing functions.

## Goal

See what the agent is doing internally — every routing decision, tool call,
and LLM generation traced and viewable in the Langfuse UI.

## Approach: `@observe` Decorators

Langfuse v3 Python SDK offers three integration patterns. We use **`@observe()`
decorators** directly on existing functions because:

- Minimal code changes — just add decorators, no wrapper layers
- Auto-nesting based on Python call hierarchy matches our architecture
- Works with async functions and class methods
- No-op when `LANGFUSE_TRACING_ENABLED=false` or credentials are invalid
- Doesn't conflict with Instructor (which wraps the OpenAI client)
- Supports typed spans: `as_type="generation"`, `"agent"`, `"tool"`

**Rejected alternatives:**
- OpenAI drop-in replacement — conflicts with Instructor wrapping
- Low-level context managers — too verbose, no benefit over decorators

## Trace Hierarchy

Every user query creates one trace with nested spans:

```
Trace: "user-query" (session_id, tags=["chat"])
+-- Span: "classify_query"
|   +-- Generation: get_structured_output (model=qwen3:4b)
+-- Agent: "BaseAgent.run"
|   +-- Span: "step_1"
|   |   +-- Generation: get_structured_output (model=llama-3.3-70b, AgentStep)
|   |   +-- Tool: "rag_search" (input, output, latency)
|   +-- Span: "step_2"
|       +-- Generation: get_structured_output (model=llama-3.3-70b, final_answer)
```

## Files

### New Files

| File | Purpose |
|------|---------|
| `observability/tracer.py` | Init function, flush helper, re-exports `observe` and `get_client` |
| `infra/docker-compose.langfuse.yml` | Self-hosted Langfuse v3 stack (Postgres, ClickHouse, Redis, MinIO, web, worker) |
| `tests/unit/test_tracer.py` | Unit tests for init, flush, no-op behavior |

### Modified Files

| File | Change |
|------|--------|
| `config.py` | Add `LANGFUSE_TRACING_ENABLED`, `LANGFUSE_SAMPLE_RATE` |
| `web/app.py` | Call `init_langfuse()` on startup, `flush_traces()` on shutdown |
| `web/routers/chat.py` | `@observe()` on `_agent_response`, set trace session_id/tags |
| `routing/router.py` | `@observe()` on `classify_query` |
| `agents/base.py` | `@observe(as_type="agent")` on `run()` |
| `llm/structured.py` | `@observe(as_type="generation")` on `get_structured_output`, record model metadata |
| `tools/registry.py` | `@observe(as_type="tool")` on `execute()`, record tool_name in metadata |

## Implementation Details

### `observability/tracer.py`

```python
def init_langfuse() -> None:
    """Configure Langfuse singleton from config vars. Call once at startup."""
    # Pushes LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
    # into env vars, then eagerly initializes the client via get_client().
    # No-op if LANGFUSE_TRACING_ENABLED is false.

def flush_traces() -> None:
    """Flush buffered traces. Call on shutdown."""
```

### Instrumentation Pattern

Each module adds a decorator and optionally updates span metadata:

```python
from langfuse import observe, get_client

@observe(as_type="generation")
async def get_structured_output(...) -> T:
    langfuse = get_client()
    langfuse.update_current_generation(
        model=model,
        model_parameters={"temperature": temperature, "max_tokens": max_tokens},
    )
    # ... existing code unchanged ...
```

The `@observe` decorator captures function inputs/outputs automatically.
`update_current_generation()` adds LLM-specific metadata (model name, params).

### Docker Compose

Six containers on a shared network:

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `langfuse-web` | `langfuse/langfuse:3` | 3000 | Web UI + API |
| `langfuse-worker` | `langfuse/langfuse-worker:3` | 3030 | Background processing |
| `postgres` | `postgres:16` | 5432 | Metadata storage |
| `clickhouse` | `clickhouse/clickhouse-server` | 8123 | Trace/event storage |
| `redis` | `redis:7` | 6379 | Queue/cache |
| `minio` | `minio/minio` | 9090 | S3-compatible blob storage |

All use named volumes for persistence. The Compose file is standalone — user
copies it and runs `docker compose up -d`.

### Config Additions

```python
# config.py
LANGFUSE_TRACING_ENABLED = os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() == "true"
LANGFUSE_SAMPLE_RATE = float(os.getenv("LANGFUSE_SAMPLE_RATE", "1.0"))
```

## Testing

### Unit Tests (`test_tracer.py`, ~5 tests)

| Test | Verifies |
|------|----------|
| `test_init_langfuse_sets_env_vars` | Config values pushed to env |
| `test_init_langfuse_disabled` | No-op when tracing disabled |
| `test_flush_traces_no_client` | Safe when never initialized |
| `test_flush_traces_with_mock_client` | Calls `client.flush()` |
| `test_observe_noop_when_disabled` | Decorated function returns normally |

### Existing Tests (240)

Unchanged. SDK degrades to no-op without valid credentials. `@observe`
decorators become passthrough — zero impact.

## Gate

Ask a question in the browser, then open Langfuse UI. Find the trace, see
the full routing -> agent -> tool -> response span tree with model metadata.
