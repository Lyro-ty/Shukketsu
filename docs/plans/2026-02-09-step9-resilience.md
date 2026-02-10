# Step 9: Resilience

> Design spec for circuit breakers, retry logic, agent guardrails, and graceful
> degradation. Builds on Steps 1-8 (195 tests passing).

## Goal

The system handles failures gracefully instead of crashing. External service
outages degrade functionality instead of breaking the entire system. Agent loops
are detected and terminated. Retries use exponential backoff.

---

## New Modules

### 1. Circuit Breaker (`resilience/circuit_breaker.py`)

Generic async circuit breaker with three states:

- **CLOSED** (normal): Requests pass through. Tracks consecutive failures.
  After `failure_threshold` failures, transitions to OPEN.
- **OPEN** (failing): Requests rejected immediately with `CircuitOpenError`.
  After `recovery_timeout` seconds, transitions to HALF_OPEN.
- **HALF_OPEN** (testing): Allows one request through. Success resets to
  CLOSED. Failure returns to OPEN.

```python
class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5,
                 recovery_timeout: float = 60.0): ...

    async def call(self, fn: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute fn through the breaker. Raises CircuitOpenError if open."""

    @property
    def state(self) -> CircuitState: ...

    def reset(self) -> None:
        """Manually reset to CLOSED state (for testing/admin)."""
```

Four named instances:

| Name | Threshold | Recovery | Rationale |
|------|-----------|----------|-----------|
| `vllm_breaker` | 3 | 30s | Critical service, fast retry |
| `ollama_router_breaker` | 5 | 60s | Has fallback (skip routing) |
| `ollama_embed_breaker` | 5 | 60s | Has fallback (FTS5 only) |
| `brave_breaker` | 5 | 120s | External API, slower recovery |

### 2. Retry Decorator (`resilience/retry.py`)

Async decorator with exponential backoff and jitter:

```python
def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable: tuple[type[Exception], ...] = (ConnectionError, TimeoutError),
) -> Callable:
    """Retry an async function with exponential backoff + jitter.

    Delay formula: min(base_delay * 2^attempt + random(0, base_delay), max_delay)
    Jitter prevents thundering herd on concurrent retries.
    """
```

- Only retries exception types listed in `retryable`
- Logs each retry at WARNING level (attempt, exception, delay)
- Re-raises the last exception if all attempts exhausted

### 3. Agent Guardrails (`agents/guardrails.py`)

Loop detector that inspects the agent scratchpad:

```python
class LoopDetector:
    def __init__(self, *, max_consecutive_same: int = 3,
                 max_total_repeats: int = 3,
                 max_token_budget: int = config.MAX_TOTAL_TOKENS): ...

    def check(self, scratchpad: list[dict]) -> str | None:
        """Returns error message if loop detected, None if OK."""
```

Four checks:
1. **Consecutive identical calls**: Last N entries have same `(tool_name, tool_input)`.
2. **Total repeats**: Same `(tool_name, tool_input)` appears N times anywhere.
3. **Token budget**: Estimated total tokens (chars / 4) exceeds budget.
4. **Max iterations**: Redundant safety net (defense in depth).

When triggered, forces a `final_answer` with whatever partial observations
exist in the scratchpad.

---

## Wiring + Graceful Degradation

Seven fallback paths:

### vLLM timeout → retry once, then error

**File**: `llm/structured.py`

Wrap `get_structured_output` with `@with_retry(max_attempts=2, retryable=(ConnectionError, TimeoutError))`. After retries exhausted, `LLMUnavailableError` propagates up and the chat handler returns an error to the user.

### Ollama/Qwen down → skip routing

**File**: `routing/router.py`

Wrap the instructor call with `ollama_router_breaker.call(...)`. On `CircuitOpenError`, return a fallback `RoutingDecision(complexity=MODERATE, category=RETRIEVAL)` that routes everything to the Llama 70B agent. The existing fallback logic in `classify_query` already handles `ConnectionError` — extend it to also handle `CircuitOpenError`.

### Brave Search fails → KB-only answers

**File**: `tools/research/web_search.py`

Wrap the httpx call with `brave_breaker.call(...)`. On `CircuitOpenError`, return observation: "Web search is temporarily unavailable. Try answering from the knowledge base." The agent sees this and falls back to `rag_search`.

### Tool execution error → already handled

**File**: `tools/registry.py`

The registry already catches exceptions and returns error strings. No changes needed.

### Agent loop detected → force final answer

**File**: `agents/base.py`

After each tool execution, call `loop_detector.check(scratchpad)`. If it returns a message, log a warning and return a synthesized answer from partial observations (or the graceful failure message if no observations exist).

### Embedding model down → FTS5-only search

**File**: `rag/search.py`

Wrap the embedding call with `ollama_embed_breaker.call(...)`. On failure, skip vector search and use FTS5 keyword search only. Log a warning that results are keyword-only. The RRF fusion handles single-source results correctly (tested in Step 6).

### SQLite locked → already handled

**File**: `db/connection.py`

`PRAGMA busy_timeout=5000` already provides a 5-second retry. No changes needed.

---

## New Error Class

Add to `resilience/errors.py`:

```python
class CircuitOpenError(ShukketsuError):
    """Raised when a circuit breaker is open and rejecting requests."""

    def __init__(self, breaker_name: str):
        super().__init__(
            f"Circuit breaker '{breaker_name}' is open — service unavailable",
            FailureMode.MODEL_UNAVAILABLE,
        )
        self.breaker_name = breaker_name
```

---

## Config Additions

Add to `config.py`:

```python
# Circuit breaker defaults
CB_VLLM_FAILURE_THRESHOLD = 3
CB_VLLM_RECOVERY_TIMEOUT = 30.0
CB_OLLAMA_ROUTER_FAILURE_THRESHOLD = 5
CB_OLLAMA_ROUTER_RECOVERY_TIMEOUT = 60.0
CB_OLLAMA_EMBED_FAILURE_THRESHOLD = 5
CB_OLLAMA_EMBED_RECOVERY_TIMEOUT = 60.0
CB_BRAVE_FAILURE_THRESHOLD = 5
CB_BRAVE_RECOVERY_TIMEOUT = 120.0

# Retry defaults
RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 30.0

# Loop detection
LOOP_MAX_CONSECUTIVE_SAME = 3
LOOP_MAX_TOTAL_REPEATS = 3
```

---

## Test Plan

### New test files

**`tests/unit/test_circuit_breaker.py`** (~10 tests):
- test CLOSED state passes calls through
- test failure increments counter
- test threshold reached transitions to OPEN
- test OPEN state raises CircuitOpenError
- test recovery timeout transitions to HALF_OPEN
- test HALF_OPEN success transitions to CLOSED
- test HALF_OPEN failure transitions to OPEN
- test success resets failure counter
- test reset() method returns to CLOSED
- test concurrent calls handled correctly

**`tests/unit/test_retry.py`** (~6 tests):
- test successful call returns immediately
- test retries on retryable exception
- test gives up after max_attempts
- test exponential backoff timing
- test non-retryable exception raises immediately
- test logs retry attempts

**`tests/unit/test_guardrails.py`** (~7 tests):
- test no loop returns None
- test consecutive identical calls detected
- test non-consecutive repeats detected
- test token budget exceeded detected
- test different tool calls not flagged
- test similar but not identical inputs not flagged
- test empty scratchpad returns None

### Modified test files

**`tests/unit/test_base_agent.py`**: Add tests for loop detection integration
- test agent stops on loop detection
- test agent returns partial answer on loop

**`tests/unit/test_router.py`**: Add test for circuit breaker fallback
- test circuit open returns fallback decision

**`tests/unit/test_errors.py`**: Add test for CircuitOpenError

---

## Files Summary

**Created (3):**
- `resilience/circuit_breaker.py`
- `resilience/retry.py`
- `agents/guardrails.py`

**Modified (6):**
- `resilience/errors.py` — add CircuitOpenError
- `config.py` — add CB/retry/loop constants
- `agents/base.py` — integrate LoopDetector
- `routing/router.py` — wrap with ollama_router_breaker
- `tools/research/web_search.py` — wrap with brave_breaker
- `rag/search.py` — wrap embedding with ollama_embed_breaker, FTS5 fallback

**Test files created (3):**
- `tests/unit/test_circuit_breaker.py`
- `tests/unit/test_retry.py`
- `tests/unit/test_guardrails.py`

**Test files modified (3):**
- `tests/unit/test_base_agent.py`
- `tests/unit/test_router.py`
- `tests/unit/test_errors.py`

**Expected test count**: 195 existing + ~28 new = ~223 tests
