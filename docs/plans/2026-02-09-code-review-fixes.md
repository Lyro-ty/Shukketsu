# Code Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all Critical, Important, and Minor issues identified in the full-project code review of Steps 1-9.

**Architecture:** Each task targets a specific review finding. Tasks are ordered by priority (Critical first, then Important, then Minor). Most tasks are isolated — each modifies 1-2 source files and 1 test file.

**Tech Stack:** Python 3.12, asyncio, sqlite3, pytest, httpx, Pydantic v2

**Baseline:** 227 tests passing. All changes must keep the test suite green.

---

### Task 1: Fix ingest pipeline transaction safety (C4)

**Files:**
- Modify: `code/shukketsu/ingest/pipeline.py:56-113`
- Modify: `tests/unit/test_pipeline.py` (add test)

**Why:** If `embed_texts()` fails after the source row is written, the source gets a `content_hash` but no chunks. Re-ingest then sees matching hash and returns `already_existed=True` — content is silently lost.

**Step 1: Write the failing test**

Add to `tests/unit/test_pipeline.py`:

```python
async def test_embedding_failure_rolls_back_source(self, test_db: sqlite3.Connection) -> None:
    """If embedding fails, the source row should be rolled back so re-ingest works."""
    embedder = _mock_embedder()
    embedder.embed_texts = AsyncMock(side_effect=EmbeddingError("GPU OOM"))
    pipeline = IngestPipeline(conn=test_db, embedder=embedder)

    with pytest.raises(EmbeddingError):
        await pipeline.ingest(text="Some real content.", url="https://example.com/fail", title="Fail")

    # Source should NOT exist — it was rolled back
    row = test_db.execute("SELECT id FROM sources WHERE url = ?", ("https://example.com/fail",)).fetchone()
    assert row is None

    # Re-ingest with a working embedder should succeed
    pipeline2 = IngestPipeline(conn=test_db, embedder=_mock_embedder())
    result = await pipeline2.ingest(text="Some real content.", url="https://example.com/fail", title="Fail")
    assert result.already_existed is False
    assert result.chunk_count >= 1
```

Add `EmbeddingError` import at top of test file:
```python
from code.shukketsu.resilience.errors import EmbeddingError
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_pipeline.py::TestIngestPipeline::test_embedding_failure_rolls_back_source -v`
Expected: FAIL (source row exists after failure, re-ingest returns `already_existed=True`)

**Step 3: Implement transaction safety**

In `code/shukketsu/ingest/pipeline.py`, wrap the ingest in an explicit transaction. Replace the `ingest` method body (lines 54-121) with:

```python
    async def ingest(
        self,
        text: str,
        url: str,
        title: str,
        source_type: str = "guide",
    ) -> IngestResult:
        """Ingest text into the knowledge base.

        Args:
            text: The full text content to ingest.
            url: Source URL (used as unique key for dedup).
            title: Human-readable title for the source.
            source_type: Category of source (guide, forum, wiki, etc.).

        Returns:
            IngestResult with source_id, chunk_count, and dedup status.
        """
        content_hash = hashlib.sha256(text.encode()).hexdigest() if text.strip() else ""

        # Check for existing source with the same URL
        existing = self._conn.execute("SELECT id, content_hash FROM sources WHERE url = ?", (url,)).fetchone()

        if existing:
            if existing["content_hash"] == content_hash:
                chunk_count = self._conn.execute(
                    "SELECT chunk_count FROM sources WHERE id = ?", (existing["id"],)
                ).fetchone()["chunk_count"]
                return IngestResult(
                    source_id=existing["id"],
                    chunk_count=chunk_count,
                    already_existed=True,
                )

        # Embed BEFORE touching the database (this is the most likely failure point)
        if text.strip():
            chunks = chunk_text(text)
            embeddings = await self._embedder.embed_texts([c.content for c in chunks])
        else:
            chunks = []
            embeddings = []

        # Now write everything in a single transaction
        try:
            if existing:
                source_id = existing["id"]
                self._delete_chunks_and_vectors(source_id)
                self._conn.execute(
                    "UPDATE sources SET title = ?, source_type = ?, content_hash = ?, "
                    "fetched_at = ?, chunk_count = 0 WHERE id = ?",
                    (title, source_type, content_hash, datetime.now(UTC).isoformat(), source_id),
                )
            else:
                cursor = self._conn.execute(
                    "INSERT INTO sources (url, title, source_type, content_hash, fetched_at) VALUES (?, ?, ?, ?, ?)",
                    (url, title, source_type, content_hash, datetime.now(UTC).isoformat()),
                )
                source_id = cursor.lastrowid

            for chunk, embedding in zip(chunks, embeddings):
                cursor = self._conn.execute(
                    "INSERT INTO chunks (source_id, content, chunk_index, metadata_json) VALUES (?, ?, ?, NULL)",
                    (source_id, chunk.content, chunk.chunk_index),
                )
                chunk_id = cursor.lastrowid
                embedding_blob = struct.pack(f"{len(embedding)}f", *embedding)
                self._conn.execute(
                    "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                    (chunk_id, embedding_blob),
                )

            self._conn.execute(
                "UPDATE sources SET chunk_count = ? WHERE id = ?",
                (len(chunks), source_id),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        logger.info("Ingested %d chunks from %s (%s)", len(chunks), url, title)

        return IngestResult(
            source_id=source_id,
            chunk_count=len(chunks),
            already_existed=False,
        )
```

The key change: embedding happens BEFORE any database writes, and all DB writes are wrapped in try/except with rollback.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_pipeline.py -v`
Expected: All pipeline tests pass (8 existing + 1 new)

**Step 5: Commit**

```bash
git add code/shukketsu/ingest/pipeline.py tests/unit/test_pipeline.py
git commit -m "fix: wrap ingest pipeline in explicit transaction with rollback (C4)"
```

---

### Task 2: Fix `datetime.utcnow()` and add trust scoring tests (I2 + I10)

**Files:**
- Modify: `code/shukketsu/trust/scoring.py:1-31`
- Create: `tests/unit/test_trust_scoring.py`

**Why:** `datetime.utcnow()` is deprecated in Python 3.12 and returns naive datetimes. If `fetched_at` is timezone-aware (as stored by `IngestPipeline`), the subtraction will raise `TypeError`. Also, this module has zero tests.

**Step 1: Write the tests**

Create `tests/unit/test_trust_scoring.py`:

```python
"""Tests for trust scoring with time-based decay."""

from datetime import UTC, datetime, timedelta

from code.shukketsu.trust.scoring import SOURCE_TRUST, effective_trust


class TestSourceTrust:
    """Tests for the SOURCE_TRUST mapping."""

    def test_game_data_highest(self) -> None:
        assert SOURCE_TRUST["game_data"] == 1.0

    def test_unknown_lowest(self) -> None:
        assert SOURCE_TRUST["unknown"] == 0.3

    def test_all_values_between_zero_and_one(self) -> None:
        for value in SOURCE_TRUST.values():
            assert 0.0 < value <= 1.0


class TestEffectiveTrust:
    """Tests for the effective_trust decay function."""

    def test_within_max_age_returns_base(self) -> None:
        now = datetime.now(UTC)
        result = effective_trust(0.8, fetched_at=now, max_age=timedelta(days=30), decay_factor=0.5)
        assert result == 0.8

    def test_exactly_at_max_age_returns_base(self) -> None:
        now = datetime.now(UTC)
        fetched = now - timedelta(days=30)
        result = effective_trust(0.8, fetched_at=fetched, max_age=timedelta(days=30), decay_factor=0.5)
        assert result == 0.8

    def test_one_period_past_max_age_applies_decay(self) -> None:
        now = datetime.now(UTC)
        fetched = now - timedelta(days=60)  # 30 days past max_age of 30
        result = effective_trust(0.8, fetched_at=fetched, max_age=timedelta(days=30), decay_factor=0.5)
        assert abs(result - 0.4) < 0.01  # 0.8 * 0.5^1

    def test_two_periods_past_applies_double_decay(self) -> None:
        now = datetime.now(UTC)
        fetched = now - timedelta(days=90)  # 60 days past max_age of 30
        result = effective_trust(0.8, fetched_at=fetched, max_age=timedelta(days=30), decay_factor=0.5)
        assert abs(result - 0.2) < 0.01  # 0.8 * 0.5^2

    def test_result_is_float(self) -> None:
        now = datetime.now(UTC)
        result = effective_trust(1.0, fetched_at=now, max_age=timedelta(days=30), decay_factor=0.9)
        assert isinstance(result, float)

    def test_timezone_aware_fetched_at(self) -> None:
        """Ensure timezone-aware datetimes (as stored by IngestPipeline) work."""
        now = datetime.now(UTC)
        fetched = now - timedelta(hours=1)
        result = effective_trust(0.75, fetched_at=fetched, max_age=timedelta(days=7), decay_factor=0.5)
        assert result == 0.75
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_trust_scoring.py -v`
Expected: FAIL — `TypeError: can't subtract offset-naive and offset-aware datetimes` on tests that pass timezone-aware datetimes.

**Step 3: Fix the `datetime.utcnow()` bug**

Replace `code/shukketsu/trust/scoring.py` contents:

```python
"""Source trust scoring and decay computation."""

from datetime import UTC, datetime, timedelta

SOURCE_TRUST = {
    "game_data": 1.0,
    "simulation": 0.9,
    "combat_logs": 0.85,
    "expert_guide": 0.75,
    "archived_theory": 0.7,
    "community": 0.5,
    "unknown": 0.3,
}


def effective_trust(
    base_trust: float,
    fetched_at: datetime,
    max_age: timedelta,
    decay_factor: float,
) -> float:
    """Compute effective trust with time-based decay.

    Trust stays at base_trust until max_age, then decays
    by decay_factor for each additional max_age period.
    """
    age = datetime.now(UTC) - fetched_at
    if age <= max_age:
        return base_trust
    periods_past = (age - max_age) / max_age
    return float(base_trust * (decay_factor**periods_past))
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_trust_scoring.py -v`
Expected: All 7 tests pass.

Run: `python3 -m pytest tests/unit/ -v`
Expected: All tests pass (227 existing + 7 new = 234).

**Step 5: Commit**

```bash
git add code/shukketsu/trust/scoring.py tests/unit/test_trust_scoring.py
git commit -m "fix: replace deprecated datetime.utcnow() and add trust scoring tests (I2, I10)"
```

---

### Task 3: Add asyncio.Lock to circuit breaker HALF_OPEN state (C1)

**Files:**
- Modify: `code/shukketsu/resilience/circuit_breaker.py:1-102`
- Modify: `tests/unit/test_circuit_breaker.py` (add test)

**Why:** Multiple concurrent coroutines can all pass the `HALF_OPEN` state check simultaneously, defeating the "test one request" semantics.

**Step 1: Write the failing test**

Add to `tests/unit/test_circuit_breaker.py`:

```python
class TestCircuitBreakerConcurrency:
    """Tests for async-safety of the circuit breaker."""

    async def test_half_open_allows_only_one_concurrent_request(self) -> None:
        """In HALF_OPEN, only one request should pass; others get CircuitOpenError."""
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        # Trip the breaker
        fn = AsyncMock(side_effect=ConnectionError("down"))
        with pytest.raises(ConnectionError):
            await cb.call(fn)
        await asyncio.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

        # A slow function that simulates work
        call_count = 0
        results = []

        async def slow_fn():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return "ok"

        # Launch two concurrent calls
        tasks = [asyncio.create_task(cb.call(slow_fn)) for _ in range(2)]
        done = await asyncio.gather(*tasks, return_exceptions=True)

        # Exactly one should succeed, the other should get CircuitOpenError
        successes = [r for r in done if r == "ok"]
        errors = [r for r in done if isinstance(r, CircuitOpenError)]
        assert len(successes) == 1
        assert len(errors) == 1
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_circuit_breaker.py::TestCircuitBreakerConcurrency -v`
Expected: FAIL (both concurrent calls pass through)

**Step 3: Add asyncio.Lock to CircuitBreaker**

Modify `code/shukketsu/resilience/circuit_breaker.py`. Add `import asyncio` at the top. Then modify the class:

```python
class CircuitBreaker:
    """Async circuit breaker that protects against cascading failures.

    Tracks consecutive failures for a named service. After reaching
    the failure threshold, the breaker opens and rejects requests
    immediately. After a recovery timeout, it transitions to half-open
    and allows one test request through.
    """

    def __init__(self, name: str, *, failure_threshold: int = 5, recovery_timeout: float = 60.0) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._state = CircuitState.CLOSED
        self._half_open_lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current breaker state, accounting for recovery timeout."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    async def call(self, fn: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """Execute fn through the circuit breaker.

        Args:
            fn: Async callable to execute.
            *args: Positional arguments for fn.
            **kwargs: Keyword arguments for fn.

        Returns:
            The return value of fn.

        Raises:
            CircuitOpenError: If the breaker is open and rejecting requests.
            Exception: Any exception raised by fn (also tracked as a failure).
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            logger.warning("Circuit breaker '%s' is OPEN — rejecting request", self.name)
            raise CircuitOpenError(self.name)

        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_lock.locked():
                raise CircuitOpenError(self.name)
            async with self._half_open_lock:
                return await self._execute(fn, *args, **kwargs)

        return await self._execute(fn, *args, **kwargs)

    async def _execute(self, fn: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """Execute fn and record success/failure."""
        try:
            result = await fn(*args, **kwargs)
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        return result

    def reset(self) -> None:
        """Manually reset the breaker to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        logger.info("Circuit breaker '%s' manually reset to CLOSED", self.name)

    def _record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit breaker '%s' opened after %d failures",
                self.name,
                self._failure_count,
            )

    def _record_success(self) -> None:
        """Record a success and reset counters."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_circuit_breaker.py -v`
Expected: All circuit breaker tests pass (10 existing + 1 new).

Run: `python3 -m pytest tests/unit/ -v`
Expected: Full suite passes.

**Step 5: Commit**

```bash
git add code/shukketsu/resilience/circuit_breaker.py tests/unit/test_circuit_breaker.py
git commit -m "fix: add asyncio.Lock to circuit breaker HALF_OPEN state (C1)"
```

---

### Task 4: Add `FailureMode.ACCESS_DENIED` and fix `RobotsDisallowedError` (I8)

**Files:**
- Modify: `code/shukketsu/resilience/errors.py:6-100`
- Modify: `tests/unit/test_errors.py` (add test)

**Why:** `RobotsDisallowedError` uses `FailureMode.RATE_LIMITED`, but robots.txt blocking is a permanent policy denial, not rate limiting. This could cause incorrect retry behavior.

**Step 1: Write the failing test**

Add to `tests/unit/test_errors.py`:

```python
def test_robots_disallowed_has_access_denied_failure_mode() -> None:
    err = RobotsDisallowedError("blocked")
    assert err.failure_mode == FailureMode.ACCESS_DENIED
```

Add imports at the top if not already present:
```python
from code.shukketsu.resilience.errors import RobotsDisallowedError
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_errors.py::test_robots_disallowed_has_access_denied_failure_mode -v`
Expected: FAIL (either `AttributeError` for missing enum value or assertion failure)

**Step 3: Add the enum value and fix the error**

In `code/shukketsu/resilience/errors.py`, add `ACCESS_DENIED` to the `FailureMode` enum after `HTTP_ERROR`:

```python
    # Network failures
    NETWORK_TIMEOUT = "network_timeout"
    RATE_LIMITED = "rate_limited"
    HTTP_ERROR = "http_error"
    ACCESS_DENIED = "access_denied"
```

Then change `RobotsDisallowedError`:

```python
class RobotsDisallowedError(ShukketsuError):
    """Raised when robots.txt disallows access to a URL."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.ACCESS_DENIED)
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_errors.py -v`
Expected: All error tests pass.

Run: `python3 -m pytest tests/unit/ -v`
Expected: Full suite passes.

**Step 5: Commit**

```bash
git add code/shukketsu/resilience/errors.py tests/unit/test_errors.py
git commit -m "fix: use ACCESS_DENIED failure mode for RobotsDisallowedError (I8)"
```

---

### Task 5: Fix exception chaining — add `from exc` consistently (I7)

**Files:**
- Modify: `code/shukketsu/llm/clients.py:42-50`
- Modify: `code/shukketsu/llm/structured.py:108-113`

**Why:** Re-raises without `from exc` lose the original traceback. The `embedder.py` does it correctly; these two files don't.

**Step 1: No new test needed** — exception chaining is not easily testable via unit test, and existing tests already cover the exception types. This is a code quality fix.

**Step 2: Fix `clients.py`**

In `code/shukketsu/llm/clients.py`, change the exception handlers:

```python
    except (httpx.ConnectError, APIConnectionError) as exc:
        raise LLMUnavailableError(
            f"Cannot connect to LLM server at {config.VLLM_BASE_URL}. Is vLLM running on port 8000?"
        ) from exc
    except httpx.TimeoutException as exc:
        raise LLMUnavailableError(
            f"LLM server at {config.VLLM_BASE_URL} did not respond within "
            f"{config.LLM_TIMEOUT_SECONDS} seconds. It may be loading the model."
        ) from exc
```

And the second handler:
```python
    except httpx.ReadError as exc:
        raise LLMUnavailableError(f"Connection to LLM server lost during response: {exc}") from exc
```

**Step 3: Fix `structured.py`**

In `code/shukketsu/llm/structured.py`, change:

```python
    except (httpx.ConnectError, APIConnectionError) as exc:
        raise LLMUnavailableError(f"Cannot connect to {backend.value} server. Is it running?") from exc
    except httpx.TimeoutException as exc:
        raise LLMUnavailableError(f"{backend.value} server did not respond within {config.LLM_TIMEOUT_SECONDS}s.") from exc
    except instructor.core.exceptions.InstructorRetryException as exc:
        raise StructuredOutputError(f"Failed to get valid {response_model.__name__} after {max_retries} retries.") from exc
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_llm_clients.py tests/unit/test_structured_client.py -v`
Expected: All tests pass (no behavior change, just traceback quality).

**Step 5: Commit**

```bash
git add code/shukketsu/llm/clients.py code/shukketsu/llm/structured.py
git commit -m "fix: add 'from exc' to all exception re-raises for proper chaining (I7)"
```

---

### Task 6: Add input length validation to WebSocket handler (I5)

**Files:**
- Modify: `code/shukketsu/config.py` (add constant)
- Modify: `code/shukketsu/web/routers/chat.py:109-112`
- Modify: `tests/unit/test_chat_handler.py` (add test)

**Why:** No message length limit means a multi-megabyte message gets sent to the LLM.

**Step 1: Add config constant**

In `code/shukketsu/config.py`, add after `CHAT_MAX_HISTORY_PAIRS = 20`:

```python
CHAT_MAX_MESSAGE_LENGTH = 10_000  # Max characters per user message
```

**Step 2: Write the failing test**

Add to `tests/unit/test_chat_handler.py`:

```python
async def test_rejects_oversized_message(self) -> None:
    """Messages exceeding CHAT_MAX_MESSAGE_LENGTH should be rejected."""
    ws = AsyncMock()
    session = ChatSession()
    huge_message = "x" * 10_001
    await _handle_message(ws, session, {"type": "message", "content": huge_message})
    ws.send_json.assert_called_once()
    response = ws.send_json.call_args[0][0]
    assert response["type"] == "error"
    assert "too long" in response["content"].lower()
```

Import `_handle_message` and `ChatSession` if not already imported.

**Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_chat_handler.py::TestChatHandler::test_rejects_oversized_message -v`
Expected: FAIL (message is accepted, no error sent)

**Step 4: Add the length check**

In `code/shukketsu/web/routers/chat.py`, after the emptiness check (line 112), add:

```python
    if len(content) > config.CHAT_MAX_MESSAGE_LENGTH:
        await websocket.send_json({
            "type": "error",
            "content": f"Message too long ({len(content)} chars). Maximum is {config.CHAT_MAX_MESSAGE_LENGTH}.",
        })
        return
```

**Step 5: Run tests**

Run: `python3 -m pytest tests/unit/test_chat_handler.py -v`
Expected: All chat handler tests pass.

**Step 6: Commit**

```bash
git add code/shukketsu/config.py code/shukketsu/web/routers/chat.py tests/unit/test_chat_handler.py
git commit -m "feat: add message length validation to WebSocket handler (I5)"
```

---

### Task 7: Fix `_synthesize_partial_answer` to include all observations (I6)

**Files:**
- Modify: `code/shukketsu/agents/base.py:95-101`
- Modify: `tests/unit/test_base_agent.py` (add/modify test)

**Why:** When the agent is stopped due to a loop, only the last observation is returned. Earlier useful tool results are lost.

**Step 1: Write the test**

Add to `tests/unit/test_base_agent.py` (find the appropriate class):

```python
async def test_partial_answer_includes_all_observations(self) -> None:
    """When a loop is detected, all unique observations should be included."""
    from code.shukketsu.agents.base import BaseAgent

    agent = BaseAgent(tool_registry=mock_registry())
    scratchpad = [
        {"observation": "Rogues have a 9% hit cap.", "tool_name": "rag_search", "tool_input": {"query": "hit cap"}, "reasoning": "look up hit cap"},
        {"observation": "Combat swords is the best spec.", "tool_name": "rag_search", "tool_input": {"query": "best spec"}, "reasoning": "look up spec"},
        {"observation": "Rogues have a 9% hit cap.", "tool_name": "rag_search", "tool_input": {"query": "hit cap"}, "reasoning": "look up hit cap"},
    ]
    result = agent._synthesize_partial_answer(scratchpad)
    assert "9% hit cap" in result
    assert "Combat swords" in result
```

Note: You will need to check the test file for the correct helper function (`mock_registry`) — adapt to whatever pattern exists in `test_base_agent.py`.

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_base_agent.py::test_partial_answer_includes_all_observations -v`
Expected: FAIL ("Combat swords" not in result since only last unique observation is returned)

**Step 3: Fix the method**

In `code/shukketsu/agents/base.py`, replace `_synthesize_partial_answer`:

```python
    def _synthesize_partial_answer(self, scratchpad: list[dict[str, Any]]) -> str:
        """Build an answer from partial observations when a loop is detected."""
        observations = [e["observation"] for e in scratchpad if e.get("observation")]
        if observations:
            unique = list(dict.fromkeys(observations))
            joined = "\n\n".join(unique)
            return f"Based on partial results:\n\n{joined}"
        return config.AGENT_GRACEFUL_FAILURE
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_base_agent.py -v`
Expected: All agent tests pass.

**Step 5: Commit**

```bash
git add code/shukketsu/agents/base.py tests/unit/test_base_agent.py
git commit -m "fix: include all unique observations in partial answers (I6)"
```

---

### Task 8: Remove unused `MAX_AGENT_ITERATIONS` config (M3)

**Files:**
- Modify: `code/shukketsu/config.py:37`

**Why:** `MAX_AGENT_ITERATIONS = 15` is never used — the agent uses `AGENT_MAX_ITERATIONS = 5`. Two similarly-named constants are confusing.

**Step 1: Verify it's unused**

Search the codebase for `MAX_AGENT_ITERATIONS` (not `AGENT_MAX_ITERATIONS`). Only `config.py` should reference it.

**Step 2: Remove it**

In `code/shukketsu/config.py`, delete line 37:
```python
MAX_AGENT_ITERATIONS = 15
```

**Step 3: Run tests**

Run: `python3 -m pytest tests/unit/ -v`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add code/shukketsu/config.py
git commit -m "chore: remove unused MAX_AGENT_ITERATIONS config constant (M3)"
```

---

### Task 9: Fix test style — use `pytest.raises` in `test_fetcher.py` (M5)

**Files:**
- Modify: `tests/unit/test_fetcher.py:50-100`

**Why:** Three tests use `try/assert False/except` instead of the standard `with pytest.raises()` pattern.

**Step 1: Replace the patterns**

In `test_robots_disallowed_raises`:
```python
    async def test_robots_disallowed_raises(self) -> None:
        fetcher = _make_fetcher(robots_allowed=False)
        with pytest.raises(RobotsDisallowedError):
            await fetcher.fetch("https://example.com/page")
```

In `test_http_error_raises_scraping_error`:
```python
    async def test_http_error_raises_scraping_error(self) -> None:
        fetcher = _make_fetcher()
        mock_resp = _mock_response(status_code=403)
        with patch("code.shukketsu.scraping.fetcher.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client
            with pytest.raises(ScrapingError, match="403"):
                await fetcher.fetch("https://example.com/page")
```

In `test_timeout_raises_scraping_error`:
```python
    async def test_timeout_raises_scraping_error(self) -> None:
        fetcher = _make_fetcher()
        with patch("code.shukketsu.scraping.fetcher.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            mock_client_cls.return_value = mock_client
            with pytest.raises(ScrapingError, match="(?i)timed out"):
                await fetcher.fetch("https://example.com/page")
```

In `test_non_html_raises_scraping_error`:
```python
    async def test_non_html_raises_scraping_error(self) -> None:
        fetcher = _make_fetcher()
        mock_resp = _mock_response(content_type="application/pdf")
        with patch("code.shukketsu.scraping.fetcher.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client
            with pytest.raises(ScrapingError, match="(?i)html"):
                await fetcher.fetch("https://example.com/file.pdf")
```

Add `import pytest` at the top of the file if not already there.

**Step 2: Run tests**

Run: `python3 -m pytest tests/unit/test_fetcher.py -v`
Expected: All 7 fetcher tests pass.

**Step 3: Commit**

```bash
git add tests/unit/test_fetcher.py
git commit -m "style: use pytest.raises instead of try/assert False/except (M5)"
```

---

### Task 10: Add `reset_all_breakers()` utility and autouse fixture (C3 — partial)

**Files:**
- Modify: `code/shukketsu/resilience/circuit_breaker.py` (add function)
- Modify: `tests/conftest.py` (add autouse fixture)

**Why:** Module-level circuit breaker singletons can leak state between tests. Adding a reset utility and autouse fixture prevents test pollution.

**Step 1: Add `reset_all_breakers()`**

At the bottom of `code/shukketsu/resilience/circuit_breaker.py`, after the named instances, add:

```python
def reset_all_breakers() -> None:
    """Reset all named circuit breakers to CLOSED. Used by test fixtures."""
    for breaker in (vllm_breaker, ollama_router_breaker, ollama_embed_breaker, brave_breaker):
        breaker.reset()
```

**Step 2: Add autouse fixture**

In `tests/conftest.py`, add:

```python
@pytest.fixture(autouse=True)
def _reset_breakers():
    """Reset all circuit breakers before each test to prevent state leakage."""
    from code.shukketsu.resilience.circuit_breaker import reset_all_breakers
    reset_all_breakers()
    yield
    reset_all_breakers()
```

**Step 3: Run tests**

Run: `python3 -m pytest tests/unit/ -v`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add code/shukketsu/resilience/circuit_breaker.py tests/conftest.py
git commit -m "fix: add reset_all_breakers() and autouse fixture to prevent test pollution (C3)"
```

---

### Task 11: Final verification

**Step 1: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v`
Expected: All tests pass (227 original + ~10 new tests).

**Step 2: Run linting**

Run: `ruff check code/ tests/`
Expected: No errors.

Run: `ruff format --check code/ tests/`
Expected: No formatting issues.

**Step 3: Commit any formatting fixes if needed**

```bash
ruff format code/ tests/
git add -u
git commit -m "style: apply ruff formatting"
```
