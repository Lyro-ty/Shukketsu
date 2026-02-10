# Step 10: Observability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Langfuse v3 tracing to every user query — routing, agent loop, LLM calls, and tool execution — viewable in the Langfuse UI.

**Architecture:** `@observe()` decorators on existing functions auto-nest spans based on Python's call hierarchy. A thin `tracer.py` module handles init/flush. Docker Compose provides the self-hosted Langfuse stack.

**Tech Stack:** Langfuse Python SDK v3, Docker Compose (Postgres, ClickHouse, Redis, MinIO), pytest

**Design doc:** `docs/plans/2026-02-10-step10-observability.md`

**Test command:** `python3 -m pytest tests/unit/ -v` (must use `python3 -m pytest`, not bare `pytest`)

---

### Task 1: Install Langfuse and add config values

**Files:**
- Modify: `code/shukketsu/config.py:26-28`
- Modify: `variables.env:18-21`

**Step 1: Install langfuse**

Run: `pip install --break-system-packages langfuse>=3.0`
Expected: Successfully installed langfuse-3.x.x

**Step 2: Verify import works**

Run: `python3 -c "import langfuse; print(langfuse.__version__)"`
Expected: 3.x.x

**Step 3: Add config values**

In `code/shukketsu/config.py`, after the existing Langfuse block (lines 26-28), add:

```python
LANGFUSE_TRACING_ENABLED = os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() == "true"
LANGFUSE_SAMPLE_RATE = float(os.getenv("LANGFUSE_SAMPLE_RATE", "1.0"))
```

In `variables.env`, update the Langfuse section to:

```env
# Langfuse (self-hosted)
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=local-pk
LANGFUSE_SECRET_KEY=local-sk
LANGFUSE_TRACING_ENABLED=true
LANGFUSE_SAMPLE_RATE=1.0
```

**Step 4: Run all tests to verify no regressions**

Run: `python3 -m pytest tests/unit/ -v --tb=short -q`
Expected: 240 passed

**Step 5: Commit**

```bash
git add code/shukketsu/config.py variables.env
git commit -m "feat(config): add Langfuse tracing config values"
```

---

### Task 2: Create `observability/tracer.py` with tests (TDD)

**Files:**
- Create: `code/shukketsu/observability/tracer.py`
- Create: `tests/unit/test_tracer.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_tracer.py`:

```python
"""Tests for observability/tracer.py — Langfuse initialization and flush."""

from unittest.mock import MagicMock, patch

import pytest


class TestInitLangfuse:
    """Tests for init_langfuse()."""

    def test_sets_env_vars_from_config(self, monkeypatch):
        """init_langfuse pushes config values into environment."""
        monkeypatch.setattr("code.shukketsu.config.LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setattr("code.shukketsu.config.LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setattr("code.shukketsu.config.LANGFUSE_HOST", "http://test:3000")
        monkeypatch.setattr("code.shukketsu.config.LANGFUSE_SAMPLE_RATE", 0.5)
        monkeypatch.setattr("code.shukketsu.config.LANGFUSE_TRACING_ENABLED", True)

        import os

        from code.shukketsu.observability.tracer import init_langfuse

        with patch("code.shukketsu.observability.tracer.get_client") as mock_get:
            init_langfuse()

        assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-test"
        assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-test"
        assert os.environ["LANGFUSE_HOST"] == "http://test:3000"
        assert os.environ["LANGFUSE_SAMPLE_RATE"] == "0.5"
        mock_get.assert_called_once()

    def test_noop_when_tracing_disabled(self, monkeypatch):
        """init_langfuse does nothing when LANGFUSE_TRACING_ENABLED is False."""
        monkeypatch.setattr("code.shukketsu.config.LANGFUSE_TRACING_ENABLED", False)

        from code.shukketsu.observability.tracer import init_langfuse

        with patch("code.shukketsu.observability.tracer.get_client") as mock_get:
            init_langfuse()

        mock_get.assert_not_called()

    def test_sets_tracing_enabled_env_var_false(self, monkeypatch):
        """When disabled, sets LANGFUSE_TRACING_ENABLED=false in env."""
        monkeypatch.setattr("code.shukketsu.config.LANGFUSE_TRACING_ENABLED", False)

        import os

        from code.shukketsu.observability.tracer import init_langfuse

        with patch("code.shukketsu.observability.tracer.get_client"):
            init_langfuse()

        assert os.environ.get("LANGFUSE_TRACING_ENABLED") == "false"


class TestFlushTraces:
    """Tests for flush_traces()."""

    def test_flush_no_error_when_not_initialized(self):
        """flush_traces doesn't crash when Langfuse was never initialized."""
        from code.shukketsu.observability.tracer import flush_traces

        flush_traces()  # Should not raise

    def test_flush_calls_client_flush(self):
        """flush_traces calls flush() on the Langfuse client."""
        from code.shukketsu.observability.tracer import flush_traces

        mock_client = MagicMock()
        with patch("code.shukketsu.observability.tracer.get_client", return_value=mock_client):
            flush_traces()

        mock_client.flush.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_tracer.py -v`
Expected: ERRORS — `ModuleNotFoundError` or `ImportError` (tracer.py doesn't exist yet)

**Step 3: Write the implementation**

Create `code/shukketsu/observability/tracer.py`:

```python
"""Langfuse tracing initialization and helpers.

Configures the Langfuse SDK from values in config.py. Call init_langfuse()
once at app startup and flush_traces() on shutdown. The @observe decorator
from the langfuse package is used directly on functions in other modules.
"""

import logging
import os

from langfuse import get_client, observe  # noqa: F401 — re-exported

from code.shukketsu import config

logger = logging.getLogger(__name__)

_initialized = False


def init_langfuse() -> None:
    """Configure Langfuse from config values and eagerly create the client.

    Sets environment variables that the Langfuse SDK reads automatically,
    then initializes the singleton client. No-op when tracing is disabled.
    """
    global _initialized  # noqa: PLW0603

    if not config.LANGFUSE_TRACING_ENABLED:
        os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
        logger.info("Langfuse tracing disabled")
        return

    os.environ["LANGFUSE_PUBLIC_KEY"] = config.LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = config.LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_HOST"] = config.LANGFUSE_HOST
    os.environ["LANGFUSE_SAMPLE_RATE"] = str(config.LANGFUSE_SAMPLE_RATE)

    get_client()
    _initialized = True
    logger.info("Langfuse tracing initialized (host=%s)", config.LANGFUSE_HOST)


def flush_traces() -> None:
    """Flush any buffered traces to Langfuse. Safe to call even if not initialized."""
    try:
        client = get_client()
        client.flush()
        logger.debug("Langfuse traces flushed")
    except Exception:
        logger.debug("Langfuse flush skipped (not initialized or unavailable)")
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_tracer.py -v`
Expected: 5 passed

**Step 5: Run full suite to check for regressions**

Run: `python3 -m pytest tests/unit/ -v --tb=short -q`
Expected: 245 passed

**Step 6: Commit**

```bash
git add code/shukketsu/observability/tracer.py tests/unit/test_tracer.py
git commit -m "feat(observability): add tracer module with init and flush"
```

---

### Task 3: Create Docker Compose for Langfuse

**Files:**
- Create: `infra/docker-compose.langfuse.yml`

**Step 1: Write the Docker Compose file**

Create `infra/docker-compose.langfuse.yml`:

```yaml
# Langfuse v3 self-hosted stack for Shukketsu observability.
#
# Usage:
#   docker compose -f infra/docker-compose.langfuse.yml up -d
#   docker compose -f infra/docker-compose.langfuse.yml down
#
# Once running, access Langfuse UI at http://localhost:3000
# Create an account on first visit, then generate API keys
# and update variables.env with LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.

services:
  postgres:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: langfuse
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse
    volumes:
      - langfuse_postgres:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 5s
      timeout: 3s
      retries: 10

  clickhouse:
    image: clickhouse/clickhouse-server:latest
    restart: unless-stopped
    user: "101:101"
    environment:
      CLICKHOUSE_DB: default
      CLICKHOUSE_USER: clickhouse
      CLICKHOUSE_PASSWORD: clickhouse
    volumes:
      - langfuse_clickhouse_data:/var/lib/clickhouse
      - langfuse_clickhouse_logs:/var/log/clickhouse-server
    healthcheck:
      test: ["CMD-SHELL", "wget --no-verbose --tries=1 --spider http://localhost:8123/ping || exit 1"]
      interval: 5s
      timeout: 3s
      retries: 10

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --requirepass langfuse
    volumes:
      - langfuse_redis:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "langfuse", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10

  minio:
    image: minio/minio
    restart: unless-stopped
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: langfuse
      MINIO_ROOT_PASSWORD: langfuse
    volumes:
      - langfuse_minio:/data
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 5s
      timeout: 3s
      retries: 10

  # Create the default bucket on first start
  minio-init:
    image: minio/mc
    depends_on:
      minio:
        condition: service_healthy
    entrypoint: >
      /bin/sh -c "
      mc alias set local http://minio:9000 langfuse langfuse;
      mc mb local/langfuse --ignore-existing;
      exit 0;
      "

  langfuse-worker:
    image: langfuse/langfuse-worker:3
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
      clickhouse:
        condition: service_healthy
      redis:
        condition: service_healthy
      minio:
        condition: service_healthy
    environment: &langfuse-env
      DATABASE_URL: postgresql://langfuse:langfuse@postgres:5432/langfuse
      NEXTAUTH_URL: http://localhost:3000
      SALT: shukketsu-salt-change-me
      ENCRYPTION_KEY: "0000000000000000000000000000000000000000000000000000000000000000"
      CLICKHOUSE_MIGRATION_URL: clickhouse://clickhouse:9000
      CLICKHOUSE_URL: http://clickhouse:8123
      CLICKHOUSE_USER: clickhouse
      CLICKHOUSE_PASSWORD: clickhouse
      REDIS_HOST: redis
      REDIS_PORT: "6379"
      REDIS_AUTH: langfuse
      LANGFUSE_S3_EVENT_UPLOAD_BUCKET: langfuse
      LANGFUSE_S3_EVENT_UPLOAD_REGION: auto
      LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID: langfuse
      LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY: langfuse
      LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT: http://minio:9000
      LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE: "true"
      LANGFUSE_S3_MEDIA_UPLOAD_BUCKET: langfuse
      LANGFUSE_S3_MEDIA_UPLOAD_REGION: auto
      LANGFUSE_S3_MEDIA_UPLOAD_ACCESS_KEY_ID: langfuse
      LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY: langfuse
      LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT: http://minio:9000
      LANGFUSE_S3_MEDIA_UPLOAD_FORCE_PATH_STYLE: "true"
      TELEMETRY_ENABLED: "false"

  langfuse-web:
    image: langfuse/langfuse:3
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
      clickhouse:
        condition: service_healthy
      redis:
        condition: service_healthy
      minio:
        condition: service_healthy
    ports:
      - "3000:3000"
    environment:
      <<: *langfuse-env
      NEXTAUTH_SECRET: shukketsu-secret-change-me

volumes:
  langfuse_postgres:
  langfuse_clickhouse_data:
  langfuse_clickhouse_logs:
  langfuse_redis:
  langfuse_minio:
```

**Step 2: Validate YAML syntax**

Run: `python3 -c "import yaml; yaml.safe_load(open('infra/docker-compose.langfuse.yml'))" 2>&1 || echo "Install pyyaml or just check with docker compose config"`
Run: `docker compose -f infra/docker-compose.langfuse.yml config --quiet 2>&1 && echo "Valid" || echo "Invalid"`

**Step 3: Commit**

```bash
git add infra/docker-compose.langfuse.yml
git commit -m "infra: add Langfuse v3 self-hosted Docker Compose"
```

---

### Task 4: Instrument `llm/structured.py` (generation spans)

**Files:**
- Modify: `code/shukketsu/llm/structured.py`

This is the innermost instrumentation point — LLM calls. We add `@observe(as_type="generation")` and record model metadata.

**Step 1: Add the decorator and metadata update**

In `code/shukketsu/llm/structured.py`, add imports at the top (after the existing imports):

```python
from langfuse import get_client, observe
```

Add the decorator to `get_structured_output` (above the existing `@with_retry` decorator — `@observe` must be the outermost decorator so the span wraps the retries):

```python
@observe(as_type="generation")
@with_retry(max_attempts=2, base_delay=1.0, retryable=(LLMUnavailableError,))
async def get_structured_output[T: BaseModel](...) -> T:
```

Inside the function body, after `model` is resolved (after line `model = config.REASONING_MODEL if ...`), add:

```python
    langfuse = get_client()
    langfuse.update_current_generation(
        model=model,
        model_parameters={"temperature": temperature, "max_tokens": max_tokens},
    )
```

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v --tb=short -q`
Expected: 245 passed (no regressions — `@observe` is no-op without valid Langfuse)

**Step 3: Commit**

```bash
git add code/shukketsu/llm/structured.py
git commit -m "feat(observability): trace LLM generations in structured.py"
```

---

### Task 5: Instrument `tools/registry.py` (tool spans)

**Files:**
- Modify: `code/shukketsu/tools/registry.py`

**Step 1: Add the decorator**

Add imports:

```python
from langfuse import get_client, observe
```

Decorate the `execute` method:

```python
    @observe(as_type="tool")
    async def execute(self, name: str, tool_input: dict[str, Any]) -> str:
```

Inside `execute`, at the top of the method (before the `if name not in` check), add:

```python
        langfuse = get_client()
        langfuse.update_current_span(metadata={"tool_name": name, "tool_input": tool_input})
```

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v --tb=short -q`
Expected: 245 passed

**Step 3: Commit**

```bash
git add code/shukketsu/tools/registry.py
git commit -m "feat(observability): trace tool execution in registry.py"
```

---

### Task 6: Instrument `routing/router.py` (routing span)

**Files:**
- Modify: `code/shukketsu/routing/router.py`

**Step 1: Add the decorator**

Add import:

```python
from langfuse import observe
```

Decorate `classify_query`:

```python
@observe()
async def classify_query(query: str) -> RoutingDecision:
```

No additional metadata needed — the nested `get_structured_output` call auto-creates a generation span inside this span.

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v --tb=short -q`
Expected: 245 passed

**Step 3: Commit**

```bash
git add code/shukketsu/routing/router.py
git commit -m "feat(observability): trace query routing in router.py"
```

---

### Task 7: Instrument `agents/base.py` (agent span)

**Files:**
- Modify: `code/shukketsu/agents/base.py`

**Step 1: Add the decorator**

Add import:

```python
from langfuse import observe
```

Decorate the `run` method:

```python
    @observe(as_type="agent")
    async def run(self, query: str) -> str:
```

No additional metadata needed — `get_structured_output` and `tool_registry.execute` calls inside the loop auto-nest as child spans.

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v --tb=short -q`
Expected: 245 passed

**Step 3: Commit**

```bash
git add code/shukketsu/agents/base.py
git commit -m "feat(observability): trace agent ReAct loop in base.py"
```

---

### Task 8: Instrument `web/routers/chat.py` (root trace) and wire up `web/app.py`

**Files:**
- Modify: `code/shukketsu/web/routers/chat.py`
- Modify: `code/shukketsu/web/app.py`

**Step 1: Add root trace decorator to chat handler**

In `code/shukketsu/web/routers/chat.py`, add imports:

```python
from langfuse import get_client, observe
```

Decorate `_agent_response`:

```python
@observe()
async def _agent_response(websocket: WebSocket, session: ChatSession, content: str) -> None:
```

Inside `_agent_response`, at the start of the `try` block (after `session.is_streaming = True`), add:

```python
        langfuse = get_client()
        langfuse.update_current_trace(
            session_id=str(id(session)),
            tags=["chat"],
            input=content,
        )
```

**Step 2: Wire init/flush into app lifecycle**

In `code/shukketsu/web/app.py`, add a lifespan context manager. Replace the plain `FastAPI(...)` with:

```python
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from code.shukketsu.observability.tracer import flush_traces, init_langfuse


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown hooks."""
    init_langfuse()
    yield
    flush_traces()


app = FastAPI(
    title="Shukketsu",
    description="TBC Rogue Research Agent — Web Knowledgebase",
    version="0.1.0",
    lifespan=lifespan,
)
```

**Step 3: Run full test suite**

Run: `python3 -m pytest tests/unit/ -v --tb=short -q`
Expected: 245 passed

**Step 4: Commit**

```bash
git add code/shukketsu/web/routers/chat.py code/shukketsu/web/app.py
git commit -m "feat(observability): add root trace in chat handler, wire lifespan"
```

---

### Task 9: Lint, format, and final verification

**Files:**
- All modified files

**Step 1: Run ruff check and format**

Run: `ruff check code/ tests/ --fix && ruff format code/ tests/`
Expected: All checks passed, files formatted

**Step 2: Run full test suite one final time**

Run: `python3 -m pytest tests/unit/ -v`
Expected: 245 passed (240 existing + 5 tracer tests)

**Step 3: Commit any formatting changes**

```bash
git add -A
git commit -m "style: apply ruff formatting for Step 10"
```

---

### Task 10: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update the key files list**

In CLAUDE.md line 9, add `observability/tracer.py` to the list of key files with real code.

**Step 2: Update Step 10 status**

Change line 165 from:
```
10. **Observability (Langfuse tracing)** ← current
```
to:
```
10. ~~Observability (Langfuse tracing)~~ **DONE**
```

**Step 3: Update test count**

Update the text "240 unit tests passing" to "245 unit tests passing".

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Step 10 completion"
```
