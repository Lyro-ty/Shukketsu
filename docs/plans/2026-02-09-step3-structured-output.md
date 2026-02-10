# Step 3: Structured Output — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** LLM returns validated Pydantic models instead of free text, using Instructor in JSON mode with automatic retry on validation failure. Both vLLM and Ollama backends supported via a factory pattern.

**Architecture:** A `ModelBackend` enum selects between vLLM (Llama 70B) and Ollama (Qwen 4B). A `_get_client()` factory lazily creates and caches Instructor-wrapped `AsyncOpenAI` clients per backend, both using `Mode.JSON`. A generic `get_structured_output()` async function accepts any Pydantic `response_model`, handles connection errors separately from validation retries, and returns a typed instance. Pydantic schemas (`ToolCall`, `AgentStep`) define the ReAct loop's structured contract for Step 4.

**Tech Stack:** Instructor (>=1.7), Pydantic v2, AsyncOpenAI, httpx

---

## Task 0: Add Config Constants

Add structured output defaults to the config module.

**Files:**
- Modify: `code/shukketsu/config.py` (append after line 51)

**Step 1: Add the constants**

Append after the `LLM_TIMEOUT_SECONDS` line:

```python

# Structured output defaults
STRUCTURED_TEMPERATURE = 0.1
STRUCTURED_MAX_TOKENS = 4096
STRUCTURED_MAX_RETRIES = 3
```

**Step 2: Verify import**

Run: `python3 -c "from code.shukketsu.config import STRUCTURED_TEMPERATURE, STRUCTURED_MAX_TOKENS, STRUCTURED_MAX_RETRIES; print(STRUCTURED_TEMPERATURE, STRUCTURED_MAX_TOKENS, STRUCTURED_MAX_RETRIES)"`
Expected: `0.1 4096 3`

**Step 3: Lint**

Run: `ruff check code/shukketsu/config.py`
Expected: No errors

**Step 4: Commit**

```bash
git add code/shukketsu/config.py
git commit -m "feat: add structured output config constants"
```

---

## Task 1: Pydantic Schemas (TDD)

Create the response models that the LLM will return. These are pure Pydantic — no LLM or Instructor dependency.

**Files:**
- Create: `tests/unit/test_structured_schemas.py`
- Create: `code/shukketsu/llm/schemas.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_structured_schemas.py`:

```python
"""Tests for LLM structured output Pydantic schemas."""

import pytest
from pydantic import ValidationError


class TestToolCall:
    """Tests for the ToolCall schema."""

    def test_valid_tool_call(self) -> None:
        """ToolCall should accept valid thought, tool_name, tool_input."""
        from code.shukketsu.llm.schemas import ToolCall

        tc = ToolCall(
            thought="Need to search the knowledge base",
            tool_name="rag_search",
            tool_input={"query": "hit cap for combat rogues"},
        )
        assert tc.thought == "Need to search the knowledge base"
        assert tc.tool_name == "rag_search"
        assert tc.tool_input == {"query": "hit cap for combat rogues"}

    def test_missing_required_fields(self) -> None:
        """ToolCall should reject missing required fields."""
        from code.shukketsu.llm.schemas import ToolCall

        with pytest.raises(ValidationError):
            ToolCall(thought="test")  # type: ignore[call-arg]


class TestAgentStep:
    """Tests for the AgentStep schema with conditional validation."""

    def test_valid_tool_call_action(self) -> None:
        """AgentStep with action=tool_call and populated tool_call should pass."""
        from code.shukketsu.llm.schemas import ActionType, AgentStep, ToolCall

        step = AgentStep(
            reasoning="I need to look this up",
            action=ActionType.TOOL_CALL,
            tool_call=ToolCall(
                thought="Search for hit cap info",
                tool_name="rag_search",
                tool_input={"query": "hit cap"},
            ),
        )
        assert step.action == ActionType.TOOL_CALL
        assert step.tool_call is not None
        assert step.answer is None

    def test_valid_final_answer_action(self) -> None:
        """AgentStep with action=final_answer and populated answer should pass."""
        from code.shukketsu.llm.schemas import ActionType, AgentStep

        step = AgentStep(
            reasoning="I have enough information to answer",
            action=ActionType.FINAL_ANSWER,
            answer="The hit cap for combat rogues is 9% (142 hit rating).",
        )
        assert step.action == ActionType.FINAL_ANSWER
        assert step.answer is not None
        assert step.tool_call is None

    def test_tool_call_action_requires_tool_call(self) -> None:
        """AgentStep should reject action=tool_call when tool_call is None."""
        from code.shukketsu.llm.schemas import ActionType, AgentStep

        with pytest.raises(ValidationError, match="tool_call required"):
            AgentStep(
                reasoning="I need to search",
                action=ActionType.TOOL_CALL,
                tool_call=None,
            )

    def test_final_answer_action_requires_answer(self) -> None:
        """AgentStep should reject action=final_answer when answer is None."""
        from code.shukketsu.llm.schemas import ActionType, AgentStep

        with pytest.raises(ValidationError, match="answer required"):
            AgentStep(
                reasoning="I know the answer",
                action=ActionType.FINAL_ANSWER,
                answer=None,
            )

    def test_missing_required_fields(self) -> None:
        """AgentStep should reject missing reasoning or action."""
        from code.shukketsu.llm.schemas import AgentStep

        with pytest.raises(ValidationError):
            AgentStep()  # type: ignore[call-arg]


class TestActionTypeSerialization:
    """Tests for ActionType JSON round-trip behavior."""

    def test_serializes_to_string(self) -> None:
        """ActionType values should serialize as plain strings in model_dump."""
        from code.shukketsu.llm.schemas import ActionType, AgentStep

        step = AgentStep(
            reasoning="Done",
            action=ActionType.FINAL_ANSWER,
            answer="42",
        )
        dumped = step.model_dump()
        assert dumped["action"] == "final_answer"
        assert isinstance(dumped["action"], str)

    def test_round_trip_json(self) -> None:
        """AgentStep should survive JSON serialization and deserialization."""
        from code.shukketsu.llm.schemas import ActionType, AgentStep, ToolCall

        original = AgentStep(
            reasoning="Search needed",
            action=ActionType.TOOL_CALL,
            tool_call=ToolCall(
                thought="look up",
                tool_name="rag_search",
                tool_input={"q": "test"},
            ),
        )
        json_str = original.model_dump_json()
        restored = AgentStep.model_validate_json(json_str)
        assert restored == original
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_structured_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'code.shukketsu.llm.schemas'`

**Step 3: Write the implementation**

Create `code/shukketsu/llm/schemas.py`:

```python
"""Pydantic response models for LLM structured output.

These schemas define the contract between the LLM and the agent loop.
Instructor validates LLM responses against these models and retries
on validation failure.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, model_validator


class ActionType(StrEnum):
    """Actions an agent can take in a ReAct step."""

    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"


class ToolCall(BaseModel):
    """A tool invocation requested by the agent."""

    thought: str
    tool_name: str
    tool_input: dict[str, Any]


class AgentStep(BaseModel):
    """A single step in the agent's ReAct loop.

    The LLM returns one of these per iteration. Either it calls a tool
    (action=tool_call with tool_call populated) or gives a final answer
    (action=final_answer with answer populated).
    """

    reasoning: str
    action: ActionType
    tool_call: ToolCall | None = None
    answer: str | None = None

    @model_validator(mode="after")
    def _check_action_fields(self) -> "AgentStep":
        if self.action == ActionType.TOOL_CALL and self.tool_call is None:
            raise ValueError("tool_call required when action is tool_call")
        if self.action == ActionType.FINAL_ANSWER and self.answer is None:
            raise ValueError("answer required when action is final_answer")
        return self
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_structured_schemas.py -v`
Expected: 7 passed

**Step 5: Lint**

Run: `ruff check code/shukketsu/llm/schemas.py tests/unit/test_structured_schemas.py`
Expected: No errors

**Step 6: Commit**

```bash
git add code/shukketsu/llm/schemas.py tests/unit/test_structured_schemas.py
git commit -m "feat: add Pydantic schemas for structured LLM output"
```

---

## Task 2: Structured Output Client (TDD)

Build the Instructor client factory and generic `get_structured_output()` function. This wraps the Instructor library and handles both vLLM and Ollama backends.

**Files:**
- Create: `tests/unit/test_structured_client.py`
- Create: `code/shukketsu/llm/structured.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_structured_client.py`:

```python
"""Tests for the Instructor-based structured output client."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError
from pydantic import BaseModel

from code.shukketsu.resilience.errors import LLMUnavailableError, StructuredOutputError


# --- Simple test model ---


class SimpleResponse(BaseModel):
    """A trivial model for testing structured output."""

    name: str
    value: int


# --- Fixtures ---


@pytest.fixture(autouse=True)
def _clear_client_cache() -> None:
    """Reset the client cache before each test."""
    from code.shukketsu.llm.structured import clear_clients

    clear_clients()


# --- Tests ---


class TestGetStructuredOutput:
    """Tests for the get_structured_output function."""

    @patch("code.shukketsu.llm.structured._get_client")
    async def test_returns_validated_model(self, mock_get_client: MagicMock) -> None:
        """Should return a validated Pydantic model instance."""
        from code.shukketsu.llm.structured import ModelBackend, get_structured_output

        expected = SimpleResponse(name="test", value=42)
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=expected)
        mock_get_client.return_value = mock_client

        result = await get_structured_output(
            response_model=SimpleResponse,
            messages=[{"role": "user", "content": "test"}],
            backend=ModelBackend.VLLM,
        )

        assert result == expected
        assert isinstance(result, SimpleResponse)

    @patch("code.shukketsu.llm.structured._get_client")
    async def test_default_model_vllm(self, mock_get_client: MagicMock) -> None:
        """VLLM backend should use REASONING_MODEL by default."""
        from code.shukketsu.llm.structured import ModelBackend, get_structured_output

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=SimpleResponse(name="x", value=1)
        )
        mock_get_client.return_value = mock_client

        await get_structured_output(
            response_model=SimpleResponse,
            messages=[{"role": "user", "content": "test"}],
            backend=ModelBackend.VLLM,
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "llama-3.3-70b-instruct-awq"

    @patch("code.shukketsu.llm.structured._get_client")
    async def test_default_model_ollama(self, mock_get_client: MagicMock) -> None:
        """OLLAMA backend should use ROUTER_MODEL by default."""
        from code.shukketsu.llm.structured import ModelBackend, get_structured_output

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=SimpleResponse(name="x", value=1)
        )
        mock_get_client.return_value = mock_client

        await get_structured_output(
            response_model=SimpleResponse,
            messages=[{"role": "user", "content": "test"}],
            backend=ModelBackend.OLLAMA,
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "qwen3:4b"

    @patch("code.shukketsu.llm.structured._get_client")
    async def test_explicit_model_override(self, mock_get_client: MagicMock) -> None:
        """Passing model= should override the backend default."""
        from code.shukketsu.llm.structured import ModelBackend, get_structured_output

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=SimpleResponse(name="x", value=1)
        )
        mock_get_client.return_value = mock_client

        await get_structured_output(
            response_model=SimpleResponse,
            messages=[{"role": "user", "content": "test"}],
            backend=ModelBackend.VLLM,
            model="custom-model",
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "custom-model"

    @patch("code.shukketsu.llm.structured._get_client")
    async def test_connection_error_raises_llm_unavailable(
        self, mock_get_client: MagicMock
    ) -> None:
        """ConnectError should raise LLMUnavailableError, not retry."""
        from code.shukketsu.llm.structured import get_structured_output

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_get_client.return_value = mock_client

        with pytest.raises(LLMUnavailableError, match="Cannot connect"):
            await get_structured_output(
                response_model=SimpleResponse,
                messages=[{"role": "user", "content": "test"}],
            )

    @patch("code.shukketsu.llm.structured._get_client")
    async def test_api_connection_error_raises_llm_unavailable(
        self, mock_get_client: MagicMock
    ) -> None:
        """APIConnectionError should raise LLMUnavailableError."""
        from code.shukketsu.llm.structured import get_structured_output

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=APIConnectionError(request=MagicMock())
        )
        mock_get_client.return_value = mock_client

        with pytest.raises(LLMUnavailableError, match="Cannot connect"):
            await get_structured_output(
                response_model=SimpleResponse,
                messages=[{"role": "user", "content": "test"}],
            )

    @patch("code.shukketsu.llm.structured._get_client")
    async def test_timeout_raises_llm_unavailable(
        self, mock_get_client: MagicMock
    ) -> None:
        """TimeoutException should raise LLMUnavailableError."""
        from code.shukketsu.llm.structured import get_structured_output

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=httpx.TimeoutException("Timed out")
        )
        mock_get_client.return_value = mock_client

        with pytest.raises(LLMUnavailableError, match="did not respond"):
            await get_structured_output(
                response_model=SimpleResponse,
                messages=[{"role": "user", "content": "test"}],
            )

    @patch("code.shukketsu.llm.structured._get_client")
    async def test_retries_exhausted_raises_structured_output_error(
        self, mock_get_client: MagicMock
    ) -> None:
        """InstructorRetryException should raise StructuredOutputError."""
        from instructor.core.exceptions import InstructorRetryException

        from code.shukketsu.llm.structured import get_structured_output

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=InstructorRetryException(
                n_attempts=3,
                messages=[],
                last_completion=None,
            )
        )
        mock_get_client.return_value = mock_client

        with pytest.raises(StructuredOutputError, match="Failed to get valid"):
            await get_structured_output(
                response_model=SimpleResponse,
                messages=[{"role": "user", "content": "test"}],
            )


class TestClientCaching:
    """Tests for the client factory caching behavior."""

    @patch("code.shukketsu.llm.structured.instructor")
    def test_client_is_cached(self, mock_instructor: MagicMock) -> None:
        """Second call with same backend should reuse the cached client."""
        from code.shukketsu.llm.structured import ModelBackend, _get_client

        mock_instructor.from_openai.return_value = MagicMock()

        client1 = _get_client(ModelBackend.VLLM)
        client2 = _get_client(ModelBackend.VLLM)

        assert client1 is client2
        assert mock_instructor.from_openai.call_count == 1

    @patch("code.shukketsu.llm.structured.instructor")
    def test_clear_clients_resets_cache(self, mock_instructor: MagicMock) -> None:
        """clear_clients should force new client creation on next call."""
        from code.shukketsu.llm.structured import ModelBackend, _get_client, clear_clients

        mock_instructor.from_openai.return_value = MagicMock()

        _get_client(ModelBackend.VLLM)
        clear_clients()
        _get_client(ModelBackend.VLLM)

        assert mock_instructor.from_openai.call_count == 2

    @patch("code.shukketsu.llm.structured.instructor")
    def test_ollama_url_has_v1_suffix(self, mock_instructor: MagicMock) -> None:
        """Ollama client should be created with /v1 appended to base URL."""
        from code.shukketsu.llm.structured import ModelBackend, _get_client

        mock_instructor.from_openai.return_value = MagicMock()

        _get_client(ModelBackend.OLLAMA)

        # Inspect the AsyncOpenAI constructor call
        from_openai_call = mock_instructor.from_openai.call_args
        openai_client = from_openai_call.args[0]
        assert str(openai_client.base_url).rstrip("/").endswith("/v1")
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_structured_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'code.shukketsu.llm.structured'`

**Step 3: Write the implementation**

Create `code/shukketsu/llm/structured.py`:

```python
"""Instructor-wrapped LLM clients for structured output.

Provides a generic function to get validated Pydantic models from
either the vLLM (Llama 70B) or Ollama (Qwen 4B) backend. Uses
Instructor in JSON mode with automatic retry on validation failure.
"""

import logging
from enum import StrEnum
from typing import TypeVar

import httpx
import instructor
from openai import APIConnectionError, AsyncOpenAI
from pydantic import BaseModel

from code.shukketsu import config
from code.shukketsu.resilience.errors import LLMUnavailableError, StructuredOutputError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ModelBackend(StrEnum):
    """Available LLM backends for structured output."""

    VLLM = "vllm"
    OLLAMA = "ollama"


_clients: dict[ModelBackend, instructor.AsyncInstructor] = {}


def _get_client(backend: ModelBackend) -> instructor.AsyncInstructor:
    """Return a cached Instructor client for the given backend.

    Creates the client on first call, then caches it. Both backends
    use Mode.JSON for consistent structured output behavior.
    """
    if backend not in _clients:
        if backend == ModelBackend.VLLM:
            base_url = config.VLLM_BASE_URL
        else:
            base_url = f"{config.OLLAMA_BASE_URL}/v1"

        openai_client = AsyncOpenAI(
            base_url=base_url,
            api_key="not-needed",
            timeout=httpx.Timeout(timeout=config.LLM_TIMEOUT_SECONDS, connect=10.0),
        )
        _clients[backend] = instructor.from_openai(
            openai_client,
            mode=instructor.Mode.JSON,
        )
        logger.info("Created Instructor client for %s backend", backend.value)

    return _clients[backend]


def clear_clients() -> None:
    """Clear the client cache. Used by tests to reset state."""
    _clients.clear()


async def get_structured_output(
    response_model: type[T],
    messages: list[dict[str, str]],
    *,
    backend: ModelBackend = ModelBackend.VLLM,
    model: str | None = None,
    temperature: float = config.STRUCTURED_TEMPERATURE,
    max_tokens: int = config.STRUCTURED_MAX_TOKENS,
    max_retries: int = config.STRUCTURED_MAX_RETRIES,
) -> T:
    """Get validated structured output from an LLM.

    Args:
        response_model: Pydantic model class to validate the response against.
        messages: Chat messages to send to the LLM.
        backend: Which LLM backend to use (vLLM or Ollama).
        model: Model name override. Defaults to REASONING_MODEL for vLLM,
            ROUTER_MODEL for Ollama.
        temperature: Sampling temperature (default 0.1 for consistency).
        max_tokens: Maximum tokens in the response (default 4096).
        max_retries: Number of validation retry attempts (default 3).

    Returns:
        A validated instance of response_model.

    Raises:
        LLMUnavailableError: If the backend server is unreachable or times out.
        StructuredOutputError: If all validation retries are exhausted.
    """
    if model is None:
        model = config.REASONING_MODEL if backend == ModelBackend.VLLM else config.ROUTER_MODEL

    client = _get_client(backend)

    try:
        return await client.chat.completions.create(
            model=model,
            response_model=response_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
    except (httpx.ConnectError, APIConnectionError):
        raise LLMUnavailableError(
            f"Cannot connect to {backend.value} server. Is it running?"
        )
    except httpx.TimeoutException:
        raise LLMUnavailableError(
            f"{backend.value} server did not respond within {config.LLM_TIMEOUT_SECONDS}s."
        )
    except instructor.core.exceptions.InstructorRetryException:
        raise StructuredOutputError(
            f"Failed to get valid {response_model.__name__} after {max_retries} retries."
        )
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_structured_client.py -v`
Expected: 10 passed

**Step 5: Lint**

Run: `ruff check code/shukketsu/llm/structured.py tests/unit/test_structured_client.py`
Expected: No errors

**Step 6: Commit**

```bash
git add code/shukketsu/llm/structured.py tests/unit/test_structured_client.py
git commit -m "feat: add Instructor client factory with structured output"
```

---

## Task 3: Final Verification

Run the full verification suite to confirm nothing is broken.

**Files:** None (verification only)

**Step 1: Ruff lint**

Run: `ruff check code/ tests/`
Expected: No errors. Fix any issues before proceeding.

**Step 2: Ruff format check**

Run: `ruff format --check code/ tests/`
Expected: All files formatted. If not, run `ruff format code/ tests/` to fix.

**Step 3: Mypy type check**

Run: `python3 -m mypy code/shukketsu/`
Expected: No errors (or only pre-existing warnings from untyped libraries).

**Step 4: Full test suite**

Run: `python3 -m pytest tests/unit/ -v`
Expected: 46 passed (7 chat + 7 LLM client + 15 database + 7 schemas + 10 structured client)

**Step 5: Verify imports**

Run: `python3 -c "from code.shukketsu.llm.structured import ModelBackend, get_structured_output; from code.shukketsu.llm.schemas import AgentStep, ToolCall, ActionType; print('All imports OK')"`
Expected: `All imports OK`

**Step 6: Commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: address lint/format/mypy issues from step 3 verification"
```

---

## Task 4: Update CLAUDE.md

Update the project instructions to reflect Step 3 completion.

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update status**

In the CLAUDE.md file, update these references:

1. Change `Currently at **Phase 1, Step 3**` to `Currently at **Phase 1, Step 4**`
2. Change `Steps 1-2 are complete with 29 unit tests passing` to `Steps 1-3 are complete with 46 unit tests passing`
3. Update the step list to mark Step 3 as DONE:
   - Change `3. **Structured output (Instructor + Pydantic)** ← current` to `3. ~~Structured output (Instructor + Pydantic)~~ **DONE**`
   - Change `4. First tool + ReAct loop` to `4. **First tool + ReAct loop (BaseAgent + rag_search)** ← current`
4. Add `llm/structured.py` and `llm/schemas.py` to the "Key files with real code" list in the overview paragraph

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Step 3 completion"
```

---

## Files Created/Modified Summary

| Action | File | Task |
|--------|------|------|
| Modify | `code/shukketsu/config.py` | 0 |
| Create | `tests/unit/test_structured_schemas.py` | 1 |
| Create | `code/shukketsu/llm/schemas.py` | 1 |
| Create | `tests/unit/test_structured_client.py` | 2 |
| Create | `code/shukketsu/llm/structured.py` | 2 |
| Modify | `CLAUDE.md` | 4 |

**Total: 4 files created, 2 files modified, 17 new unit tests (46 cumulative)**

## Edge Cases Addressed

1. **Validation retries vs. connection failures**: Connection errors (`httpx.ConnectError`, `APIConnectionError`, `TimeoutException`) are caught outside Instructor's retry loop and raise `LLMUnavailableError` immediately. Only schema validation failures trigger Instructor retries.

2. **Ollama URL `/v1` suffix**: The factory appends `/v1` to `OLLAMA_BASE_URL` when constructing the Ollama client. Config stays as-is for native Ollama API compatibility.

3. **AgentStep conditional validation**: A `@model_validator(mode="after")` enforces that `tool_call` is populated when `action=tool_call` and `answer` is populated when `action=final_answer`. Pydantic raises `ValidationError`, which Instructor catches and re-prompts.

4. **Lazy client creation**: Clients are created on first use and cached in a module-level dict. `AsyncOpenAI` does not connect at construction time, so import never fails. `clear_clients()` resets the cache for test isolation.

5. **Truncated JSON from max_tokens**: Default `max_tokens=4096` for structured output (vs. 2048 for streaming chat) provides headroom for JSON schema overhead plus response content.
