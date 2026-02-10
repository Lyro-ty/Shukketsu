# Step 1: Chat UI + Direct LLM — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Type a question in the browser, get a streamed response from Llama 70B over WebSocket.

**Architecture:** Browser connects via WebSocket to a FastAPI handler, which calls vLLM's OpenAI-compatible API with `stream=True` and forwards token chunks back over the WebSocket as JSON messages. Chat history is held in-memory per connection (max 20 pairs). The UI is a WoW Rogue-themed dark interface with Tailwind CSS (Play CDN), Markdown rendering (marked.js + DOMPurify), and vanilla JavaScript.

**Tech Stack:** FastAPI, AsyncOpenAI, Jinja2, Tailwind Play CDN, marked.js, DOMPurify, vanilla JS

---

## Task 0: Fix Package Import Path

The `code/` directory is missing `__init__.py`, which causes `import code.shukketsu` to collide with Python's stdlib `code` module.

**Files:**
- Create: `code/__init__.py`

**Step 1: Create the file**

```python
```

Empty file. Its existence makes `code/` a Python package that shadows the stdlib `code` module.

**Step 2: Verify**

Run: `python3 -c "from code.shukketsu.config import VLLM_BASE_URL; print(VLLM_BASE_URL)"`
Expected: `http://localhost:8000/v1`

**Step 3: Commit**

```bash
git add code/__init__.py
git commit -m "fix: add code/__init__.py to fix package imports"
```

---

## Task 1: Config Constants + LLMUnavailableError

Add chat-specific configuration and the error class needed by the LLM client.

**Files:**
- Modify: `code/shukketsu/config.py` (append new constants)
- Modify: `code/shukketsu/resilience/errors.py` (add LLMUnavailableError)

**Step 1: Add chat config constants to `config.py`**

Append after line 40 (`REFLECTION_THRESHOLD = 0.7`):

```python
# Chat defaults
SYSTEM_PROMPT = (
    "You are Shukketsu, a research assistant specializing in "
    "World of Warcraft: The Burning Crusade Rogue class. "
    "Answer questions accurately and concisely."
)
CHAT_MAX_HISTORY_PAIRS = 20
CHAT_TEMPERATURE = 0.7
CHAT_MAX_TOKENS = 2048
LLM_TIMEOUT_SECONDS = 30.0
```

**Step 2: Add LLMUnavailableError to `errors.py`**

Append after the `AgentLoopError` class (after line 51):

```python
class LLMUnavailableError(ShukketsuError):
    """Raised when the LLM server cannot be reached."""

    def __init__(self, message: str):
        super().__init__(message, FailureMode.MODEL_UNAVAILABLE)
```

**Step 3: Verify imports**

Run: `python3 -c "from code.shukketsu.config import SYSTEM_PROMPT, CHAT_MAX_TOKENS; print(SYSTEM_PROMPT[:20])"`
Expected: `You are Shukketsu, a`

Run: `python3 -c "from code.shukketsu.resilience.errors import LLMUnavailableError; print(LLMUnavailableError.__name__)"`
Expected: `LLMUnavailableError`

**Step 4: Lint**

Run: `ruff check code/shukketsu/config.py code/shukketsu/resilience/errors.py`
Expected: No errors

**Step 5: Commit**

```bash
git add code/shukketsu/config.py code/shukketsu/resilience/errors.py
git commit -m "feat: add chat config constants and LLMUnavailableError"
```

---

## Task 2: LLM Client — stream_chat()

Build the async generator that talks to vLLM and yields tokens.

**Files:**
- Create: `code/shukketsu/llm/clients.py`
- Create: `tests/unit/test_llm_clients.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_llm_clients.py`:

```python
"""Tests for the vLLM streaming client."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from code.shukketsu.resilience.errors import LLMUnavailableError


# --- Helpers ---

def make_chunk(content: str | None) -> MagicMock:
    """Create a mock OpenAI stream chunk."""
    chunk = MagicMock()
    choice = MagicMock()
    choice.delta.content = content
    chunk.choices = [choice]
    return chunk


def make_empty_chunk() -> MagicMock:
    """Create a mock chunk with no choices."""
    chunk = MagicMock()
    chunk.choices = []
    return chunk


class MockAsyncStream:
    """Simulates an OpenAI AsyncStream that yields chunks."""

    def __init__(self, chunks: list[MagicMock]) -> None:
        self._chunks = chunks
        self._index = 0

    def __aiter__(self) -> "MockAsyncStream":
        return self

    async def __anext__(self) -> MagicMock:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    async def close(self) -> None:
        pass


class MockAsyncStreamWithError:
    """Simulates a stream that errors after yielding some chunks."""

    def __init__(self, chunks: list[MagicMock]) -> None:
        self._chunks = chunks
        self._index = 0

    def __aiter__(self) -> "MockAsyncStreamWithError":
        return self

    async def __anext__(self) -> MagicMock:
        if self._index >= len(self._chunks):
            raise httpx.ReadError("Connection lost")
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    async def close(self) -> None:
        pass


# --- Tests ---

@patch("code.shukketsu.llm.clients._client")
async def test_stream_chat_yields_tokens(mock_client: MagicMock) -> None:
    """stream_chat should yield each non-None token from the stream."""
    from code.shukketsu.llm.clients import stream_chat

    chunks = [make_chunk("Hello"), make_chunk(" "), make_chunk("world")]
    mock_client.chat.completions.create = AsyncMock(
        return_value=MockAsyncStream(chunks)
    )

    tokens: list[str] = []
    async for token in stream_chat([{"role": "user", "content": "test"}]):
        tokens.append(token)

    assert tokens == ["Hello", " ", "world"]


@patch("code.shukketsu.llm.clients._client")
async def test_stream_chat_skips_none_content(mock_client: MagicMock) -> None:
    """stream_chat should skip chunks with None content."""
    from code.shukketsu.llm.clients import stream_chat

    chunks = [make_chunk("Hello"), make_chunk(None), make_chunk(" world")]
    mock_client.chat.completions.create = AsyncMock(
        return_value=MockAsyncStream(chunks)
    )

    tokens: list[str] = []
    async for token in stream_chat([{"role": "user", "content": "test"}]):
        tokens.append(token)

    assert tokens == ["Hello", " world"]


@patch("code.shukketsu.llm.clients._client")
async def test_stream_chat_skips_empty_choices(mock_client: MagicMock) -> None:
    """stream_chat should skip chunks with no choices."""
    from code.shukketsu.llm.clients import stream_chat

    chunks = [make_chunk("Hi"), make_empty_chunk()]
    mock_client.chat.completions.create = AsyncMock(
        return_value=MockAsyncStream(chunks)
    )

    tokens: list[str] = []
    async for token in stream_chat([{"role": "user", "content": "test"}]):
        tokens.append(token)

    assert tokens == ["Hi"]


@patch("code.shukketsu.llm.clients._client")
async def test_stream_chat_raises_on_connect_error(mock_client: MagicMock) -> None:
    """stream_chat should raise LLMUnavailableError when vLLM is unreachable."""
    from code.shukketsu.llm.clients import stream_chat

    mock_client.chat.completions.create = AsyncMock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    with pytest.raises(LLMUnavailableError, match="Cannot connect"):
        async for _ in stream_chat([{"role": "user", "content": "test"}]):
            pass


@patch("code.shukketsu.llm.clients._client")
async def test_stream_chat_raises_on_timeout(mock_client: MagicMock) -> None:
    """stream_chat should raise LLMUnavailableError on timeout."""
    from code.shukketsu.llm.clients import stream_chat

    mock_client.chat.completions.create = AsyncMock(
        side_effect=httpx.TimeoutException("Timed out")
    )

    with pytest.raises(LLMUnavailableError, match="did not respond"):
        async for _ in stream_chat([{"role": "user", "content": "test"}]):
            pass


@patch("code.shukketsu.llm.clients._client")
async def test_stream_chat_raises_on_mid_stream_error(mock_client: MagicMock) -> None:
    """stream_chat should raise LLMUnavailableError if connection drops mid-stream."""
    from code.shukketsu.llm.clients import stream_chat

    chunks = [make_chunk("Hello")]
    mock_client.chat.completions.create = AsyncMock(
        return_value=MockAsyncStreamWithError(chunks)
    )

    tokens: list[str] = []
    with pytest.raises(LLMUnavailableError, match="lost during response"):
        async for token in stream_chat([{"role": "user", "content": "test"}]):
            tokens.append(token)

    assert tokens == ["Hello"]
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_llm_clients.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'code.shukketsu.llm.clients'`

**Step 3: Write the implementation**

Create `code/shukketsu/llm/clients.py`:

```python
"""LLM client for vLLM streaming chat completions."""

import logging
from collections.abc import AsyncGenerator

import httpx
from openai import AsyncOpenAI

from code.shukketsu import config
from code.shukketsu.resilience.errors import LLMUnavailableError

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(
    base_url=config.VLLM_BASE_URL,
    api_key="not-needed",
    timeout=httpx.Timeout(timeout=config.LLM_TIMEOUT_SECONDS, connect=10.0),
)


async def stream_chat(
    messages: list[dict[str, str]],
    *,
    model: str = config.REASONING_MODEL,
    temperature: float = config.CHAT_TEMPERATURE,
    max_tokens: int = config.CHAT_MAX_TOKENS,
) -> AsyncGenerator[str, None]:
    """Stream chat completion tokens from vLLM.

    Yields token strings one at a time. Raises LLMUnavailableError
    if the server is unreachable or the connection drops mid-stream.
    """
    try:
        stream = await _client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except httpx.ConnectError:
        raise LLMUnavailableError(
            f"Cannot connect to LLM server at {config.VLLM_BASE_URL}. "
            "Is vLLM running on port 8000?"
        )
    except httpx.TimeoutException:
        raise LLMUnavailableError(
            f"LLM server at {config.VLLM_BASE_URL} did not respond within "
            f"{config.LLM_TIMEOUT_SECONDS} seconds. It may be loading the model."
        )

    try:
        async for chunk in stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
    except httpx.ReadError as exc:
        raise LLMUnavailableError(
            f"Connection to LLM server lost during response: {exc}"
        )
    finally:
        await stream.close()
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_llm_clients.py -v`
Expected: 7 passed

**Step 5: Lint**

Run: `ruff check code/shukketsu/llm/clients.py tests/unit/test_llm_clients.py`
Expected: No errors

**Step 6: Commit**

```bash
git add code/shukketsu/llm/clients.py tests/unit/test_llm_clients.py
git commit -m "feat: add stream_chat LLM client with tests"
```

---

## Task 3: WebSocket Chat Handler

Build the WebSocket endpoint that manages chat sessions and streams LLM responses.

**Files:**
- Create: `code/shukketsu/web/routers/chat.py`
- Create: `tests/unit/test_chat_handler.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_chat_handler.py`:

```python
"""Tests for the WebSocket chat handler."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from code.shukketsu.resilience.errors import LLMUnavailableError


# --- Mock stream generators ---

async def mock_stream_tokens(*args, **kwargs):
    """Yields two tokens."""
    for token in ["Hello", " world"]:
        yield token


async def mock_stream_empty(*args, **kwargs):
    """Yields nothing."""
    return
    yield  # noqa: unreachable — makes this an async generator


async def mock_stream_error(*args, **kwargs):
    """Raises LLMUnavailableError immediately."""
    raise LLMUnavailableError("vLLM is not running on port 8000")
    yield  # noqa: unreachable — makes this an async generator


# --- Tests ---

def _get_app():
    """Import app fresh to pick up router registration."""
    from code.shukketsu.web.app import app
    return app


@patch("code.shukketsu.web.routers.chat.stream_chat", new=mock_stream_tokens)
def test_websocket_connects_and_sends_status() -> None:
    """WebSocket should accept connection and send a status message."""
    client = TestClient(_get_app())
    with client.websocket_connect("/ws/chat") as ws:
        msg = ws.receive_json()
        assert msg == {"type": "status", "content": "connected"}


@patch("code.shukketsu.web.routers.chat.stream_chat", new=mock_stream_tokens)
def test_chat_message_streams_tokens() -> None:
    """User message should produce token messages then a done message."""
    client = TestClient(_get_app())
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # skip status

        ws.send_json({"type": "message", "content": "Hi"})

        tok1 = ws.receive_json()
        assert tok1 == {"type": "token", "content": "Hello"}

        tok2 = ws.receive_json()
        assert tok2 == {"type": "token", "content": " world"}

        done = ws.receive_json()
        assert done == {"type": "done", "content": "Hello world"}


@patch("code.shukketsu.web.routers.chat.stream_chat", new=mock_stream_tokens)
def test_chat_rejects_empty_message() -> None:
    """Empty message content should return an error."""
    client = TestClient(_get_app())
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # skip status

        ws.send_json({"type": "message", "content": "  "})

        err = ws.receive_json()
        assert err["type"] == "error"
        assert "empty" in err["content"].lower()


@patch("code.shukketsu.web.routers.chat.stream_chat", new=mock_stream_tokens)
def test_chat_rejects_unknown_type() -> None:
    """Unknown message type should return an error."""
    client = TestClient(_get_app())
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # skip status

        ws.send_json({"type": "dance", "content": "boogie"})

        err = ws.receive_json()
        assert err["type"] == "error"
        assert "unknown" in err["content"].lower()


@patch("code.shukketsu.web.routers.chat.stream_chat", new=mock_stream_tokens)
def test_chat_rejects_missing_type() -> None:
    """Message without type field should return an error."""
    client = TestClient(_get_app())
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # skip status

        ws.send_json({"content": "hi"})

        err = ws.receive_json()
        assert err["type"] == "error"


@patch("code.shukketsu.web.routers.chat.stream_chat", new=mock_stream_error)
def test_chat_sends_error_on_llm_failure() -> None:
    """LLM unavailable should send an error message, not crash."""
    client = TestClient(_get_app())
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # skip status

        ws.send_json({"type": "message", "content": "Hello"})

        err = ws.receive_json()
        assert err["type"] == "error"
        assert "vLLM" in err["content"]


@patch("code.shukketsu.web.routers.chat.stream_chat", new=mock_stream_tokens)
def test_chat_allows_second_message_after_first_completes() -> None:
    """After first response completes, a second message should work."""
    client = TestClient(_get_app())
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # skip status

        # First message
        ws.send_json({"type": "message", "content": "First"})
        ws.receive_json()  # token
        ws.receive_json()  # token
        ws.receive_json()  # done

        # Second message
        ws.send_json({"type": "message", "content": "Second"})
        tok1 = ws.receive_json()
        assert tok1["type"] == "token"
        tok2 = ws.receive_json()
        assert tok2["type"] == "token"
        done = ws.receive_json()
        assert done["type"] == "done"
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_chat_handler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'code.shukketsu.web.routers.chat'` (or ImportError from app.py once router is referenced)

**Step 3: Write the implementation**

Create `code/shukketsu/web/routers/chat.py`:

```python
"""WebSocket chat handler for streaming LLM conversations."""

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from code.shukketsu import config
from code.shukketsu.llm.clients import stream_chat
from code.shukketsu.resilience.errors import ShukketsuError

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatSession:
    """Per-connection chat state."""

    def __init__(self) -> None:
        self.history: list[dict[str, str]] = []
        self.is_streaming: bool = False
        self._stop_event: asyncio.Event = asyncio.Event()

    def add_message(self, role: str, content: str) -> None:
        """Append a message and trim history to max pairs."""
        self.history.append({"role": role, "content": content})
        max_messages = config.CHAT_MAX_HISTORY_PAIRS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def build_messages(self) -> list[dict[str, str]]:
        """Build the full messages array with system prompt."""
        return [{"role": "system", "content": config.SYSTEM_PROMPT}] + self.history

    def request_stop(self) -> None:
        """Signal the streaming loop to stop."""
        self._stop_event.set()

    def reset_stop(self) -> None:
        """Clear the stop signal for the next request."""
        self._stop_event.clear()

    @property
    def stop_requested(self) -> bool:
        """Check if stop has been requested."""
        return self._stop_event.is_set()


@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket) -> None:
    """Handle a WebSocket chat connection."""
    await websocket.accept()
    session = ChatSession()
    await websocket.send_json({"type": "status", "content": "connected"})

    try:
        while True:
            data = await websocket.receive_json()
            await _handle_message(websocket, session, data)
    except WebSocketDisconnect:
        logger.info("Chat WebSocket disconnected")


async def _handle_message(
    websocket: WebSocket, session: ChatSession, data: dict[str, str]
) -> None:
    """Dispatch a single incoming WebSocket message."""
    msg_type = data.get("type")

    if msg_type is None:
        await websocket.send_json(
            {"type": "error", "content": "Missing 'type' field in message."}
        )
        return

    if msg_type == "stop":
        session.request_stop()
        return

    if msg_type != "message":
        await websocket.send_json(
            {"type": "error", "content": f"Unknown message type: {msg_type}"}
        )
        return

    content = data.get("content", "").strip()
    if not content:
        await websocket.send_json(
            {"type": "error", "content": "Message content cannot be empty."}
        )
        return

    if session.is_streaming:
        await websocket.send_json(
            {"type": "error", "content": "Please wait for the current response to finish."}
        )
        return

    await _stream_response(websocket, session, content)


async def _stream_response(
    websocket: WebSocket, session: ChatSession, content: str
) -> None:
    """Stream an LLM response for the given user message."""
    session.is_streaming = True
    session.reset_stop()
    session.add_message("user", content)
    full_response = ""

    try:
        async for token in stream_chat(session.build_messages()):
            if session.stop_requested:
                break
            full_response += token
            await websocket.send_json({"type": "token", "content": token})

        session.add_message("assistant", full_response)
        await websocket.send_json({"type": "done", "content": full_response})
    except ShukketsuError as exc:
        if full_response:
            session.add_message("assistant", full_response)
            await websocket.send_json({"type": "done", "content": full_response})
        else:
            # Remove the unanswered user message
            if session.history and session.history[-1]["role"] == "user":
                session.history.pop()
        await websocket.send_json({"type": "error", "content": str(exc)})
    finally:
        session.is_streaming = False
```

**Step 4: Update app.py to include the router (needed for tests)**

This is the minimal change to make tests pass. Full app wiring is in Task 7.

Replace the contents of `code/shukketsu/web/app.py` with:

```python
"""FastAPI application for the Shukketsu web knowledgebase."""

from fastapi import FastAPI

from code.shukketsu.web.routers.chat import router as chat_router

app = FastAPI(
    title="Shukketsu",
    description="TBC Rogue Research Agent — Web Knowledgebase",
    version="0.1.0",
)

app.include_router(chat_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint for Workbench app registration."""
    return {"status": "ok"}
```

**Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_chat_handler.py -v`
Expected: 7 passed

Run: `python3 -m pytest tests/unit/ -v`
Expected: 14 passed (7 client + 7 handler)

**Step 6: Lint**

Run: `ruff check code/shukketsu/web/routers/chat.py tests/unit/test_chat_handler.py`
Expected: No errors

**Step 7: Commit**

```bash
git add code/shukketsu/web/routers/chat.py tests/unit/test_chat_handler.py code/shukketsu/web/app.py
git commit -m "feat: add WebSocket chat handler with streaming and tests"
```

---

## Task 4: Theme CSS

Write the WoW Rogue-themed stylesheet.

**Files:**
- Create: `code/shukketsu/web/static/css/theme.css`

**Step 1: Write the theme**

Create `code/shukketsu/web/static/css/theme.css`:

```css
/* Shukketsu — WoW Rogue Dark Theme */

:root {
    --bg-primary: #0f0f1a;
    --bg-secondary: #1a1a2e;
    --bg-tertiary: #16213e;
    --text-primary: #e8dcc4;
    --text-secondary: #a89b8c;
    --accent-gold: #f5c518;
    --accent-energy: #fff468;
    --accent-shadow: #6c3483;
    --error-red: #e74c3c;
    --success-green: #2ecc71;
    --warning-amber: #f39c12;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
    background: #333;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #555;
}

/* Connection status pulse animation */
@keyframes status-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

.status-pulse {
    animation: status-pulse 2s ease-in-out infinite;
}

/* Markdown rendered content */
.markdown-body {
    line-height: 1.7;
}

.markdown-body h1,
.markdown-body h2,
.markdown-body h3 {
    color: var(--accent-gold);
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.markdown-body h1 { font-size: 1.5rem; }
.markdown-body h2 { font-size: 1.25rem; }
.markdown-body h3 { font-size: 1.1rem; }

.markdown-body p {
    margin-bottom: 0.75rem;
}

.markdown-body strong {
    color: var(--accent-gold);
}

.markdown-body code {
    background: rgba(0, 0, 0, 0.3);
    padding: 0.15rem 0.4rem;
    border-radius: 0.25rem;
    font-size: 0.875rem;
    color: var(--accent-energy);
}

.markdown-body pre {
    background: rgba(0, 0, 0, 0.4);
    padding: 1rem;
    border-radius: 0.5rem;
    overflow-x: auto;
    margin-bottom: 0.75rem;
}

.markdown-body pre code {
    background: none;
    padding: 0;
    font-size: 0.85rem;
}

.markdown-body ul,
.markdown-body ol {
    padding-left: 1.5rem;
    margin-bottom: 0.75rem;
}

.markdown-body li {
    margin-bottom: 0.25rem;
}

.markdown-body blockquote {
    border-left: 3px solid var(--accent-gold);
    padding-left: 1rem;
    color: var(--text-secondary);
    margin-bottom: 0.75rem;
}

.markdown-body a {
    color: var(--accent-gold);
    text-decoration: underline;
}

.markdown-body a:hover {
    color: var(--accent-energy);
}

.markdown-body table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 0.75rem;
}

.markdown-body th,
.markdown-body td {
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 0.5rem 0.75rem;
    text-align: left;
}

.markdown-body th {
    background: rgba(0, 0, 0, 0.3);
    color: var(--accent-gold);
    font-weight: 600;
}

.markdown-body hr {
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    margin: 1rem 0;
}
```

**Step 2: Commit**

```bash
git add code/shukketsu/web/static/css/theme.css
git commit -m "feat: add WoW Rogue dark theme CSS"
```

---

## Task 5: HTML Templates

Write the base layout and chat page templates.

**Files:**
- Create: `code/shukketsu/web/templates/base.html`
- Create: `code/shukketsu/web/templates/chat.html`

**Step 1: Write `base.html`**

Create `code/shukketsu/web/templates/base.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Shukketsu{% endblock %}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
    tailwind.config = {
        theme: {
            extend: {
                colors: {
                    'wow-gold': '#f5c518',
                    'rogue-energy': '#fff468',
                    'shadow': {
                        DEFAULT: '#1a1a2e',
                        deep: '#0f0f1a',
                        mid: '#16213e',
                    },
                    'parchment': {
                        DEFAULT: '#e8dcc4',
                        dim: '#a89b8c',
                    },
                },
            },
        },
    }
    </script>
    <link rel="stylesheet" href="/static/css/theme.css">
    {% block head %}{% endblock %}
</head>
<body class="bg-shadow-deep text-parchment min-h-screen flex flex-col">
    <nav class="bg-shadow border-b border-white/10 px-6 py-3 flex items-center justify-between shrink-0">
        <div class="flex items-center gap-3">
            <span class="text-wow-gold font-bold text-xl tracking-wide">Shukketsu</span>
            <span class="text-parchment-dim text-sm">出血</span>
        </div>
        <div id="connection-status" class="flex items-center gap-2 text-sm">
            <span id="status-dot" class="w-2 h-2 rounded-full bg-gray-500"></span>
            <span id="status-text" class="text-parchment-dim">Connecting...</span>
        </div>
    </nav>
    <main class="flex-1 flex flex-col overflow-hidden">
        {% block content %}{% endblock %}
    </main>
</body>
</html>
```

**Step 2: Write `chat.html`**

Create `code/shukketsu/web/templates/chat.html`:

```html
{% extends "base.html" %}

{% block title %}Chat — Shukketsu{% endblock %}

{% block head %}
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify/dist/purify.min.js"></script>
{% endblock %}

{% block content %}
<div class="flex-1 flex flex-col max-w-4xl w-full mx-auto overflow-hidden">
    <!-- Chat messages -->
    <div id="chat-messages" class="flex-1 overflow-y-auto p-6 space-y-4">
        <!-- Welcome state -->
        <div id="welcome-state" class="flex flex-col items-center justify-center h-full text-center select-none">
            <h1 class="text-wow-gold text-3xl font-bold mb-1">Shukketsu</h1>
            <p class="text-parchment-dim text-lg mb-8">TBC Rogue Research Assistant</p>
            <p class="text-parchment-dim mb-6">Ask me anything about the Rogue class in World of Warcraft: TBC</p>
            <div class="flex flex-col gap-3 w-full max-w-md">
                <button class="example-prompt px-4 py-3 bg-shadow border border-white/10 rounded-lg
                               hover:border-wow-gold/50 hover:text-wow-gold transition-colors text-left text-sm"
                        data-prompt="What's the hit cap for combat rogues?">
                    What's the hit cap for combat rogues?
                </button>
                <button class="example-prompt px-4 py-3 bg-shadow border border-white/10 rounded-lg
                               hover:border-wow-gold/50 hover:text-wow-gold transition-colors text-left text-sm"
                        data-prompt="Compare combat vs assassination for Karazhan">
                    Compare combat vs assassination for Karazhan
                </button>
                <button class="example-prompt px-4 py-3 bg-shadow border border-white/10 rounded-lg
                               hover:border-wow-gold/50 hover:text-wow-gold transition-colors text-left text-sm"
                        data-prompt="Explain the Seal Fate talent">
                    Explain the Seal Fate talent
                </button>
            </div>
        </div>
    </div>

    <!-- Input area -->
    <div class="border-t border-white/10 p-4 shrink-0">
        <div class="flex gap-3 items-end">
            <textarea id="chat-input"
                      class="flex-1 bg-shadow border border-white/10 rounded-lg px-4 py-3
                             text-parchment placeholder-parchment-dim/50 resize-none
                             focus:outline-none focus:border-wow-gold/50 transition-colors"
                      placeholder="Ask about TBC Rogues... (Shift+Enter for new line)"
                      rows="1"></textarea>
            <button id="send-btn"
                    class="px-5 py-3 bg-wow-gold/20 border border-wow-gold/50 rounded-lg
                           text-wow-gold hover:bg-wow-gold/30 transition-colors font-medium
                           disabled:opacity-30 disabled:cursor-not-allowed">
                Send
            </button>
            <button id="stop-btn"
                    class="hidden px-5 py-3 bg-red-500/20 border border-red-500/50 rounded-lg
                           text-red-400 hover:bg-red-500/30 transition-colors font-medium">
                Stop
            </button>
        </div>
    </div>
</div>

<script src="/static/js/chat.js"></script>
{% endblock %}
```

**Step 3: Commit**

```bash
git add code/shukketsu/web/templates/base.html code/shukketsu/web/templates/chat.html
git commit -m "feat: add base layout and chat page templates"
```

---

## Task 6: Chat JavaScript

Write the WebSocket client, message rendering, and UI state management.

**Files:**
- Create: `code/shukketsu/web/static/js/chat.js`

**Step 1: Write `chat.js`**

Create `code/shukketsu/web/static/js/chat.js`:

```javascript
/**
 * Shukketsu Chat — WebSocket client with streaming LLM responses.
 *
 * Connects to /ws/chat, sends user messages as JSON, receives
 * streamed tokens, and renders final responses as sanitized Markdown.
 */

// --- State ---
let socket = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 10;
let currentAssistantEl = null;
let fullResponse = "";
let isStreaming = false;

// --- DOM ---
const messagesEl = document.getElementById("chat-messages");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const stopBtn = document.getElementById("stop-btn");
const welcomeEl = document.getElementById("welcome-state");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

// --- Markdown config ---
marked.setOptions({ breaks: true, gfm: true });

// ============================================================
// WebSocket
// ============================================================

function connectWebSocket() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${protocol}//${location.host}/ws/chat`);

    socket.onopen = () => {
        reconnectAttempts = 0;
        updateStatus("connected");
    };

    socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
    };

    socket.onclose = () => {
        updateStatus("disconnected");
        if (isStreaming) endStreaming();
        scheduleReconnect();
    };

    socket.onerror = () => {
        // onclose fires after this — no action needed here
    };
}

function scheduleReconnect() {
    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        updateStatus("failed");
        return;
    }
    const delay = Math.min(2000 * Math.pow(1.5, reconnectAttempts), 30000);
    reconnectAttempts++;
    updateStatus("reconnecting");
    setTimeout(connectWebSocket, delay);
}

function updateStatus(state) {
    const states = {
        connected:    { color: "bg-green-500",                  text: "Connected" },
        disconnected: { color: "bg-gray-500",                   text: "Disconnected" },
        reconnecting: { color: "bg-yellow-500 status-pulse",    text: "Reconnecting..." },
        failed:       { color: "bg-red-500",                    text: "Connection failed" },
    };
    const s = states[state] || states.disconnected;
    statusDot.className = `w-2 h-2 rounded-full ${s.color}`;
    statusText.textContent = s.text;
}

// ============================================================
// Message Handling
// ============================================================

function handleMessage(msg) {
    switch (msg.type) {
        case "token":
            if (currentAssistantEl) {
                fullResponse += msg.content;
                currentAssistantEl.textContent = fullResponse;
                scrollToBottom();
            }
            break;

        case "done":
            if (currentAssistantEl) {
                fullResponse = msg.content;
                if (fullResponse) {
                    currentAssistantEl.innerHTML =
                        DOMPurify.sanitize(marked.parse(fullResponse));
                    currentAssistantEl.classList.add("markdown-body");
                }
            }
            endStreaming();
            break;

        case "error":
            appendError(msg.content);
            if (isStreaming) endStreaming();
            break;

        case "status":
            // Connection status — handled by onopen
            break;
    }
}

// ============================================================
// UI Helpers
// ============================================================

function appendUserMessage(text) {
    hideWelcome();
    const el = document.createElement("div");
    el.className = "flex justify-end";
    const bubble = document.createElement("div");
    bubble.className =
        "max-w-[80%] bg-shadow-mid border border-wow-gold/20 rounded-lg px-4 py-3";
    const p = document.createElement("p");
    p.className = "whitespace-pre-wrap";
    p.textContent = text;
    bubble.appendChild(p);
    el.appendChild(bubble);
    messagesEl.appendChild(el);
    scrollToBottom();
}

function appendAssistantBubble() {
    hideWelcome();
    const wrapper = document.createElement("div");
    wrapper.className = "flex justify-start";
    const bubble = document.createElement("div");
    bubble.className =
        "max-w-[80%] bg-shadow border border-white/10 rounded-lg px-4 py-3 min-h-[2rem]";
    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);
    scrollToBottom();
    return bubble;
}

function appendError(text) {
    const el = document.createElement("div");
    el.className = "flex justify-start";
    const bubble = document.createElement("div");
    bubble.className =
        "max-w-[80%] bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm";
    bubble.textContent = text;
    el.appendChild(bubble);
    messagesEl.appendChild(el);
    scrollToBottom();
}

function hideWelcome() {
    if (welcomeEl) {
        welcomeEl.style.display = "none";
    }
}

function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ============================================================
// Streaming State
// ============================================================

function startStreaming() {
    isStreaming = true;
    inputEl.disabled = true;
    inputEl.classList.add("opacity-50");
    sendBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");
    currentAssistantEl = appendAssistantBubble();
    fullResponse = "";
}

function endStreaming() {
    isStreaming = false;
    inputEl.disabled = false;
    inputEl.classList.remove("opacity-50");
    sendBtn.classList.remove("hidden");
    stopBtn.classList.add("hidden");
    currentAssistantEl = null;
    fullResponse = "";
    inputEl.focus();
}

// ============================================================
// Send / Stop
// ============================================================

function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || !socket || socket.readyState !== WebSocket.OPEN) return;
    if (isStreaming) return;

    appendUserMessage(text);
    socket.send(JSON.stringify({ type: "message", content: text }));
    inputEl.value = "";
    inputEl.style.height = "auto";
    startStreaming();
}

function stopGeneration() {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "stop" }));
    }
}

// ============================================================
// Input Handling
// ============================================================

inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
    if (e.key === "Escape") {
        inputEl.value = "";
        inputEl.style.height = "auto";
    }
});

// Auto-resize textarea (1 line to max 6 lines)
inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 150) + "px";
});

sendBtn.addEventListener("click", sendMessage);
stopBtn.addEventListener("click", stopGeneration);

// Example prompts
document.querySelectorAll(".example-prompt").forEach((btn) => {
    btn.addEventListener("click", () => {
        inputEl.value = btn.dataset.prompt;
        sendMessage();
    });
});

// ============================================================
// Init
// ============================================================

inputEl.focus();
connectWebSocket();
```

**Step 2: Commit**

```bash
git add code/shukketsu/web/static/js/chat.js
git commit -m "feat: add chat WebSocket client with streaming and Markdown"
```

---

## Task 7: App Wiring

Update `app.py` to mount static files, templates, and the chat page route.

**Files:**
- Modify: `code/shukketsu/web/app.py`

**Step 1: Write the full app.py**

Replace `code/shukketsu/web/app.py` with:

```python
"""FastAPI application for the Shukketsu web knowledgebase."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from code.shukketsu.web.routers.chat import router as chat_router

_WEB_DIR = Path(__file__).parent

app = FastAPI(
    title="Shukketsu",
    description="TBC Rogue Research Agent — Web Knowledgebase",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=_WEB_DIR / "static"), name="static")
app.include_router(chat_router)

templates = Jinja2Templates(directory=_WEB_DIR / "templates")


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint for Workbench app registration."""
    return {"status": "ok"}


@app.get("/")
async def root() -> RedirectResponse:
    """Redirect root to chat page."""
    return RedirectResponse(url="/chat")


@app.get("/chat")
async def chat_page(request: Request) -> Response:
    """Serve the chat interface."""
    return templates.TemplateResponse("chat.html", {"request": request})
```

**Step 2: Verify all tests still pass**

Run: `python3 -m pytest tests/unit/ -v`
Expected: 14 passed

**Step 3: Verify the server starts**

Run: `timeout 5 python3 -m uvicorn code.shukketsu.web.app:app --host 0.0.0.0 --port 9000 2>&1 || true`
Expected: Output includes `Uvicorn running on http://0.0.0.0:9000` (then timeout kills it)

**Step 4: Commit**

```bash
git add code/shukketsu/web/app.py
git commit -m "feat: wire up app with static files, templates, and chat route"
```

---

## Task 8: Final Verification

Run the full verification suite: lint, type check, tests.

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
Expected: 14 passed (7 client + 7 handler)

**Step 5: Manual smoke test**

Start the server:
```bash
python3 -m uvicorn code.shukketsu.web.app:app --host 0.0.0.0 --port 9000
```

Open browser to `http://localhost:9000/chat`. Verify:
- [ ] WoW-themed dark UI renders with gold "Shukketsu" in nav
- [ ] Connection status shows green "Connected"
- [ ] Welcome state shows with 3 example prompts
- [ ] Clicking an example prompt sends the message
- [ ] Welcome state disappears after first message
- [ ] If vLLM is running: tokens stream in, Markdown renders on completion
- [ ] If vLLM is NOT running: friendly error message appears inline
- [ ] Input auto-resizes with multi-line text
- [ ] Enter sends, Shift+Enter adds newline
- [ ] Stop button appears during streaming
- [ ] Input is disabled during streaming, re-enabled after

**Step 6: Final commit (if any fixes were needed)**

```bash
git add -A
git commit -m "fix: address lint/type issues from final verification"
```

---

## Files Created/Modified Summary

| Action | File | Task |
|--------|------|------|
| Create | `code/__init__.py` | 0 |
| Modify | `code/shukketsu/config.py` | 1 |
| Modify | `code/shukketsu/resilience/errors.py` | 1 |
| Create | `code/shukketsu/llm/clients.py` | 2 |
| Create | `tests/unit/test_llm_clients.py` | 2 |
| Create | `code/shukketsu/web/routers/chat.py` | 3 |
| Create | `tests/unit/test_chat_handler.py` | 3 |
| Create | `code/shukketsu/web/static/css/theme.css` | 4 |
| Create | `code/shukketsu/web/templates/base.html` | 5 |
| Create | `code/shukketsu/web/templates/chat.html` | 5 |
| Create | `code/shukketsu/web/static/js/chat.js` | 6 |
| Modify | `code/shukketsu/web/app.py` | 7 |

**Total: 8 files created, 3 files modified, 14 unit tests**
