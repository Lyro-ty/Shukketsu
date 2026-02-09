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
