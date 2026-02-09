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
    yield  # makes this an async generator


async def mock_stream_error(*args, **kwargs):
    """Raises LLMUnavailableError immediately."""
    raise LLMUnavailableError("vLLM is not running on port 8000")
    yield  # makes this an async generator


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
