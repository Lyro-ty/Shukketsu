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


async def _handle_message(websocket: WebSocket, session: ChatSession, data: dict[str, str]) -> None:
    """Dispatch a single incoming WebSocket message."""
    msg_type = data.get("type")

    if msg_type is None:
        await websocket.send_json({"type": "error", "content": "Missing 'type' field in message."})
        return

    if msg_type == "stop":
        session.request_stop()
        return

    if msg_type != "message":
        await websocket.send_json({"type": "error", "content": f"Unknown message type: {msg_type}"})
        return

    content = data.get("content", "").strip()
    if not content:
        await websocket.send_json({"type": "error", "content": "Message content cannot be empty."})
        return

    if session.is_streaming:
        await websocket.send_json({"type": "error", "content": "Please wait for the current response to finish."})
        return

    await _stream_response(websocket, session, content)


async def _stream_response(websocket: WebSocket, session: ChatSession, content: str) -> None:
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
