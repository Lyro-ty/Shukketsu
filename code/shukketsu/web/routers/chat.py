"""WebSocket chat handler for agent-based conversations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langfuse import get_client, observe

from code.shukketsu import config
from code.shukketsu.resilience.errors import ShukketsuError
from code.shukketsu.routing.models import TaskComplexity
from code.shukketsu.routing.router import classify_query

if TYPE_CHECKING:
    from code.shukketsu.agents.base import BaseAgent

logger = logging.getLogger(__name__)

router = APIRouter()

_agent_instance: BaseAgent | None = None


class ChatSession:
    """Per-connection chat state."""

    def __init__(self) -> None:
        self.history: list[dict[str, str]] = []
        self.is_streaming: bool = False

    def add_message(self, role: str, content: str) -> None:
        """Append a message and trim history to max pairs."""
        self.history.append({"role": role, "content": content})
        max_messages = config.CHAT_MAX_HISTORY_PAIRS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]


def _get_agent() -> BaseAgent:
    """Get or create the agent singleton.

    Lazy initialization avoids import-time side effects (DB connection,
    extension loading). The agent is created once and reused.
    """
    global _agent_instance  # noqa: PLW0603

    if _agent_instance is None:
        from code.shukketsu.agents.base import BaseAgent
        from code.shukketsu.db.connection import get_connection, init_db
        from code.shukketsu.ingest.embedder import get_embedder
        from code.shukketsu.ingest.pipeline import IngestPipeline
        from code.shukketsu.scraping.fetcher import WebFetcher
        from code.shukketsu.scraping.rate_limiter import RateLimiter
        from code.shukketsu.scraping.robots import RobotsChecker
        from code.shukketsu.tools.knowledge.search import RagSearchTool
        from code.shukketsu.tools.registry import ToolRegistry
        from code.shukketsu.tools.research.web_ingest import WebIngestTool
        from code.shukketsu.tools.research.web_search import WebSearchTool

        conn = get_connection()
        init_db(conn)
        embedder = get_embedder()

        registry = ToolRegistry()
        registry.register(RagSearchTool(conn=conn, embed_fn=embedder.embed_query))

        # Web search + ingest tools
        fetcher = WebFetcher(rate_limiter=RateLimiter(), robots_checker=RobotsChecker())
        pipeline = IngestPipeline(conn=conn, embedder=embedder)
        registry.register(WebSearchTool())
        registry.register(WebIngestTool(fetcher=fetcher, pipeline=pipeline))

        _agent_instance = BaseAgent(tool_registry=registry)

    return _agent_instance


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
        return

    if msg_type != "message":
        await websocket.send_json({"type": "error", "content": f"Unknown message type: {msg_type}"})
        return

    content = data.get("content", "").strip()
    if not content:
        await websocket.send_json({"type": "error", "content": "Message content cannot be empty."})
        return

    if len(content) > config.CHAT_MAX_MESSAGE_LENGTH:
        await websocket.send_json(
            {
                "type": "error",
                "content": f"Message too long ({len(content)} chars). Maximum is {config.CHAT_MAX_MESSAGE_LENGTH}.",
            }
        )
        return

    if session.is_streaming:
        await websocket.send_json({"type": "error", "content": "Please wait for the current response to finish."})
        return

    await _agent_response(websocket, session, content)


@observe()
async def _agent_response(websocket: WebSocket, session: ChatSession, content: str) -> None:
    """Get an agent response for the given user message.

    Routes through Qwen 4B first: trivial queries get a direct answer,
    everything else goes to the Llama 70B agent with tools.
    """
    session.is_streaming = True
    session.add_message("user", content)

    try:
        langfuse = get_client()
        langfuse.update_current_trace(
            session_id=str(id(session)),
            tags=["chat"],
            input=content,
        )
        await websocket.send_json({"type": "status", "content": "routing..."})
        decision = await classify_query(content)
        logger.info("Route: %s → %s", decision.complexity, decision.category)

        if (
            decision.complexity == TaskComplexity.TRIVIAL
            and decision.direct_answer is not None
            and decision.direct_answer.strip()
        ):
            session.add_message("assistant", decision.direct_answer)
            await websocket.send_json({"type": "done", "content": decision.direct_answer})
        else:
            await websocket.send_json({"type": "status", "content": "thinking..."})
            agent = _get_agent()

            async def _send_status(msg: str) -> None:
                await websocket.send_json({"type": "status", "content": msg})

            answer = await agent.run(content, on_status=_send_status)
            session.add_message("assistant", answer)
            await websocket.send_json({"type": "done", "content": answer})
    except ShukketsuError as exc:
        if session.history and session.history[-1]["role"] == "user":
            session.history.pop()
        await websocket.send_json({"type": "error", "content": str(exc)})
    except Exception as exc:
        logger.exception("Unexpected error in agent response")
        if session.history and session.history[-1]["role"] == "user":
            session.history.pop()
        await websocket.send_json(
            {"type": "error", "content": "An unexpected error occurred. Please try again."}
        )
    finally:
        session.is_streaming = False
