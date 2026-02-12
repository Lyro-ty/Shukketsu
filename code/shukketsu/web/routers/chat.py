"""WebSocket chat handler for agent-based conversations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langfuse import get_client, observe

from code.shukketsu import config
from code.shukketsu.agents.tasks import AgentTask, OrchestratorResult, ResearchTask
from code.shukketsu.resilience.errors import ShukketsuError
from code.shukketsu.routing.models import TaskComplexity
from code.shukketsu.routing.router import classify_query

if TYPE_CHECKING:
    from code.shukketsu.agents.base import BaseAgent
    from code.shukketsu.memory.manager import MemoryManager
    from code.shukketsu.memory.models import SessionMemory

logger = logging.getLogger(__name__)

router = APIRouter()

_researcher_instance: BaseAgent | None = None
_orchestrator_instance: BaseAgent | None = None
_memory_manager_instance: MemoryManager | None = None


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


def _get_agents() -> tuple[BaseAgent, BaseAgent]:
    """Get or create agent singletons (researcher + orchestrator).

    Lazy initialization avoids import-time side effects (DB connection,
    extension loading). Agents are created once and reused.
    """
    global _researcher_instance, _orchestrator_instance  # noqa: PLW0603

    if _researcher_instance is None:
        from code.shukketsu.agents.factory import AgentFactory
        from code.shukketsu.agents.tasks import AgentRole
        from code.shukketsu.db.connection import get_connection, init_db
        from code.shukketsu.ingest.embedder import get_embedder
        from code.shukketsu.ingest.pipeline import IngestPipeline
        from code.shukketsu.knowledge.manager import KnowledgeManager
        from code.shukketsu.scraping.fetcher import WebFetcher
        from code.shukketsu.scraping.rate_limiter import RateLimiter
        from code.shukketsu.scraping.robots import RobotsChecker
        from code.shukketsu.tools.knowledge.graph_search import GraphSearchTool
        from code.shukketsu.tools.knowledge.search import RagSearchTool
        from code.shukketsu.tools.registry import ToolRegistry
        from code.shukketsu.tools.research.web_ingest import WebIngestTool
        from code.shukketsu.tools.research.web_search import WebSearchTool

        conn = get_connection()
        init_db(conn)
        embedder = get_embedder()

        registry = ToolRegistry()
        registry.register(RagSearchTool(conn=conn, embed_fn=embedder.embed_query))
        registry.register(GraphSearchTool(conn=conn))

        # Web search + ingest tools
        fetcher = WebFetcher(rate_limiter=RateLimiter(), robots_checker=RobotsChecker())
        pipeline = IngestPipeline(conn=conn, embedder=embedder)
        registry.register(WebSearchTool())
        registry.register(WebIngestTool(fetcher=fetcher, pipeline=pipeline))

        factory = AgentFactory()
        km = KnowledgeManager(conn, config.WIKI_PATH)

        _researcher_instance = factory.create(
            AgentRole.RESEARCHER,
            tool_registry=registry,
        )
        _orchestrator_instance = factory.create(
            AgentRole.ORCHESTRATOR,
            tool_registry=registry,
            factory=factory,
            knowledge_manager=km,
        )

    assert _orchestrator_instance is not None  # Set in same block as _researcher_instance
    return _researcher_instance, _orchestrator_instance


def _get_memory_manager() -> MemoryManager:
    """Get or create the MemoryManager singleton.

    Lazy initialization to avoid import-time side effects.
    Reuses the same DB connection and embedder as the agents.
    """
    global _memory_manager_instance  # noqa: PLW0603

    if _memory_manager_instance is None:
        from code.shukketsu.db.connection import get_connection, init_db
        from code.shukketsu.ingest.embedder import get_embedder
        from code.shukketsu.memory.manager import MemoryManager as _MemoryManager

        conn = get_connection()
        init_db(conn)
        embedder = get_embedder()
        _memory_manager_instance = _MemoryManager(conn=conn, embed_fn=embedder.embed_query)

    return _memory_manager_instance


def _format_memory_context(memories: list[SessionMemory]) -> str:
    """Format recalled memories into a context string for the agent.

    Each memory is rendered with its summary and key facts,
    truncated to MEMORY_MAX_CONTEXT_CHARS.
    """
    if not memories:
        return ""

    parts: list[str] = []
    total_chars = 0

    for mem in memories:
        entry = f"- **{mem.query}**: {mem.answer_summary}"
        if mem.key_facts:
            entry += " (" + "; ".join(mem.key_facts) + ")"

        if total_chars + len(entry) > config.MEMORY_MAX_CONTEXT_CHARS:
            break
        parts.append(entry)
        total_chars += len(entry)

    return "\n".join(parts)


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
    moderate queries go to the Researcher, complex queries go to the
    Orchestrator for multi-agent coordination.
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

        # Memory recall: fetch relevant context from previous sessions
        memory_context: str | None = None
        if config.MEMORY_ENABLED:
            try:
                mm = _get_memory_manager()
                memories = await mm.recall_relevant(content)
                memory_context = _format_memory_context(memories) or None
            except Exception:
                logger.warning("Memory recall failed, continuing without context", exc_info=True)

        await websocket.send_json({"type": "status", "content": "routing..."})
        decision = await classify_query(content)
        logger.info("Route: %s → %s", decision.complexity, decision.category)

        async def _send_status(msg: str | dict) -> None:
            if isinstance(msg, dict):
                await websocket.send_json(msg)
            else:
                await websocket.send_json({"type": "status", "content": msg})

        if (
            decision.complexity == TaskComplexity.TRIVIAL
            and decision.direct_answer is not None
            and decision.direct_answer.strip()
        ):
            answer = decision.direct_answer
        elif decision.complexity == TaskComplexity.MODERATE:
            await websocket.send_json({"type": "status", "content": "researching..."})
            researcher, _ = _get_agents()
            result = await researcher.execute(
                ResearchTask(
                    query=content,
                    context={"memory_context": memory_context} if memory_context else {},
                ),
                on_status=_send_status,
            )
            answer = result.output
        else:
            await websocket.send_json({"type": "status", "content": "planning..."})
            _, orchestrator = _get_agents()

            # Recall strategy hints for the Orchestrator
            strategy_hints = ""
            if config.MEMORY_ENABLED:
                try:
                    mm = _get_memory_manager()
                    strategies = await mm.recall_strategies(top_k=config.MEMORY_STRATEGY_TOP_K)
                    if strategies:
                        hint_parts = [
                            f"- {s['query_pattern']}: {', '.join(s['successful_tools'])}"
                            f" (quality: {s['avg_quality']:.1f})"
                            for s in strategies
                        ]
                        strategy_hints = "\n".join(hint_parts)
                except Exception:
                    logger.warning("Strategy recall failed", exc_info=True)

            context: dict[str, str] = {}
            if memory_context:
                context["memory_context"] = memory_context
            if strategy_hints:
                context["strategy_hints"] = strategy_hints

            result = await orchestrator.execute(
                AgentTask(
                    query=content,
                    context=context,
                ),
                on_status=_send_status,
            )
            answer = result.output
            if isinstance(result, OrchestratorResult):
                if result.article_path:
                    answer += f"\n\n---\n*Draft article created: {result.article_path}*"
                if result.needs_human_review:
                    answer += "\n*Article pending review in Wiki*"

        session.add_message("assistant", answer)
        await websocket.send_json({"type": "done", "content": answer})

        # Memory extraction: store key facts from this conversation (fire-and-forget)
        if config.MEMORY_ENABLED:
            try:
                mm = _get_memory_manager()
                await mm.extract_session_memory(
                    query=content,
                    answer=answer,
                    trajectory=[],
                    session_id=str(id(session)),
                )
            except Exception:
                logger.warning("Memory extraction failed", exc_info=True)

            try:
                mm = _get_memory_manager()
                await mm.record_strategy(
                    query=content,
                    tools_used=[],
                    quality=0.5,
                )
            except Exception:
                logger.warning("Strategy recording failed", exc_info=True)
    except ShukketsuError as exc:
        if session.history and session.history[-1]["role"] == "user":
            session.history.pop()
        await websocket.send_json({"type": "error", "content": str(exc)})
    except Exception:
        logger.exception("Unexpected error in agent response")
        if session.history and session.history[-1]["role"] == "user":
            session.history.pop()
        await websocket.send_json({"type": "error", "content": "An unexpected error occurred. Please try again."})
    finally:
        session.is_streaming = False
