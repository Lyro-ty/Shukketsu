"""WebSocket chat handler for agent-based conversations."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langfuse import get_client, observe

from code.shukketsu import config
from code.shukketsu.agents.tasks import AgentTask, AnalysisTask, OrchestratorResult, ResearchTask
from code.shukketsu.resilience.errors import FailureMode, ShukketsuError
from code.shukketsu.routing.models import TaskCategory, TaskComplexity
from code.shukketsu.routing.router import classify_query

if TYPE_CHECKING:
    import sqlite3

    from code.shukketsu.agents.base import BaseAgent
    from code.shukketsu.ingest.embedder import Embedder
    from code.shukketsu.memory.manager import MemoryManager
    from code.shukketsu.memory.models import SessionMemory

logger = logging.getLogger(__name__)

router = APIRouter()

# User-facing error messages mapped from internal FailureMode enum.
# Prevents leaking internal details (service names, URLs, error bodies) to clients.
_USER_ERROR_MESSAGES: dict[FailureMode, str] = {
    FailureMode.MODEL_UNAVAILABLE: "The AI model is temporarily unavailable. Please try again shortly.",
    FailureMode.LLM_TIMEOUT: "The response took too long. Please try a simpler question or try again.",
    FailureMode.LLM_LOOP: "The agent got stuck in a loop. Please rephrase your question.",
    FailureMode.LLM_MALFORMED_OUTPUT: "The AI returned an unexpected response. Please try again.",
    FailureMode.RATE_LIMITED: "Too many requests. Please wait a moment and try again.",
    FailureMode.NETWORK_TIMEOUT: "A network request timed out. Please try again.",
    FailureMode.WCL_API: "Could not retrieve Warcraft Logs data. Please try again later.",
    FailureMode.DB_ERROR: "A database error occurred. Please try again.",
    FailureMode.EMBEDDING_ERROR: "The search system is temporarily unavailable. Please try again.",
    FailureMode.SIM_ERROR: "The simulation encountered an error. Please check your configuration.",
    FailureMode.SIM_TIMEOUT: "The simulation took too long. Try reducing the number of iterations.",
}

_db_conn: sqlite3.Connection | None = None
_researcher_instance: BaseAgent | None = None
_orchestrator_instance: BaseAgent | None = None
_analyst_instance: BaseAgent | None = None
_memory_manager_instance: MemoryManager | None = None


class ChatSession:
    """Per-connection chat state."""

    def __init__(self) -> None:
        self.session_id: str = str(uuid.uuid4())
        self.history: list[dict[str, str]] = []
        self.is_streaming: bool = False

    def add_message(self, role: str, content: str) -> None:
        """Append a message and trim history to max pairs."""
        self.history.append({"role": role, "content": content})
        max_messages = config.CHAT_MAX_HISTORY_PAIRS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]


def _get_agents() -> tuple[BaseAgent, BaseAgent, BaseAgent]:
    """Get or create agent singletons (researcher + orchestrator + analyst).

    Lazy initialization avoids import-time side effects (DB connection,
    extension loading). Agents are created once and reused. All agents
    share a single DB connection for consistency.
    """
    global _researcher_instance, _orchestrator_instance, _analyst_instance  # noqa: PLW0603

    if _researcher_instance is None:
        global _db_conn  # noqa: PLW0603
        from code.shukketsu.agents.factory import AgentFactory
        from code.shukketsu.agents.tasks import AgentRole
        from code.shukketsu.db.connection import get_connection, init_db
        from code.shukketsu.ingest.embedder import get_embedder
        from code.shukketsu.ingest.pipeline import IngestPipeline
        from code.shukketsu.knowledge.manager import KnowledgeManager
        from code.shukketsu.scraping.fetcher import WebFetcher
        from code.shukketsu.scraping.rate_limiter import RateLimiter
        from code.shukketsu.scraping.robots import RobotsChecker
        from code.shukketsu.tools.analysis import SimCompareTool, SimOptimizeTool, SimRunTool
        from code.shukketsu.tools.knowledge.graph_search import GraphSearchTool
        from code.shukketsu.tools.knowledge.search import RagSearchTool
        from code.shukketsu.tools.registry import ToolRegistry
        from code.shukketsu.tools.research.web_ingest import WebIngestTool
        from code.shukketsu.tools.research.web_search import WebSearchTool

        conn = get_connection()
        _db_conn = conn
        init_db(conn)
        embedder = get_embedder()

        # Shared tool registry for researcher/orchestrator
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

        # Analyst with sim tools
        analyst_registry = ToolRegistry()
        analyst_registry.register(SimRunTool())
        analyst_registry.register(SimCompareTool())
        analyst_registry.register(SimOptimizeTool())
        _analyst_instance = factory.create(
            AgentRole.ANALYST,
            tool_registry=analyst_registry,
        )

        # Also initialize the memory manager with the shared connection
        _init_memory_manager(conn, embedder)

    if _orchestrator_instance is None or _analyst_instance is None:
        raise RuntimeError("Agent instances not initialized")
    return _researcher_instance, _orchestrator_instance, _analyst_instance


def _init_memory_manager(conn: sqlite3.Connection, embedder: Embedder) -> None:
    """Initialize the MemoryManager with the shared DB connection."""
    global _memory_manager_instance  # noqa: PLW0603

    if _memory_manager_instance is None:
        from code.shukketsu.memory.manager import MemoryManager as _MemoryManager

        _memory_manager_instance = _MemoryManager(conn=conn, embed_fn=embedder.embed_query)


def _get_memory_manager() -> MemoryManager:
    """Get the MemoryManager singleton. Must be called after _get_agents().

    Raises RuntimeError if agents haven't been initialized yet.
    """
    if _memory_manager_instance is None:
        # Force agent initialization which also creates memory manager
        _get_agents()

    if _memory_manager_instance is None:
        raise RuntimeError("MemoryManager not initialized")
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
            try:
                data = await websocket.receive_json()
            except ValueError:
                # Invalid JSON from client
                await websocket.send_json({"type": "error", "content": "Invalid JSON message."})
                continue
            await _handle_message(websocket, session, data)
    except WebSocketDisconnect:
        logger.info("Chat WebSocket disconnected")
    except Exception:
        logger.warning("Chat WebSocket error", exc_info=True)


async def _handle_message(websocket: WebSocket, session: ChatSession, data: dict[str, Any]) -> None:
    """Dispatch a single incoming WebSocket message."""
    msg_type = data.get("type")

    if msg_type is None:
        await websocket.send_json({"type": "error", "content": "Missing 'type' field in message."})
        return

    if msg_type == "stop":
        # Signal cancellation — cooperative; running agent will finish current iteration
        session.is_streaming = False
        return

    if msg_type == "feedback":
        trace_id = data.get("trace_id")
        score = data.get("score")
        if trace_id is not None and score is not None:
            try:
                langfuse = get_client()
                langfuse.create_score(
                    trace_id=trace_id,
                    name="user_feedback",
                    value=float(score),
                    comment=f"thumbs {'up' if float(score) > 0 else 'down'}",
                )
            except Exception:
                logger.warning("Failed to record feedback", exc_info=True)
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
        try:
            langfuse = get_client()
            langfuse.update_current_trace(
                session_id=session.session_id,
                tags=["chat"],
                input=content,
            )
        except Exception:
            logger.warning("Langfuse trace update failed, continuing without tracing", exc_info=True)

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

        # Track trajectory for memory extraction
        trajectory: list[dict] = []
        tools_used: list[str] = []

        if (
            decision.complexity == TaskComplexity.TRIVIAL
            and decision.direct_answer is not None
            and decision.direct_answer.strip()
        ):
            answer = decision.direct_answer
        elif decision.category == TaskCategory.ANALYSIS:
            await websocket.send_json({"type": "status", "content": "analyzing..."})
            _, _, analyst = _get_agents()
            analysis_ctx: dict[str, str] = {}
            if memory_context:
                analysis_ctx["memory_context"] = memory_context
            result = await analyst.execute(
                AnalysisTask(query=content, context=analysis_ctx),
                on_status=_send_status,
            )
            answer = result.output
            trajectory = [{"tool_name": t.tool_name, "tool_input": t.tool_input} for t in result.trajectory]
            tools_used = list(dict.fromkeys(t.tool_name for t in result.trajectory))
        elif decision.complexity == TaskComplexity.MODERATE:
            await websocket.send_json({"type": "status", "content": "researching..."})
            researcher, _, _ = _get_agents()
            result = await researcher.execute(
                ResearchTask(
                    query=content,
                    context={"memory_context": memory_context} if memory_context else {},
                ),
                on_status=_send_status,
                model_name=config.FAST_MODEL,
            )
            answer = result.output
            trajectory = [{"tool_name": t.tool_name, "tool_input": t.tool_input} for t in result.trajectory]
            tools_used = list(dict.fromkeys(t.tool_name for t in result.trajectory))
        else:
            await websocket.send_json({"type": "status", "content": "planning..."})
            _, orchestrator, _ = _get_agents()

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
            trajectory = [{"tool_name": t.tool_name, "tool_input": t.tool_input} for t in result.trajectory]
            tools_used = list(dict.fromkeys(t.tool_name for t in result.trajectory))
            if isinstance(result, OrchestratorResult):
                if result.article_path:
                    answer += f"\n\n---\n*Draft article created: {result.article_path}*"
                if result.needs_human_review:
                    answer += "\n*Article pending review in Wiki*"

        try:
            trace_id = get_client().get_current_trace_id()
        except Exception:
            logger.warning("Failed to get Langfuse trace ID", exc_info=True)
            trace_id = None
        session.add_message("assistant", answer)
        await websocket.send_json({"type": "done", "content": answer, "trace_id": trace_id})

        # Memory extraction: store key facts from this conversation (fire-and-forget)
        if config.MEMORY_ENABLED:
            try:
                mm = _get_memory_manager()
                await mm.extract_session_memory(
                    query=content,
                    answer=answer,
                    trajectory=trajectory,
                    session_id=session.session_id,
                )
            except Exception:
                logger.warning("Memory extraction failed", exc_info=True)

            if trajectory:
                try:
                    mm = _get_memory_manager()
                    # Estimate strategy quality from result trajectory:
                    # base 0.5, +0.1 per useful tool call (max 1.0)
                    strategy_quality = min(1.0, 0.5 + 0.1 * len(tools_used))
                    await mm.record_strategy(
                        query=content,
                        tools_used=tools_used,
                        quality=strategy_quality,
                    )
                except Exception:
                    logger.warning("Strategy recording failed", exc_info=True)
    except ShukketsuError as exc:
        if session.history and session.history[-1]["role"] == "user":
            session.history.pop()
        logger.warning("ShukketsuError: %s", exc)
        user_msg = _USER_ERROR_MESSAGES.get(
            exc.failure_mode,
            "An error occurred. Please try again.",
        )
        await websocket.send_json({"type": "error", "content": user_msg})
    except WebSocketDisconnect:
        raise
    except Exception:
        logger.exception("Unexpected error in agent response")
        if session.history and session.history[-1]["role"] == "user":
            session.history.pop()
        await websocket.send_json({"type": "error", "content": "An unexpected error occurred. Please try again."})
    finally:
        session.is_streaming = False
