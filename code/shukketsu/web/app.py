"""FastAPI application for the Shukketsu web knowledgebase."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from code.shukketsu import config
from code.shukketsu.observability.tracer import flush_traces, init_langfuse
from code.shukketsu.web.routers.backup import router as backup_router
from code.shukketsu.web.routers.chat import router as chat_router
from code.shukketsu.web.routers.freshness import router as freshness_router
from code.shukketsu.web.routers.wiki import router as wiki_router

_WEB_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)

_OLLAMA_KEEP_ALIVE = "24h"


async def _warmup_ollama() -> None:
    """Pin models in Ollama VRAM with a long keep_alive.

    The 70B model takes minutes to reload from disk. By sending a tiny
    request with keep_alive=24h on startup, we prevent Ollama from
    evicting it after the default 5-minute idle timeout.
    """
    models = [config.REASONING_MODEL, config.ROUTER_MODEL, config.EMBEDDING_MODEL]
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=300.0, connect=10.0)) as client:
        for model in models:
            try:
                if "embed" in model:
                    await client.post(
                        f"{config.OLLAMA_BASE_URL}/api/embed",
                        json={"model": model, "input": "warmup", "keep_alive": _OLLAMA_KEEP_ALIVE},
                    )
                else:
                    await client.post(
                        f"{config.OLLAMA_BASE_URL}/api/chat",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": "hi"}],
                            "keep_alive": _OLLAMA_KEEP_ALIVE,
                            "stream": False,
                        },
                    )
                logger.info("Ollama warmup: %s pinned (keep_alive=%s)", model, _OLLAMA_KEEP_ALIVE)
            except Exception:
                logger.warning("Ollama warmup failed for %s", model, exc_info=True)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown hooks."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    init_langfuse()

    # Pin models in VRAM so first chat request isn't penalised by model loading
    await _warmup_ollama()

    # Run freshness sweep on startup (non-blocking — stale sources just get flagged)
    try:
        from code.shukketsu.db.connection import get_connection, init_db
        from code.shukketsu.freshness.checker import run_freshness_sweep

        conn = get_connection()
        init_db(conn)
        results = await run_freshness_sweep(conn)
        stale_count = sum(1 for r in results if r.changed)
        logger.info("Freshness sweep: %d checked, %d changed", len(results), stale_count)
    except Exception:
        logger.exception("Freshness sweep failed on startup")

    yield
    flush_traces()


app = FastAPI(
    title="Shukketsu",
    description="TBC Rogue Research Agent — Web Knowledgebase",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=_WEB_DIR / "static"), name="static")
app.include_router(backup_router)
app.include_router(chat_router)
app.include_router(freshness_router)
app.include_router(wiki_router)

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
    return templates.TemplateResponse(request, "chat.html")
