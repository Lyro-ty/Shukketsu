"""FastAPI application for the Shukketsu web knowledgebase."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from code.shukketsu.observability.tracer import flush_traces, init_langfuse
from code.shukketsu.web.routers.backup import router as backup_router
from code.shukketsu.web.routers.chat import router as chat_router
from code.shukketsu.web.routers.freshness import router as freshness_router
from code.shukketsu.web.routers.wiki import router as wiki_router

_WEB_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown hooks."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    init_langfuse()
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
