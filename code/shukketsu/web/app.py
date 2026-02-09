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
