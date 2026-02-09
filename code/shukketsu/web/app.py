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
