"""FastAPI application for the Shukketsu web knowledgebase."""

from fastapi import FastAPI

app = FastAPI(
    title="Shukketsu",
    description="TBC Rogue Research Agent — Web Knowledgebase",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint for Workbench app registration."""
    return {"status": "ok"}
