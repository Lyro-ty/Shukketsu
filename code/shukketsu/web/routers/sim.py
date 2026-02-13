"""Simulation web UI routes.

Provides HTML pages for character import, gear display, simulation execution,
and result visualization, plus JSON API endpoints for programmatic access.
"""

import html
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.responses import Response

from code.shukketsu.sim.models import GearSlot, SimConfig
from code.shukketsu.sim.runner import SimRunner

logger = logging.getLogger(__name__)

_WEB_DIR = Path(__file__).parent.parent
_templates = Jinja2Templates(directory=_WEB_DIR / "templates")

router = APIRouter(tags=["sim"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CompareRequest(BaseModel):
    """Request body for comparing two simulation configurations."""

    config_a: SimConfig
    config_b: SimConfig


# ---------------------------------------------------------------------------
# HTML page routes
# ---------------------------------------------------------------------------


@router.get("/sim/", response_class=HTMLResponse)
async def sim_page(request: Request) -> Response:
    """Render the simulation page with empty initial state."""
    return _templates.TemplateResponse(request, "sim/index.html", {"presets": _get_presets()})


@router.post("/sim/import", response_class=HTMLResponse)
async def sim_import(request: Request, import_text: str = Form(...)) -> Response:
    """Parse import string and return gear table partial."""
    try:
        runner = SimRunner()
        config = runner.build_config_from_import(import_text)
        return _templates.TemplateResponse(
            request,
            "sim/partials/gear_table.html",
            {
                "config": config,
                "gear_slots": list(GearSlot),
                "import_text": import_text,
            },
        )
    except Exception as exc:
        logger.debug("Import failed: %s", exc)
        return _templates.TemplateResponse(request, "sim/partials/import_error.html", {"error": str(exc)})


@router.post("/sim/run", response_class=HTMLResponse)
async def sim_run(request: Request) -> Response:
    """Run simulation and return results panel partial."""
    form = await request.form()
    try:
        runner = SimRunner()
        import_text = str(form.get("import_text", ""))
        config = runner.build_config_from_import(import_text)

        iterations = int(str(form.get("iterations", "1000")))
        config = config.model_copy(update={"iterations": iterations})

        result = await runner.sim_run(config)
        return _templates.TemplateResponse(request, "sim/partials/results_panel.html", {"result": result})
    except Exception as exc:
        logger.exception("Simulation run failed")
        return HTMLResponse(f'<div class="text-red-400 p-4">Error: {html.escape(str(exc))}</div>')


# ---------------------------------------------------------------------------
# JSON API routes
# ---------------------------------------------------------------------------


@router.post("/api/sim/run")
async def api_sim_run(config: SimConfig) -> dict[str, Any]:
    """JSON API for running a simulation."""
    runner = SimRunner()
    result = await runner.sim_run(config)
    return result.model_dump()


@router.post("/api/sim/compare")
async def api_sim_compare(body: CompareRequest) -> dict[str, Any]:
    """JSON API for comparing two configurations."""
    runner = SimRunner()
    result = await runner.sim_compare(body.config_a, body.config_b)
    return result.model_dump()


@router.get("/api/sim/presets")
async def api_sim_presets() -> list[str]:
    """List available buff presets."""
    return _get_presets()


@router.get("/api/sim/items/{slot}")
async def api_sim_items(slot: str) -> list[dict[str, Any]]:
    """Search items available for a given gear slot."""
    from code.shukketsu.sim.items import ItemDatabase

    db = ItemDatabase()
    try:
        gear_slot = GearSlot(slot)
    except ValueError:
        return []
    items = db.items_for_slot(gear_slot)
    return [{"id": i.id, "name": i.name, "item_level": i.item_level, "phase": i.phase} for i in items]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_presets() -> list[str]:
    """Return available buff preset names."""
    return ["full_25man", "full_10man", "self_only"]
