"""Simulation web UI routes.

Provides HTML pages for character import, gear display, simulation execution,
and result visualization, plus JSON API endpoints for programmatic access.
"""

import html
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.responses import Response

from code.shukketsu.db.connection import get_connection, init_db
from code.shukketsu.sim.comparator import ValidationReport
from code.shukketsu.sim.models import GearSlot, SimConfig
from code.shukketsu.sim.runner import SimRunner
from code.shukketsu.sim.validation_pipeline import ValidationPipeline

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

        try:
            iterations = int(str(form.get("iterations", "1000")))
        except (ValueError, TypeError):
            return HTMLResponse('<div class="text-red-400 p-4">Error: Invalid iterations value.</div>')
        if iterations < 1 or iterations > 1_000_000:
            return HTMLResponse(
                '<div class="text-red-400 p-4">Error: Iterations must be between 1 and 1,000,000.</div>'
            )
        config = config.model_copy(update={"iterations": iterations})

        result = await runner.sim_run(config)
        return _templates.TemplateResponse(request, "sim/partials/results_panel.html", {"result": result})
    except Exception as exc:
        logger.exception("Simulation run failed")
        return HTMLResponse(f'<div class="text-red-400 p-4">Error: {html.escape(str(exc))}</div>')


# ---------------------------------------------------------------------------
# Validation routes
# ---------------------------------------------------------------------------


@router.get("/sim/validate/", response_class=HTMLResponse)
async def validate_page(request: Request) -> Response:
    """Render validation dashboard with history of past runs."""
    runs: list[dict[str, Any]] = []
    try:
        conn = get_connection()
        init_db(conn)
        cursor = conn.execute(
            """SELECT id, character_name, run_type, total_fights, included_fights,
                      overall_dps_drift_pct, overall_status, created_at
               FROM validation_runs ORDER BY created_at DESC LIMIT 20"""
        )
        for row in cursor.fetchall():
            runs.append(
                {
                    "id": row[0],
                    "character_name": row[1],
                    "run_type": row[2],
                    "total_fights": row[3],
                    "included_fights": row[4],
                    "overall_dps_drift_pct": row[5],
                    "overall_status": row[6],
                    "created_at": row[7],
                }
            )
    except Exception:
        logger.debug("Could not load validation runs", exc_info=True)
    return _templates.TemplateResponse(request, "sim/validate/index.html", {"runs": runs})


@router.post("/sim/validate/run", response_class=HTMLResponse)
async def validate_run(request: Request, character_name: str = Form(...)) -> Response:
    """Trigger a validation run and return the report partial."""
    try:
        conn = get_connection()
        init_db(conn)
        pipeline = ValidationPipeline(conn)
        report = await pipeline.run_validation(character_name)
        return _templates.TemplateResponse(request, "sim/validate/partials/report.html", {"report": report})
    except Exception as exc:
        logger.exception("Validation run failed")
        return HTMLResponse(f'<div class="text-red-400 p-4">Error: {html.escape(str(exc))}</div>')


@router.get("/sim/validate/report/{run_id}", response_class=HTMLResponse)
async def validate_report(request: Request, run_id: int) -> Response:
    """Render a stored validation report."""
    try:
        conn = get_connection()
        init_db(conn)
        row = conn.execute("SELECT report_json FROM validation_runs WHERE id = ?", (run_id,)).fetchone()
    except Exception:
        logger.debug("Could not load validation report %d", run_id, exc_info=True)
        raise HTTPException(status_code=404, detail="Report not found")
    if row is None:
        raise HTTPException(status_code=404, detail="Report not found")
    report = ValidationReport(**json.loads(row[0]))
    return _templates.TemplateResponse(request, "sim/validate/partials/report.html", {"report": report})


# ---------------------------------------------------------------------------
# JSON API routes
# ---------------------------------------------------------------------------


@router.post("/api/sim/run")
async def api_sim_run(config: SimConfig) -> dict[str, Any]:
    """JSON API for running a simulation."""
    try:
        runner = SimRunner()
        result = await runner.sim_run(config)
        return result.model_dump()
    except Exception:
        logger.exception("Simulation API run failed")
        raise HTTPException(status_code=500, detail="Simulation run failed")


@router.post("/api/sim/compare")
async def api_sim_compare(body: CompareRequest) -> dict[str, Any]:
    """JSON API for comparing two configurations."""
    try:
        runner = SimRunner()
        result = await runner.sim_compare(body.config_a, body.config_b)
        return result.model_dump()
    except Exception:
        logger.exception("Simulation comparison failed")
        raise HTTPException(status_code=500, detail="Simulation comparison failed")


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
