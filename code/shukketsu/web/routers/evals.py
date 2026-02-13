"""Evaluation dashboard routes.

Provides a lean in-app dashboard for triggering eval runs, viewing
latest results, and tracking metric trends over time. Detailed
trace drill-down links to Langfuse.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langfuse import Langfuse

from code.shukketsu import config
from code.shukketsu.evals.dataset import EvalDatasetManager

logger = logging.getLogger(__name__)

_WEB_DIR = Path(__file__).parent.parent
_templates = Jinja2Templates(directory=_WEB_DIR / "templates")

router = APIRouter(prefix="/evals", tags=["evals"])

# In-memory run tracking (simple — one run at a time)
_current_run: dict[str, Any] | None = None
_background_tasks: set[asyncio.Task[None]] = set()


def _get_langfuse() -> Langfuse:
    """Get the Langfuse client singleton."""
    from code.shukketsu.observability.tracer import get_client

    return get_client()


def _get_dataset_manager() -> EvalDatasetManager:
    """Create an EvalDatasetManager with the current Langfuse client."""
    return EvalDatasetManager(langfuse_client=_get_langfuse())


@router.get("/")
async def evals_dashboard(request: Request) -> HTMLResponse:
    """Render the eval dashboard page."""
    dm = _get_dataset_manager()
    try:
        runs = dm.list_runs(limit=1)
        latest = runs[0] if runs else None
    except Exception:
        latest = None

    return _templates.TemplateResponse(
        request,
        "evals/index.html",
        {"latest_run": latest, "langfuse_host": config.LANGFUSE_HOST, "is_running": _current_run is not None},
    )


@router.post("/run")
async def trigger_eval_run(
    request: Request,
    tier: str | None = Query(default=None),
) -> HTMLResponse:
    """Trigger an eval run. Returns HTMX fragment with initial status."""
    global _current_run  # noqa: PLW0603

    if _current_run is not None:
        return _templates.TemplateResponse(
            request,
            "evals/partials/run_status.html",
            {"status": "already_running", "progress": 0, "total": 0},
        )

    _current_run = {"status": "starting", "progress": 0, "total": 0, "run_name": None}

    # Launch eval in background (set holds reference to prevent GC)
    task = asyncio.create_task(_run_eval_background(tier))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return _templates.TemplateResponse(
        request,
        "evals/partials/run_status.html",
        {"status": "starting", "progress": 0, "total": 0},
    )


async def _run_eval_background(tier: str | None) -> None:
    """Background task that runs the eval suite."""
    global _current_run  # noqa: PLW0603
    try:
        from code.shukketsu.evals.runner import EvalRunner

        dm = _get_dataset_manager()
        dm.sync_dataset()

        runner = EvalRunner(dataset_manager=dm, langfuse_client=_get_langfuse())

        async def _on_progress(completed: int, total: int) -> None:
            if _current_run is not None:
                _current_run["progress"] = completed
                _current_run["total"] = total

        report = await runner.run(tier=tier, on_progress=_on_progress)

        if _current_run is not None:
            _current_run["status"] = "complete"
            _current_run["report"] = {
                "passed": report.passed,
                "avg_faithfulness": report.avg_faithfulness,
                "avg_answer_relevancy": report.avg_answer_relevancy,
                "avg_trajectory_precision": report.avg_trajectory_precision,
                "avg_domain_accuracy": report.avg_domain_accuracy,
                "tier_breakdown": report.tier_breakdown,
                "question_count": len(report.question_results),
            }
    except Exception:
        logger.exception("Eval run failed")
        if _current_run is not None:
            _current_run["status"] = "failed"


@router.get("/run/status")
async def run_status(request: Request) -> HTMLResponse:
    """HTMX polling endpoint for run progress."""
    global _current_run  # noqa: PLW0603

    if _current_run is None:
        return _templates.TemplateResponse(
            request,
            "evals/partials/run_status.html",
            {"status": "idle", "progress": 0, "total": 0},
        )

    status = _current_run["status"]
    ctx: dict[str, Any] = {
        "status": status,
        "progress": _current_run.get("progress", 0),
        "total": _current_run.get("total", 0),
    }

    if status in ("complete", "failed"):
        ctx["report"] = _current_run.get("report")
        # Reset for next run
        _current_run = None

    return _templates.TemplateResponse(request, "evals/partials/run_status.html", ctx)


@router.get("/history")
async def eval_history() -> JSONResponse:
    """Return recent runs as JSON for Chart.js."""
    dm = _get_dataset_manager()
    try:
        runs = dm.list_runs(limit=20)
    except Exception:
        runs = []

    return JSONResponse(content={"runs": runs})


@router.get("/latest")
async def latest_summary(request: Request) -> HTMLResponse:
    """HTMX fragment: summary card for latest run."""
    dm = _get_dataset_manager()
    try:
        runs = dm.list_runs(limit=1)
        latest = runs[0] if runs else None
    except Exception:
        latest = None

    return _templates.TemplateResponse(
        request,
        "evals/partials/summary_card.html",
        {"latest_run": latest, "langfuse_host": config.LANGFUSE_HOST},
    )
