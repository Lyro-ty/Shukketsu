"""Combat log upload and parsing routes.

Provides HTML pages for uploading CLEU combat log files and parsing
them into fight metrics via the CLEUParser.
"""

import html
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from code.shukketsu import config
from code.shukketsu.sim.log_parser import CLEUParser

logger = logging.getLogger(__name__)

_WEB_DIR = Path(__file__).parent.parent
_templates = Jinja2Templates(directory=_WEB_DIR / "templates")

router = APIRouter(tags=["logs"])

_MAX_UPLOAD_BYTES = config.LOG_UPLOAD_MAX_SIZE_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# HTML page routes
# ---------------------------------------------------------------------------


@router.get("/logs/", response_class=HTMLResponse)
async def logs_page(request: Request) -> Response:
    """Render the log upload page."""
    return _templates.TemplateResponse(request, "logs/index.html")


@router.post("/logs/upload", response_class=HTMLResponse)
async def logs_upload(
    request: Request,
    character_name: str = Form(...),
    file: UploadFile = File(...),
) -> Response:
    """Parse an uploaded CLEU combat log file and return fight metrics.

    Accepts a multipart file upload and a character name field. Validates
    file size, reads the content, and parses it through CLEUParser.

    Args:
        request: The incoming HTTP request.
        character_name: Character name to filter combat events for.
        file: The uploaded CLEU combat log file.

    Returns:
        HTML partial with parsed fight metrics, or an error partial.
    """
    try:
        content = await file.read()

        if len(content) > _MAX_UPLOAD_BYTES:
            return _templates.TemplateResponse(
                request,
                "logs/error.html",
                {"error": f"File too large ({len(content) / 1024 / 1024:.1f} MB). Maximum is 50 MB."},
            )

        if len(content) == 0:
            return _templates.TemplateResponse(
                request,
                "logs/error.html",
                {"error": "Uploaded file is empty."},
            )

        log_text = content.decode("utf-8", errors="replace")

        parser = CLEUParser()
        fights = parser.parse(log_text, character_name)

        if not fights:
            return _templates.TemplateResponse(
                request,
                "logs/error.html",
                {"error": f"No encounters found for character '{html.escape(character_name)}'."},
            )

        return _templates.TemplateResponse(
            request,
            "logs/fights.html",
            {"fights": fights, "character_name": character_name},
        )
    except UnicodeDecodeError:
        return _templates.TemplateResponse(
            request,
            "logs/error.html",
            {"error": "File does not appear to be a valid text combat log."},
        )
    except Exception as exc:
        logger.exception("Log upload failed")
        return _templates.TemplateResponse(
            request,
            "logs/error.html",
            {"error": f"Parse error: {html.escape(str(exc))}"},
        )
