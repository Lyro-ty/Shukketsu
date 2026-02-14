"""Wiki article browser, reader, and review routes."""

import html
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from code.shukketsu import config
from code.shukketsu.knowledge.manager import ArticleStatus, KnowledgeManager, Spec
from code.shukketsu.web.wiki_render import render_markdown

logger = logging.getLogger(__name__)

_WEB_DIR = Path(__file__).parent.parent
_templates = Jinja2Templates(directory=_WEB_DIR / "templates")

router = APIRouter(prefix="/wiki", tags=["wiki"])

_km_instance: KnowledgeManager | None = None
_db_conn = None  # Exposed for shutdown cleanup in lifespan


def _get_km() -> KnowledgeManager:
    """Get or create KnowledgeManager singleton."""
    global _km_instance, _db_conn  # noqa: PLW0603
    if _km_instance is None:
        from code.shukketsu.db.connection import get_connection, init_db

        conn = get_connection()
        init_db(conn)
        _db_conn = conn
        _km_instance = KnowledgeManager(conn, config.WIKI_PATH)
    return _km_instance


def _filter_articles(
    km: KnowledgeManager,
    spec: str | None = None,
    status: str | None = None,
    q: str | None = None,
) -> list:
    """Query and filter articles. Shared by full-page and HTMX routes."""
    try:
        status_enum = ArticleStatus(status) if status else None
        spec_enum = Spec(spec) if spec else None
    except ValueError:
        return []
    articles = km.list_articles(status=status_enum, spec=spec_enum)
    if q:
        q_lower = q.lower()
        articles = [a for a in articles if q_lower in a.title.lower()]
    return articles


# --- Full page routes ---


@router.get("/", response_class=HTMLResponse)
async def wiki_browser(
    request: Request,
    spec: str | None = None,
    status: str | None = None,
    q: str | None = None,
    km: KnowledgeManager = Depends(_get_km),
) -> HTMLResponse:
    """Article browser with filters and search."""
    articles = _filter_articles(km, spec, status, q)
    return _templates.TemplateResponse(
        request,
        "wiki/browser.html",
        {"articles": articles, "spec": spec, "status": status, "q": q},
    )


@router.get("/review", response_class=HTMLResponse)
async def wiki_review(
    request: Request,
    km: KnowledgeManager = Depends(_get_km),
) -> HTMLResponse:
    """Review queue -- articles pending human approval."""
    articles = km.list_articles(status=ArticleStatus.REVIEW)
    return _templates.TemplateResponse(
        request,
        "wiki/review.html",
        {"articles": articles},
    )


@router.get("/view/{path:path}", response_class=HTMLResponse)
async def wiki_article(
    request: Request,
    path: str,
    km: KnowledgeManager = Depends(_get_km),
) -> HTMLResponse:
    """Render a single article with metadata sidebar."""
    if not path.endswith(".md"):
        path = path + ".md"
    try:
        meta, content = km.read_article(path)
    except FileNotFoundError:
        return _templates.TemplateResponse(
            request,
            "wiki/browser.html",
            {"articles": [], "spec": None, "status": None, "q": None},
            status_code=404,
        )
    # Read authoritative status from DB (set_status only updates DB, not frontmatter)
    db_status = km.get_article_status(path)
    if db_status is not None:
        meta = meta.model_copy(update={"status": db_status})
    # Compute stale source count for warning badge
    stale_source_count = 0
    if meta.sources:
        from code.shukketsu.freshness.checker import count_stale_sources

        source_urls = [s.url for s in meta.sources]
        stale_source_count = count_stale_sources(km.get_db_connection(), source_urls)

    article_html = render_markdown(content)
    return _templates.TemplateResponse(
        request,
        "wiki/article.html",
        {"meta": meta, "article_html": article_html, "stale_source_count": stale_source_count},
    )


# --- Action routes (HTMX) ---


@router.post("/approve/{article_id}", response_class=HTMLResponse)
async def wiki_approve(
    request: Request,
    article_id: int,
    km: KnowledgeManager = Depends(_get_km),
) -> HTMLResponse:
    """Approve an article: review -> published."""
    try:
        path = km.get_path_by_id(article_id)
    except ValueError:
        return HTMLResponse(
            '<div class="text-red-400 text-sm p-2">Article not found.</div>',
            status_code=404,
        )
    try:
        km.set_status(path, ArticleStatus.PUBLISHED)
    except ValueError as exc:
        return HTMLResponse(
            f'<div class="text-red-400 text-sm p-2">{html.escape(str(exc))}</div>',
            status_code=400,
        )
    return HTMLResponse(
        '<div class="flash-message text-green-400 bg-green-500/10 border border-green-500/30">Article published.</div>'
    )


@router.post("/reject/{article_id}", response_class=HTMLResponse)
async def wiki_reject(
    request: Request,
    article_id: int,
    reason: str = Form(default=""),
    km: KnowledgeManager = Depends(_get_km),
) -> HTMLResponse:
    """Reject an article: review -> draft."""
    try:
        path = km.get_path_by_id(article_id)
    except ValueError:
        return HTMLResponse(
            '<div class="text-red-400 text-sm p-2">Article not found.</div>',
            status_code=404,
        )
    try:
        km.reject_article(path, reason=reason)
    except ValueError as exc:
        return HTMLResponse(
            f'<div class="text-red-400 text-sm p-2">{html.escape(str(exc))}</div>',
            status_code=400,
        )
    return HTMLResponse(
        '<div class="flash-message text-yellow-400 bg-yellow-500/10 border border-yellow-500/30">'
        "Article returned to draft.</div>"
    )


# --- HTMX fragment routes ---


@router.get("/htmx/articles", response_class=HTMLResponse)
async def wiki_htmx_articles(
    request: Request,
    spec: str | None = None,
    status: str | None = None,
    q: str | None = None,
    km: KnowledgeManager = Depends(_get_km),
) -> HTMLResponse:
    """Return filtered article list fragment (no base template)."""
    articles = _filter_articles(km, spec, status, q)
    return _templates.TemplateResponse(
        request,
        "wiki/partials/article_list.html",
        {"articles": articles},
    )


@router.get("/htmx/review", response_class=HTMLResponse)
async def wiki_htmx_review(
    request: Request,
    km: KnowledgeManager = Depends(_get_km),
) -> HTMLResponse:
    """Return review list fragment (no base template)."""
    articles = km.list_articles(status=ArticleStatus.REVIEW)
    return _templates.TemplateResponse(
        request,
        "wiki/partials/review_list.html",
        {"articles": articles},
    )
