# Phase 2 Step 8: Wiki UI

> Design document for Phase 2, Step 8. Validated through brainstorming session 2026-02-11.

## Overview

The Wiki UI adds a web interface for browsing, reading, and managing wiki articles produced by the
Writer and Editor agents. Users can browse all articles with filters (spec, status, search),
read rendered Markdown with a metadata sidebar showing confidence scores and verification summaries,
and approve or reject articles through a dedicated review queue.

The UI builds on the existing FastAPI + Jinja2 + Tailwind stack. HTMX is added for dynamic
interactions (filtering, search, approve/reject) without full page reloads. Markdown is rendered
server-side using Python's `markdown` library, which pairs naturally with HTMX's fragment-swap
pattern. The existing dark WoW theme is extended with wiki-specific styles for confidence bars,
status badges, and article cards.

Navigation is a simple top nav bar with two links: Chat and Wiki. The KnowledgeManager (built in
Step 5) handles all data operations — the wiki routes are a thin HTTP layer on top of it, plus a
new `reject_article()` method for the review workflow.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Navigation | Simple top nav with Chat + Wiki links | Only two sections; sidebar is overkill. Easy to extend later. |
| Markdown rendering | Server-side via Python `markdown` library | HTMX works best with server-rendered HTML. No client-side post-processing needed. Consistent rendering. |
| HTMX integration | CDN script in base.html, fragment endpoints | Standard HTMX pattern. Debounced search, filter swaps, approve/reject without page reload. |
| Article detail layout | Two-column: content (70%) + metadata sidebar (30%) | Clean reading experience with verification data accessible but not intrusive. Collapses on mobile. |
| Claim display | Summary counts on article view, full detail on review page only | Keeps reading experience clean. Reviewers need the detail, readers don't. |
| Article search | Title substring via SQL LIKE | Few articles expected. FTS5 on articles is overkill for Phase 2. |
| Review workflow | Inline approve/reject on review page | Single page with all review-status articles. HTMX removes cards on action. |
| Rejection handling | New `reject_article()` method on KnowledgeManager | Keeps `set_status()` strict for forward transitions. Rejection is a special backward flow with reason tracking. |
| Route prefix for articles | `/wiki/view/{path:path}` | Avoids catch-all path ambiguity with `/wiki/review`. Clean URL structure. |
| Approve/reject by article ID | Look up `path` from `id` via DB query, then delegate to KnowledgeManager | HTMX sends integer ID (clean URLs, no path-encoding issues). Small helper resolves ID → path. |
| Wiki router templates | Own `Jinja2Templates` instance in wiki router | Avoids circular import with `app.py`. Points to same templates directory. Standard FastAPI pattern. |
| Connection status | Move from `base.html` to `chat.html` | Connection indicator only relevant on chat page (WebSocket). Wiki pages have no WebSocket. |
| CSS approach | Tailwind utility classes + small `wiki.css` for custom elements | Consistent with existing chat UI. Custom CSS only for confidence bars and animations. |

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `web/routers/wiki.py` | Wiki routes: browser, review, article detail, approve/reject, HTMX fragments |
| `web/templates/wiki/browser.html` | Full page: article list with search + filters |
| `web/templates/wiki/review.html` | Full page: review queue with approve/reject actions |
| `web/templates/wiki/article.html` | Full page: rendered article with metadata sidebar |
| `web/templates/wiki/partials/article_list.html` | HTMX fragment: filtered article cards |
| `web/templates/wiki/partials/review_list.html` | HTMX fragment: review cards with action buttons |
| `web/static/css/wiki.css` | Confidence bars, status badges, article card hover styles |
| `web/wiki_render.py` | Thin wrapper: `render_markdown(text) -> str` using `markdown` with extensions |

### Modified Files

| File | Change |
|------|--------|
| `web/app.py` | Mount wiki router at `/wiki` |
| `web/templates/base.html` | Add Chat/Wiki nav links, add HTMX CDN script tag, move connection status to chat.html |
| `web/templates/chat.html` | Receive connection status indicator moved from base.html |
| `knowledge/manager.py` | Add `reject_article(path, reason)` method, add `id` to `ArticleSummary` + `list_articles()` query, add `get_path_by_id()` helper, add `rejection_reason` to `ArticleMeta`, add `verified_claims`/`unverified_claims` to `ArticleSummary` |

### Unchanged Files

- `web/routers/chat.py` — Chat functionality untouched
- `web/static/js/chat.js` — Chat JS untouched (connection status element ID unchanged, just moved to chat.html)
- `requirements.txt` — `markdown>=3.7` already present, no change needed
- `web/static/css/theme.css` — Existing theme preserved; wiki styles in separate file
- `agents/` — No agent changes
- `db/schema.sql` — No schema changes (rejection reason stored in frontmatter, not DB)
- `routing/` — No routing changes
- `tools/` — No tool changes

## Component 1: Markdown Renderer (`web/wiki_render.py`)

A thin utility module that configures Python's `markdown` library:

```python
import markdown

_MD = markdown.Markdown(extensions=["tables", "fenced_code", "toc"])

def render_markdown(text: str) -> str:
    """Render Markdown text to HTML. Resets state between calls."""
    _MD.reset()
    return _MD.convert(text)
```

The renderer is stateless between calls (`.reset()` clears internal state). Extensions enabled:
- `tables` — gear comparisons, stat breakdowns
- `fenced_code` — code/config snippets with syntax hints
- `toc` — generates table of contents metadata (available via `_MD.toc` if needed later)

No sanitization needed since all article content is system-generated (Writer agent output) and
stored in git-tracked files. User-submitted content never enters this path.

### Edge Cases
- Empty string → returns empty string
- Frontmatter delimiters (`---`) in content → not an issue since KnowledgeManager strips
  frontmatter before returning content
- Very long articles → no concern, `markdown` handles large documents efficiently

## Component 2: Wiki Routes (`web/routers/wiki.py`)

### Route Table

```
GET  /wiki/                    # Article browser (full page)
GET  /wiki/review              # Review queue (full page)
GET  /wiki/view/{path:path}    # Article detail (full page)
POST /wiki/approve/{article_id} # Approve article (HTMX)
POST /wiki/reject/{article_id}  # Reject article (HTMX)
GET  /wiki/htmx/articles       # Article list fragment (HTMX)
GET  /wiki/htmx/review         # Review list fragment (HTMX)
```

### Module Setup and Dependencies

The wiki router creates its own `Jinja2Templates` instance (avoids circular import with `app.py`)
and uses a lazy-singleton pattern for the `KnowledgeManager` (same approach as the chat router
uses for agents):

```python
from pathlib import Path

from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from code.shukketsu import config
from code.shukketsu.knowledge.manager import KnowledgeManager

_WEB_DIR = Path(__file__).parent.parent
templates = Jinja2Templates(directory=_WEB_DIR / "templates")

router = APIRouter(prefix="/wiki", tags=["wiki"])

_km_instance: KnowledgeManager | None = None

def _get_km() -> KnowledgeManager:
    """Get or create KnowledgeManager singleton.

    Lazy initialization: creates DB connection on first call.
    """
    global _km_instance
    if _km_instance is None:
        from code.shukketsu.db.connection import get_connection, init_db

        conn = get_connection()
        init_db(conn)
        _km_instance = KnowledgeManager(conn, config.WIKI_PATH)
    return _km_instance
```

**Note**: The wiki router and chat router may end up with separate `sqlite3.Connection` objects.
This is safe because SQLite WAL mode supports concurrent readers. If we later want a single
shared connection, we can extract a shared `get_km()` helper — but for now, separate is simpler
and avoids coupling the two routers.

### GET `/wiki/` — Article Browser

Full page render. Loads the browser template with initial article list. Query parameters for
initial state: `spec`, `status`, `q` (search term). Defaults to showing all articles.

```python
@router.get("/", response_class=HTMLResponse)
async def wiki_browser(
    request: Request,
    spec: str | None = None,
    status: str | None = None,
    q: str | None = None,
    km: KnowledgeManager = Depends(_get_km),
) -> HTMLResponse:
```

### GET `/wiki/review` — Review Queue

Full page render. Shows only articles with `status=review`. Includes Editor verification
data for each article (claim counts, overall confidence).

```python
@router.get("/review", response_class=HTMLResponse)
async def wiki_review(
    request: Request,
    km: KnowledgeManager = Depends(_get_km),
) -> HTMLResponse:
```

### GET `/wiki/view/{path:path}` — Article Detail

Reads the article via `KnowledgeManager.read_article()`, renders Markdown server-side, and
passes both rendered HTML and parsed frontmatter to the template.

```python
@router.get("/view/{path:path}", response_class=HTMLResponse)
async def wiki_article(
    request: Request,
    path: str,
    km: KnowledgeManager = Depends(_get_km),
) -> HTMLResponse:
```

Returns 404 if the article doesn't exist. The `path` parameter captures the full article path
(e.g., `combat/gear/phase-1-bis`). The `.md` extension is added internally if not present.

### POST `/wiki/approve/{article_id}` — Approve Article

Looks up the article path from the integer ID via `KnowledgeManager.get_path_by_id()`, then
transitions from `review` → `published` via `KnowledgeManager.set_status()`. Returns an HTMX
fragment.

```python
@router.post("/approve/{article_id}", response_class=HTMLResponse)
async def wiki_approve(
    request: Request,
    article_id: int,
    km: KnowledgeManager = Depends(_get_km),
) -> HTMLResponse:
```

Flow:
1. `path = km.get_path_by_id(article_id)` — raises 404 if not found
2. `km.set_status(path, ArticleStatus.PUBLISHED)` — raises 400 if wrong status
3. Returns an empty div with a brief "Published" flash message (the card disappears
   from the review list via `hx-swap="outerHTML"`)

### POST `/wiki/reject/{article_id}` — Reject Article

Looks up path from ID, then transitions from `review` → `draft` via
`KnowledgeManager.reject_article()`. Accepts an optional rejection reason via form data.

```python
@router.post("/reject/{article_id}", response_class=HTMLResponse)
async def wiki_reject(
    request: Request,
    article_id: int,
    reason: str = Form(default=""),
    km: KnowledgeManager = Depends(_get_km),
) -> HTMLResponse:
```

### GET `/wiki/htmx/articles` — Article List Fragment

Returns only the article list HTML (no base template). Used by HTMX for dynamic filtering
and search.

```python
@router.get("/htmx/articles", response_class=HTMLResponse)
async def wiki_htmx_articles(
    request: Request,
    spec: str | None = None,
    status: str | None = None,
    q: str | None = None,
    km: KnowledgeManager = Depends(_get_km),
) -> HTMLResponse:
```

Query filtering logic:
1. Start with `KnowledgeManager.list_articles(status=status, spec=spec)`
2. If `q` is provided, filter results where `q.lower()` is a substring of the article title
3. Return rendered `partials/article_list.html` fragment

### GET `/wiki/htmx/review` — Review List Fragment

Same pattern as article list but for the review page. Returns review cards with action buttons.

### Error Handling

- Article not found → 404 with a styled "Article not found" page
- Invalid status transition (e.g., approve a draft) → 400 with error message
- KnowledgeManager errors → 500 with generic error (logged server-side)

## Component 3: Templates

### `base.html` Changes

Three changes:

1. **Add HTMX and wiki CSS** in `<head>`:
```html
<script src="https://unpkg.com/htmx.org@2.0.4"></script>
<link rel="stylesheet" href="/static/css/wiki.css">
```

2. **Add navigation links** in the nav bar (between the logo and the right side):
```html
<nav class="flex items-center gap-6">
    <a href="/chat" class="...">Chat</a>
    <a href="/wiki/" class="...">Wiki</a>
</nav>
```
Active link highlighted based on current path (Jinja2 `request.url.path.startswith()` check).

3. **Move connection status indicator** out of `base.html` into `chat.html`. The `#connection-status`
div (with `#status-dot` and `#status-text`) is only relevant on the chat page where the WebSocket
exists. On wiki pages, it would permanently show "Connecting..." since there's no WebSocket. Move
it into a `{% block nav_right %}` block that `chat.html` overrides:

```html
<!-- base.html -->
<nav class="...">
    <div class="flex items-center gap-3"><!-- logo --></div>
    <div class="flex items-center gap-6"><!-- Chat | Wiki links --></div>
    {% block nav_right %}{% endblock %}
</nav>

<!-- chat.html -->
{% block nav_right %}
<div id="connection-status" class="flex items-center gap-2 text-sm">
    <span id="status-dot" class="w-2 h-2 rounded-full bg-gray-500"></span>
    <span id="status-text" class="text-parchment-dim">Connecting...</span>
</div>
{% endblock %}
```

This preserves the chat page behavior exactly while keeping wiki pages clean.

### `wiki/browser.html`

Extends `base.html`. Structure:

```
┌─────────────────────────────────────────┐
│  Wiki Articles                          │
│  [Search...____] [Spec ▼] [Status ▼]   │
├─────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ Article │ │ Article │ │ Article │  │
│  │  Card   │ │  Card   │ │  Card   │  │
│  └─────────┘ └─────────┘ └─────────┘  │
│  ...                                    │
└─────────────────────────────────────────┘
```

- Search input: `hx-get="/wiki/htmx/articles" hx-trigger="keyup changed delay:300ms" hx-target="#article-list" hx-include="[name='spec'],[name='status']"`
- Spec dropdown: `<select name="spec">` with options for combat/assassination/subtlety/general/all
- Status dropdown: `<select name="status">` with draft/review/published/all
- Both dropdowns: `hx-get="/wiki/htmx/articles" hx-trigger="change" hx-target="#article-list" hx-include="[name='q'],[name='spec'],[name='status']"`
- Empty state: "No articles yet. Articles are created by the Writer agent during research."

### `wiki/review.html`

Extends `base.html`. Similar to browser but:
- No status filter (always review)
- Title: "Articles Pending Review"
- Each card shows claim verification summary + approve/reject buttons
- Empty state: "No articles pending review."

### `wiki/article.html`

Extends `base.html`. Two-column layout:

```
┌──────────────────────────┬─────────────┐
│                          │  Status     │
│  # Article Title         │  ● Review   │
│                          │             │
│  ## Section 1            │  Confidence │
│  Article content...      │  ████░░ 72% │
│                          │             │
│  ## Section 2            │  Updated    │
│  More content...         │  Feb 11     │
│                          │             │
│  | Table | Data |        │  Claims     │
│  |-------|------|        │  4✓ 1? 0✗   │
│                          │             │
│                          │  Sources    │
│                          │  • url (0.9)│
│                          │             │
│                          │  Tags       │
│                          │  [combat]   │
│                          │  [gear]     │
└──────────────────────────┴─────────────┘
```

- Content area: `{{ article_html | safe }}` inside a `.markdown-body` div (reuses existing
  markdown styles from `theme.css`)
- Sidebar: Tailwind-styled metadata cards
- Confidence bar: CSS gradient with color based on value (green/yellow/red)
- Responsive: on small screens, sidebar moves below content (`md:flex-row` / `flex-col`)

### `wiki/partials/article_list.html`

Fragment (no base template). Renders a grid of article cards:

```html
{% for article in articles %}
<a href="/wiki/view/{{ article.path }}" class="article-card ...">
    <div class="flex justify-between items-start">
        <h3>{{ article.title }}</h3>
        <span class="status-badge status-{{ article.status }}">{{ article.status }}</span>
    </div>
    <div class="flex gap-2 mt-2">
        <span class="text-sm">{{ article.spec }}</span>
        <span class="text-sm">{{ article.category }}</span>
    </div>
    <div class="confidence-bar mt-2" style="--confidence: {{ article.confidence }}"></div>
</a>
{% else %}
<div class="text-center text-parchment-400 py-12">
    No articles found.
</div>
{% endfor %}
```

### `wiki/partials/review_list.html`

Similar to article_list but each card includes:
- Claim summary: "X verified, Y uncertain, Z contradicted"
- Overall confidence with colored bar
- Approve button: `hx-post="/wiki/approve/{{ article.id }}" hx-target="closest .review-card" hx-swap="outerHTML"`
- Reject button: opens a small inline form with reason text input
- Reject form: `hx-post="/wiki/reject/{{ article.id }}" hx-target="closest .review-card" hx-swap="outerHTML"`

## Component 4: Wiki Styles (`web/static/css/wiki.css`)

Minimal custom CSS for elements that are awkward with pure Tailwind:

```css
/* Confidence bar — uses CSS custom property for width */
.confidence-bar {
    height: 6px;
    border-radius: 3px;
    background: var(--shadow-700);
    overflow: hidden;
}
.confidence-bar::after {
    content: '';
    display: block;
    height: 100%;
    width: calc(var(--confidence) * 100%);
    border-radius: 3px;
    background: linear-gradient(
        90deg,
        #ef4444 0%,      /* red at 0% */
        #eab308 50%,     /* yellow at 50% */
        #22c55e 100%     /* green at 100% */
    );
    background-size: 200% 100%;
    background-position: calc((1 - var(--confidence)) * 100%) 0;
}

/* Status badges */
.status-badge { /* base styles */ }
.status-draft { color: #94a3b8; border-color: #475569; }
.status-review { color: #eab308; border-color: #854d0e; }
.status-published { color: #22c55e; border-color: #166534; }

/* Article card hover */
.article-card {
    transition: border-color 0.15s ease;
}
.article-card:hover {
    border-color: var(--wow-gold);
}

/* Review action buttons */
.btn-approve { /* green accent */ }
.btn-reject { /* red accent */ }

/* HTMX loading indicator */
.htmx-indicator {
    opacity: 0;
    transition: opacity 0.2s ease;
}
.htmx-request .htmx-indicator {
    opacity: 1;
}
```

## Component 5: KnowledgeManager Changes (`knowledge/manager.py`)

Four changes to the existing KnowledgeManager:

### 5a. Add `id` and claim counts to `ArticleSummary`

The DB `articles` table has `id`, `verified_claims`, and `unverified_claims` columns, but
`ArticleSummary` doesn't include them. The wiki UI needs `id` for approve/reject routes and
claim counts for the review page.

```python
class ArticleSummary(BaseModel):
    id: int                        # NEW: needed for approve/reject route URLs
    path: str
    title: str
    spec: str
    category: str
    status: ArticleStatus
    confidence_score: float
    verified_claims: int = 0       # NEW: for review page claim summary
    unverified_claims: int = 0     # NEW: for review page claim summary
    last_updated: str
```

Update `list_articles()` SELECT to include `id, verified_claims, unverified_claims`.

### 5b. Add `get_path_by_id()` helper

```python
def get_path_by_id(self, article_id: int) -> str:
    """Look up article path by integer ID. Raises ValueError if not found."""
    row = self._conn.execute("SELECT path FROM articles WHERE id = ?", (article_id,)).fetchone()
    if row is None:
        raise ValueError(f"Article not found: id={article_id}")
    return row["path"]
```

### 5c. Add `rejection_reason` to `ArticleMeta`

```python
class ArticleMeta(BaseModel):
    ...
    rejection_reason: str | None = None  # Set when article is rejected from review
```

This field is omitted from frontmatter when `None` (using `exclude_none=True` during YAML
serialization).

### 5d. Add `reject_article()` method

New method for the REVIEW → DRAFT backward transition:

```python
def reject_article(self, path: str, reason: str = "") -> None:
    """Reject an article in review, transitioning back to draft.

    Args:
        path: Relative article path.
        reason: Optional rejection reason stored in frontmatter.

    Raises:
        ValueError: If article is not in REVIEW status.
    """
```

Implementation:
1. Read current article via `read_article(path)`
2. Verify status is `REVIEW` — raise `ValueError` if not
3. Update frontmatter: set `status: draft`, add `rejection_reason: reason` (if non-empty),
   clear `rejection_reason` if reason is empty
4. Write updated Markdown file to disk
5. Update the DB row: `UPDATE articles SET status = 'draft', last_updated = ? WHERE path = ?`

**Note**: Uses `ValueError` (not `ShukketsuError`) to match the existing `set_status()` pattern.
The `rejection_reason` field is stored only in YAML frontmatter, not in the DB. It's
informational for the Writer agent if the article is re-drafted.

## Component 6: App Wiring (`web/app.py`)

Mount the wiki router alongside the existing chat router:

```python
from code.shukketsu.web.routers import wiki

app.include_router(wiki.router)
```

The wiki router is self-contained with its own prefix (`/wiki`). No other changes to `app.py`
beyond the import and mount.

## Risks and Unknowns

| Risk | Impact | Mitigation |
|------|--------|------------|
| `{path:path}` catch-all conflicts with `/review` route | Wrong route matched | `/wiki/view/` prefix separates article paths from other wiki routes |
| No articles exist yet | Empty UI on first visit | Graceful empty states in both browser and review templates |
| HTMX CDN unavailable | Filters and actions broken | Fallback: forms still submit normally (progressive enhancement). Low risk for local deployment. |
| `markdown` library rendering differs from `marked.js` in chat | Visual inconsistency | Both use `.markdown-body` CSS class from `theme.css`. Minor differences acceptable. |
| Large article count slows LIKE query | Slow search | Not a concern for Phase 2 (< 100 articles expected). Add FTS5 on articles later if needed. |
| Review page has no claim-by-claim data from DB | Can't show verification details | Read article frontmatter at render time — claims/verification stored in YAML. Slightly slower but correct. |
| Rejection reason not in DB | Can't query rejected articles by reason | Acceptable — rejection reason is informational, not a filter dimension. Stored in frontmatter for Writer reference. |

## What This Does NOT Include

- **Article editing in the browser** — No in-browser Markdown editor. Articles are edited by the Writer agent. Human editing is a Phase 4+ feature.
- **Entity tag links** — Entity pills are displayed but not clickable/linked to graph search. Wiring that needs a graph search UI (future).
- **Full-text search on article content** — Title-only LIKE search. FTS5 on articles deferred until article count warrants it.
- **Article versioning/history** — No diff view or revision history. Git tracks file history but there's no UI for it.
- **Comments/annotations** — No inline commenting on articles. Review is binary approve/reject.
- **Pagination** — Article list loads all articles at once. Pagination deferred until article count warrants it.
- **Mobile-optimized review** — Responsive layout works on mobile but approve/reject UX isn't touch-optimized.
- **Real-time updates** — Review page doesn't auto-refresh when new articles enter review. Manual refresh or HTMX polling could be added later.

## Dependencies

- **Step 5** (`knowledge/manager.py`): `KnowledgeManager`, `ArticleMeta`, `ArticleSummary`, `ArticleStatus`, `Spec`
- **Step 5** (`db/schema.sql`): `articles` table with status, confidence_score columns
- **Phase 1** (`web/app.py`): FastAPI app, Jinja2 templates, static file mounting
- **Phase 1** (`web/templates/base.html`): Base template with Tailwind CSS
- **Phase 1** (`web/static/css/theme.css`): `.markdown-body` styles for rendered content
- **External**: `markdown` Python package (already in `requirements.txt` as `markdown>=3.7`)

## Test Plan

All tests in `tests/unit/`. Tests use FastAPI `TestClient` with mocked `KnowledgeManager`.

### `tests/unit/test_wiki_render.py` (~4 tests)

- `test_render_basic_markdown` — Headings, paragraphs, bold, italic
- `test_render_tables` — Markdown tables become HTML `<table>` elements
- `test_render_fenced_code` — Fenced code blocks render as `<pre><code>`
- `test_render_empty_string` — Empty input returns empty string

### `tests/unit/test_wiki_routes.py` (~15 tests)

- `test_browser_returns_200` — GET /wiki/ renders successfully
- `test_browser_empty_state` — No articles shows empty message
- `test_browser_lists_articles` — Articles appear in response
- `test_review_returns_200` — GET /wiki/review renders successfully
- `test_review_empty_state` — No review articles shows empty message
- `test_article_returns_200` — GET /wiki/view/{path} renders article
- `test_article_not_found_returns_404` — Nonexistent path returns 404
- `test_article_renders_markdown` — Article content is HTML, not raw Markdown
- `test_article_shows_sidebar_data` — Confidence, status, sources in response
- `test_approve_transitions_to_published` — POST approve changes status
- `test_approve_non_review_returns_400` — Can't approve a draft
- `test_reject_transitions_to_draft` — POST reject changes status
- `test_reject_stores_reason` — Rejection reason passed to KnowledgeManager
- `test_htmx_articles_returns_fragment` — No base template wrapper
- `test_htmx_articles_filters_by_spec` — Spec parameter filters results

### `tests/unit/test_knowledge_manager_wiki.py` (~6 tests)

- `test_reject_review_article` — REVIEW → DRAFT succeeds
- `test_reject_non_review_raises` — Rejecting a DRAFT or PUBLISHED raises error
- `test_reject_stores_reason_in_frontmatter` — Reason appears in ArticleMeta after rejection
- `test_get_path_by_id_found` — Returns correct path for existing article
- `test_get_path_by_id_not_found` — Raises ValueError for nonexistent ID
- `test_list_articles_includes_id_and_claims` — ArticleSummary has id, verified_claims, unverified_claims

**Starting count: 620 tests → Estimated after this step: ~645 tests**
