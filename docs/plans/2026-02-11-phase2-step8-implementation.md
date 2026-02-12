# Phase 2 Step 8: Wiki UI — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a web interface for browsing, reading, and managing wiki articles with HTMX-powered filtering, search, and approve/reject review workflow.

**Architecture:** Server-side Markdown rendering via Python `markdown` library. HTMX fragment endpoints for dynamic interactions. FastAPI router with Jinja2 templates extending the existing Tailwind dark theme. KnowledgeManager handles all data; wiki routes are a thin HTTP layer.

**Tech Stack:** FastAPI, Jinja2, HTMX 2.0, Tailwind CSS (CDN), Python `markdown` library, SQLite

---

## Task 1: KnowledgeManager — Add `id` and claim counts to `ArticleSummary`

**Files:**
- Modify: `code/shukketsu/knowledge/manager.py:53-63` (ArticleSummary model)
- Modify: `code/shukketsu/knowledge/manager.py:192-226` (list_articles query)
- Test: `tests/unit/test_knowledge_manager_wiki.py` (new)

**Step 1: Write the failing test**

Create `tests/unit/test_knowledge_manager_wiki.py`:

```python
"""Tests for KnowledgeManager wiki UI extensions."""

import sqlite3
from datetime import UTC, datetime

import pytest

from code.shukketsu.knowledge.manager import (
    ArticleMeta,
    ArticleStatus,
    ClaimRef,
    KnowledgeManager,
    SourceRef,
    Spec,
)


@pytest.fixture
def km(test_db: sqlite3.Connection, tmp_path) -> KnowledgeManager:
    """KnowledgeManager with test DB and temp knowledge dir."""
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    return KnowledgeManager(test_db, knowledge_dir)


def _sample_meta(**overrides) -> ArticleMeta:
    """Create a sample ArticleMeta for testing."""
    now = datetime.now(tz=UTC)
    defaults = dict(
        title="Phase 1 Trinkets",
        spec=Spec.COMBAT,
        category="gear",
        status=ArticleStatus.DRAFT,
        confidence=0.85,
        created_at=now,
        updated_at=now,
        sources=[SourceRef(url="https://example.com", trust=0.7)],
        claims=[ClaimRef(text="DST is BiS", verified=False, evidence=["chunk:42"])],
        entity_refs=["Dragonspine Trophy"],
        tags=["combat", "trinkets"],
    )
    defaults.update(overrides)
    return ArticleMeta(**defaults)


class TestArticleSummaryFields:
    def test_list_articles_includes_id(
        self, km: KnowledgeManager, test_db: sqlite3.Connection
    ) -> None:
        """ArticleSummary should include the integer id from the DB."""
        km.create_draft(_sample_meta(), "Content")
        articles = km.list_articles()
        assert len(articles) == 1
        assert isinstance(articles[0].id, int)
        assert articles[0].id > 0

    def test_list_articles_includes_claim_counts(
        self, km: KnowledgeManager, test_db: sqlite3.Connection
    ) -> None:
        """ArticleSummary should include verified_claims and unverified_claims."""
        path = km.create_draft(_sample_meta(), "Content")
        # Set claim counts directly in DB (Editor sets these during verification)
        test_db.execute(
            "UPDATE articles SET verified_claims = 3, unverified_claims = 1 WHERE path = ?",
            (path,),
        )
        test_db.commit()
        articles = km.list_articles()
        assert articles[0].verified_claims == 3
        assert articles[0].unverified_claims == 1

    def test_list_articles_claim_counts_default_zero(
        self, km: KnowledgeManager
    ) -> None:
        """Claim counts should default to 0 for new articles."""
        km.create_draft(_sample_meta(), "Content")
        articles = km.list_articles()
        assert articles[0].verified_claims == 0
        assert articles[0].unverified_claims == 0
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_knowledge_manager_wiki.py::TestArticleSummaryFields -v`
Expected: FAIL — `ArticleSummary` has no `id` field

**Step 3: Write minimal implementation**

In `code/shukketsu/knowledge/manager.py`, update `ArticleSummary`:

```python
class ArticleSummary(BaseModel):
    """Lightweight article record from DB queries."""

    id: int
    path: str
    title: str
    spec: str
    category: str
    status: ArticleStatus
    confidence_score: float
    verified_claims: int = 0
    unverified_claims: int = 0
    last_updated: str
```

Update `list_articles()` SELECT and row mapping:

```python
query = "SELECT id, path, title, spec, category, status, confidence_score, verified_claims, unverified_claims, last_updated FROM articles"
```

And in the row-mapping list comprehension, add the three new fields:

```python
ArticleSummary(
    id=row["id"],
    path=row["path"],
    ...
    verified_claims=row["verified_claims"],
    unverified_claims=row["unverified_claims"],
    last_updated=row["last_updated"],
)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_knowledge_manager_wiki.py::TestArticleSummaryFields -v`
Expected: PASS (3 tests)

**Step 5: Run full test suite to check for regressions**

Run: `python3 -m pytest tests/unit/ -v`
Expected: All 620 existing tests still pass. The `ArticleSummary` change affects `TestListArticles` in `test_knowledge_manager.py` — those tests don't check `id` but should still work since the DB rows have `id` by default.

**Step 6: Commit**

```bash
git add code/shukketsu/knowledge/manager.py tests/unit/test_knowledge_manager_wiki.py
git commit -m "feat(knowledge): add id and claim counts to ArticleSummary"
```

---

## Task 2: KnowledgeManager — Add `get_path_by_id()` and `rejection_reason`

**Files:**
- Modify: `code/shukketsu/knowledge/manager.py:65-78` (ArticleMeta model)
- Modify: `code/shukketsu/knowledge/manager.py` (new methods at end)
- Test: `tests/unit/test_knowledge_manager_wiki.py` (append)

**Step 1: Write the failing tests**

Append to `tests/unit/test_knowledge_manager_wiki.py`:

```python
class TestGetPathById:
    def test_found(self, km: KnowledgeManager, test_db: sqlite3.Connection) -> None:
        path = km.create_draft(_sample_meta(), "Content")
        row = test_db.execute("SELECT id FROM articles WHERE path = ?", (path,)).fetchone()
        assert km.get_path_by_id(row["id"]) == path

    def test_not_found(self, km: KnowledgeManager) -> None:
        with pytest.raises(ValueError, match="not found"):
            km.get_path_by_id(99999)


class TestRejectArticle:
    def test_review_to_draft(
        self, km: KnowledgeManager, test_db: sqlite3.Connection
    ) -> None:
        path = km.create_draft(_sample_meta(), "Content")
        km.set_status(path, ArticleStatus.REVIEW)
        km.reject_article(path, reason="Needs more sources")
        row = test_db.execute("SELECT status FROM articles WHERE path = ?", (path,)).fetchone()
        assert row["status"] == "draft"

    def test_rejection_reason_in_frontmatter(self, km: KnowledgeManager) -> None:
        path = km.create_draft(_sample_meta(), "Content")
        km.set_status(path, ArticleStatus.REVIEW)
        km.reject_article(path, reason="Inaccurate hit cap value")
        meta, _ = km.read_article(path)
        assert meta.rejection_reason == "Inaccurate hit cap value"

    def test_reject_draft_raises(self, km: KnowledgeManager) -> None:
        path = km.create_draft(_sample_meta(), "Content")
        with pytest.raises(ValueError, match="not in review"):
            km.reject_article(path, reason="bad")

    def test_reject_published_raises(
        self, km: KnowledgeManager, test_db: sqlite3.Connection
    ) -> None:
        path = km.create_draft(_sample_meta(), "Content")
        km.set_status(path, ArticleStatus.REVIEW)
        km.set_status(path, ArticleStatus.PUBLISHED)
        with pytest.raises(ValueError, match="not in review"):
            km.reject_article(path, reason="bad")

    def test_reject_empty_reason(self, km: KnowledgeManager) -> None:
        """Empty reason should not add rejection_reason to frontmatter."""
        path = km.create_draft(_sample_meta(), "Content")
        km.set_status(path, ArticleStatus.REVIEW)
        km.reject_article(path)
        meta, _ = km.read_article(path)
        assert meta.rejection_reason is None
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_knowledge_manager_wiki.py -v`
Expected: FAIL — `get_path_by_id` and `reject_article` don't exist, `ArticleMeta` has no `rejection_reason`

**Step 3: Implement**

Add `rejection_reason` to `ArticleMeta` (after `tags` field):

```python
class ArticleMeta(BaseModel):
    """Pydantic model for article YAML frontmatter."""

    title: str
    spec: Spec
    category: str
    status: ArticleStatus = ArticleStatus.DRAFT
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    created_at: datetime
    updated_at: datetime
    sources: list[SourceRef] = Field(default_factory=list)
    claims: list[ClaimRef] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    rejection_reason: str | None = None
```

Add two new methods to the end of `KnowledgeManager`:

```python
def get_path_by_id(self, article_id: int) -> str:
    """Look up article path by integer ID. Raises ValueError if not found."""
    row = self._conn.execute(
        "SELECT path FROM articles WHERE id = ?", (article_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Article not found: id={article_id}")
    return row["path"]

def reject_article(self, path: str, reason: str = "") -> None:
    """Reject an article in review, transitioning back to draft.

    Args:
        path: Relative article path.
        reason: Optional rejection reason stored in frontmatter.

    Raises:
        ValueError: If article is not in REVIEW status.
    """
    meta, content = self.read_article(path)
    if meta.status != ArticleStatus.REVIEW:
        raise ValueError(
            f"Cannot reject article with status '{meta.status}': not in review"
        )

    updated = meta.model_copy(
        update={
            "status": ArticleStatus.DRAFT,
            "rejection_reason": reason if reason else None,
            "updated_at": datetime.now(tz=meta.updated_at.tzinfo),
        }
    )

    # Write file with updated frontmatter
    full_path = self._knowledge_dir / path
    post = frontmatter.Post(content, **updated.model_dump(mode="json", exclude_none=True))
    full_path.write_text(frontmatter.dumps(post), encoding="utf-8")

    # Update DB
    self._conn.execute(
        "UPDATE articles SET status = 'draft', last_updated = ? WHERE path = ?",
        (updated.updated_at.isoformat(), path),
    )
    self._conn.commit()
    logger.info("Rejected article %s: %s", path, reason or "(no reason)")
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_knowledge_manager_wiki.py -v`
Expected: PASS (all 8 tests)

**Step 5: Run full suite**

Run: `python3 -m pytest tests/unit/ -v`
Expected: All passing. The `rejection_reason: str | None = None` default means existing frontmatter without this field still parses correctly.

**Step 6: Commit**

```bash
git add code/shukketsu/knowledge/manager.py tests/unit/test_knowledge_manager_wiki.py
git commit -m "feat(knowledge): add get_path_by_id, reject_article, rejection_reason"
```

---

## Task 3: Markdown Renderer

**Files:**
- Create: `code/shukketsu/web/wiki_render.py`
- Test: `tests/unit/test_wiki_render.py` (new)

**Step 1: Write the failing tests**

Create `tests/unit/test_wiki_render.py`:

```python
"""Tests for server-side Markdown rendering."""

from code.shukketsu.web.wiki_render import render_markdown


class TestRenderMarkdown:
    def test_headings(self) -> None:
        html = render_markdown("# Title\n\n## Subtitle")
        assert "<h1>" in html
        assert "<h2>" in html
        assert "Title" in html

    def test_bold_and_italic(self) -> None:
        html = render_markdown("**bold** and *italic*")
        assert "<strong>bold</strong>" in html
        assert "<em>italic</em>" in html

    def test_tables(self) -> None:
        md = "| Head |\n|------|\n| Cell |"
        html = render_markdown(md)
        assert "<table>" in html
        assert "<th>" in html
        assert "Cell" in html

    def test_fenced_code(self) -> None:
        md = "```python\nprint('hello')\n```"
        html = render_markdown(md)
        assert "<pre>" in html
        assert "<code>" in html
        assert "print" in html

    def test_empty_string(self) -> None:
        assert render_markdown("") == ""

    def test_multiple_calls_independent(self) -> None:
        """State shouldn't leak between calls (reset works)."""
        html1 = render_markdown("# First")
        html2 = render_markdown("# Second")
        assert "First" in html1
        assert "Second" in html2
        assert "First" not in html2
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_wiki_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'code.shukketsu.web.wiki_render'`

**Step 3: Implement**

Create `code/shukketsu/web/wiki_render.py`:

```python
"""Server-side Markdown rendering for wiki articles."""

import markdown

_MD = markdown.Markdown(extensions=["tables", "fenced_code", "toc"])


def render_markdown(text: str) -> str:
    """Render Markdown text to HTML.

    Uses tables, fenced_code, and toc extensions. Resets internal state
    between calls to prevent leakage.
    """
    if not text:
        return ""
    _MD.reset()
    return _MD.convert(text)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_wiki_render.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add code/shukketsu/web/wiki_render.py tests/unit/test_wiki_render.py
git commit -m "feat(web): add server-side Markdown renderer for wiki"
```

---

## Task 4: Base Template Changes — Nav Links, HTMX, Connection Status Move

**Files:**
- Modify: `code/shukketsu/web/templates/base.html`
- Modify: `code/shukketsu/web/templates/chat.html`

**Step 1: Update `base.html`**

Replace the full file content. Key changes:
1. Add HTMX CDN script and wiki.css link in `<head>`
2. Add Chat/Wiki nav links with active-state highlighting
3. Move connection status into `{% block nav_right %}` block

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Shukketsu{% endblock %}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
    tailwind.config = {
        theme: {
            extend: {
                colors: {
                    'wow-gold': '#f5c518',
                    'rogue-energy': '#fff468',
                    'shadow': {
                        DEFAULT: '#1a1a2e',
                        deep: '#0f0f1a',
                        mid: '#16213e',
                    },
                    'parchment': {
                        DEFAULT: '#e8dcc4',
                        dim: '#a89b8c',
                    },
                },
            },
        },
    }
    </script>
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
    <link rel="stylesheet" href="/static/css/theme.css">
    <link rel="stylesheet" href="/static/css/wiki.css">
    {% block head %}{% endblock %}
</head>
<body class="bg-shadow-deep text-parchment min-h-screen flex flex-col">
    <nav class="bg-shadow border-b border-white/10 px-6 py-3 flex items-center justify-between shrink-0">
        <div class="flex items-center gap-6">
            <div class="flex items-center gap-3">
                <span class="text-wow-gold font-bold text-xl tracking-wide">Shukketsu</span>
                <span class="text-parchment-dim text-sm">出血</span>
            </div>
            <div class="flex items-center gap-4 ml-4">
                <a href="/chat"
                   class="text-sm transition-colors {% if request.url.path.startswith('/chat') or request.url.path == '/' %}text-wow-gold{% else %}text-parchment-dim hover:text-parchment{% endif %}">
                    Chat
                </a>
                <a href="/wiki/"
                   class="text-sm transition-colors {% if request.url.path.startswith('/wiki') %}text-wow-gold{% else %}text-parchment-dim hover:text-parchment{% endif %}">
                    Wiki
                </a>
            </div>
        </div>
        {% block nav_right %}{% endblock %}
    </nav>
    <main class="flex-1 flex flex-col overflow-hidden">
        {% block content %}{% endblock %}
    </main>
</body>
</html>
```

**Step 2: Update `chat.html`**

Add the `{% block nav_right %}` override to bring back the connection status on the chat page. Insert after `{% block head %}...{% endblock %}` and before `{% block content %}`:

```html
{% extends "base.html" %}

{% block title %}Chat — Shukketsu{% endblock %}

{% block head %}
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify/dist/purify.min.js"></script>
{% endblock %}

{% block nav_right %}
<div id="connection-status" class="flex items-center gap-2 text-sm">
    <span id="status-dot" class="w-2 h-2 rounded-full bg-gray-500"></span>
    <span id="status-text" class="text-parchment-dim">Connecting...</span>
</div>
{% endblock %}

{% block content %}
<div class="flex-1 flex flex-col max-w-4xl w-full mx-auto overflow-hidden">
    <!-- Chat messages -->
    <div id="chat-messages" class="flex-1 overflow-y-auto p-6 space-y-4">
        <!-- Welcome state -->
        <div id="welcome-state" class="flex flex-col items-center justify-center h-full text-center select-none">
            <h1 class="text-wow-gold text-3xl font-bold mb-1">Shukketsu</h1>
            <p class="text-parchment-dim text-lg mb-8">TBC Rogue Research Assistant</p>
            <p class="text-parchment-dim mb-6">Ask me anything about the Rogue class in World of Warcraft: TBC</p>
            <div class="flex flex-col gap-3 w-full max-w-md">
                <button class="example-prompt px-4 py-3 bg-shadow border border-white/10 rounded-lg
                               hover:border-wow-gold/50 hover:text-wow-gold transition-colors text-left text-sm"
                        data-prompt="What's the hit cap for combat rogues?">
                    What's the hit cap for combat rogues?
                </button>
                <button class="example-prompt px-4 py-3 bg-shadow border border-white/10 rounded-lg
                               hover:border-wow-gold/50 hover:text-wow-gold transition-colors text-left text-sm"
                        data-prompt="Compare combat vs assassination for Karazhan">
                    Compare combat vs assassination for Karazhan
                </button>
                <button class="example-prompt px-4 py-3 bg-shadow border border-white/10 rounded-lg
                               hover:border-wow-gold/50 hover:text-wow-gold transition-colors text-left text-sm"
                        data-prompt="Explain the Seal Fate talent">
                    Explain the Seal Fate talent
                </button>
            </div>
        </div>
    </div>

    <!-- Input area -->
    <div class="border-t border-white/10 p-4 shrink-0">
        <div class="flex gap-3 items-end">
            <textarea id="chat-input"
                      class="flex-1 bg-shadow border border-white/10 rounded-lg px-4 py-3
                             text-parchment placeholder-parchment-dim/50 resize-none
                             focus:outline-none focus:border-wow-gold/50 transition-colors"
                      placeholder="Ask about TBC Rogues... (Shift+Enter for new line)"
                      rows="1"></textarea>
            <button id="send-btn"
                    class="px-5 py-3 bg-wow-gold/20 border border-wow-gold/50 rounded-lg
                           text-wow-gold hover:bg-wow-gold/30 transition-colors font-medium
                           disabled:opacity-30 disabled:cursor-not-allowed">
                Send
            </button>
            <button id="stop-btn"
                    class="hidden px-5 py-3 bg-red-500/20 border border-red-500/50 rounded-lg
                           text-red-400 hover:bg-red-500/30 transition-colors font-medium">
                Stop
            </button>
        </div>
    </div>
</div>

<script src="/static/js/chat.js"></script>
{% endblock %}
```

**Step 3: Run existing chat tests to verify no regression**

Run: `python3 -m pytest tests/unit/test_chat_handler.py -v`
Expected: All passing — connection status element IDs unchanged, just moved from base.html to chat.html via block override.

**Step 4: Commit**

```bash
git add code/shukketsu/web/templates/base.html code/shukketsu/web/templates/chat.html
git commit -m "feat(web): add nav links, HTMX, move connection status to chat"
```

---

## Task 5: Wiki CSS

**Files:**
- Create: `code/shukketsu/web/static/css/wiki.css`

**Step 1: Create wiki styles**

Create `code/shukketsu/web/static/css/wiki.css`:

```css
/* Wiki UI — custom styles for elements awkward with pure Tailwind */

/* Confidence bar — uses CSS custom property for width */
.confidence-bar {
    height: 6px;
    border-radius: 3px;
    background: rgba(255, 255, 255, 0.1);
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
        #ef4444 0%,
        #eab308 50%,
        #22c55e 100%
    );
    background-size: 200% 100%;
    background-position: calc((1 - var(--confidence)) * 100%) 0;
}

/* Status badges */
.status-badge {
    font-size: 0.75rem;
    padding: 0.125rem 0.5rem;
    border: 1px solid;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
}

.status-draft {
    color: #94a3b8;
    border-color: #475569;
}

.status-review {
    color: #eab308;
    border-color: #854d0e;
}

.status-published {
    color: #22c55e;
    border-color: #166534;
}

/* Article card hover */
.article-card {
    display: block;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 0.5rem;
    padding: 1rem;
    background: #1a1a2e;
    transition: border-color 0.15s ease;
    text-decoration: none;
    color: inherit;
}

.article-card:hover {
    border-color: #f5c518;
}

/* Review card */
.review-card {
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 0.5rem;
    padding: 1rem;
    background: #1a1a2e;
    transition: border-color 0.15s ease;
}

/* Action buttons */
.btn-approve {
    padding: 0.375rem 0.75rem;
    font-size: 0.875rem;
    font-weight: 500;
    border-radius: 0.375rem;
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.4);
    color: #22c55e;
    cursor: pointer;
    transition: background 0.15s ease;
}

.btn-approve:hover {
    background: rgba(34, 197, 94, 0.25);
}

.btn-reject {
    padding: 0.375rem 0.75rem;
    font-size: 0.875rem;
    font-weight: 500;
    border-radius: 0.375rem;
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #ef4444;
    cursor: pointer;
    transition: background 0.15s ease;
}

.btn-reject:hover {
    background: rgba(239, 68, 68, 0.25);
}

/* HTMX loading indicator */
.htmx-indicator {
    opacity: 0;
    transition: opacity 0.2s ease;
}

.htmx-request .htmx-indicator {
    opacity: 1;
}

/* Flash message (shown after approve/reject) */
.flash-message {
    padding: 0.75rem 1rem;
    border-radius: 0.375rem;
    font-size: 0.875rem;
    animation: flash-fade 3s ease forwards;
}

@keyframes flash-fade {
    0%, 70% { opacity: 1; }
    100% { opacity: 0; }
}

/* Sidebar metadata cards */
.sidebar-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 0.375rem;
    padding: 0.75rem;
}

/* Entity tag pills */
.entity-tag {
    display: inline-block;
    font-size: 0.75rem;
    padding: 0.125rem 0.5rem;
    background: rgba(245, 197, 24, 0.1);
    border: 1px solid rgba(245, 197, 24, 0.3);
    border-radius: 9999px;
    color: #f5c518;
}
```

**Step 2: Commit**

```bash
git add code/shukketsu/web/static/css/wiki.css
git commit -m "feat(web): add wiki CSS styles"
```

---

## Task 6: Wiki Templates — Partials + Full Pages

**Files:**
- Create: `code/shukketsu/web/templates/wiki/partials/article_list.html`
- Create: `code/shukketsu/web/templates/wiki/partials/review_list.html`
- Create: `code/shukketsu/web/templates/wiki/browser.html`
- Create: `code/shukketsu/web/templates/wiki/review.html`
- Create: `code/shukketsu/web/templates/wiki/article.html`

**Step 1: Create partials directory and article_list partial**

Create `code/shukketsu/web/templates/wiki/partials/article_list.html`:

```html
{% for article in articles %}
<a href="/wiki/view/{{ article.path | replace('.md', '') }}" class="article-card">
    <div class="flex justify-between items-start">
        <h3 class="text-parchment font-medium">{{ article.title }}</h3>
        <span class="status-badge status-{{ article.status }}">{{ article.status }}</span>
    </div>
    <div class="flex gap-3 mt-2 text-sm text-parchment-dim">
        <span>{{ article.spec }}</span>
        <span>&middot;</span>
        <span>{{ article.category }}</span>
    </div>
    <div class="confidence-bar mt-3" style="--confidence: {{ article.confidence_score }}"></div>
</a>
{% else %}
<div class="text-center text-parchment-dim py-12 col-span-full">
    No articles found.
</div>
{% endfor %}
```

**Step 2: Create review_list partial**

Create `code/shukketsu/web/templates/wiki/partials/review_list.html`:

```html
{% for article in articles %}
<div class="review-card" id="review-card-{{ article.id }}">
    <div class="flex justify-between items-start">
        <a href="/wiki/view/{{ article.path | replace('.md', '') }}" class="text-parchment font-medium hover:text-wow-gold transition-colors">
            {{ article.title }}
        </a>
        <span class="status-badge status-review">review</span>
    </div>
    <div class="flex gap-3 mt-2 text-sm text-parchment-dim">
        <span>{{ article.spec }}</span>
        <span>&middot;</span>
        <span>{{ article.category }}</span>
    </div>
    <div class="flex gap-4 mt-2 text-sm">
        <span class="text-green-400">{{ article.verified_claims }} verified</span>
        <span class="text-parchment-dim">{{ article.unverified_claims }} unverified</span>
    </div>
    <div class="confidence-bar mt-3" style="--confidence: {{ article.confidence_score }}"></div>

    <div class="flex gap-3 mt-4 items-center">
        <button class="btn-approve"
                hx-post="/wiki/approve/{{ article.id }}"
                hx-target="#review-card-{{ article.id }}"
                hx-swap="outerHTML">
            Approve
        </button>
        <button class="btn-reject"
                onclick="this.nextElementSibling.classList.toggle('hidden')">
            Reject
        </button>
        <form class="hidden flex gap-2 items-center flex-1"
              hx-post="/wiki/reject/{{ article.id }}"
              hx-target="#review-card-{{ article.id }}"
              hx-swap="outerHTML">
            <input type="text" name="reason" placeholder="Reason (optional)"
                   class="flex-1 bg-shadow-deep border border-white/10 rounded px-3 py-1.5
                          text-sm text-parchment placeholder-parchment-dim/50
                          focus:outline-none focus:border-wow-gold/50">
            <button type="submit" class="btn-reject text-xs">Confirm</button>
        </form>
    </div>
</div>
{% else %}
<div class="text-center text-parchment-dim py-12">
    No articles pending review.
</div>
{% endfor %}
```

**Step 3: Create browser.html**

Create `code/shukketsu/web/templates/wiki/browser.html`:

```html
{% extends "base.html" %}

{% block title %}Wiki — Shukketsu{% endblock %}

{% block content %}
<div class="flex-1 flex flex-col max-w-6xl w-full mx-auto p-6 overflow-y-auto">
    <h1 class="text-wow-gold text-2xl font-bold mb-6">Wiki Articles</h1>

    <!-- Filters -->
    <div class="flex flex-wrap gap-3 mb-6">
        <input type="text" name="q" placeholder="Search by title..."
               value="{{ q or '' }}"
               class="flex-1 min-w-[200px] bg-shadow border border-white/10 rounded-lg px-4 py-2
                      text-parchment placeholder-parchment-dim/50 text-sm
                      focus:outline-none focus:border-wow-gold/50 transition-colors"
               hx-get="/wiki/htmx/articles"
               hx-trigger="keyup changed delay:300ms"
               hx-target="#article-list"
               hx-include="[name='spec'],[name='status']">

        <select name="spec"
                class="bg-shadow border border-white/10 rounded-lg px-3 py-2 text-sm text-parchment
                       focus:outline-none focus:border-wow-gold/50"
                hx-get="/wiki/htmx/articles"
                hx-trigger="change"
                hx-target="#article-list"
                hx-include="[name='q'],[name='status']">
            <option value="">All Specs</option>
            <option value="combat" {% if spec == 'combat' %}selected{% endif %}>Combat</option>
            <option value="assassination" {% if spec == 'assassination' %}selected{% endif %}>Assassination</option>
            <option value="subtlety" {% if spec == 'subtlety' %}selected{% endif %}>Subtlety</option>
            <option value="general" {% if spec == 'general' %}selected{% endif %}>General</option>
        </select>

        <select name="status"
                class="bg-shadow border border-white/10 rounded-lg px-3 py-2 text-sm text-parchment
                       focus:outline-none focus:border-wow-gold/50"
                hx-get="/wiki/htmx/articles"
                hx-trigger="change"
                hx-target="#article-list"
                hx-include="[name='q'],[name='spec']">
            <option value="">All Statuses</option>
            <option value="draft" {% if status == 'draft' %}selected{% endif %}>Draft</option>
            <option value="review" {% if status == 'review' %}selected{% endif %}>Review</option>
            <option value="published" {% if status == 'published' %}selected{% endif %}>Published</option>
        </select>

        <a href="/wiki/review"
           class="bg-shadow border border-wow-gold/30 rounded-lg px-4 py-2 text-sm text-wow-gold
                  hover:bg-wow-gold/10 transition-colors">
            Review Queue
        </a>
    </div>

    <!-- Article grid -->
    <div id="article-list" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {% include "wiki/partials/article_list.html" %}
    </div>
</div>
{% endblock %}
```

**Step 4: Create review.html**

Create `code/shukketsu/web/templates/wiki/review.html`:

```html
{% extends "base.html" %}

{% block title %}Review Queue — Shukketsu{% endblock %}

{% block content %}
<div class="flex-1 flex flex-col max-w-4xl w-full mx-auto p-6 overflow-y-auto">
    <div class="flex items-center justify-between mb-6">
        <h1 class="text-wow-gold text-2xl font-bold">Articles Pending Review</h1>
        <a href="/wiki/"
           class="text-sm text-parchment-dim hover:text-parchment transition-colors">
            &larr; Back to Wiki
        </a>
    </div>

    <div id="review-list" class="flex flex-col gap-4">
        {% include "wiki/partials/review_list.html" %}
    </div>
</div>
{% endblock %}
```

**Step 5: Create article.html**

Create `code/shukketsu/web/templates/wiki/article.html`:

```html
{% extends "base.html" %}

{% block title %}{{ meta.title }} — Shukketsu Wiki{% endblock %}

{% block content %}
<div class="flex-1 flex flex-col md:flex-row max-w-6xl w-full mx-auto p-6 gap-6 overflow-y-auto">
    <!-- Article content -->
    <div class="flex-1 min-w-0">
        <a href="/wiki/" class="text-sm text-parchment-dim hover:text-parchment transition-colors mb-4 inline-block">
            &larr; Back to Wiki
        </a>
        <div class="markdown-body mt-2">
            {{ article_html | safe }}
        </div>
    </div>

    <!-- Metadata sidebar -->
    <div class="w-full md:w-72 shrink-0 flex flex-col gap-3">
        <!-- Status -->
        <div class="sidebar-card">
            <div class="text-xs text-parchment-dim uppercase tracking-wide mb-1">Status</div>
            <span class="status-badge status-{{ meta.status }}">{{ meta.status }}</span>
        </div>

        <!-- Confidence -->
        <div class="sidebar-card">
            <div class="text-xs text-parchment-dim uppercase tracking-wide mb-1">Confidence</div>
            <div class="flex items-center gap-2">
                <div class="confidence-bar flex-1" style="--confidence: {{ meta.confidence }}"></div>
                <span class="text-sm">{{ (meta.confidence * 100) | int }}%</span>
            </div>
        </div>

        <!-- Last Updated -->
        <div class="sidebar-card">
            <div class="text-xs text-parchment-dim uppercase tracking-wide mb-1">Updated</div>
            <span class="text-sm">{{ meta.updated_at.strftime('%b %d, %Y') if meta.updated_at else 'Unknown' }}</span>
        </div>

        <!-- Claim Summary -->
        {% set verified = meta.claims | selectattr('verified') | list | length %}
        {% set unverified = (meta.claims | length) - verified %}
        {% if meta.claims %}
        <div class="sidebar-card">
            <div class="text-xs text-parchment-dim uppercase tracking-wide mb-1">Claims</div>
            <div class="flex gap-3 text-sm">
                <span class="text-green-400">{{ verified }} verified</span>
                <span class="text-parchment-dim">{{ unverified }} unverified</span>
            </div>
        </div>
        {% endif %}

        <!-- Sources -->
        {% if meta.sources %}
        <div class="sidebar-card">
            <div class="text-xs text-parchment-dim uppercase tracking-wide mb-1">Sources</div>
            <ul class="text-sm space-y-1">
                {% for source in meta.sources %}
                <li class="flex justify-between">
                    <a href="{{ source.url }}" target="_blank" rel="noopener"
                       class="text-wow-gold hover:text-rogue-energy truncate">
                        {{ source.url | truncate(40) }}
                    </a>
                    <span class="text-parchment-dim ml-2 shrink-0">{{ source.trust }}</span>
                </li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        <!-- Entity Refs -->
        {% if meta.entity_refs %}
        <div class="sidebar-card">
            <div class="text-xs text-parchment-dim uppercase tracking-wide mb-1">Entities</div>
            <div class="flex flex-wrap gap-1.5">
                {% for entity in meta.entity_refs %}
                <span class="entity-tag">{{ entity }}</span>
                {% endfor %}
            </div>
        </div>
        {% endif %}

        <!-- Tags -->
        {% if meta.tags %}
        <div class="sidebar-card">
            <div class="text-xs text-parchment-dim uppercase tracking-wide mb-1">Tags</div>
            <div class="flex flex-wrap gap-1.5">
                {% for tag in meta.tags %}
                <span class="text-xs px-2 py-0.5 bg-white/5 border border-white/10 rounded text-parchment-dim">{{ tag }}</span>
                {% endfor %}
            </div>
        </div>
        {% endif %}
    </div>
</div>
{% endblock %}
```

**Step 6: Commit**

```bash
git add code/shukketsu/web/templates/wiki/
git commit -m "feat(web): add wiki templates (browser, review, article, partials)"
```

---

## Task 7: Wiki Router + App Wiring

**Files:**
- Create: `code/shukketsu/web/routers/wiki.py`
- Modify: `code/shukketsu/web/app.py:15-16` (add wiki router import and mount)
- Test: `tests/unit/test_wiki_routes.py` (new)

**Step 1: Write the failing tests**

Create `tests/unit/test_wiki_routes.py`:

```python
"""Tests for wiki routes."""

import sqlite3
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from code.shukketsu.knowledge.manager import (
    ArticleMeta,
    ArticleStatus,
    ClaimRef,
    KnowledgeManager,
    SourceRef,
    Spec,
)


def _get_app():
    from code.shukketsu.web.app import app

    return app


@pytest.fixture
def km(test_db: sqlite3.Connection, tmp_path) -> KnowledgeManager:
    """KnowledgeManager with test DB and temp knowledge dir."""
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    return KnowledgeManager(test_db, knowledge_dir)


def _sample_meta(**overrides) -> ArticleMeta:
    now = datetime.now(tz=UTC)
    defaults = dict(
        title="Phase 1 Trinkets",
        spec=Spec.COMBAT,
        category="gear",
        status=ArticleStatus.DRAFT,
        confidence=0.85,
        created_at=now,
        updated_at=now,
        sources=[SourceRef(url="https://example.com", trust=0.7)],
        claims=[ClaimRef(text="DST is BiS", verified=True, evidence=["chunk:42"])],
        entity_refs=["Dragonspine Trophy"],
        tags=["combat", "trinkets"],
    )
    defaults.update(overrides)
    return ArticleMeta(**defaults)


@pytest.fixture
def client(km: KnowledgeManager):
    """TestClient with KnowledgeManager dependency overridden."""
    from code.shukketsu.web.routers.wiki import _get_km

    app = _get_app()
    app.dependency_overrides[_get_km] = lambda: km
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestBrowser:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/wiki/")
        assert resp.status_code == 200
        assert "Wiki Articles" in resp.text

    def test_empty_state(self, client: TestClient) -> None:
        resp = client.get("/wiki/")
        assert "No articles found" in resp.text

    def test_lists_articles(self, client: TestClient, km: KnowledgeManager) -> None:
        km.create_draft(_sample_meta(), "# Test\n\nBody")
        resp = client.get("/wiki/")
        assert "Phase 1 Trinkets" in resp.text


class TestReview:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/wiki/review")
        assert resp.status_code == 200
        assert "Pending Review" in resp.text

    def test_empty_state(self, client: TestClient) -> None:
        resp = client.get("/wiki/review")
        assert "No articles pending review" in resp.text


class TestArticleDetail:
    def test_returns_200(self, client: TestClient, km: KnowledgeManager) -> None:
        path = km.create_draft(_sample_meta(), "# Trinkets\n\n**Bold text**")
        view_path = path.replace(".md", "")
        resp = client.get(f"/wiki/view/{view_path}")
        assert resp.status_code == 200

    def test_not_found_returns_404(self, client: TestClient) -> None:
        resp = client.get("/wiki/view/nonexistent/path")
        assert resp.status_code == 404

    def test_renders_markdown(self, client: TestClient, km: KnowledgeManager) -> None:
        path = km.create_draft(_sample_meta(), "# Title\n\n**Bold text**")
        view_path = path.replace(".md", "")
        resp = client.get(f"/wiki/view/{view_path}")
        assert "<strong>Bold text</strong>" in resp.text

    def test_shows_sidebar_data(self, client: TestClient, km: KnowledgeManager) -> None:
        path = km.create_draft(_sample_meta(), "# Test\n\nBody")
        view_path = path.replace(".md", "")
        resp = client.get(f"/wiki/view/{view_path}")
        assert "85%" in resp.text  # confidence
        assert "draft" in resp.text  # status
        assert "example.com" in resp.text  # source URL


class TestApprove:
    def test_approve_review_article(
        self, client: TestClient, km: KnowledgeManager, test_db: sqlite3.Connection
    ) -> None:
        path = km.create_draft(_sample_meta(), "Content")
        km.set_status(path, ArticleStatus.REVIEW)
        article_id = test_db.execute("SELECT id FROM articles WHERE path = ?", (path,)).fetchone()["id"]
        resp = client.post(f"/wiki/approve/{article_id}")
        assert resp.status_code == 200
        row = test_db.execute("SELECT status FROM articles WHERE path = ?", (path,)).fetchone()
        assert row["status"] == "published"

    def test_approve_draft_returns_400(
        self, client: TestClient, km: KnowledgeManager, test_db: sqlite3.Connection
    ) -> None:
        path = km.create_draft(_sample_meta(), "Content")
        article_id = test_db.execute("SELECT id FROM articles WHERE path = ?", (path,)).fetchone()["id"]
        resp = client.post(f"/wiki/approve/{article_id}")
        assert resp.status_code == 400


class TestReject:
    def test_reject_review_article(
        self, client: TestClient, km: KnowledgeManager, test_db: sqlite3.Connection
    ) -> None:
        path = km.create_draft(_sample_meta(), "Content")
        km.set_status(path, ArticleStatus.REVIEW)
        article_id = test_db.execute("SELECT id FROM articles WHERE path = ?", (path,)).fetchone()["id"]
        resp = client.post(f"/wiki/reject/{article_id}", data={"reason": "Needs work"})
        assert resp.status_code == 200
        row = test_db.execute("SELECT status FROM articles WHERE path = ?", (path,)).fetchone()
        assert row["status"] == "draft"

    def test_reject_stores_reason(
        self, client: TestClient, km: KnowledgeManager, test_db: sqlite3.Connection
    ) -> None:
        path = km.create_draft(_sample_meta(), "Content")
        km.set_status(path, ArticleStatus.REVIEW)
        article_id = test_db.execute("SELECT id FROM articles WHERE path = ?", (path,)).fetchone()["id"]
        client.post(f"/wiki/reject/{article_id}", data={"reason": "Bad sources"})
        meta, _ = km.read_article(path)
        assert meta.rejection_reason == "Bad sources"


class TestHtmxFragments:
    def test_articles_returns_fragment(self, client: TestClient) -> None:
        """Fragment should NOT contain base template elements like <html>."""
        resp = client.get("/wiki/htmx/articles")
        assert resp.status_code == 200
        assert "<html" not in resp.text
        assert "No articles found" in resp.text

    def test_articles_filters_by_spec(
        self, client: TestClient, km: KnowledgeManager
    ) -> None:
        km.create_draft(_sample_meta(title="Combat Guide", spec=Spec.COMBAT), "A")
        km.create_draft(_sample_meta(title="Mut Guide", spec=Spec.ASSASSINATION), "B")
        resp = client.get("/wiki/htmx/articles?spec=combat")
        assert "Combat Guide" in resp.text
        assert "Mut Guide" not in resp.text

    def test_articles_filters_by_title_search(
        self, client: TestClient, km: KnowledgeManager
    ) -> None:
        km.create_draft(_sample_meta(title="Phase 1 Trinkets"), "A")
        km.create_draft(_sample_meta(title="Rotation Guide"), "B")
        resp = client.get("/wiki/htmx/articles?q=trinket")
        assert "Phase 1 Trinkets" in resp.text
        assert "Rotation Guide" not in resp.text
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_wiki_routes.py -v`
Expected: FAIL — wiki router doesn't exist yet

**Step 3: Create wiki router**

Create `code/shukketsu/web/routers/wiki.py`:

```python
"""Wiki article browser, reader, and review routes."""

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


def _get_km() -> KnowledgeManager:
    """Get or create KnowledgeManager singleton."""
    global _km_instance  # noqa: PLW0603
    if _km_instance is None:
        from code.shukketsu.db.connection import get_connection, init_db

        conn = get_connection()
        init_db(conn)
        _km_instance = KnowledgeManager(conn, config.WIKI_PATH)
    return _km_instance


def _filter_articles(
    km: KnowledgeManager,
    spec: str | None = None,
    status: str | None = None,
    q: str | None = None,
) -> list:
    """Query and filter articles. Shared by full-page and HTMX routes."""
    status_enum = ArticleStatus(status) if status else None
    spec_enum = Spec(spec) if spec else None
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
        "wiki/browser.html",
        {"request": request, "articles": articles, "spec": spec, "status": status, "q": q},
    )


@router.get("/review", response_class=HTMLResponse)
async def wiki_review(
    request: Request,
    km: KnowledgeManager = Depends(_get_km),
) -> HTMLResponse:
    """Review queue — articles pending human approval."""
    articles = km.list_articles(status=ArticleStatus.REVIEW)
    return _templates.TemplateResponse(
        "wiki/review.html",
        {"request": request, "articles": articles},
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
        return HTMLResponse(
            _templates.TemplateResponse(
                "wiki/browser.html",
                {"request": request, "articles": [], "spec": None, "status": None, "q": None},
            ).body,
            status_code=404,
        )
    article_html = render_markdown(content)
    return _templates.TemplateResponse(
        "wiki/article.html",
        {"request": request, "meta": meta, "article_html": article_html},
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
            f'<div class="text-red-400 text-sm p-2">{exc}</div>',
            status_code=400,
        )
    return HTMLResponse(
        '<div class="flash-message text-green-400 bg-green-500/10 border border-green-500/30">'
        "Article published.</div>"
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
            f'<div class="text-red-400 text-sm p-2">{exc}</div>',
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
        "wiki/partials/article_list.html",
        {"request": request, "articles": articles},
    )


@router.get("/htmx/review", response_class=HTMLResponse)
async def wiki_htmx_review(
    request: Request,
    km: KnowledgeManager = Depends(_get_km),
) -> HTMLResponse:
    """Return review list fragment (no base template)."""
    articles = km.list_articles(status=ArticleStatus.REVIEW)
    return _templates.TemplateResponse(
        "wiki/partials/review_list.html",
        {"request": request, "articles": articles},
    )
```

**Step 4: Mount wiki router in `app.py`**

Add after the chat router import and mount:

```python
from code.shukketsu.web.routers.wiki import router as wiki_router
```

And after `app.include_router(chat_router)`:

```python
app.include_router(wiki_router)
```

**Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_wiki_routes.py -v`
Expected: PASS (15 tests)

**Step 6: Run full suite**

Run: `python3 -m pytest tests/unit/ -v`
Expected: All passing

**Step 7: Commit**

```bash
git add code/shukketsu/web/routers/wiki.py code/shukketsu/web/app.py tests/unit/test_wiki_routes.py
git commit -m "feat(web): add wiki router with browser, review, article detail, HTMX"
```

---

## Task 8: Lint, Type Check, Final Verification

**Files:**
- Potentially any file that needs lint fixes

**Step 1: Run ruff check and format**

```bash
ruff check code/ tests/ --fix && ruff format code/ tests/
```

Fix any issues that arise.

**Step 2: Run mypy**

```bash
python3 -m mypy code/shukketsu/web/wiki_render.py code/shukketsu/web/routers/wiki.py code/shukketsu/knowledge/manager.py
```

Fix any type errors.

**Step 3: Run full test suite**

```bash
python3 -m pytest tests/unit/ -v
```

Expected: All tests pass (~645 tests).

**Step 4: Commit any lint/type fixes**

```bash
git add -A && git commit -m "chore: lint and type fixes for wiki UI"
```

(Only if there are changes to commit.)

---

## Task 9: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update step status and test counts**

In CLAUDE.md, update:
1. The step status line for Step 8: mark as COMPLETE with test count
2. Update the header line about Phase 2 Steps complete (1-8)
3. Add the implementation plan doc to the plans table

Add to plans table:
```
| `2026-02-11-phase2-step8-implementation.md` | Step 8 implementation plan (complete) |
```

Update step 8 line:
```
8. ~~Wiki UI~~ — COMPLETE (N tests: wiki routes, markdown render, KM extensions)
```

**Step 2: Commit**

```bash
git add CLAUDE.md docs/plans/2026-02-11-phase2-step8-implementation.md
git commit -m "docs: update CLAUDE.md for Phase 2 Step 8 completion"
```
