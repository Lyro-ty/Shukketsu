# Phase 2, Step 5: Writer Agent + Wiki Backend

> Design document for Phase 2, Step 5. Validated through brainstorming session 2026-02-11.

## Overview

The Writer is the second specialist agent. Unlike the Researcher (which uses a ReAct loop to
search for information), the Writer uses **direct generation** — it receives a typed
`ResearchResult` and produces a wiki article in a single LLM pass, followed by a claims/entity
extraction pass. No ReAct loop is needed because the Writer's input is fully structured and its
output is deterministic: one article per task.

Articles are git-tracked Markdown files in `knowledge/{spec}/{category}/{slug}.md` with YAML
frontmatter (parsed via `python-frontmatter`). The `KnowledgeManager` handles all file + DB
operations: creating drafts, reading articles, updating content, querying by status/spec, and
enforcing lifecycle transitions (draft → review → published).

The database schema is upgraded to v3 to add `status`, `spec`, `category`, and `created_at`
columns to the `articles` table. Since the table has never contained data, this is a clean
replacement rather than a migration.

This step also tightens `WriteTask.research` from `AgentResult` to `ResearchResult`, and evolves
the `AgentFactory` to forward `**kwargs` to agent constructors — enabling the Writer to receive
its `KnowledgeManager` dependency via the factory without changing the factory signature per role.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Directory layout | Spec-first: `knowledge/{spec}/{category}/{slug}.md` | Most queries are spec-scoped. Existing dirs are empty — zero cost to reorganize. |
| Articles table | Schema v3 with `status`, `spec`, `category`, `created_at` | Clean columns beat encoding status in `needs_review` boolean. No data to migrate. |
| `WriteTask.research` type | Tighten to `ResearchResult` | Writer requires findings/gaps/sufficient. Bare `AgentResult` would produce garbage. Fail at validation, not at runtime. |
| Agent architecture | Direct generation (no ReAct loop) | Writer's input is fully structured. No tools needed. One LLM call for generation, one for extraction. |
| Existing article handling | Overwrite drafts, refuse review/published | Drafts are cheap; published articles are human-approved. Merge logic is YAGNI until Editor exists. |
| KnowledgeManager connection | Injected `sqlite3.Connection` | Consistent with `GraphStore`. Caller controls lifecycle. Testable with `test_db` fixture. |
| YAML frontmatter | `python-frontmatter` library | Battle-tested parse/serialize. Rolling our own `---` parser is asking for bugs. |
| Article generation | Two-pass: generate prose, then extract claims/entities | Same pattern as Researcher structuring pass. Each call has a focused job. Better quality. |
| Factory dependency injection | `**kwargs` forwarding to agent constructor | Writer needs `KnowledgeManager`, Editor will too. Generic forwarding scales without signature changes. |
| KnowledgeManager sync/async | Sync methods | SQLite and filesystem I/O are blocking. Consistent with `GraphStore`. |
| Slug generation | `re.sub` regex (lowercase, hyphens, strip) | No slugify library needed for simple title → filename conversion. |

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `code/shukketsu/knowledge/__init__.py` | Package init, exports `KnowledgeManager`, `ArticleStatus`, `ArticleMeta` |
| `code/shukketsu/knowledge/manager.py` | `KnowledgeManager` class + `ArticleStatus`, `ArticleMeta`, `SourceRef`, `ClaimRef` models |
| `code/shukketsu/agents/writer.py` | `Writer` subclass with direct generation `execute()` override |
| `code/shukketsu/llm/prompts/writer.py` | Writer system prompt + extraction prompt constants |
| `tests/unit/test_knowledge_manager.py` | All KnowledgeManager tests |
| `tests/unit/test_writer.py` | All Writer-specific tests |

### Modified Files

| File | Change |
|------|--------|
| `code/shukketsu/agents/tasks.py` | Add `WriteResult` model; tighten `WriteTask.research` from `AgentResult` to `ResearchResult` |
| `code/shukketsu/agents/factory.py` | Add `Writer` to `_ROLE_CLASSES`; change `create()` to forward `**kwargs` to constructor |
| `code/shukketsu/agents/__init__.py` | Export `Writer`, `WriteResult` |
| `code/shukketsu/config.py` | Import `WRITER_SYSTEM_PROMPT` from `llm.prompts.writer` instead of inline placeholder |
| `code/shukketsu/db/schema.sql` | Replace `articles` table with v3 schema (add `status`, `spec`, `category`, `created_at`) |
| `requirements.txt` | Add `python-frontmatter` |

### Directory Changes

| Change | Detail |
|--------|--------|
| Delete `knowledge/encounters/`, `knowledge/fundamentals/`, `knowledge/gearing/`, `knowledge/pvp/`, `knowledge/specs/` | Empty dirs from old topic-based layout |
| Create `knowledge/.gitkeep` | Preserve the directory in git; subdirs created dynamically by KnowledgeManager |

### Unchanged Files

These related files will NOT be touched (prevents scope creep):

- `agents/researcher.py` — Researcher is complete, produces `ResearchResult` consumed by Writer
- `agents/base.py` — No changes needed; Writer overrides `execute()` but doesn't touch the ReAct loop
- `tools/` — Writer has no tools; all search tools remain unchanged
- `rag/reranker.py` — Reranker is complete
- `llm/structured.py` — `get_structured_output()` already supports any Pydantic model
- `routing/router.py` — Routing changes deferred to Step 7 (Orchestrator)
- `web/routers/chat.py` — Chat handler changes deferred to Step 10 (Integration)
- `ingest/pipeline.py` — Ingest pipeline unchanged; articles are not ingested back into chunks

## Component 1: Schema v3 — Articles Table

### Change to `db/schema.sql`

Replace the existing `articles` table definition:

```sql
CREATE TABLE articles (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,                     -- "combat/gear/trinkets.md"
    title TEXT NOT NULL,
    spec TEXT NOT NULL,                            -- "combat", "assassination", "subtlety", "general"
    category TEXT NOT NULL,                        -- "gear", "rotation", "mechanics", etc.
    status TEXT NOT NULL DEFAULT 'draft',           -- "draft", "review", "published"
    confidence_score REAL NOT NULL DEFAULT 0.0,    -- 0.0–1.0, average of finding confidences
    verified_claims INTEGER NOT NULL DEFAULT 0,
    unverified_claims INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_updated TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_articles_status ON articles(status);
CREATE INDEX idx_articles_spec ON articles(spec);
```

Schema version bumped from 2 to 3. Since the `articles` table has never contained data,
this is a clean replacement — no migration logic needed.

### Changes from v2

- Added: `spec TEXT NOT NULL`, `category TEXT NOT NULL`, `status TEXT NOT NULL DEFAULT 'draft'`, `created_at TEXT NOT NULL`
- Changed: `last_updated` now has `DEFAULT (datetime('now'))` and is `NOT NULL`
- Changed: `confidence_score` now `NOT NULL DEFAULT 0.0`
- Removed: `needs_review` boolean (replaced by `status`)
- Added: indexes on `status` and `spec` for KnowledgeManager queries

## Component 2: KnowledgeManager

### Location: `code/shukketsu/knowledge/manager.py`

### Supporting Models

```python
class ArticleStatus(StrEnum):
    DRAFT = "draft"
    REVIEW = "review"
    PUBLISHED = "published"

class SourceRef(BaseModel):
    """A source referenced in an article."""
    url: str
    trust: float = Field(ge=0.0, le=1.0)

class ClaimRef(BaseModel):
    """A factual claim in an article, tracked for verification."""
    text: str
    verified: bool = False
    evidence: list[str] = Field(default_factory=list)  # chunk IDs or URLs

class ArticleMeta(BaseModel):
    """Pydantic model for article YAML frontmatter."""
    title: str
    spec: str
    category: str
    status: ArticleStatus = ArticleStatus.DRAFT
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    created_at: datetime
    updated_at: datetime
    sources: list[SourceRef] = Field(default_factory=list)
    claims: list[ClaimRef] = Field(default_factory=list)
    entity_refs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
```

### Class Signature

```python
class KnowledgeManager:
    def __init__(self, conn: sqlite3.Connection, knowledge_dir: Path) -> None:
        self._conn = conn
        self._knowledge_dir = knowledge_dir
```

### Methods

#### `slugify(title: str) -> str`

Module-level utility. Converts title to URL-safe slug.

```
"Phase 1 BiS Gear Guide" → "phase-1-bis-gear-guide"
"Combat Swords: Rotation Priority" → "combat-swords-rotation-priority"
```

Implementation: lowercase → replace non-alphanumeric with hyphens → collapse multiple hyphens → strip leading/trailing hyphens.

#### `derive_path(spec: str, category: str, title: str) -> str`

Returns relative path: `{spec}/{category}/{slugify(title)}.md`

Example: `derive_path("combat", "gear", "Phase 1 BiS Gear Guide")` → `"combat/gear/phase-1-bis-gear-guide.md"`

#### `create_draft(meta: ArticleMeta, content: str) -> str`

1. Derive path from `meta.spec`, `meta.category`, `meta.title`
2. Create parent directories under `knowledge_dir` if needed (`mkdir -p`)
3. Serialize `meta` to YAML frontmatter + `content` to markdown via `python-frontmatter`
4. Write file to `knowledge_dir / path`
5. INSERT row into `articles` table: `path`, `title`, `spec`, `category`, `status='draft'`, `confidence_score`, `created_at`, `last_updated`
6. Commit
7. Return the relative path string

Error cases:
- If file already exists at that path, raise `ValueError` (caller should use `exists()` + `update_draft()`)

#### `read_article(path: str) -> tuple[ArticleMeta, str]`

1. Read file from `knowledge_dir / path`
2. Parse via `frontmatter.loads()` → metadata dict + content string
3. Validate metadata into `ArticleMeta` via Pydantic
4. Return `(meta, content)`

Error cases:
- `FileNotFoundError` if file doesn't exist

#### `update_draft(path: str, meta: ArticleMeta, content: str) -> None`

1. Query DB for current status at `path`
2. If status is not `draft`, raise `ValueError("Cannot update article with status '{status}'")`
3. Update `meta.updated_at` to now
4. Serialize and write file (overwrite)
5. UPDATE DB row: `title`, `confidence_score`, `verified_claims`, `unverified_claims`, `last_updated`
6. Commit

#### `list_articles(*, status: ArticleStatus | None = None, spec: str | None = None) -> list[dict]`

Query the `articles` table with optional `WHERE` clauses. Returns list of dicts with
`path`, `title`, `spec`, `category`, `status`, `confidence_score`, `last_updated`.

Uses parameterized query building (no f-strings in SQL).

#### `set_status(path: str, new_status: ArticleStatus) -> None`

1. Query current status
2. Validate transition:
   - `draft` → `review` ✓
   - `review` → `published` ✓
   - All other transitions → raise `ValueError`
3. UPDATE `status` and `last_updated`
4. Commit

#### `exists(spec: str, category: str, title: str) -> str | None`

1. Derive path from args
2. Query `articles` table for that path
3. Return path if found, `None` otherwise

Note: checks DB, not filesystem. The DB is the source of truth for article existence.

## Component 3: Writer Agent

### Location: `code/shukketsu/agents/writer.py`

### Class Signature

```python
class Writer(BaseAgent):
    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        role: AgentRole | None = None,
        max_iterations: int = config.AGENT_MAX_ITERATIONS,
        system_prompt: str = config.SYSTEM_PROMPT,
        knowledge_manager: KnowledgeManager,
    ) -> None:
        super().__init__(
            tool_registry=tool_registry,
            role=role,
            max_iterations=max_iterations,
            system_prompt=system_prompt,
        )
        self._km = knowledge_manager
```

### `execute()` Flow

```python
@observe(as_type="agent")
async def execute(self, task: AgentTask, *, on_status=None) -> WriteResult:
```

**Step 1: Validate input**
- Verify `self.role` is set
- Verify `task` is a `WriteTask` (isinstance check)
- Verify `task.research` is a `ResearchResult`

**Step 2: Check for existing article**
- Call `self._km.exists(spec, category, title)` where title is derived from query + article_type
- If exists:
  - Read current article via `self._km.read_article(path)`
  - If status is `draft` → will overwrite in step 4
  - If status is `review` or `published` → return `WriteResult` with `status=FAILED` and explanation

**Step 3: Generate article content (Pass 1 — Llama 70B)**
- Build messages with writer system prompt + formatted findings
- Input includes: query, article_type, findings (claim + evidence + confidence each), gaps
- Call `get_structured_output()` with a `GeneratedArticle` schema:
  ```python
  class GeneratedArticle(BaseModel):
      title: str
      content: str  # Markdown body (no frontmatter)
  ```
- The LLM generates a well-structured Markdown article from the findings

**Step 4: Extract claims and entities (Pass 2 — Llama 70B)**
- Input: the generated article content
- Call `get_structured_output()` with an `ArticleExtraction` schema:
  ```python
  class ArticleExtraction(BaseModel):
      claims: list[str]           # Factual claims in the article
      entity_refs: list[str]      # Game entities mentioned
  ```
- This mirrors the Researcher's structuring pass pattern

**Step 5: Build metadata and write**
- Compute `confidence` as weighted average of finding confidences (weighted by how many
  findings the article used vs total findings available)
- Build `SourceRef` list by deduplicating evidence URLs from `ResearchResult.findings`
- Build `ClaimRef` list from extraction pass claims (all `verified=False` initially)
- Build `ArticleMeta` with all fields
- Call `self._km.create_draft(meta, content)` or `self._km.update_draft(path, meta, content)`

**Step 6: Return WriteResult**
```python
return WriteResult(
    task_id=task.task_id,
    agent_role=self.role,
    status=TaskStatus.SUCCESS,
    output=f"Article written: {title}",
    evidence=sources_used,
    article_path=path,
    title=title,
    claims=extracted_claims,
    research_gaps=task.research.gaps,
    word_count=len(content.split()),
    entity_refs=extracted_entity_refs,
)
```

### Error Handling

- If article generation fails (LLM error): return `WriteResult` with `status=FAILED`
- If extraction pass fails: fall back to empty claims/entity_refs (article is still written)
- If KnowledgeManager write fails: propagate exception (filesystem/DB error is not recoverable)

### Title Derivation

The article title comes from the generation pass (Pass 1). The LLM produces a title based on
the query, article_type, and findings. The Writer does not hardcode titles.

For the `exists()` check before generation, the Writer uses a preliminary title derived from
`task.query` + `task.article_type` to check for conflicts. After generation, if the LLM
produces a different title (and thus different slug), there's no conflict issue — the new
path simply won't match any existing article.

## Component 4: WriteResult Model

### Location: `code/shukketsu/agents/tasks.py`

```python
class WriteResult(AgentResult):
    """Structured output from the Writer agent."""
    article_path: str
    title: str
    claims: list[str] = Field(default_factory=list)
    research_gaps: list[str] = Field(default_factory=list)
    word_count: int = 0
    entity_refs: list[str] = Field(default_factory=list)
```

Also in `tasks.py`, tighten `WriteTask`:

```python
class WriteTask(AgentTask):
    research: ResearchResult    # was AgentResult
    article_type: ArticleType
```

## Component 5: Writer System Prompt

### Location: `code/shukketsu/llm/prompts/writer.py`

Same pattern as `researcher.py` — pure string constants with zero imports.
`config.py` imports from here.

### `WRITER_SYSTEM_PROMPT`

```
You are a Wiki Writer for WoW: The Burning Crusade (TBC) Rogue content. You produce
clear, accurate, well-structured Markdown articles from research findings.

## Writing Rules

STRUCTURE every article with:
  - A title that clearly identifies the topic and scope
  - A concise overview paragraph (2-3 sentences)
  - Logical sections with ## headers
  - Specific numbers and values (never vague)
  - Source attribution for key claims

TONE: Authoritative but approachable. Like an experienced raider explaining to a guild
member. Use "you" to address the reader.

CLAIMS: Every factual claim must trace back to a research finding. Never invent facts
the research didn't provide. If a finding has low confidence (< 0.5), hedge with
"likely" or "appears to".

COMPLETENESS: If the research has gaps, insert a <!-- NEEDS RESEARCH: [topic] -->
HTML comment at the relevant location rather than guessing.

FORMAT:
  - Use tables for gear comparisons, stat breakdowns, priority lists
  - Use **bold** for item names, spell names, stat values
  - Use ordered lists for priorities/rankings
  - Keep paragraphs short (3-5 sentences max)
  - Use > blockquotes for important callouts or tips

## Article Types

GUIDE: Step-by-step or priority-based. "How to gear your combat rogue in Phase 1."
  - Structure: Overview → Priority List → Detailed Breakdown → Tips

REFERENCE: Factual lookup. "Stat caps and hit table mechanics."
  - Structure: Overview → Core Mechanics → Values Table → Edge Cases

ANALYSIS: Comparative or analytical. "Combat vs Mutilate in Phase 2."
  - Structure: Overview → Methodology → Comparison → Conclusion

## Example 1: GUIDE from Research Findings

Input findings:
  1. "DST is BiS trinket for combat" (confidence: 0.9, evidence: [chunk:42, chunk:78])
  2. "Hit cap is 142 rating (9%)" (confidence: 0.95, evidence: [chunk:15, chunk:91])
  3. "Expertise soft cap is 23" (confidence: 0.7, evidence: [chunk:91])

Output:

# Combat Swords: Phase 1 Stat Priority

## Overview

Combat Swords is the dominant PvE rogue spec in Phase 1 of TBC. Reaching your hit and
expertise caps before stacking attack power and crit is essential for maximizing DPS.

## Stat Priority

1. **Hit Rating** to cap (142 rating / 9%)
2. **Expertise** to soft cap (23 rating)
3. **Attack Power**
4. **Crit Rating**
...

## Example 2: REFERENCE from Research Findings

Input findings:
  1. "Dual wield miss penalty is 19%" (confidence: 0.85, evidence: [chunk:15])
  2. "Special attacks use single-roll hit table" (confidence: 0.8, evidence: [chunk:203])

Output:

# Hit Table Mechanics for Rogues

## Overview

Understanding the hit table is fundamental to gearing decisions. Rogues face a dual wield
miss penalty that makes hit rating the most valuable stat until capped.

## How the Hit Table Works

Auto-attacks use a **two-roll system**...
...
```

### `EXTRACTION_PROMPT`

Short prompt for the second pass (claims + entity extraction from generated article):

```
You are an extraction assistant. Given a wiki article about WoW TBC Rogue content,
extract:

1. CLAIMS: Every factual claim in the article. Each claim should be a single,
   verifiable statement. Examples: "Hit cap is 142 rating", "DST drops from Gruul",
   "Combat Swords is the highest DPS spec in Phase 1".

2. ENTITY_REFS: Every WoW game entity mentioned in the article. Include item names,
   spell names, talent names, boss names, instance names, stat names, spec names.
   Use the canonical form (e.g., "Dragonspine Trophy" not "DST").

Be exhaustive. Extract every claim and every entity, not just the main ones.
```

## Component 6: Factory Changes

### `agents/factory.py`

Two changes:

1. Add Writer to `_ROLE_CLASSES`:
```python
from code.shukketsu.agents.writer import Writer

_ROLE_CLASSES: dict[AgentRole, type[BaseAgent]] = {
    AgentRole.RESEARCHER: Researcher,
    AgentRole.WRITER: Writer,
}
```

2. Forward `**kwargs` to constructor:
```python
def create(self, role: AgentRole, *, tool_registry=None, **kwargs) -> BaseAgent:
    ...
    agent = cls(
        tool_registry=registry,
        role=role,
        max_iterations=max_iter,
        system_prompt=prompt,
        **kwargs,
    )
```

This means creating a Writer requires:
```python
factory.create(AgentRole.WRITER, knowledge_manager=km)
```

If `knowledge_manager` is omitted, `Writer.__init__` raises `TypeError` — clear and immediate.

## Risks and Unknowns

| Risk | Impact | Mitigation |
|------|--------|------------|
| `python-frontmatter` not available for ARM64 | Can't parse/serialize YAML frontmatter | Pure Python package, no native extensions. Very low risk. Verify on install. |
| LLM generates poor article titles | Slug collisions or nonsensical paths | Title comes from structured output (Pydantic validated). Fallback: derive title from query + article_type if generation fails. |
| Two-pass latency (~5-6s total) | Slow article generation | Acceptable — article writing is a background task, not real-time chat. |
| Extraction pass misses claims | Unverified claims slip through to published articles | Editor (Step 6) independently verifies. Extraction is best-effort for frontmatter; Editor is the safety net. |
| Schema v3 breaks existing `init_db()` | DB initialization fails | No data in `articles` table. Schema is applied on fresh DB or test fixtures. `init_db()` uses `CREATE TABLE IF NOT EXISTS` — existing empty tables are fine. |
| `WriteTask.research` type tightening | Breaks any code passing `AgentResult` | Only `tasks.py` and tests reference `WriteTask`. No production callers yet. Clean break. |

## What This Does NOT Include

- **Article merging/diffing** — When a published article exists and new research arrives, the Writer refuses. Merge logic is deferred until it's actually needed (likely Phase 2 Step 10 integration or Phase 4).
- **Article deletion** — `KnowledgeManager` has no `delete()` method. Articles are never deleted programmatically. If needed, it's a manual git operation.
- **Article search/indexing** — Articles are NOT ingested back into the chunks/embeddings pipeline. They're a separate knowledge representation. If we want to search articles by content, that's a future step.
- **Writer tools** — The Writer has no tools (empty `ToolRegistry`). It doesn't search, browse, or interact — it generates from structured input. If the Writer ever needs to look things up, that's an architecture change.
- **Review/approval UI** — The wiki UI (Step 8) handles human review. `KnowledgeManager.set_status()` exists but is only called programmatically in tests and by the Editor (Step 6).
- **Automatic re-writing** — The Writer doesn't detect stale articles or trigger re-research. Content freshness (Step 9) handles that.
- **Category taxonomy enforcement** — Category values (`gear`, `rotation`, `mechanics`, etc.) are free-form strings, not an enum. Standardization can wait until we see what categories actually emerge.

## Dependencies

What must exist before this can be implemented:

- **Step 4 complete** — `ResearchResult`, `Finding`, `Researcher`, structuring pass pattern (all done)
- **Step 1 complete** — `AgentTask`/`AgentResult`, `WriteTask`/`ArticleType`, `AgentFactory` (all done)
- **`get_structured_output()`** — Used for both generation and extraction passes (exists in `llm/structured.py`)
- **`test_db` fixture** — Creates DB with full schema, used for KnowledgeManager tests (exists in `tests/conftest.py`)
- **`python-frontmatter`** — New pip dependency, must be added to `requirements.txt`

## Test Plan

Starting count: **473 tests** → Estimated after this step: **~510-515 tests**

### `tests/unit/test_knowledge_manager.py` (~20 tests)

**Slug generation:**
- `test_slugify_basic` — simple title to slug
- `test_slugify_special_characters` — colons, apostrophes, parentheses stripped
- `test_slugify_multiple_hyphens_collapsed` — "foo---bar" → "foo-bar"
- `test_slugify_leading_trailing_stripped` — no leading/trailing hyphens

**Path derivation:**
- `test_derive_path` — spec + category + title → correct relative path

**create_draft:**
- `test_create_draft_writes_file` — file exists at expected path with correct content
- `test_create_draft_inserts_db_row` — articles table has matching row
- `test_create_draft_frontmatter_roundtrip` — read back file, frontmatter matches input meta
- `test_create_draft_creates_parent_dirs` — nested dirs created automatically
- `test_create_draft_duplicate_raises` — creating at existing path raises ValueError

**read_article:**
- `test_read_article_parses_frontmatter` — returns correct ArticleMeta + content
- `test_read_article_missing_file_raises` — FileNotFoundError for nonexistent path

**update_draft:**
- `test_update_draft_overwrites_content` — file content changes
- `test_update_draft_bumps_timestamp` — updated_at changes
- `test_update_draft_refuses_review` — ValueError when status is "review"
- `test_update_draft_refuses_published` — ValueError when status is "published"

**list_articles:**
- `test_list_articles_no_filter` — returns all articles
- `test_list_articles_filter_by_status` — only matching status
- `test_list_articles_filter_by_spec` — only matching spec
- `test_list_articles_empty` — returns empty list when no articles

**set_status:**
- `test_set_status_draft_to_review` — valid transition succeeds
- `test_set_status_review_to_published` — valid transition succeeds
- `test_set_status_invalid_transition` — draft→published raises ValueError
- `test_set_status_backward_raises` — published→review raises ValueError

**exists:**
- `test_exists_returns_path_when_found` — article exists, returns path string
- `test_exists_returns_none_when_missing` — no article, returns None

### `tests/unit/test_writer.py` (~15-18 tests)

**Input validation:**
- `test_execute_requires_role` — ValueError when role is None
- `test_execute_requires_write_task` — fails gracefully on non-WriteTask input

**Article generation:**
- `test_execute_produces_article_file` — file exists at expected path after execute
- `test_execute_returns_write_result` — returns WriteResult with correct fields
- `test_execute_article_has_frontmatter` — generated file has valid YAML frontmatter
- `test_execute_claims_extracted` — WriteResult.claims is non-empty
- `test_execute_entity_refs_extracted` — WriteResult.entity_refs is non-empty
- `test_execute_word_count_accurate` — word_count matches content
- `test_execute_gaps_carried_through` — research_gaps from ResearchResult appear in WriteResult
- `test_execute_confidence_computed` — confidence is weighted average of finding confidences

**Existing article handling:**
- `test_execute_overwrites_draft` — existing draft article is replaced
- `test_execute_refuses_published` — returns FAILED status for published article
- `test_execute_refuses_review` — returns FAILED status for review article

**Error handling:**
- `test_execute_generation_failure_returns_failed` — LLM error → FAILED status
- `test_execute_extraction_failure_still_writes` — extraction error → article written, empty claims

**Factory integration:**
- `test_factory_creates_writer` — `AgentFactory.create(WRITER, knowledge_manager=km)` returns Writer
- `test_factory_writer_requires_knowledge_manager` — omitting kwarg raises TypeError

### `tests/unit/test_factory.py` (modify existing, ~2-3 new tests)

- `test_factory_kwargs_forwarded` — verify `**kwargs` reach the constructor
- `test_factory_writer_in_role_classes` — Writer in `_ROLE_CLASSES` registry

### Schema test (in existing `tests/unit/test_db_schema.py`, ~1-2 new tests)

- `test_articles_table_v3_columns` — verify `status`, `spec`, `category`, `created_at` columns exist
- `test_articles_status_index_exists` — verify index on `status`
