# Phase 2 Step 9: Content Freshness + Automated Backups

> Design document for Phase 2, Step 9. Validated through brainstorming session 2026-02-11.

## Overview

Content Freshness + Automated Backups adds two maintenance capabilities to the system: detecting
when ingested web sources have changed (and flagging them for re-ingestion), and creating verified
SQLite database backups with automatic pruning.

The freshness checker finds sources whose `last_checked` timestamp exceeds their
`check_interval_hours`, then performs a two-stage check: first a HEAD request for ETag/Last-Modified
headers, then a full fetch and content hash comparison if needed. Sources with confirmed content
changes are flagged as `is_stale = 1`. The checker does NOT re-ingest automatically — it flags
sources so the user or Orchestrator can decide when to act. Wiki articles that cite stale sources
display a warning badge on their detail page, but remain published (non-destructive).

The backup manager uses Python's built-in `sqlite3.Connection.backup()` API to create timestamped
copies of the database while it's in use (WAL mode). Each backup is integrity-checked after
creation. Old backups are pruned to keep disk usage bounded.

Both features are exposed via API endpoints for manual triggering. A freshness sweep also runs
automatically on application startup to catch staleness accumulated while the server was down.
No background scheduler (APScheduler) is used — this is a local single-user system where manual
control and startup checks are sufficient. Periodic scheduling can be added in a later phase
without changing the core logic.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Staleness model | Use existing `check_interval_hours` per-source column | Schema already has the column with default 168h. No new enum or migration needed. Domain heuristics set the interval during ingest. |
| Domain interval mapping | Config dict in `config.py` mapping domain → hours | Configuration data, not logic. Consistent with centralized config pattern. Easy to override later. |
| Scheduling | Startup sweep + manual API endpoints | Local single-user system. APScheduler adds complexity for marginal benefit. Manual endpoints are trivially testable. Can add periodic scheduling later without changing core logic. |
| Freshness check depth | HEAD request + full fetch + content hash comparison | HEAD alone is unreliable (many WoW sites lack ETag/Last-Modified). Content hash comparison is definitive. `content_hash_previous` and `change_count` columns already exist in schema. |
| Article staleness display | Stale source count computed on read, shown on article detail only | Non-destructive — published articles stay published with a warning badge. Computed via SQL query against `sources.is_stale`, always accurate, no sync issues. |
| Article list view | No stale warning on list cards | Avoids N extra queries per article in the list view. Users check individual articles or the `/api/freshness/stale` endpoint. |
| Backup scope | Database file only | `knowledge/` directory is git-tracked. Git is a better versioning system for Markdown than timestamped tarballs. |
| Backup API | Sync methods (not async) | File I/O and SQLite backup are blocking but fast. No benefit from async for local file copies. |
| HTTP client for freshness | Direct `httpx.AsyncClient` | Not the existing `WebFetcher` — freshness checks are internal maintenance, not user-facing scraping. Don't need robots.txt compliance or rate limiting for HEAD requests to our own cached sources. |
| Re-ingest cleanup | Ingest pipeline clears `is_stale` and updates `content_hash_previous` | Closes the loop: checker flags stale → re-ingest clears flag. `change_count` provides historical signal. |

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `freshness/checker.py` | `find_stale_sources()`, `check_source_freshness()`, `run_freshness_sweep()`, `count_stale_sources()` |
| `backup/manager.py` | `BackupManager` class: create, verify, list, prune backups |
| `web/routers/freshness.py` | API endpoints: POST check, GET stale, POST clear |
| `web/routers/backup.py` | API endpoints: POST create, GET list, POST prune |
| `tests/unit/test_freshness_checker.py` | Unit tests for freshness detection logic |
| `tests/unit/test_backup_manager.py` | Unit tests for backup create/verify/prune |
| `tests/unit/test_freshness_routes.py` | Unit tests for freshness API endpoints |
| `tests/unit/test_backup_routes.py` | Unit tests for backup API endpoints |
| `tests/unit/test_ingest_freshness.py` | Unit tests for ingest pipeline freshness integration |

### Modified Files

| File | Change |
|------|--------|
| `config.py` | Add `FRESHNESS_HTTP_TIMEOUT`, `DOMAIN_CHECK_INTERVALS`, `DEFAULT_CHECK_INTERVAL_HOURS`, `BACKUP_KEEP_COUNT` (plus unused-for-now `FRESHNESS_CHECK_INTERVAL_HOURS`, `BACKUP_INTERVAL_HOURS` for future scheduling) |
| `web/app.py` | Mount freshness and backup routers. Add freshness sweep to lifespan startup. |
| `ingest/pipeline.py` | Set `check_interval_hours` from domain heuristic on new sources. Update `content_hash_previous`, `change_count`, clear `is_stale` on re-ingest. |
| `web/templates/wiki/article.html` | Add stale source warning badge in metadata sidebar |
| `web/routers/wiki.py` | Call `count_stale_sources()` when rendering article detail, pass count to template |

### Unchanged Files

- `db/schema.sql` — No schema changes. All freshness columns already exist on `sources` table.
- `agents/` — No agent changes. Agents are unaware of freshness.
- `knowledge/manager.py` — No changes. Article status is not affected by staleness.
- `routing/` — No routing changes.
- `tools/` — No tool changes.
- `rag/` — No RAG changes.
- `scraping/fetcher.py` — Not used for freshness checks (direct httpx instead).
- `web/routers/chat.py` — Chat functionality untouched.
- `backup/scheduler.py` — Not created. No scheduler in this step.
- `freshness/scheduler.py` — Not created. No scheduler in this step.

## Component 1: Freshness Checker (`freshness/checker.py`)

### Models

```python
class StaleSource(BaseModel):
    id: int
    url: str
    content_hash: str | None
    check_interval_hours: int

class FreshnessResult(BaseModel):
    source_id: int
    url: str
    changed: bool
    old_hash: str | None
    new_hash: str | None
    checked_at: str          # ISO 8601
    head_only: bool          # True if HEAD was sufficient (no full fetch needed)
    error: str | None = None # Non-None if check failed (network error, timeout)
```

### Functions

```python
def find_stale_sources(conn: Connection) -> list[StaleSource]:
    """Find sources where now - last_checked > check_interval_hours.

    Sources with last_checked = NULL are always considered stale.
    Returns list ordered by staleness (most overdue first).
    """

async def check_source_freshness(
    source: StaleSource, conn: Connection, timeout: int = 10
) -> FreshnessResult:
    """Check a single source for content changes.

    Flow:
    1. Send HEAD request to source URL
    2. If ETag or Last-Modified present and match stored values → mark fresh
       (update last_checked only, return head_only=True)
    3. If headers missing or suggest change → full GET request
    4. Compute SHA-256 hash of fetched content
    5. Compare against stored content_hash
    6. If unchanged → update last_checked only
    7. If changed → update content_hash_previous = old hash,
       content_hash = new hash, change_count += 1, is_stale = 1, last_checked

    Network errors are caught and returned in FreshnessResult.error.
    The source row is not modified on error (will be retried next sweep).
    """

async def run_freshness_sweep(
    conn: Connection, timeout: int = 10
) -> list[FreshnessResult]:
    """Find all stale sources and check each. Returns results for all checked sources."""

def count_stale_sources(conn: Connection, source_urls: list[str]) -> int:
    """Count how many of the given URLs have is_stale = 1.

    Used by wiki article detail to show stale source warning badge.
    Returns 0 if source_urls is empty.
    """
```

### HEAD Request Logic

The HEAD check is an optimization, not a requirement. Many WoW community sites don't set
ETag or Last-Modified headers. The logic:

```
HEAD request →
  if response has ETag → compare with stored ETag (stored where? see note)
  if response has Last-Modified → compare with stored timestamp
  if neither header present → fall through to full fetch
  if both missing and HEAD fails → fall through to full fetch
```

**Note on ETag storage**: The current `sources` table has no `etag` or `last_modified` column.
Rather than adding schema columns for an optimization, the HEAD check simply compares the
Last-Modified date against `last_checked`. If the content was modified after our last check,
do a full fetch. If Last-Modified is before our last check, skip the full fetch. If no
Last-Modified header, always do the full fetch. This avoids any schema changes.

### Error Handling

- Network timeout → `FreshnessResult(error="Timeout after Ns")`
- Connection refused → `FreshnessResult(error="Connection refused")`
- HTTP 404/410 → `FreshnessResult(error="Source gone (404)")` — source may have been removed
- HTTP 403 → `FreshnessResult(error="Access denied (403)")`
- Any other exception → `FreshnessResult(error=str(e))`

Errors are logged but don't stop the sweep. The source row is unchanged on error, so it
remains stale and will be retried on the next sweep.

## Component 2: Config Additions (`config.py`)

```python
# Freshness checking
FRESHNESS_CHECK_INTERVAL_HOURS = int(os.getenv("FRESHNESS_CHECK_INTERVAL_HOURS", "6"))
FRESHNESS_HTTP_TIMEOUT = int(os.getenv("FRESHNESS_HTTP_TIMEOUT", "10"))

# Domain → check_interval_hours mapping for new sources during ingest
DOMAIN_CHECK_INTERVALS: dict[str, int] = {
    "wowhead.com": 720,          # 30 days — guides update slowly
    "icy-veins.com": 720,        # 30 days
    "shadowpanther.net": 2160,   # 90 days — TBC content is static
    "silentshadows.net": 2160,   # 90 days
    "tbcdb.com": 2160,           # 90 days — database, patch-locked
    "warcraftlogs.com": 24,      # 1 day — rankings change constantly
}
DEFAULT_CHECK_INTERVAL_HOURS = int(os.getenv("DEFAULT_CHECK_INTERVAL_HOURS", "168"))

# Backup
BACKUP_KEEP_COUNT = int(os.getenv("BACKUP_KEEP_COUNT", "7"))
```

A helper function in config for the domain mapping:

```python
def get_check_interval(url: str) -> int:
    """Extract domain from URL, return check_interval_hours from mapping."""
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.lower()
    # Strip www. prefix
    if domain.startswith("www."):
        domain = domain[4:]
    return DOMAIN_CHECK_INTERVALS.get(domain, DEFAULT_CHECK_INTERVAL_HOURS)
```

## Component 3: Backup Manager (`backup/manager.py`)

```python
class BackupResult(BaseModel):
    path: str            # Absolute path to backup file
    size_bytes: int
    created_at: str      # ISO 8601
    integrity_ok: bool   # Result of PRAGMA integrity_check

class BackupManager:
    def __init__(self, db_path: Path, backup_dir: Path):
        """Initialize with paths. Creates backup_dir if it doesn't exist."""

    def create_backup(self) -> BackupResult:
        """Create timestamped backup using sqlite3.Connection.backup().

        Filename: shukketsu-YYYYMMDD-HHMMSS.db
        Flow:
        1. Open source connection to db_path (read-only)
        2. Create destination connection to backup_dir/shukketsu-YYYYMMDD-HHMMSS.db
        3. source.backup(destination)
        4. Close both connections
        5. Run verify_integrity() on the backup
        6. Return BackupResult with file size and integrity status
        """

    def verify_integrity(self, backup_path: Path) -> bool:
        """Run PRAGMA integrity_check on a backup file.

        Opens a read-only connection to the backup, runs the pragma,
        returns True if result is 'ok', False otherwise.
        """

    def list_backups(self) -> list[BackupResult]:
        """List existing backups sorted newest first.

        Scans backup_dir for files matching shukketsu-*.db pattern.
        Returns BackupResult for each (integrity_ok is not re-checked;
        set to True since it was verified at creation time).
        """

    def prune_old_backups(self, keep: int | None = None) -> int:
        """Delete all but the most recent `keep` backups.

        Uses BACKUP_KEEP_COUNT from config if keep is None.
        Returns count of deleted files. Sorts by filename
        (timestamp naming makes this chronological).
        """
```

All methods are synchronous. SQLite backup and file operations are fast for a local database.

## Component 4: API Endpoints

### Freshness Routes (`web/routers/freshness.py`)

```python
router = APIRouter(prefix="/api/freshness", tags=["freshness"])

@router.post("/check")
async def freshness_check() -> dict:
    """Run freshness sweep. Returns summary of results."""
    # Returns: {"checked": N, "changed": N, "errors": N, "results": [...]}

@router.get("/stale")
async def list_stale_sources() -> dict:
    """List all currently stale sources (is_stale = 1)."""
    # Returns: {"stale_sources": [{"id": N, "url": "...", "change_count": N}, ...]}

@router.post("/clear/{source_id}")
async def clear_stale_flag(source_id: int) -> dict:
    """Clear is_stale flag for a source (e.g., after manual re-ingest)."""
    # Returns: {"cleared": True, "source_id": N}
```

### Backup Routes (`web/routers/backup.py`)

```python
router = APIRouter(prefix="/api/backup", tags=["backup"])

@router.post("/create")
async def create_backup() -> dict:
    """Create a new backup. Returns BackupResult."""

@router.get("/list")
async def list_backups() -> dict:
    """List existing backups."""
    # Returns: {"backups": [...], "count": N}

@router.post("/prune")
async def prune_backups() -> dict:
    """Prune old backups. Returns count deleted."""
    # Returns: {"deleted": N, "remaining": N}
```

### Startup Freshness Sweep (`web/app.py`)

Added to the existing lifespan context manager:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing startup logic ...

    # Run freshness sweep (non-blocking — stale sources just get flagged)
    try:
        results = await run_freshness_sweep(conn)
        stale_count = sum(1 for r in results if r.changed)
        logger.info("Freshness sweep: %d checked, %d changed", len(results), stale_count)
    except Exception:
        logger.exception("Freshness sweep failed on startup")

    yield
```

Fire-and-forget with exception handling. If it fails (no network, etc.), the app still starts.

## Component 5: Ingest Pipeline Changes (`ingest/pipeline.py`)

Two modifications to the existing `ingest()` method:

### 5a. Set `check_interval_hours` on new sources

When inserting a new source row, use the domain heuristic instead of the schema default:

```python
from code.shukketsu.config import get_check_interval

# In the INSERT for new sources:
check_interval = get_check_interval(url)
self._conn.execute(
    """INSERT INTO sources (url, title, source_type, trust_score, fetched_at,
       content_hash, chunk_count, check_interval_hours, last_checked)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
    (url, title, source_type, trust_score, now, content_hash,
     chunk_count, check_interval, now),
)
```

### 5b. Track content changes on re-ingest

When an existing source has changed content (hash mismatch), update tracking columns:

```python
# Existing: content_hash differs → proceed with re-ingest
self._conn.execute(
    """UPDATE sources
       SET content_hash_previous = content_hash,
           content_hash = ?,
           change_count = change_count + 1,
           is_stale = 0,
           last_checked = ?
       WHERE id = ?""",
    (content_hash, now, existing["id"]),
)
# ... then proceed with chunk replacement / re-embedding
```

This closes the freshness loop: checker sets `is_stale = 1` → user/Orchestrator triggers
re-ingest → pipeline clears `is_stale` and records the change.

## Component 6: Wiki UI Changes

### Article Detail Sidebar (`web/templates/wiki/article.html`)

Add a stale source warning below the existing "Sources" section in the sidebar:

```html
{% if stale_source_count > 0 %}
<div class="mt-3 p-2 rounded border border-yellow-700 bg-yellow-900/20">
    <span class="text-yellow-400 text-sm font-medium">
        {{ stale_source_count }} source{{ "s" if stale_source_count != 1 }}
        may be outdated
    </span>
</div>
{% endif %}
```

### Wiki Route Change (`web/routers/wiki.py`)

In the `wiki_article()` handler, after reading the article, compute the stale count:

```python
from code.shukketsu.freshness.checker import count_stale_sources

# Extract source URLs from article frontmatter
source_urls = [s.url for s in article.meta.sources] if article.meta.sources else []
stale_count = count_stale_sources(conn, source_urls) if source_urls else 0

# Pass to template
return templates.TemplateResponse("wiki/article.html", {
    ...,
    "stale_source_count": stale_count,
})
```

## Risks and Unknowns

| Risk | Impact | Mitigation |
|------|--------|------------|
| WoW sites block HEAD requests | HEAD optimization never works, always falls through to full fetch | Acceptable — full fetch is the fallback by design. HEAD is an optimization, not a requirement. |
| Sites block our User-Agent | Freshness checks fail for some sources | Use the same User-Agent string as the existing fetcher. Log errors so user knows which sources can't be checked. |
| Large database makes backup slow | Startup blocked if backup is triggered | Backup is manual-only, not on startup. Freshness sweep on startup doesn't touch backup. |
| Startup freshness sweep is slow with many sources | App startup delayed | Log progress. For Phase 2 scale (< 100 sources), this is sub-minute. If it grows, make the sweep async/background. |
| `is_stale` accumulates with no one clearing it | Stale flags pile up | `/api/freshness/stale` endpoint makes them visible. Article warning badge surfaces them to users. Re-ingest clears them. |
| Content hash changes due to ad/layout changes, not actual content | False positive stale flags | Acceptable for Phase 2. Content-only hashing (stripping boilerplate) is a Phase 4 refinement. |

## What This Does NOT Include

- **Automatic re-ingestion** — The checker flags stale sources but does not re-ingest. Re-ingestion is a separate manual action or future Orchestrator capability.
- **APScheduler / periodic background jobs** — No scheduler. Manual endpoints + startup sweep. Scheduler can be added later without changing core logic.
- **Freshness log table** — The phase plan mentioned a `freshness_log` table. Deferred — the `change_count` and `last_checked` columns provide sufficient history for now. Add audit logging if needed.
- **Content-aware diffing** — Hash comparison is binary (changed/unchanged). No semantic diffing or change summarization.
- **Source removal detection** — If a source returns 404, it's logged as an error but not automatically removed from the DB.
- **Backup encryption or compression** — Backups are plain SQLite files. Encryption/compression deferred.
- **Backup restoration endpoint** — No API to restore from backup. Manual operation via SQLite CLI if needed.
- **Article list stale badges** — Stale source warnings only on article detail page, not in the browser list view.
- **Freshness UI page** — No dedicated web page for freshness status. API endpoints only. UI can be added if needed.

## Dependencies

- **Phase 1** (`db/schema.sql`): `sources` table with `last_checked`, `check_interval_hours`, `content_hash`, `content_hash_previous`, `change_count`, `is_stale` columns
- **Phase 1** (`config.py`): `BACKUP_PATH`, `DB_PATH` configuration values
- **Phase 1** (`web/app.py`): FastAPI app with lifespan context manager
- **Step 5** (`knowledge/manager.py`): `ArticleMeta.sources` for extracting source URLs
- **Step 8** (`web/routers/wiki.py`): Article detail route to add stale source count
- **Step 8** (`web/templates/wiki/article.html`): Article sidebar template to add warning badge
- **External**: `httpx` (already in requirements), `sqlite3` (stdlib)

## Test Plan

All tests in `tests/unit/`. Freshness checker tests mock `httpx` responses. Backup tests use
`tmp_path` for isolated file operations. Route tests use FastAPI `TestClient` with mocked
dependencies.

### `tests/unit/test_freshness_checker.py` (~14 tests)

- `test_find_stale_sources_returns_overdue` — source past check_interval is returned
- `test_find_stale_sources_skips_fresh` — source within interval is not returned
- `test_find_stale_sources_null_last_checked` — NULL last_checked is always stale
- `test_find_stale_sources_empty_table` — no sources returns empty list
- `test_check_head_last_modified_fresh` — Last-Modified before last_checked → head_only=True, changed=False
- `test_check_head_last_modified_stale` — Last-Modified after last_checked → triggers full fetch
- `test_check_head_no_headers` — no ETag/Last-Modified → triggers full fetch
- `test_check_full_fetch_unchanged` — same content hash → changed=False, last_checked updated
- `test_check_full_fetch_changed` — different hash → changed=True, is_stale=1, content_hash_previous set
- `test_check_change_count_increments` — change_count goes up on each confirmed change
- `test_check_network_error` — timeout/connection error → FreshnessResult.error set, source unchanged
- `test_check_http_404` — gone source → error logged, source unchanged
- `test_count_stale_sources_some_stale` — returns correct count for mixed stale/fresh URLs
- `test_count_stale_sources_empty_list` — empty URL list returns 0

### `tests/unit/test_backup_manager.py` (~8 tests)

- `test_create_backup_produces_file` — backup file exists at expected path
- `test_create_backup_filename_format` — filename matches shukketsu-YYYYMMDD-HHMMSS.db pattern
- `test_create_backup_integrity_ok` — integrity check passes on fresh backup
- `test_verify_integrity_good_db` — returns True for valid SQLite file
- `test_verify_integrity_bad_file` — returns False for corrupted/non-SQLite file
- `test_list_backups_sorted_newest_first` — backups returned in reverse chronological order
- `test_prune_keeps_correct_count` — prune(keep=3) with 5 backups → 2 deleted, 3 remain
- `test_prune_empty_directory` — no backups → returns 0 deleted

### `tests/unit/test_freshness_routes.py` (~5 tests)

- `test_post_check_returns_results` — POST /api/freshness/check returns summary
- `test_get_stale_returns_list` — GET /api/freshness/stale returns stale sources
- `test_get_stale_empty` — no stale sources returns empty list
- `test_clear_stale_flag` — POST /api/freshness/clear/{id} clears is_stale
- `test_clear_nonexistent_source_404` — clearing nonexistent source returns 404

### `tests/unit/test_backup_routes.py` (~4 tests)

- `test_post_create_backup` — POST /api/backup/create returns BackupResult
- `test_get_list_backups` — GET /api/backup/list returns backup list
- `test_post_prune_backups` — POST /api/backup/prune returns deleted count
- `test_prune_empty` — prune with no backups returns 0

### `tests/unit/test_ingest_freshness.py` (~3 tests)

- `test_new_source_gets_domain_check_interval` — wowhead.com source gets 720h, unknown gets 168h
- `test_reingest_changed_content_updates_tracking` — content_hash_previous set, change_count incremented, is_stale cleared
- `test_reingest_same_content_no_tracking_change` — unchanged content doesn't modify tracking columns

**Starting count: 652 tests → Estimated after this step: ~686 tests**
