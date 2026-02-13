# WCL API Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Warcraft Logs v2 GraphQL client to ingest combat log data (rankings, gear, damage, buffs) for TBC Rogue analysis with personal character tracking.

**Architecture:** New `code/shukketsu/apis/wcl/` package with OAuth2 auth, rate-limit-aware GraphQL client, Pydantic models for all WCL response types, SQLite storage (schema v5), and CLI for batch ingest. Follows existing patterns: async httpx, circuit breakers, ShukketsuError taxonomy, TDD.

**Tech Stack:** httpx (async HTTP), Pydantic v2 (models), SQLite (storage), pytest + AsyncMock (testing)

**Design doc:** `docs/plans/2026-02-13-wcl-api-integration.md`

---

## Task 1: Config + Error Types

**Files:**
- Modify: `code/shukketsu/config.py`
- Modify: `code/shukketsu/resilience/errors.py`
- Test: `tests/unit/test_wcl_config.py`

**Step 1: Write test for new config constants**

```python
# tests/unit/test_wcl_config.py
"""Tests for WCL configuration constants."""

from code.shukketsu import config


def test_wcl_endpoints_defined() -> None:
    assert config.WCL_TOKEN_URL == "https://www.warcraftlogs.com/oauth/token"
    assert "classic.warcraftlogs.com" in config.WCL_CLASSIC_ENDPOINT
    assert "fresh.warcraftlogs.com" in config.WCL_FRESH_ENDPOINT


def test_wcl_rate_limit_defaults() -> None:
    assert config.WCL_RATE_LIMIT_BUDGET == 3600
    assert config.WCL_RATE_LIMIT_BUFFER == 600
    assert config.WCL_RATE_CHECK_INTERVAL == 10


def test_wcl_query_timeout() -> None:
    assert config.WCL_QUERY_TIMEOUT == 30.0


def test_wcl_tracked_characters_default() -> None:
    assert isinstance(config.WCL_TRACKED_CHARACTERS, list)
    lyroo = config.WCL_TRACKED_CHARACTERS[0]
    assert lyroo["wcl_id"] == 104956434
    assert lyroo["name"] == "Lyroo"
    assert lyroo["endpoint"] == "fresh"


def test_wcl_circuit_breaker_config() -> None:
    assert config.CB_WCL_FAILURE_THRESHOLD == 3
    assert config.CB_WCL_RECOVERY_TIMEOUT == 60.0
```

**Step 2: Run test, verify fails** — `python3 -m pytest tests/unit/test_wcl_config.py -v`

**Step 3: Add config constants to `config.py`**

Add after the existing circuit breaker section:

```python
# WCL API
WCL_TOKEN_URL = "https://www.warcraftlogs.com/oauth/token"
WCL_CLASSIC_ENDPOINT = "https://classic.warcraftlogs.com/api/v2/client"
WCL_FRESH_ENDPOINT = "https://fresh.warcraftlogs.com/api/v2/client"
WCL_RATE_LIMIT_BUDGET = 3600
WCL_RATE_LIMIT_BUFFER = int(os.getenv("WCL_RATE_LIMIT_BUFFER", "600"))
WCL_RATE_CHECK_INTERVAL = int(os.getenv("WCL_RATE_CHECK_INTERVAL", "10"))
WCL_QUERY_TIMEOUT = float(os.getenv("WCL_QUERY_TIMEOUT", "30.0"))
WCL_TRACKED_CHARACTERS: list[dict[str, str | int]] = [
    {"wcl_id": 104956434, "name": "Lyroo", "server": "nightslayer", "region": "us", "endpoint": "fresh"},
]
CB_WCL_FAILURE_THRESHOLD = int(os.getenv("CB_WCL_FAILURE_THRESHOLD", "3"))
CB_WCL_RECOVERY_TIMEOUT = float(os.getenv("CB_WCL_RECOVERY_TIMEOUT", "60"))
```

**Step 4: Add error types to `resilience/errors.py`**

Add `WCL_API` to the `FailureMode` enum. Add three error classes:

```python
# In FailureMode enum:
WCL_API = "wcl_api"

# New error classes:
class WCLAuthError(ShukketsuError):
    """WCL OAuth2 authentication failure."""
    def __init__(self, message: str):
        super().__init__(message, FailureMode.WCL_API)

class WCLRateLimitError(ShukketsuError):
    """WCL API rate limit exceeded."""
    def __init__(self, message: str, points_reset_in: float = 0):
        super().__init__(message, FailureMode.WCL_API)
        self.points_reset_in = points_reset_in

class WCLQueryError(ShukketsuError):
    """WCL GraphQL query error (invalid query, archived report, etc)."""
    def __init__(self, message: str):
        super().__init__(message, FailureMode.WCL_API)
```

**Step 5: Run tests, verify passes** — `python3 -m pytest tests/unit/test_wcl_config.py -v`

**Step 6: Commit**

```bash
git add code/shukketsu/config.py code/shukketsu/resilience/errors.py tests/unit/test_wcl_config.py
git commit -m "feat(wcl): add config constants and error types for WCL API"
```

---

## Task 2: Pydantic Models

**Files:**
- Create: `code/shukketsu/apis/__init__.py`
- Create: `code/shukketsu/apis/wcl/__init__.py`
- Create: `code/shukketsu/apis/wcl/models.py`
- Test: `tests/unit/apis/__init__.py`
- Test: `tests/unit/apis/wcl/__init__.py`
- Test: `tests/unit/apis/wcl/test_wcl_models.py`

**Step 1: Write tests for all Pydantic models**

Test each model can be constructed from realistic WCL response data. Use fixture data captured during API exploration. Key models to test:

- `WCLRanking` — from characterRankings response
- `WCLFight` — from report fights
- `WCLGearItem` + `WCLGem` — from CombatantInfo gear array
- `WCLCombatantInfo` — from events(CombatantInfo) response
- `WCLDamageEntry` + `WCLAbilitySummary` — from DamageDone table
- `WCLBuffAura` + `WCLBuffBand` — from Buffs table
- `WCLDamageEvent` + `WCLHitType` enum — from raw damage events
- `WCLFightRanking` — from report rankings

Use real response data from the exploration session as test fixtures (the Ganick Brutallus data, the CN Void Reaver data, and Lyroo's AQ40 data).

**Step 2: Run tests, verify fails**

**Step 3: Implement all models in `models.py`**

Follow Pydantic v2 BaseModel patterns. Use `str` enums for JSON round-trip. All fields typed with 3.12 syntax (`str | None`, `list[X]`). Use `model_config = ConfigDict(populate_by_name=True)` where WCL field names differ from Python conventions (e.g., `fightID` → `fight_id` with `alias`).

Key design decisions:
- `WCLHitType` as IntEnum: NORMAL=1, CRIT=2, ABSORB=3, DODGE=7, MISS=8, IMMUNE=10, PARTIAL_RESIST=16
- `WCLEndpoint` as StrEnum: CLASSIC="classic", FRESH="fresh"
- Gear items use optional fields (gems, enchant can be absent)
- CombatantInfo stat fields all optional (not every stat appears for every class)

**Step 4: Run tests, verify passes**

**Step 5: Commit**

```bash
git add code/shukketsu/apis/ tests/unit/apis/
git commit -m "feat(wcl): add Pydantic models for all WCL response types"
```

---

## Task 3: Database Schema (v5 Migration)

**Files:**
- Create: `code/shukketsu/apis/wcl/schema.sql`
- Modify: `code/shukketsu/db/connection.py` (add v5 migration)
- Test: `tests/unit/apis/wcl/test_wcl_schema.py`

**Step 1: Write tests for WCL table creation**

Test that after `init_db()`, all WCL tables exist with correct columns:
- `wcl_tracked_characters`
- `wcl_rankings`
- `wcl_reports`
- `wcl_fights`
- `wcl_combatants`
- `wcl_damage`
- `wcl_buffs`
- `wcl_casts`
- `wcl_fight_rankings`
- `wcl_character_log`

Test indexes exist. Test UNIQUE constraints prevent duplicate inserts.

**Step 2: Run tests, verify fails**

**Step 3: Create `schema.sql` with all WCL tables**

Copy the exact SQL from the design doc section "Database Schema". Save as `code/shukketsu/apis/wcl/schema.sql`.

**Step 4: Add v5 migration to `connection.py`**

Add `_migrate_v4_to_v5(conn)` that reads and executes `apis/wcl/schema.sql`. Update `CURRENT_SCHEMA_VERSION = 5`. Follow existing migration pattern (check version, execute SQL, update version).

**Step 5: Run tests, verify passes**

**Step 6: Commit**

```bash
git add code/shukketsu/apis/wcl/schema.sql code/shukketsu/db/connection.py tests/unit/apis/wcl/test_wcl_schema.py
git commit -m "feat(wcl): add WCL database schema v5 migration"
```

---

## Task 4: OAuth2 Auth Module

**Files:**
- Create: `code/shukketsu/apis/wcl/auth.py`
- Test: `tests/unit/apis/wcl/test_wcl_auth.py`

**Step 1: Write tests for auth module**

Test cases:
1. `test_get_token_acquires_new_token` — mock httpx POST, verify correct URL/body, return token
2. `test_get_token_caches_token` — second call returns cached token without HTTP request
3. `test_get_token_refreshes_expired` — set expiry in past, verify new request made
4. `test_get_token_raises_on_auth_failure` — mock 401 response, expect WCLAuthError
5. `test_get_token_raises_on_network_error` — mock ConnectError, expect WCLAuthError
6. `test_get_token_uses_body_params` — verify client_id and client_secret sent as POST body (not Basic Auth)

Mock `httpx.AsyncClient.post` for all tests.

**Step 2: Run tests, verify fails**

**Step 3: Implement `WCLAuth` class**

```python
class WCLAuth:
    """Manages OAuth2 client credentials token for WCL API."""

    def __init__(self, client_id: str = "", client_secret: str = "") -> None:
        self._client_id = client_id or config.WCL_CLIENT_ID
        self._client_secret = client_secret or config.WCL_CLIENT_SECRET
        self._token: str | None = None
        self._expires_at: float = 0.0

    async def get_token(self) -> str:
        if self._token and time.time() < self._expires_at:
            return self._token
        return await self._acquire_token()

    async def _acquire_token(self) -> str:
        # POST to WCL_TOKEN_URL with body params
        # Parse access_token + expires_in
        # Cache and return
```

**Step 4: Run tests, verify passes**

**Step 5: Commit**

```bash
git add code/shukketsu/apis/wcl/auth.py tests/unit/apis/wcl/test_wcl_auth.py
git commit -m "feat(wcl): add OAuth2 token management"
```

---

## Task 5: GraphQL Query Library

**Files:**
- Create: `code/shukketsu/apis/wcl/queries.py`
- Test: `tests/unit/apis/wcl/test_wcl_queries.py`

**Step 1: Write tests**

Verify each query constant is a valid non-empty string containing expected GraphQL keywords. Test that parameterized query builder functions produce correct variable dicts.

Test cases:
1. Each query constant exists and contains expected field names
2. `build_rankings_query(encounter_id, page)` returns correct variables
3. `build_report_query(code, fight_ids)` returns correct variables
4. `build_character_query(wcl_id)` returns correct variables

**Step 2: Run tests, verify fails**

**Step 3: Implement query constants and builders**

All the query strings from the design doc: `ZONE_METADATA`, `ENCOUNTER_RANKINGS`, `REPORT_FIGHTS`, `REPORT_COMBATANT_INFO`, `REPORT_DAMAGE_TABLE`, `REPORT_BUFF_TABLE`, `REPORT_CAST_TABLE`, `REPORT_RANKINGS`, `CHARACTER_BY_ID`, `CHARACTER_RECENT_REPORTS`, `CHARACTER_ZONE_RANKINGS`, `RATE_LIMIT`.

Plus builder functions that return `(query_string, variables_dict)` tuples for parameterized queries.

**Step 4: Run tests, verify passes**

**Step 5: Commit**

```bash
git add code/shukketsu/apis/wcl/queries.py tests/unit/apis/wcl/test_wcl_queries.py
git commit -m "feat(wcl): add GraphQL query library"
```

---

## Task 6: GraphQL Client (Rate-Limit Aware)

**Files:**
- Create: `code/shukketsu/apis/wcl/client.py`
- Test: `tests/unit/apis/wcl/test_wcl_client.py`

**Step 1: Write tests**

Test cases for `WCLClient`:
1. `test_query_sends_correct_headers` — Bearer token in Authorization header
2. `test_query_returns_data` — successful response parsed correctly
3. `test_query_raises_on_graphql_errors` — response with `errors` key raises WCLQueryError
4. `test_query_handles_archived_report` — specific error message for archived reports
5. `test_query_selects_correct_endpoint` — "fresh" vs "classic" maps to correct URL
6. `test_rate_limit_tracking` — after N queries, client checks rate limit
7. `test_rate_limit_sleeps_when_near_budget` — when points > budget - buffer, client sleeps
8. `test_query_retries_on_429` — HTTP 429 triggers sleep + retry
9. `test_query_raises_on_persistent_429` — 3x 429 raises WCLRateLimitError
10. `test_circuit_breaker_integration` — failures trip the circuit breaker

Mock httpx + WCLAuth for all tests. Use `AsyncMock` for auth.get_token().

**Step 2: Run tests, verify fails**

**Step 3: Implement `WCLClient`**

```python
class WCLClient:
    """Rate-limit-aware async GraphQL client for WCL v2 API."""

    ENDPOINTS = {
        "classic": config.WCL_CLASSIC_ENDPOINT,
        "fresh": config.WCL_FRESH_ENDPOINT,
    }

    def __init__(self, auth: WCLAuth) -> None:
        self._auth = auth
        self._query_count = 0
        self._points_spent = 0.0
        self._points_limit = config.WCL_RATE_LIMIT_BUDGET

    async def query(self, graphql: str, variables: dict | None = None,
                    endpoint: str = "fresh") -> dict:
        # Get token, build request, check rate budget
        # POST to endpoint, parse response
        # Handle errors, track rate limit

    async def check_rate_limit(self, endpoint: str = "fresh") -> dict:
        # Query rateLimitData, update internal tracking
```

Wire up circuit breaker from `resilience/circuit_breaker.py`. Add `wcl_breaker` instance.

**Step 4: Run tests, verify passes**

**Step 5: Commit**

```bash
git add code/shukketsu/apis/wcl/client.py tests/unit/apis/wcl/test_wcl_client.py
git commit -m "feat(wcl): add rate-limit-aware GraphQL client with circuit breaker"
```

---

## Task 7: Ingest Orchestrator

**Files:**
- Create: `code/shukketsu/apis/wcl/ingest.py`
- Test: `tests/unit/apis/wcl/test_wcl_ingest.py`

**Step 1: Write tests**

Test the three collection modes with mocked WCLClient:

**Rankings ingest:**
1. `test_ingest_rankings_stores_to_db` — mock rankings response, verify rows in `wcl_rankings`
2. `test_ingest_rankings_collects_report_codes` — returns set of unique report codes
3. `test_ingest_rankings_skips_duplicates` — UNIQUE constraint doesn't crash, updates gracefully

**Report deep-dive:**
4. `test_deep_dive_stores_combatants` — mock CombatantInfo events, verify `wcl_combatants` rows
5. `test_deep_dive_stores_damage` — mock damage table, verify `wcl_damage` rows
6. `test_deep_dive_stores_buffs` — mock buff table, verify `wcl_buffs` rows
7. `test_deep_dive_stores_casts` — mock cast table, verify `wcl_casts` rows
8. `test_deep_dive_stores_rankings` — mock report rankings, verify `wcl_fight_rankings` rows
9. `test_deep_dive_skips_archived` — archived report detected, skip with log warning

**Character sync:**
10. `test_sync_character_finds_new_reports` — mock recent reports, filter against existing log
11. `test_sync_character_stores_log` — new report processed, `wcl_character_log` updated
12. `test_sync_character_detects_gear_change` — compare gear JSON between two syncs

**Step 2: Run tests, verify fails**

**Step 3: Implement ingest orchestrator**

Three main classes:
- `RankingsIngestor` — iterates encounters, fetches top 100 Rogues, stores rankings
- `ReportDiver` — deep-dives a report: CombatantInfo, damage, buffs, casts, rankings
- `CharacterSyncer` — syncs tracked characters, stores log entries, detects changes

All take `WCLClient` + `sqlite3.Connection` as constructor args. Use existing `get_connection()` pattern.

**Step 4: Run tests, verify passes**

**Step 5: Commit**

```bash
git add code/shukketsu/apis/wcl/ingest.py tests/unit/apis/wcl/test_wcl_ingest.py
git commit -m "feat(wcl): add ingest orchestrator (rankings, deep-dive, character sync)"
```

---

## Task 8: CLI Interface

**Files:**
- Modify: `code/shukketsu/apis/wcl/ingest.py` (add CLI entry point)
- Test: `tests/unit/apis/wcl/test_wcl_cli.py`

**Step 1: Write tests for CLI argument parsing**

Test `_build_parser()` returns correct args for each mode:
1. `--sync-characters` flag
2. `--rankings --zone 1052`
3. `--report TNtKz3G1H9kVAQr4`
4. `--full` flag
5. `--rate-limit` flag
6. `--stats` flag
7. `--dry-run` combined with other flags

**Step 2: Run tests, verify fails**

**Step 3: Implement CLI**

Add `_build_parser()`, `_async_main(args)`, and `main()` functions to `ingest.py` (same pattern as `code/shukketsu/ingest/batch.py`). Wire up each flag to the corresponding orchestrator class.

Entry point: `python3 -m code.shukketsu.apis.wcl.ingest`

Add `__main__.py` for module execution:
```python
# code/shukketsu/apis/wcl/__main__.py
from code.shukketsu.apis.wcl.ingest import main
main()
```

**Step 4: Run tests, verify passes**

**Step 5: Commit**

```bash
git add code/shukketsu/apis/wcl/ingest.py code/shukketsu/apis/wcl/__main__.py tests/unit/apis/wcl/test_wcl_cli.py
git commit -m "feat(wcl): add CLI interface for WCL data ingest"
```

---

## Task 9: Integration Smoke Test + Cleanup

**Files:**
- Create: `tests/integration/test_wcl_integration.py`
- Modify: `code/shukketsu/apis/wcl/__init__.py` (public exports)

**Step 1: Write integration test**

Marked `@pytest.mark.integration`. Tests that hit the real WCL API (requires credentials):
1. `test_auth_acquires_real_token` — real OAuth2 flow
2. `test_query_zone_metadata` — fetch TBC zones
3. `test_query_character_lyroo` — look up Lyroo by ID on fresh endpoint
4. `test_rate_limit_check` — verify rate limit data returned

**Step 2: Set up `__init__.py` exports**

Export the main public interface:
```python
from code.shukketsu.apis.wcl.auth import WCLAuth
from code.shukketsu.apis.wcl.client import WCLClient
from code.shukketsu.apis.wcl.ingest import CharacterSyncer, RankingsIngestor, ReportDiver
```

**Step 3: Run full test suite**

```bash
python3 -m pytest tests/unit/ -v  # All unit tests pass
ruff check code/ tests/           # No lint errors
ruff format code/ tests/           # Formatted
python3 -m mypy code/shukketsu/apis/  # Type checks pass
```

**Step 4: Run integration tests** (requires Ollama off, just WCL credentials)

```bash
python3 -m pytest tests/integration/test_wcl_integration.py -v -m integration
```

**Step 5: Commit**

```bash
git add tests/integration/test_wcl_integration.py code/shukketsu/apis/wcl/__init__.py
git commit -m "feat(wcl): add integration smoke tests and public API exports"
```

---

## Task 10: CLAUDE.md + Memory Update

**Files:**
- Modify: `CLAUDE.md` (add WCL API section)
- Modify: Memory files

**Step 1: Update CLAUDE.md**

Add to Commands section:
```bash
# WCL API ingest
python3 -m code.shukketsu.apis.wcl.ingest --sync-characters    # Sync Lyroo's latest reports
python3 -m code.shukketsu.apis.wcl.ingest --rankings --zone 1052  # Top 100 Rogue rankings
python3 -m code.shukketsu.apis.wcl.ingest --full                 # Full TBC zone ingest
```

Add to Architecture section: WCL API client description, endpoint info.

Add to Project Layout: `apis/wcl/` module listing.

Update test count.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with WCL API integration"
```

---

## Execution Summary

| Task | Description | Est. Tests |
|------|-------------|-----------|
| 1 | Config + Error Types | 6 |
| 2 | Pydantic Models | ~15 |
| 3 | Database Schema v5 | ~8 |
| 4 | OAuth2 Auth | 6 |
| 5 | Query Library | ~8 |
| 6 | GraphQL Client | 10 |
| 7 | Ingest Orchestrator | 12 |
| 8 | CLI Interface | 7 |
| 9 | Integration Smoke | 4 |
| 10 | Docs Update | 0 |
| **Total** | | **~76** |

Post-completion test count: ~889 tests (813 existing + ~76 new).
