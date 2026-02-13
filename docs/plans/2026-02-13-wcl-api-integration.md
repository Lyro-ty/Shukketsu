# WCL API Integration — Design Document

**Date**: 2026-02-13
**Status**: Approved
**Scope**: Warcraft Logs v2 GraphQL client, data ingest, personal character tracking

## Overview

Build a WCL API client to ingest combat log data for TBC Rogue analysis. Three collection modes: leaderboard rankings (top 100 per encounter), report deep-dives (gear/damage/buffs/casts), and personal character tracking (Lyroo-Nightslayer).

## API Details (Verified)

### Authentication

- **OAuth2 client credentials** via POST body params (not Basic Auth)
- Token endpoint: `https://www.warcraftlogs.com/oauth/token`
- Body: `grant_type=client_credentials&client_id=...&client_secret=...`
- Token expiry: ~360 days (re-acquire on expiry)
- Credentials stored in `variables.env` → read via `config.py`

### Endpoints

| Game Version | GraphQL Endpoint | Use Case |
|---|---|---|
| TBC Classic / Anniversary | `https://classic.warcraftlogs.com/api/v2/client` | TBC raids (zones 1007-1013, 1047-1052) |
| Classic Fresh (progression) | `https://fresh.warcraftlogs.com/api/v2/client` | Lyroo tracking, pre-TBC + TBC when it launches Feb 19 |

All requests: `POST` with `Authorization: Bearer {token}`, `Content-Type: application/json`, body `{"query": "...", "variables": {...}}`.

### Rate Limits

- **3,600 points/hour** (free tier), point cost varies by query complexity
- Check via `rateLimitData { pointsSpentThisHour limitPerHour pointsResetIn }`
- A full report drill costs ~5-10 points
- Budget: ~300-500 report deep-dives per hour

### TBC Zone IDs (Classic)

| Zone | ID | Phase | Encounters |
|---|---|---|---|
| Karazhan | 1007 / 1047 | 1 | 652-662 / 50652-50662 |
| Gruul / Magtheridon | 1008 / 1048 | 1 | 649-651 / 50649-50651 |
| SSC / TK | 1010 / 1052 | 2 | 623-628, 730-733 / 50623-50628, 50730-50733 |
| BT / Hyjal | 1011 | 3 | 601-609, 618-622 |
| Zul'Aman | 1012 | 4 | — |
| Sunwell Plateau | 1013 | 5 | 724-729 |

Zones with IDs 1047+ and encounter IDs 50xxx are Anniversary/Fresh versions. Active zones (non-frozen) are where new data appears.

### Classic Fresh Zone IDs (Lyroo's current progression)

| Zone | ID | Phase |
|---|---|---|
| Molten Core | 1049 | 1 |
| Blackwing Lair | 1034 | 2 |
| Temple of Ahn'Qiraj | 1035 | 3 |
| Naxxramas | 1036 | 4 |

TBC zones will appear on this endpoint when TBC launches Feb 19.

## Key Findings from Schema Introspection

### CombatantInfo Events (richest per-player data)

Query: `events(dataType: CombatantInfo, fightIDs: [...], limit: 100) { data nextPageTimestamp }`

Each event contains:
- `sourceID`, `specID`, `faction`, `expansion`
- **Full stat block**: strength, agility, stamina, intellect, spirit, dodge, parry, block, armor, critMelee/Ranged/Spell, hasteMelee/Ranged/Spell, hitMelee/Ranged/Spell, expertise, resilienceCritTaken, resilienceDamageTaken
- **Gear**: 19 slots, each with id, slot, quality, name, itemLevel, permanentEnchant, permanentEnchantName, gems[] (id, itemLevel, icon), setID
- **Talents**: guid + type entries (names require DB lookup — shows "UseDatabaseForName")
- **Auras**: pre-pull buffs at encounter start
- `talentTree`, `pvpTalents`, `customPowerSet`

### Damage Table (per-player breakdown)

Query: `table(dataType: DamageDone, fightIDs: [...], sourceClass: "Rogue")`

Each entry contains:
- name, id, guid, type, icon, itemLevel
- **total** damage, **activeTime**, activeTimeReduced
- **abilities[]**: per-ability name + total + type
- **damageAbilities[]**: detailed breakdown (when available)
- **targets[]**: damage per target
- **Inline gear[]** and **talents[]** (same format as CombatantInfo)

### Buff Table

Query: `table(dataType: Buffs, fightIDs: [...], sourceClass: "Rogue")`

Returns `auras[]`:
- name, guid, type, abilityIcon
- **totalUptime** (ms), **totalUses**
- **bands[]**: time ranges `{startTime, endTime}` for each buff application

### Casts Table

Query: `table(dataType: Casts, fightIDs: [...], sourceClass: "Rogue")`

Same structure as damage table but abilities[] show **cast counts** instead of damage totals.

### Raw Damage Events

Query: `events(dataType: DamageDone, fightIDs: [...], sourceClass: "Rogue", limit: N) { data nextPageTimestamp }`

Each event:
- timestamp, type ("damage"), sourceID, targetID, abilityGameID, fight
- **hitType**: 1=normal hit, 2=crit, 7=dodge, 8=miss, 10=immune, 16=partial resist
- amount, mitigated, unmitigatedAmount, resisted
- **buffs**: dot-separated string of active buff IDs at time of hit
- isAoE

### Rankings (per-report)

Query on report: `rankings`

Returns per-fight ranking data including:
- Per-player: name, server, class, spec, amount (DPS), bracket, rank, rankPercent, totalParses

### Character Rankings (leaderboard)

Query: `worldData { encounter(id: N) { characterRankings(className: "Rogue", metric: dps, page: N) } }`

100 per page, returns:
- name, class, spec, amount (DPS), duration, hardModeLevel
- report { code, fightID, startTime }
- guild { id, name, faction }
- server { id, name, region }
- bracketData, faction, size

### Important Caveats

1. **Archived reports**: Old reports (2021-2022 TBC Classic) are archived and inaccessible via free client API. Only non-archived reports can be deep-dived.
2. **playerDetails.combatantInfo**: Often empty. Use `events(dataType: CombatantInfo)` instead for reliable gear/stat data.
3. **Talent names**: CombatantInfo returns `"UseDatabaseForName"` for talent names. Must cross-reference talent GUIDs against a talent database.
4. **events sub-selection**: The `events()` field requires `{ data nextPageTimestamp }` sub-selection (it's a paginator type).
5. **Endpoint matters**: Must use `fresh.warcraftlogs.com` for Classic Fresh characters, `classic.warcraftlogs.com` for TBC Classic/Anniversary.

## Module Architecture

```
code/shukketsu/apis/
  __init__.py
  wcl/
    __init__.py
    auth.py          # OAuth2 token management
    client.py        # Rate-limit-aware GraphQL client
    queries.py       # Pre-built GraphQL query strings
    models.py        # Pydantic models for all response types
    ingest.py        # Collection orchestrator (rankings, reports, character sync)
    schema.sql       # WCL-specific database tables
```

### auth.py — Token Lifecycle

```python
class WCLAuth:
    TOKEN_URL = "https://www.warcraftlogs.com/oauth/token"

    async def get_token(self) -> str:
        # POST with client_id + client_secret as body params
        # Cache token + expiry timestamp in memory
        # Re-acquire when expired
```

### client.py — Rate-Limit-Aware GraphQL Client

```python
class WCLClient:
    ENDPOINTS = {
        "classic": "https://classic.warcraftlogs.com/api/v2/client",
        "fresh": "https://fresh.warcraftlogs.com/api/v2/client",
    }

    async def query(self, graphql: str, variables: dict | None, endpoint: str = "fresh") -> dict:
        # Check rate budget (query rateLimitData every ~10 requests)
        # Auto-sleep if approaching limit (>3000 points spent)
        # Retry on 429 with Retry-After
        # Raise on GraphQL errors

    async def query_paginated(self, ...) -> AsyncIterator[dict]:
        # For rankings pagination and events pagination (nextPageTimestamp)
```

Uses httpx async (consistent with existing scraping module). Integrates with existing circuit breaker pattern.

### queries.py — Pre-Built GraphQL Strings

Constants for all query types:
- `ZONE_METADATA` — zones + encounters (run once, cache)
- `ENCOUNTER_RANKINGS` — top 100 Rogues per encounter
- `REPORT_FIGHTS` — fight list for a report
- `REPORT_PLAYER_DETAILS` — playerDetails with includeCombatantInfo
- `REPORT_COMBATANT_INFO` — events(dataType: CombatantInfo)
- `REPORT_DAMAGE_TABLE` — table(dataType: DamageDone, sourceClass: "Rogue")
- `REPORT_BUFF_TABLE` — table(dataType: Buffs, sourceClass: "Rogue")
- `REPORT_CAST_TABLE` — table(dataType: Casts, sourceClass: "Rogue")
- `REPORT_RANKINGS` — per-fight percentiles
- `REPORT_DAMAGE_EVENTS` — raw damage events (paginated)
- `CHARACTER_LOOKUP` — by ID or name/server/region
- `CHARACTER_RECENT_REPORTS` — latest reports for a character
- `CHARACTER_ZONE_RANKINGS` — per-encounter percentiles
- `RATE_LIMIT` — current point usage

### models.py — Pydantic Models

**Rankings & Characters:**
- `WCLRanking` — leaderboard entry (name, spec, dps, duration, report code/fight, guild, server)
- `WCLCharacter` — character profile (id, name, classID, server, guilds)
- `WCLZoneRanking` — per-encounter percentile, best DPS, total kills

**Report-Level:**
- `WCLReport` — code, title, startTime, endTime, zone, archiveStatus
- `WCLFight` — id, encounterID, name, kill, duration, bossPercentage, avgItemLevel, size, difficulty
- `WCLActor` — id, name, type, subType (spec)

**Per-Player (CombatantInfo):**
- `WCLCombatantInfo` — sourceID, specID, faction, full stat block (all melee/ranged/spell stats)
- `WCLGearItem` — id, slot, quality, name, itemLevel, permanentEnchant, permanentEnchantName, gems[], setID
- `WCLGem` — id, itemLevel, icon
- `WCLTalentEntry` — guid, type, name, abilityIcon
- `WCLAuraEntry` — pre-pull buffs

**Tables:**
- `WCLDamageEntry` — per-player total, activeTime, abilities[], targets[], inline gear/talents
- `WCLAbilitySummary` — name, total, type
- `WCLBuffAura` — name, guid, totalUptime, totalUses, bands[]
- `WCLBuffBand` — startTime, endTime
- `WCLCastEntry` — per-player cast counts by ability

**Raw Events:**
- `WCLDamageEvent` — timestamp, sourceID, targetID, abilityGameID, hitType, amount, mitigated, unmitigated, resisted, buffs string, isAoE
- `WCLHitType` — enum: NORMAL_HIT=1, CRIT=2, ABSORB=3, DODGE=7, MISS=8, IMMUNE=10, PARTIAL_RESIST=16

**Fight Rankings:**
- `WCLFightRanking` — per-player name, server, class, spec, dps, rank, rankPercent, totalParses

## Database Schema (WCL-Specific Tables)

```sql
-- Tracked characters for personal monitoring
CREATE TABLE IF NOT EXISTS wcl_tracked_characters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wcl_id INTEGER NOT NULL UNIQUE,
    name TEXT NOT NULL,
    server_slug TEXT NOT NULL,
    server_region TEXT NOT NULL,
    endpoint TEXT NOT NULL DEFAULT 'fresh',  -- 'fresh' or 'classic'
    class_id INTEGER,
    last_sync_at TEXT,  -- ISO8601 timestamp
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Encounter rankings (top 100 per encounter)
CREATE TABLE IF NOT EXISTS wcl_rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id INTEGER NOT NULL,
    encounter_id INTEGER NOT NULL,
    encounter_name TEXT NOT NULL,
    player_name TEXT NOT NULL,
    spec TEXT NOT NULL,
    dps REAL NOT NULL,
    duration_ms INTEGER NOT NULL,
    report_code TEXT NOT NULL,
    fight_id INTEGER NOT NULL,
    guild_name TEXT,
    server_name TEXT NOT NULL,
    server_region TEXT NOT NULL,
    faction INTEGER,
    raid_size INTEGER,
    bracket_data INTEGER,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(encounter_id, player_name, server_name, report_code)
);

-- Report metadata
CREATE TABLE IF NOT EXISTS wcl_reports (
    code TEXT PRIMARY KEY,
    title TEXT,
    zone_id INTEGER,
    zone_name TEXT,
    start_time INTEGER,  -- epoch ms
    end_time INTEGER,
    is_archived INTEGER NOT NULL DEFAULT 0,
    endpoint TEXT NOT NULL DEFAULT 'fresh',
    fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Fight metadata within reports
CREATE TABLE IF NOT EXISTS wcl_fights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    encounter_id INTEGER NOT NULL,
    encounter_name TEXT NOT NULL,
    kill INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    boss_percentage REAL,
    avg_item_level REAL,
    raid_size INTEGER,
    difficulty INTEGER,
    UNIQUE(report_code, fight_id)
);

-- CombatantInfo per player per fight
CREATE TABLE IF NOT EXISTS wcl_combatants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    source_id INTEGER NOT NULL,
    spec_id INTEGER,
    faction INTEGER,
    -- Stats
    strength INTEGER, agility INTEGER, stamina INTEGER,
    intellect INTEGER, spirit INTEGER,
    crit_melee INTEGER, crit_ranged INTEGER, crit_spell INTEGER,
    haste_melee INTEGER, haste_ranged INTEGER, haste_spell INTEGER,
    hit_melee INTEGER, hit_ranged INTEGER, hit_spell INTEGER,
    expertise INTEGER, dodge INTEGER, parry INTEGER, block INTEGER, armor INTEGER,
    -- JSON blobs for complex nested data
    gear_json TEXT,      -- JSON array of gear items
    talents_json TEXT,   -- JSON array of talent entries
    auras_json TEXT,     -- JSON array of pre-pull auras
    UNIQUE(report_code, fight_id, source_id)
);

-- Damage breakdown per player per fight
CREATE TABLE IF NOT EXISTS wcl_damage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    player_type TEXT,  -- class
    total_damage INTEGER NOT NULL,
    active_time_ms INTEGER,
    -- JSON for ability list
    abilities_json TEXT,  -- JSON array of {name, total, type}
    targets_json TEXT,    -- JSON array of {name, total}
    UNIQUE(report_code, fight_id, player_name)
);

-- Buff uptimes per player per fight
CREATE TABLE IF NOT EXISTS wcl_buffs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    buff_name TEXT NOT NULL,
    buff_guid INTEGER NOT NULL,
    total_uptime_ms INTEGER NOT NULL,
    total_uses INTEGER NOT NULL,
    bands_json TEXT,  -- JSON array of {startTime, endTime}
    UNIQUE(report_code, fight_id, buff_guid)
);

-- Cast counts per player per fight
CREATE TABLE IF NOT EXISTS wcl_casts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    ability_name TEXT NOT NULL,
    cast_count INTEGER NOT NULL,
    UNIQUE(report_code, fight_id, player_name, ability_name)
);

-- Per-fight rankings (percentiles)
CREATE TABLE IF NOT EXISTS wcl_fight_rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_code TEXT NOT NULL REFERENCES wcl_reports(code),
    fight_id INTEGER NOT NULL,
    encounter_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    server_name TEXT,
    class TEXT NOT NULL,
    spec TEXT,
    dps REAL NOT NULL,
    rank_percent INTEGER,
    total_parses INTEGER,
    UNIQUE(report_code, fight_id, player_name)
);

-- Character sync log (tracks what we've already pulled for a tracked character)
CREATE TABLE IF NOT EXISTS wcl_character_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_wcl_id INTEGER NOT NULL,
    report_code TEXT NOT NULL,
    fight_id INTEGER NOT NULL,
    encounter_id INTEGER NOT NULL,
    encounter_name TEXT,
    dps REAL,
    rank_percent INTEGER,
    kill INTEGER,
    timestamp INTEGER,  -- epoch ms of the fight
    UNIQUE(character_wcl_id, report_code, fight_id)
);

CREATE INDEX IF NOT EXISTS idx_wcl_rankings_encounter ON wcl_rankings(encounter_id);
CREATE INDEX IF NOT EXISTS idx_wcl_rankings_player ON wcl_rankings(player_name, server_name);
CREATE INDEX IF NOT EXISTS idx_wcl_combatants_report ON wcl_combatants(report_code, fight_id);
CREATE INDEX IF NOT EXISTS idx_wcl_damage_report ON wcl_damage(report_code, fight_id);
CREATE INDEX IF NOT EXISTS idx_wcl_character_log_char ON wcl_character_log(character_wcl_id);
```

## Collection Modes

### 1. Leaderboard Ingest (Top 100 Rogues)

For each active TBC zone + encounter:
1. Query `characterRankings(className: "Rogue", metric: dps, page: 1)` → 100 rankings
2. Store in `wcl_rankings`
3. Collect unique report codes
4. For non-archived reports, run deep-dive (mode 3)

### 2. Character Sync (Lyroo Tracking)

1. Query `recentReports(limit: 10)` for tracked character
2. Compare against `wcl_character_log` to find new reports
3. For each new report: extract fights, damage, buffs, casts, CombatantInfo
4. Store dps + percentile in `wcl_character_log`
5. Detect gear changes between syncs

### 3. Report Deep-Dive

For a given report code + fight IDs:
1. Check `wcl_reports` for archive status — skip if archived
2. `events(dataType: CombatantInfo)` → `wcl_combatants`
3. `table(dataType: DamageDone, sourceClass: "Rogue")` → `wcl_damage`
4. `table(dataType: Buffs, sourceClass: "Rogue")` → `wcl_buffs`
5. `table(dataType: Casts, sourceClass: "Rogue")` → `wcl_casts`
6. `rankings` → `wcl_fight_rankings`

### Rate Budget

- 3,600 points/hour
- Full report deep-dive: ~5-10 points
- Rankings query: ~2-3 points
- Budget allows ~300-500 report deep-dives/hour
- Top 100 across ~15 TBC encounters = ~1,500 unique reports → ~3-5 hours for full initial ingest

## Configuration

```python
# config.py additions
WCL_TOKEN_URL = "https://www.warcraftlogs.com/oauth/token"
WCL_CLASSIC_ENDPOINT = "https://classic.warcraftlogs.com/api/v2/client"
WCL_FRESH_ENDPOINT = "https://fresh.warcraftlogs.com/api/v2/client"
WCL_RATE_LIMIT_BUFFER = 600  # Stop 600 points before limit
WCL_RATE_CHECK_INTERVAL = 10  # Check rate limit every N queries
WCL_QUERY_TIMEOUT = 30.0  # Seconds per GraphQL request
WCL_TRACKED_CHARACTERS = [
    {"wcl_id": 104956434, "name": "Lyroo", "server": "nightslayer", "region": "us", "endpoint": "fresh"},
]
```

## CLI Interface

```bash
# Sync tracked characters (pull new reports)
python3 -m code.shukketsu.apis.wcl.ingest --sync-characters

# Ingest top 100 rankings for a zone
python3 -m code.shukketsu.apis.wcl.ingest --rankings --zone 1052

# Deep-dive a specific report
python3 -m code.shukketsu.apis.wcl.ingest --report TNtKz3G1H9kVAQr4

# Full ingest (all active TBC zones)
python3 -m code.shukketsu.apis.wcl.ingest --full

# Check rate limit status
python3 -m code.shukketsu.apis.wcl.ingest --rate-limit

# Database stats
python3 -m code.shukketsu.apis.wcl.ingest --stats
```

## Testing Strategy

- Unit tests mock httpx responses (no real API calls)
- Test OAuth2 token caching and refresh logic
- Test GraphQL error handling (archived reports, rate limits, 429s)
- Test Pydantic model parsing against real response fixtures (saved from exploration)
- Integration tests (marked `@pytest.mark.integration`) hit real API with rate limiting
