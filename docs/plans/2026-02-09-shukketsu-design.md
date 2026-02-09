# Shukketsu — TBC Rogue Research Agent

> The premier World of Warcraft TBC Anniversary Rogue application.
> A local AI-powered research agent that searches, analyzes, simulates, and synthesizes
> everything about the Rogue class in The Burning Crusade.

## Context

- **Hardware**: NVIDIA DGX Spark (GB10 Grace Blackwell, 128GB unified memory, ARM64)
- **Environment**: NVIDIA AI Workbench container (Python, Node.js, CUDA 13.0)
- **Game**: TBC Classic Anniversary Edition (launched February 5, 2026)
- **Current Phase**: Phase 1 (Karazhan, Gruul, Magtheridon — raids opened Feb 19)
- **TBC Anniversary Changes**: Dual Spec available early, no GDKP, Guild Banks from start

## Goals

1. **Comprehensive coverage** of all 3 Rogue specs (Combat, Assassination, Subtlety) across all TBC phases, PvE and PvP
2. **Active research** — the agent discovers, ingests, and synthesizes information from across the web
3. **Empirical verification** — claims backed by simulation data and real combat logs
4. **Living knowledgebase** — a browsable wiki + searchable knowledge store + chat interface
5. **Always current** — monitors sources for new data, re-evaluates conclusions as the meta evolves
6. **Educational** — every component designed to teach AI, web dev, simulation, and data engineering

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────┐
│            Nemotron-3-Nano 30B (fp16)            │
│              Agent Brain (Ollama)                 │
│                                                   │
│  ReAct mode (quick questions)                     │
│  Plan-then-Execute mode (complex research)        │
└──────────┬──────────┬──────────┬─────────────────┘
           │          │          │
     ┌─────┴───┐ ┌───┴────┐ ┌──┴──────────┐
     │Research │ │Analysis│ │ Knowledge   │
     │ Tools   │ │ Tools  │ │ Management  │
     ├─────────┤ ├────────┤ ├─────────────┤
     │Web      │ │Log     │ │RAG Search   │
     │Search   │ │Parser  │ │(vec + FTS5) │
     │Web      │ │DPS Sim │ │Wiki Writer  │
     │Ingest   │ │Stat    │ │DB Query     │
     │WCL API  │ │Weights │ │Verifier     │
     │Blizzard │ │Log     │ │             │
     │API      │ │Analyzer│ │             │
     │WoWSims  │ │        │ │             │
     └─────────┘ └────────┘ └─────────────┘
           │          │          │
     ┌─────┴──────────┴──────────┴─────────┐
     │         Storage Layer                │
     │  SQLite (structured + vectors + FTS) │
     │  Git Markdown (wiki articles)        │
     └────────────────┬────────────────────┘
                      │
     ┌────────────────┴────────────────────┐
     │        Web Knowledgebase             │
     │  FastAPI + Jinja2 + HTMX            │
     │  Wiki · Search · Chat · Logs · Sim  │
     └─────────────────────────────────────┘
```

---

## 2. Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM (reasoning) | Nemotron-3-Nano 30B fp16 via Ollama | 63GB, fits DGX Spark with ~60GB headroom. 1M context window. 3.5B active params (MoE) = fast inference. NVIDIA-native. |
| LLM (embeddings) | nomic-embed-text via Ollama | ~270MB, runs alongside Nemotron. 768-dim embeddings for RAG. |
| Web framework | FastAPI + Uvicorn | Async, WebSocket support, production Python framework |
| Templating | Jinja2 + HTMX | Server-rendered HTML with interactivity. No JS build chain. |
| Styling | Custom WoW-themed CSS | Dark theme, rogue class colors, item rarity colors, WoW-style tooltips |
| Charts | Chart.js (~60KB) | DPS breakdowns, stat comparisons, timeline visualizations |
| Database | SQLite + sqlite-vec + FTS5 | One file for structured data + vector search + full-text search |
| Wiki storage | Git-tracked Markdown | Version-controlled articles with revision history |
| Web scraping | httpx + BeautifulSoup4 | Async HTTP + HTML parsing |
| Task scheduling | APScheduler | Periodic research tasks |
| Simulation | Custom Python engine | TBC rogue combat mechanics modeled from first principles |

### Dependencies (requirements.txt)

```
fastapi
uvicorn[standard]
jinja2
httpx
beautifulsoup4
markdown
apscheduler
sqlite-vec
chart.js (CDN)
htmx (CDN)
```

---

## 3. Project Structure

```
shukketsu/
├── agent/                      # Agent core
│   ├── brain.py                # Main agent loop (ReAct + Plan modes)
│   ├── planner.py              # Task decomposition for complex research
│   ├── context.py              # Context assembly + token budget management
│   ├── trust.py                # Source trust scoring + conflict resolution
│   ├── verifier.py             # Self-check before publishing to wiki
│   └── tools/
│       ├── registry.py         # Tool registration + dispatch
│       ├── research/
│       │   ├── web_search.py   # General web search
│       │   ├── web_ingest.py   # Fetch any URL → chunk → embed → store
│       │   ├── warcraft_logs.py # WCL v2 GraphQL API (OAuth)
│       │   ├── blizzard_api.py # Battle.net Game Data API (OAuth)
│       │   ├── wowsims_import.py # Import WoWSims rogue data from GitHub
│       │   └── log_parser.py   # WoW combat log file parser
│       ├── analysis/
│       │   ├── sim_runner.py   # Run DPS simulations
│       │   ├── log_analyzer.py # Compare logs vs optimal play
│       │   └── stat_weights.py # Calculate stat priorities
│       └── knowledge/
│           ├── search.py       # Hybrid RAG search (vector + FTS5)
│           ├── wiki_writer.py  # Create/update markdown articles
│           └── db_query.py     # Query structured game data
├── data/
│   ├── shukketsu.db            # SQLite database (all structured + vector data)
│   └── sources/cache/          # Raw HTML/JSON cache of fetched pages
├── knowledge/                  # Git-tracked Markdown wiki
│   ├── specs/
│   │   ├── combat/
│   │   ├── assassination/
│   │   └── subtlety/
│   ├── gearing/                # BiS lists, stat weights per phase
│   ├── encounters/             # Boss-specific rogue strategies
│   ├── pvp/                    # Arena comps, BG strategies, seasons
│   └── fundamentals/           # Mechanics, formulas, caps
├── sim/                        # TBC Rogue DPS simulation engine
│   ├── mechanics.py            # Core combat formulas (TBC 2.4.3)
│   ├── abilities.py            # Every rogue ability with coefficients
│   ├── talents.py              # Talent tree modeling (all 3 trees)
│   ├── items.py                # Gear stats, set bonuses, proc effects
│   ├── buffs.py                # Raid buffs, consumables, boss debuffs
│   ├── rotation.py             # Priority-based rotation logic per spec
│   ├── combat.py               # Fight simulation loop
│   └── runner.py               # Public API: config → results
├── web/                        # Knowledgebase site
│   ├── app.py                  # FastAPI application + route registration
│   ├── routers/
│   │   ├── wiki.py             # Wiki browsing + markdown rendering
│   │   ├── search.py           # Hybrid search endpoint
│   │   ├── chat.py             # Chat interface (WebSocket)
│   │   ├── logs.py             # Combat log upload + analysis
│   │   └── sim.py              # Sim runner from browser
│   ├── templates/
│   │   ├── base.html           # Shared layout (nav, sidebar, theme)
│   │   ├── wiki/
│   │   │   ├── article.html    # Single wiki article view
│   │   │   ├── index.html      # Wiki home / table of contents
│   │   │   └── history.html    # Article revision history (git log)
│   │   ├── search.html         # Search results page
│   │   ├── chat.html           # Chat interface
│   │   ├── logs/
│   │   │   ├── upload.html     # Log upload form
│   │   │   └── report.html     # Analysis report view
│   │   └── sim/
│   │       └── builder.html    # Gear/talent/sim config builder
│   └── static/
│       ├── css/
│       │   ├── theme.css       # Dark WoW-inspired base theme
│       │   ├── tooltips.css    # WoW-style item tooltip hovers
│       │   ├── talent-tree.css # In-game style talent grid
│       │   └── colors.css      # Item rarity + class color system
│       ├── js/
│       │   ├── chat.js         # WebSocket chat client
│       │   ├── tooltips.js     # Item tooltip hover engine
│       │   ├── talent-tree.js  # Interactive talent picker
│       │   └── charts.js       # DPS breakdown charts
│       └── img/
│           ├── ability-icons/  # Rogue ability icons
│           ├── item-slots/     # Equipment slot silhouettes
│           └── ui/             # Borders, backgrounds, class crest
├── ollama_setup.sh             # Ollama install + model pull script
└── requirements.txt
```

---

## 4. Data Collection Layer

### Philosophy

Instead of building dozens of source-specific adapters, Shukketsu uses **two general-purpose tools** (web search + web ingest) for most sources, with **dedicated adapters only for structured APIs** that require special handling.

### Dedicated Adapters (4 total)

| Adapter | Source | Why dedicated |
|---------|--------|---------------|
| `warcraft_logs.py` | Warcraft Logs v2 | GraphQL API with OAuth, complex query structure for parses/rankings/gear |
| `blizzard_api.py` | Battle.net API | REST with OAuth, live authoritative game data (items, spells, characters) |
| `wowsims_import.py` | WoWSims GitHub | Go source code containing validated spell coefficients + combat models |
| `log_parser.py` | WoW combat log files | Unique binary/text format specific to WoW |

### General-Purpose Tools (everything else)

**`web_search.py`** — Agent searches the web for rogue topics, discovers sources organically.

**`web_ingest.py`** — Agent feeds any URL. Pipeline:
1. Fetch page content (httpx)
2. Extract text (BeautifulSoup)
3. Chunk into ~500 token segments
4. Embed each chunk (nomic-embed-text via Ollama)
5. Store chunks + embeddings in SQLite (chunks + chunks_vec tables)
6. Store source metadata for deduplication + trust scoring

### Known High-Value Sources

The agent will be directed to ingest these during Phase 2, but can discover additional sources on its own:

**Rogue-Specific Communities (HIGH trust)**
- Silent Shadows (silentshadows.net) — premier TBC rogue community
- ShadowPanther.net TBC — oldest rogue-specific gear resource
- ClassicRogueCraft — rogue damage formulas and spreadsheets
- tbcguides.gg (Sno) — guides, BiS lists, podcast transcripts

**Theorycrafting Archives (HIGH trust)**
- Elitist Jerks Roguecraft 101 (Wayback Machine + mirrors)
- Theorycraft Archive Wiki (Fandom) — preserved EJ-era research
- CMaNGOS TBC Database (GitHub) — raw spell coefficients and proc rates

**Guide Sites (HIGH trust)**
- Icy Veins TBC Classic Rogue
- WOWTBC.GG rogue guides
- Warcraft Tavern TBC Rogue section
- WoWHead TBC Rogue guides

**Simulation Data (HIGH trust)**
- WoWSims TBC Rogue (GitHub — open source sim engine)
- Garcia Rogue DPS Spreadsheet
- RogueDPS GitHub spreadsheet

**PvP / Arena Data (HIGH trust)**
- Ironforge.pro arena leaderboards + population data
- Drustvar PvP leaderboards
- Liquipedia Classic Arena Tournament results
- Skill Capped TBC Rogue PvP guides (premium, limited access)

**Community Discussion (MEDIUM trust)**
- Reddit (r/classicwow, r/worldofwarcraft)
- Blizzard Official Rogue Forums
- MMO-Champion Rogue Forums
- Barrens Chat TBC Forum

**Video/Podcast Content (MEDIUM trust)**
- YouTube rogue creators (Sno, Simonize, Nahj, Pshero, Mir)
- Sno & Zirene TBC Podcast (RSS/YouTube transcripts)
- Classic Arena Tournament VODs
- Twitch VODs from top rogue streamers

**International Sources (MEDIUM trust)**
- NGA.cn — Chinese WoW rogue theorycrafting
- Inven — Korean WoW rogue community

**Database/Reference Sites (MEDIUM trust)**
- Wowpedia / Warcraft Wiki — ability and mechanic documentation
- WoWWiki Archive — historical TBC rogue builds
- Cavern of Time TBC Database
- wow-classic-items npm package (structured JSON item data)

**Private Server Archives (LOW trust, useful for mechanic validation)**
- Endless.gg forum rogue resources
- Warmane forum EJ mirrors
- TrueWoW "EllE's Rogue Bible"

### Source Watcher (scheduler.py)

Runs on a configurable schedule:

| Source Type | Check Frequency |
|------------|----------------|
| Warcraft Logs (new top parses) | Every 6 hours |
| Blizzard news / hotfixes | Daily |
| Reddit / forum threads | Every 12 hours |
| Guide sites | Weekly |
| WoWSims repo (updates) | Weekly |

---

## 5. Storage Layer

### Philosophy

Two stores. Each doing what it's best at. Nothing redundant.

### SQLite (`data/shukketsu.db`)

One database file containing structured data, vector embeddings, and full-text search.

```sql
-- ============================================
-- STRUCTURED GAME DATA
-- ============================================

CREATE TABLE spells (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    rank INTEGER,
    energy_cost INTEGER,
    combo_points INTEGER,
    damage_min REAL,
    damage_max REAL,
    coefficient REAL,
    cooldown REAL,
    effect_type TEXT,
    description TEXT
);

CREATE TABLE items (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    slot TEXT,
    item_level INTEGER,
    quality TEXT,          -- common, uncommon, rare, epic, legendary
    phase INTEGER,
    stats_json TEXT,       -- {"stamina": 30, "agility": 40, "hit_rating": 23}
    source TEXT,           -- "Gruul the Dragonkiller"
    drop_rate REAL,
    icon_path TEXT
);

CREATE TABLE talents (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    tree TEXT,             -- assassination, combat, subtlety
    tier INTEGER,
    column_pos INTEGER,
    max_rank INTEGER,
    effect_per_rank TEXT,
    prereq_talent_id INTEGER REFERENCES talents(id),
    icon_path TEXT
);

CREATE TABLE talent_builds (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    spec TEXT,
    build_string TEXT,     -- "20/41/0"
    talents_json TEXT,     -- detailed point allocation
    source TEXT,
    usage_pct REAL,        -- % of top parses using this
    context TEXT           -- "pve_raid", "pvp_arena_2v2", etc.
);

CREATE TABLE bosses (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    zone TEXT,
    phase INTEGER,
    level INTEGER DEFAULT 73,
    armor INTEGER,
    health INTEGER,
    fight_duration_avg REAL,
    mechanics_json TEXT,
    rogue_notes TEXT
);

-- ============================================
-- INGESTED CONTENT (RAG)
-- ============================================

CREATE TABLE sources (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE,
    title TEXT,
    source_type TEXT,      -- guide, forum, wiki, video_transcript, etc.
    trust_score REAL,      -- 0.0 to 1.0
    fetched_at TIMESTAMP,
    content_hash TEXT,     -- for deduplication
    chunk_count INTEGER
);

CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    source_id INTEGER REFERENCES sources(id),
    content TEXT NOT NULL,
    chunk_index INTEGER,
    metadata_json TEXT     -- {"author": "...", "date": "...", "topic": "..."}
);

-- Vector search via sqlite-vec
CREATE VIRTUAL TABLE chunks_vec USING vec0(
    id INTEGER PRIMARY KEY,
    embedding FLOAT[768]
);

-- Full-text search via FTS5
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    content=chunks,
    content_rowid=id
);

-- ============================================
-- AGENT KNOWLEDGE MANAGEMENT
-- ============================================

CREATE TABLE articles (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,          -- "specs/combat/overview.md"
    title TEXT,
    last_updated TIMESTAMP,
    confidence_score REAL,     -- 0.0 to 1.0
    verified_claims INTEGER,
    unverified_claims INTEGER,
    needs_review BOOLEAN DEFAULT FALSE
);

CREATE TABLE conflicts (
    id INTEGER PRIMARY KEY,
    topic TEXT,
    source_a_id INTEGER REFERENCES sources(id),
    source_b_id INTEGER REFERENCES sources(id),
    claim_a TEXT,
    claim_b TEXT,
    resolution TEXT,
    resolved_by TEXT,          -- "sim", "higher_trust", "newer_data", "human"
    resolved_at TIMESTAMP
);

CREATE TABLE log_summaries (
    id INTEGER PRIMARY KEY,
    player TEXT,
    spec TEXT,
    fight TEXT,
    boss TEXT,
    duration REAL,
    dps REAL,
    gear_json TEXT,
    talents_json TEXT,
    rotation_breakdown_json TEXT,
    warcraft_logs_url TEXT,
    parsed_at TIMESTAMP
);

-- ============================================
-- SOURCE TRUST LEVELS
-- ============================================
-- Stored in trust.py as constants:
--   game_data:       1.0   (Blizzard API, CMaNGOS)
--   simulation:      0.9   (WoWSims, our sim engine)
--   combat_logs:     0.85  (Warcraft Logs parses)
--   expert_guide:    0.75  (Icy Veins, ShadowPanther, Sno)
--   archived_theory: 0.7   (EJ Roguecraft 101)
--   community:       0.5   (Reddit, forums, Discord)
--   unknown:         0.3   (Unclassified web content)
```

### Git Markdown Wiki (`knowledge/`)

Agent-written articles stored as Markdown, tracked in git:

```
knowledge/
├── specs/
│   ├── combat/
│   │   ├── overview.md         # Combat Rogue primer
│   │   ├── rotation.md         # Rotation priority + SnD management
│   │   ├── talents.md          # Talent builds + variations
│   │   └── tips.md             # Advanced tricks + optimization
│   ├── assassination/
│   │   ├── overview.md
│   │   ├── rotation.md
│   │   ├── talents.md
│   │   └── poison-mechanics.md
│   └── subtlety/
│       ├── overview.md
│       ├── pvp-guide.md
│       ├── talents.md
│       └── arena-matchups.md
├── gearing/
│   ├── combat-p1-bis.md
│   ├── combat-p2-bis.md
│   ├── ... (per spec, per phase)
│   ├── stat-weights.md
│   ├── gems-enchants.md
│   └── profession-choices.md
├── encounters/
│   ├── karazhan/
│   │   ├── attumen.md
│   │   ├── moroes.md
│   │   ├── ... (per boss)
│   │   └── prince-malchezaar.md
│   ├── gruul/
│   └── magtheridon/
├── pvp/
│   ├── arena/
│   │   ├── rmp.md              # Rogue/Mage/Priest
│   │   ├── rogue-druid.md
│   │   ├── shadowstep-guide.md
│   │   └── season-1-meta.md
│   ├── battlegrounds.md
│   └── honor-gear.md
└── fundamentals/
    ├── hit-cap.md
    ├── expertise.md
    ├── crit-and-agi.md
    ├── haste-breakpoints.md
    ├── armor-penetration.md
    ├── weapon-speed.md
    ├── energy-system.md
    ├── combo-points.md
    ├── poisons.md
    └── stealth-mechanics.md
```

Each article includes a YAML frontmatter block:

```yaml
---
title: "Combat Rogue P1 BiS Gear"
confidence: 0.87
last_verified: 2026-02-09
sources:
  - "Warcraft Logs top 100 Karazhan parses"
  - "WoWSims validation run"
  - "ShadowPanther P1 gear chart"
tags: [combat, gearing, p1, bis]
needs_review: false
---
```

---

## 6. Agent Core

### Brain Architecture

```
agent/
├── brain.py        # Main loop — decides mode, dispatches tools, assembles responses
├── planner.py      # Decomposes complex tasks into ordered steps
├── context.py      # Assembles prompt context within token budget
├── trust.py        # Source trust hierarchy + conflict resolution rules
├── verifier.py     # Validates claims before wiki publication
└── tools/
    └── registry.py # Maps tool names → callables, generates tool descriptions
```

### Agent Loop (brain.py)

Two modes, selected automatically based on task complexity:

**ReAct Mode** — for direct questions:
```
loop:
    1. THINK:   Nemotron reasons about what it knows and what it needs
    2. DECIDE:  Pick a tool to call (or answer directly)
    3. ACT:     Execute the tool
    4. OBSERVE: Add result to context
    → repeat until answer is ready
```

**Plan-then-Execute Mode** — for complex research tasks:
```
    1. PLAN:    Nemotron breaks the task into ordered steps
    2. For each step:
        a. Execute using ReAct sub-loop
        b. Store intermediate results
    3. SYNTHESIZE: Combine all step results into final output
    4. VERIFY:  Check claims against sources + sim data
    5. PUBLISH: Write/update wiki article if applicable
```

### Tool Registry (tools/registry.py)

Each tool is a Python function with a docstring. The registry auto-generates tool descriptions for the system prompt:

```python
@tool("web_search")
def web_search(query: str) -> list[SearchResult]:
    """Search the web for information. Returns titles, URLs, and snippets."""

@tool("web_ingest")
def web_ingest(url: str) -> IngestResult:
    """Fetch a web page, extract content, chunk it, embed it, and store
    in the knowledge base. Returns chunk count and summary."""

@tool("rag_search")
def rag_search(query: str, top_k: int = 10) -> list[Chunk]:
    """Search the knowledge store using hybrid vector + keyword search.
    Returns relevant content chunks with source metadata and trust scores."""

@tool("db_query")
def db_query(sql: str) -> list[dict]:
    """Query the structured game database (items, spells, talents, bosses).
    Read-only. Returns rows as dictionaries."""

@tool("sim_run")
def sim_run(config: SimConfig) -> SimResult:
    """Run a DPS simulation with given spec, gear, talents, buffs, and
    fight parameters. Returns DPS, breakdown, and stat weights."""

@tool("log_analyze")
def log_analyze(log_url: str) -> LogReport:
    """Analyze a Warcraft Logs report for a rogue player. Compares rotation,
    cooldown usage, and DPS against optimal benchmarks."""

@tool("wiki_write")
def wiki_write(path: str, content: str) -> WriteResult:
    """Create or update a wiki article. Content should be Markdown with
    YAML frontmatter. Triggers verification before commit."""

@tool("wiki_read")
def wiki_read(path: str) -> str:
    """Read an existing wiki article's content."""
```

### Context Assembly (context.py)

Even with 1M token context, focused prompts produce better results:

```
Default budget: ~50K tokens per agent call
Reserved for: system prompt (2K) + tool descriptions (2K) + conversation (varies)

Assembly priority:
  1. System prompt + tool descriptions           (always included)
  2. RAG search results (most relevant chunks)    (5-10 chunks)
  3. Relevant structured data (SQL query results) (specific rows)
  4. Current wiki article (if updating)           (full article)
  5. Conversation history (for chat mode)         (last N turns)

Full 1M context reserved for special cases:
  - Ingesting entire long guides or forum threads
  - Analyzing full combat logs
  - Cross-referencing large datasets
```

### Source Trust (trust.py)

```python
SOURCE_TRUST = {
    "game_data":       1.0,   # Blizzard API, CMaNGOS spell data
    "simulation":      0.9,   # WoWSims, our own sim engine
    "combat_logs":     0.85,  # Warcraft Logs actual parses
    "expert_guide":    0.75,  # Icy Veins, ShadowPanther, Sno, Simonize
    "archived_theory": 0.7,   # EJ Roguecraft 101 (correct but may be outdated)
    "community":       0.5,   # Reddit, forums, Discord
    "unknown":         0.3,   # Unclassified web content
}
```

Conflict resolution strategy (encoded in system prompt):
1. Higher-trust source wins by default
2. Newer data preferred for meta-dependent questions
3. If simulatable, run sim to settle empirically
4. If still ambiguous, present both views + flag for human review

### Self-Verification (verifier.py)

Before any wiki article is committed:
1. Each factual claim checked against at least 2 sources
2. DPS/stat claims validated with sim run where possible
3. Unverified claims tagged with `[UNVERIFIED]` in markdown
4. Article confidence score computed and stored
5. Conflicts logged in `conflicts` table

---

## 7. DPS Simulation Engine

### Architecture

```
sim/
├── mechanics.py    # Core combat formulas (TBC 2.4.3 rules)
├── abilities.py    # Rogue ability database (damage, cost, cooldown, coefficients)
├── talents.py      # Talent tree with point allocation + modifier calculation
├── items.py        # Gear stats, set bonuses, proc effects (DST, Mongoose, etc.)
├── buffs.py        # Raid buffs, consumables, boss debuffs
├── rotation.py     # Per-spec priority system (Combat, Assassination, Subtlety)
├── combat.py       # Main simulation loop (discrete event, millisecond resolution)
└── runner.py       # Public API: accepts SimConfig, returns SimResult
```

### Core Mechanics Modeled

| Mechanic | Detail |
|----------|--------|
| Hit table | Two-roll system for white damage, one-roll for yellow (abilities) |
| Miss/Dodge/Glancing | Level-based rates against boss (level 73) |
| Crit calculation | Base crit + agility scaling + talents + buffs |
| Weapon speed normalization | Normalized speed for instant attacks (2.4 daggers, 2.4 swords/fists/maces) |
| Armor and penetration | Boss armor (7700 base), reduction formula, Sunder/FF/CoR debuffs |
| Dual wield penalty | +19% miss for dual wield, reduced by hit rating + Precision |
| Sword Specialization | Extra attack proc (1-5% based on talent), can chain-proc |
| Combat Potency | 20% chance on OH hit to gain 15 energy |
| Seal Fate | Extra combo point on crit (Assassination) |
| Deadly Poison stacking | 5-stack mechanic + Envenom consumption |
| Instant Poison procs | PPM-based proc rate, affected by weapon speed |
| Slice and Dice | Attack speed buff, uptime tracking is #1 priority |
| Blade Flurry | Cleave mechanic, hits additional target for 100% damage |
| Adrenaline Rush | Double energy regen for 15 seconds |
| Expose Armor vs Sunder | Raid DPS tradeoff modeling |
| Energy regeneration | Fixed 20 energy per 2-second tick |
| Combo point generation | 1 per builder + Seal Fate + Ruthlessness |
| Relentless Strikes | 20% energy refund per CP on finisher (1-5 CP scaling) |

### Data Sources for Sim Accuracy

Cross-referenced for correctness:
1. **CMaNGOS TBC Database** (GitHub SQL) — raw spell coefficients, proc rates
2. **WoWSims TBC Rogue** (GitHub Go source) — validated combat model
3. **Warcraft Logs data** — real-world validation of sim output

### Implementation Strategy

```
Phase 3a: Wrap WoWSims as a callable tool (WASM or web API)
          → Gives sim capability immediately

Phase 3b: Build our own Python sim referencing WoWSims + CMaNGOS
          → Deeper understanding, customizable

Phase 3c: Validate our sim against WoWSims output + real WCL parses
          → Ensure accuracy within 1-2% of real data
```

### Sim Runner API

```python
config = SimConfig(
    spec="combat_swords",
    talents="20/41/0",
    gear={...},                    # Item IDs per slot
    buffs=["blessing_of_kings", "windfury_totem", "battle_shout"],
    consumables=["haste_potion", "elixir_of_major_agility", "roasted_clefthoof"],
    boss=BossConfig(name="Prince Malchezaar", armor=7700, level=73),
    fight_length=300,              # 5 minutes
    iterations=10000,
    seed=None                      # Random seed for reproducibility
)

result = sim_run(config)
# result.dps_mean = 1847.3
# result.dps_std = 42.1
# result.dps_min = 1698.2
# result.dps_max = 2031.7
# result.breakdown = {"Sinister Strike": 28.3%, "Melee (MH)": 24.1%, ...}
# result.stat_weights = {"agility": 1.0, "hit_rating": 1.21, "haste_rating": 0.87, ...}
# result.timeline = [...events...]
```

---

## 8. Web Knowledgebase

### Design Philosophy

Custom WoW Rogue-themed UI. Server-rendered HTML (FastAPI + Jinja2 + HTMX) with vanilla JS for interactive components. Looks like a dedicated rogue application, not a generic docs site.

### Theme System

```css
:root {
    /* Base */
    --bg-primary: #0a0a0f;
    --bg-surface: #1a1a24;
    --bg-elevated: #242432;
    --text-primary: #e8dcc8;
    --text-secondary: #8a8a9a;
    --border-default: #2a2a3a;

    /* Rogue class color */
    --accent-rogue: #FFF468;

    /* Item quality colors (WoW standard) */
    --quality-poor: #9d9d9d;
    --quality-common: #ffffff;
    --quality-uncommon: #1eff00;
    --quality-rare: #0070dd;
    --quality-epic: #a335ee;
    --quality-legendary: #ff8000;

    /* UI accents */
    --border-gold: #4a3c28;
    --bg-tooltip: #1a0a2e;
    --border-tooltip: #6644aa;
}
```

### Five Interfaces

**1. Wiki Browser (`/wiki/`)**
- Renders git-tracked Markdown as themed HTML
- Confidence badge per article (verified / needs review / unverified)
- Revision history via git log
- Cross-linked related articles
- Item names colored by rarity, hoverable for tooltips

**2. Search (`/search/`)**
- Single search bar
- Hybrid results: FTS5 keyword + sqlite-vec semantic
- Results grouped: wiki articles first, then ingested sources, then structured data
- Trust badges on source results

**3. Chat (`/chat/`)**
- WebSocket real-time conversation with agent
- Streamed token-by-token responses
- Rich rendering: item tooltips, inline sim results, bar charts
- Conversation history preserved per session

**4. Log Analysis (`/logs/`)**
- Upload WoW combat log file or paste Warcraft Logs URL
- Report: DPS breakdown, rotation analysis, uptime tracking
- Comparison vs top percentile players
- Specific improvement suggestions
- Fight timeline visualization

**5. Sim Builder (`/sim/`)** *(Phase 4)*
- Visual spec selector + talent tree grid
- Gear slot dropdowns with item quality colors
- Buff/consumable checkboxes
- Run sim → see DPS result + stat weights + breakdown chart
- Click any talent to see live DPS impact

### WoW-Style Item Tooltips

Any item name anywhere on the site is hoverable:

```
┌──────────────────────────────────┐
│ Dragonspine Trophy               │  ← colored by quality
│ Binds when picked up             │
│ Unique                           │
│ Trinket                          │
│                                  │
│ Equip: Your melee and ranged     │
│ attacks have a chance to grant   │
│ 325 haste rating for 10 sec.    │
│                                  │
│ Drop: Gruul the Dragonkiller     │
│ Drop Rate: 15%                   │
├──────────────────────────────────┤
│ Shukketsu: BiS Combat P1        │  ← agent's annotation
│ ★ 92% usage in top parses       │
└──────────────────────────────────┘
```

The bottom annotation section is unique to Shukketsu — the agent's own metadata enriching every item tooltip.

---

## 9. Development Phases

### Phase 0: Infrastructure (Day 1)

```
[ ] Install Ollama on DGX Spark
[ ] Pull nemotron-3-nano:30b-a3b-fp16 (63GB)
[ ] Pull nomic-embed-text (~270MB)
[ ] Verify inference works from Python
[ ] Initialize shukketsu/ project structure
[ ] Set up SQLite database with full schema
[ ] Configure sqlite-vec and FTS5
[ ] FastAPI skeleton serving hello page at localhost:8000
[ ] Git init knowledge/ directory
```

Skills learned: Ollama model serving, SQLite, FastAPI basics, local LLM inference.

Milestone: Talk to Nemotron from Python, see a web page in browser.

### Phase 1: Agent Brain (Week 1)

```
[ ] ReAct agent loop (brain.py)
[ ] Tool registry + dispatch (registry.py)
[ ] web_search tool
[ ] web_ingest tool (fetch → chunk → embed → store)
[ ] rag_search tool (hybrid vector + FTS5)
[ ] db_query tool
[ ] Context assembly with token budgeting
[ ] WebSocket chat endpoint
[ ] Chat UI page with dark WoW theme
[ ] Basic CSS theme (colors.css, theme.css)
```

Skills learned: LLM agents (ReAct), tool-augmented generation, RAG, embeddings, WebSockets.

Milestone: Chat with Shukketsu in browser. It searches web, ingests pages, answers from knowledge store.

### Phase 2: Knowledge Building (Weeks 2-3)

```
[ ] Warcraft Logs API client (OAuth + GraphQL)
[ ] Blizzard Battle.net API client (OAuth + REST)
[ ] Import WoWSims rogue spell data
[ ] Import CMaNGOS base game data
[ ] Ingest first wave of high-value sources
[ ] Plan-then-Execute mode (planner.py)
[ ] Source trust scoring (trust.py)
[ ] Conflict resolution logic
[ ] wiki_writer tool + wiki_read tool
[ ] Self-verification system (verifier.py)
[ ] Agent writes first set of wiki articles
[ ] Wiki browser UI (markdown rendering + navigation)
[ ] Search UI (hybrid search page)
[ ] WoW-style item tooltips (tooltips.js + tooltips.css)
[ ] Item rarity color system (colors.css)
[ ] APScheduler for periodic source checks
```

Skills learned: OAuth, GraphQL, web scraping, RAG knowledge stores, agent planning, source evaluation.

Milestone: Real rogue knowledge. Browse wiki, search content, ask sourced questions. Auto-updates on schedule.

### Phase 3: Simulation Engine (Weeks 3-4)

```
[ ] Wrap WoWSims as callable tool (immediate sim capability)
[ ] Core combat mechanics in Python (mechanics.py)
[ ] Rogue ability modeling (abilities.py)
[ ] Talent tree system (talents.py)
[ ] Combat spec rotation logic
[ ] Assassination spec rotation logic
[ ] Subtlety spec rotation logic
[ ] Buff/debuff/consumable system (buffs.py)
[ ] Item proc modeling — DST, Mongoose, etc. (items.py)
[ ] sim_runner tool integration with agent
[ ] Sim API endpoint (/sim/)
[ ] Agent re-validates wiki articles using sim data
[ ] Log analysis tool (log_analyzer.py)
[ ] Log upload + analysis report UI
[ ] Validate our sim vs WoWSims output + real WCL parses
```

Skills learned: Discrete event simulation, game mechanic modeling, statistical analysis, model validation.

Milestone: Sim any gear/talent/rotation. Analyze logs vs optimal. Wiki articles backed by hard numbers.

### Phase 4: Polish & Growth (Ongoing)

```
[ ] Interactive talent tree UI (talent-tree.js + talent-tree.css)
[ ] Sim builder visual interface
[ ] Fight timeline visualization
[ ] DPS breakdown charts (Chart.js)
[ ] Gear comparison tool
[ ] Agent confidence dashboard
[ ] PvP/Arena section build-out
[ ] Agent proactive gap research
[ ] Export wiki as shareable static site
```

Skills learned: Data visualization, frontend interactivity, product design, self-improving AI.

Milestone: The complete Shukketsu experience.

---

## 10. Skills Learned

| Skill | Phase |
|-------|-------|
| Local LLM serving (Ollama) | 0 |
| Python web development (FastAPI) | 0-1 |
| Database design (SQLite + extensions) | 0-2 |
| LLM agent architecture (ReAct) | 1 |
| RAG pipelines (embeddings + hybrid search) | 1-2 |
| WebSocket real-time communication | 1 |
| API integration (OAuth, GraphQL, REST) | 2 |
| Web scraping + data ingestion | 2 |
| AI planning + task decomposition | 2 |
| Source evaluation + trust scoring | 2 |
| Discrete event simulation | 3 |
| TBC game mechanic modeling | 3 |
| Data validation (sim vs real) | 3 |
| Custom UI/UX design | 2-4 |
| Data visualization | 4 |
| Full-stack AI application design | All |

---

## Appendix: API Credentials Needed

| Service | Auth Type | How to Get |
|---------|----------|------------|
| Warcraft Logs v2 | OAuth2 (client credentials) | Create client at warcraftlogs.com/api/clients |
| Blizzard Battle.net | OAuth2 (client credentials) | Register at develop.battle.net |
| Ollama | None (local) | N/A |

---

*Shukketsu (出血) — bleeding. Every rogue's favorite debuff.*
