# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Shukketsu (出血) is a local AI-powered multi-agent research system for the WoW TBC Rogue class. It runs on an NVIDIA DGX Spark inside an NVIDIA AI Workbench container (PyTorch 2.6, CUDA 12.6.3, Ubuntu 24.04, ARM64). Primary language is Python 3.12 with full type hints on all functions, using ruff for linting/formatting and mypy for type checking.

Phases 1 (Agent Core), 2 (Multi-Agent + Agentic RAG), 3 (Memory, Reflection, Performance), and 4 (DPS Simulation Engine) are complete (1,408 unit tests). Post-Phase 3 additions include a batch ingest engine, fast 7B model tier, GB10 timeout tuning, and the **WCL API integration** (168 tests: OAuth2 auth, rate-limit-aware GraphQL client, Pydantic models, DB schema v5, ingest orchestrator, CLI). All other modules contain real implementation code — see Project Layout for the full listing.

## Development Workflow

This project follows a phased, step-by-step implementation pattern. Each step follows: 1) Analyze project state → 2) Collaborative design/brainstorm → 3) Produce implementation plan document → 4) Execute plan using subagent-driven development → 5) Update CLAUDE.md. Always check which step/phase we're on before starting work.

## Commands

```bash
# Run the web server (registered as Workbench app on port 9000)
python3 -m uvicorn code.shukketsu.web.app:app --host 0.0.0.0 --port 9000

# Batch ingest from source manifest
python3 -m code.shukketsu.ingest.batch --manifest data/sources/manifest.yaml
python3 -m code.shukketsu.ingest.batch --browser --priority 1    # Playwright for JS-rendered pages
python3 -m code.shukketsu.ingest.batch --extract-only             # Entity extraction pass only

# WCL API ingest
python3 -m code.shukketsu.apis.wcl.ingest --sync-characters       # Sync Lyroo's latest reports
python3 -m code.shukketsu.apis.wcl.ingest --rankings --zone 1052  # Top 100 Rogue rankings
python3 -m code.shukketsu.apis.wcl.ingest --report TNtKz3G1H9kVAQr4  # Deep-dive a report
python3 -m code.shukketsu.apis.wcl.ingest --full                  # Full zone sweep
python3 -m code.shukketsu.apis.wcl.ingest --rate-limit            # Check rate limit status
python3 -m code.shukketsu.apis.wcl.ingest --stats                 # WCL database stats

# Tests — MUST use `python3 -m pytest` (bare `pytest` hits stdlib `code` module conflict)
python3 -m pytest tests/unit/ -v                        # Unit tests (no external deps)
python3 -m pytest tests/integration/ -v -m integration  # Needs Ollama + SQLite
python3 -m pytest tests/e2e/ -v -m e2e                  # Needs all services
python3 -m pytest --cov=shukketsu --cov-report=html     # Full suite with coverage

# Linting
ruff check code/ tests/
ruff format code/ tests/
mypy code/shukketsu/
```

## Coding Style

- **Imports**: stdlib → third-party → local, blank-line separated (ruff `I` rule)
- **Type annotations**: Full on all function signatures. Use `str | None` not `Optional[str]`, `list[str]` not `List[str]` (3.12 syntax)
- **Docstrings**: Google-style, brief. Required on modules, public classes, and public functions
- **Naming**: `UPPER_SNAKE` constants, `PascalCase` classes, `snake_case` functions/variables
- **Async**: `async def` for I/O (endpoints, LLM calls, DB, HTTP). Plain `def` for pure computation
- **Logging**: `logging.getLogger(__name__)` per module
- **Errors**: All exceptions inherit `ShukketsuError` with `FailureMode` enum (see `resilience/errors.py`)
- **Pydantic**: v2 `BaseModel` for data containers + LLM schemas. `str` enums for JSON round-trip
- **Line length**: 120 (ruff enforced)

## Architecture

### Model System (All via Ollama)

All models are served via Ollama on port 11434 (llama.cpp has native Blackwell/GB10 support; PyTorch CUDA kernels are not compiled for sm_121):

- **Llama 3.3 70B** via Ollama — all substantive reasoning, tool calling, article writing (`ModelBackend.REASONING`)
- **Qwen 2.5 7B** via Ollama — moderate complexity queries routed by the Researcher (`FAST_MODEL`)
- **Qwen3 4B** via Ollama — query classification/routing, trivial answers (`ModelBackend.ROUTER`)
- **nomic-embed-text** via Ollama — 768-dim embeddings for RAG
- **cross-encoder/ms-marco-MiniLM-L-6-v2** — sentence-transformers cross-encoder for search result reranking (CPU, lazy-loaded)

All accessed via Ollama's OpenAI-compatible HTTP API (`{OLLAMA_BASE_URL}/v1`) using `instructor` + `openai` clients for structured output (Pydantic models). The 7B tier handles moderate queries that are too complex for a direct 4B answer but don't need full 70B reasoning. Instructor clients use `max_retries=0` — retries are handled by our `@with_retry` decorator to avoid OpenAI client's 15-minute exponential backoff.

**Important**: The Qwen3 router uses a custom Modelfile (`infra/Modelfile.qwen3-router`) that disables thinking mode. Qwen3 generates `<think>` tokens by default which consume the `max_tokens` budget through Ollama's `/v1` endpoint. Create the model with: `ollama create qwen3-router -f infra/Modelfile.qwen3-router`

### Multi-Agent System (no external framework)

Plain Python classes with ReAct loops, no external framework (LangChain, CrewAI, etc.). Phase 1 has a single **General Agent** that answers questions using tools. Phase 2 adds four specialists via a **Structured Task Protocol** (typed Pydantic Task/Result objects, all communication flows through the Orchestrator):

- **Orchestrator** — decomposes complex tasks, dispatches to specialists, synthesizes results
- **Researcher** — information gathering (hybrid search, graph search, web search); promoted from General Agent
- **Writer** — wiki article creation from research findings
- **Editor** — fact-checking articles against knowledge base and graph
- **Analyst** — DPS simulation analysis (sim_run, sim_compare, sim_optimize tools)

The Orchestrator routes queries through Qwen 4B first (trivial → answered directly, moderate → Researcher solo via 7B, complex → multi-agent plan via 70B). Phase 3 added cross-session memory (recall/extraction), reflection passes, context compaction, parallel subtask execution, and evidence-based trust scoring.

### Storage

Single SQLite file (`data/shukketsu.db`) with pragmas: `journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=5000`, `foreign_keys=ON`. Three extensions:
- **sqlite-vec** for vector similarity search (cosine distance, `distance_metric=cosine` in v0.1.7+)
- **FTS5** for keyword search (BM25)
- Results fused via Reciprocal Rank Fusion (RRF)

The database also stores a **knowledge graph** (schema v5): `entity_types`, `entities` (deduplicated by canonical name + type), `relationships` (deduplicated by source + target + type), Phase 3 tables: `session_memories`, `strategy_memories`, `trust_events`, and WCL tables (v5): `wcl_tracked_characters`, `wcl_rankings`, `wcl_reports`, `wcl_fights`, `wcl_combatants`, `wcl_damage`, `wcl_buffs`, `wcl_casts`, `wcl_fight_rankings`, `wcl_character_log`. Entity names are normalized and resolved through an alias table for WoW abbreviations (e.g., DST → Dragonspine Trophy). Entity extraction is performed via Llama 70B structured output during ingest (best-effort, non-blocking).

Wiki articles are git-tracked Markdown files in `knowledge/` with YAML frontmatter (confidence scores, sources, tags).

### Key Data Flow

User query → Qwen 4B router classifies complexity → trivial: direct answer → moderate: Researcher via 7B → complex: Orchestrator decomposes into sub-tasks → specialist agents execute ReAct loops with tools (parallel where independent) → results synthesized → Editor verifies → response returned via WebSocket chat. Memory recall injects relevant context from prior sessions; memory extraction stores key facts after each response.

### Warcraft Logs API

The `apis/wcl/` module provides a rate-limit-aware GraphQL client for the WCL v2 API:
- **OAuth2 client credentials** via POST body params (not Basic Auth) — token cached ~360 days
- **Two endpoints**: `fresh.warcraftlogs.com` (Classic Fresh, Lyroo's server) and `classic.warcraftlogs.com` (TBC Classic/Anniversary)
- **Rate limiting**: 3,600 points/hour, auto-sleep when approaching budget, retry on 429
- **Circuit breaker**: `wcl_breaker` in `resilience/circuit_breaker.py`
- **Three ingest modes**: `RankingsIngestor` (top 100 per encounter), `ReportDiver` (deep-dive: gear, damage, buffs, casts, rankings), `CharacterSyncer` (Lyroo tracking)
- **Tracked character**: Lyroo-Nightslayer (US), WCL ID 104956434, fresh endpoint

## Project Layout

```
code/shukketsu/          # Main Python package (import as code.shukketsu)
  config.py              # All env vars, model names, paths, agent defaults
  routing/               # Qwen 4B query classification (models.py, router.py)
  agents/                # BaseAgent, Orchestrator, Researcher, Writer, Editor, Analyst, guardrails
  llm/                   # Ollama clients, Instructor integration, prompts/
  tools/                 # Agent tools: knowledge/ (search, graph_search), research/ (web_search, web_ingest), analysis/ (sim_run, sim_compare, sim_optimize)
  rag/                   # Agentic RAG: entities, graph (GraphStore), fusion, search, reranker
  ingest/                # pipeline.py (chunk+embed+extract), batch.py (concurrent/browser ingest), manifest.py
  scraping/              # Rate limiter, robots.txt compliance, httpx fetcher
  memory/                # Cross-session memory: models.py, manager.py (recall/extract/strategy)
  apis/wcl/              # Warcraft Logs v2 API: auth, client, queries, models, ingest, schema
  sim/                   # TBC Rogue DPS simulation engine: models, mechanics, abilities, talents, items, buffs, imports, rotation, combat, runner, validation
  db/                    # SQLite connection factory, schema.sql v5 (WAL, sqlite-vec, FTS5, graph, memory, WCL)
  web/                   # FastAPI app, Jinja2 templates, HTMX, wiki_render.py
  trust/                 # Evidence-based source trust scoring with trust_events
  freshness/             # Content staleness checking and re-ingestion
  resilience/            # Circuit breakers, retries, error taxonomy
  observability/         # Langfuse tracing integration
  knowledge/             # KnowledgeManager: wiki article CRUD, frontmatter, review/approve
  evals/                 # RAG quality (Ragas), trajectory analysis, domain accuracy, phase gate
  backup/                # SQLite online backup + integrity checks
tests/                   # pytest: unit/ (pure logic), integration/ (needs services), e2e/
knowledge/               # Git-tracked Markdown wiki articles
data/                    # Git-lfs: database, backups; sources/manifest.yaml; scratch/ is gitignored
infra/                   # Docker Compose for Langfuse, Ollama config, startup scripts
```

## NVIDIA AI Workbench Conventions

- `apt.txt` — system packages installed during build
- `requirements.txt` — pip dependencies
- `variables.env` — environment variables (sourced into container; restart required on change)
- `preBuild.bash` — installs Node.js + Ollama before container build
- `postBuild.bash` — installs Claude Code CLI after build
- `.project/spec.yaml` — Workbench project config (apps, mounts, base image)
- `code/` is git-tracked, `models/` and `data/` use git-lfs, `data/scratch/` is gitignored

## MCP Tools

The following MCP servers are configured and should be used during development:

- **GitHub** (`mcp__github`): Use for all GitHub operations — creating issues for tracking work, opening PRs, browsing repo state. Prefer `gh` CLI or MCP tools over manual git push workflows. Create tracking issues before starting multi-step work.
- **Context7** (`mcp__context7`): Look up current documentation for project dependencies (FastAPI, Pydantic, Instructor, sqlite-vec, Langfuse, httpx, pytest). Always resolve the library ID first with `resolve-library-id`, then query docs. Use this instead of guessing API signatures.
- **Playwright** (`mcp__playwright`): Use for e2e testing of the web UI at `http://localhost:9000`. Can verify WebSocket chat, page rendering, and UI interactions programmatically.
- **Filesystem** (`mcp__filesystem`): Available for file operations. Prefer Claude Code's native Read/Write/Edit tools for most work; use filesystem MCP when bulk operations or directory trees are needed.
- **Sequential Thinking** (`mcp__sequential-thinking`): Use for complex multi-step reasoning — debugging tricky issues, architectural decisions, or planning multi-file changes where you need to think through implications step by step.

## Configuration

All config is read from environment variables in `code/shukketsu/config.py`. API keys for Brave Search, Warcraft Logs, and Blizzard go in `variables.env`. Model URLs default to localhost (Ollama :11434, Langfuse :3000).

Key runtime values (tuned for GB10):
- `LLM_TIMEOUT_SECONDS = 300` — 70B structured output takes 2+ min on GB10
- `OBSERVATION_MAX_CHARS = 2000` — truncate large tool observations to keep context manageable
- `COMPACTION_THRESHOLD_TOKENS = 16000` — trigger scratchpad compaction above this
- `BATCH_CONCURRENCY = 3` — parallel HTTP fetches during batch ingest
- `RESEARCHER_MAX_ITERATIONS = 6` — balanced depth vs latency for GB10
- Domain freshness intervals: warcraftlogs.com=24h, wowhead/icy-veins=720h, shadowpanther/tbcdb=2160h

## Known Issues / Platform Gotchas

These are hard-won lessons from the DGX Spark environment. Check this section before debugging mysterious failures:

- **Cross-encoder reranker must use CPU**: The `cross-encoder/ms-marco-MiniLM-L-6-v2` model defaults to CUDA but sm_121 kernels aren't compiled. The reranker explicitly forces `device="cpu"` in `rag/reranker.py`. If adding new PyTorch models, always set `device="cpu"`.
- **sqlite-vec CTE queries need `AND k = ?`**: sqlite-vec cannot see `LIMIT` inside CTEs. KNN queries in CTEs must use `WHERE embedding MATCH ? AND k = ?` instead of `LIMIT`. The `MATCH` clause returns a bytes blob via `struct.pack(f"{dim}f", *embedding)`.
- **Langfuse Docker single-node**: Must set `CLICKHOUSE_CLUSTER_ENABLED: "false"` in docker-compose. Without it, ClickHouse migrations fail trying to create `ReplicatedMergeTree` tables (requires ZooKeeper/Keeper).
- **trafilatura fails on JS-heavy sites**: Sites like Wowhead render content via JavaScript. Use the Playwright browser ingest mode (`--browser`) with `wait_until="domcontentloaded"` (not `networkidle` which times out).
- **Qwen3 thinking tokens eat max_tokens**: Through Ollama's `/v1` endpoint, `<think>` tokens count against `max_tokens`. The router model uses a custom Modelfile that strips thinking. If adding new Qwen3 models, create a Modelfile without `<think>` in the template.
- **asyncio cooperative scheduling**: Python asyncio is single-threaded with cooperative multitasking. There are NO race conditions between synchronous operations in async code (no preemptive context switches without `await`). Don't add unnecessary locks around synchronous state checks.
- **Instructor `max_retries=0`**: Our `@with_retry` decorator handles retries externally. Setting `max_retries > 0` on the Instructor client triggers OpenAI's built-in exponential backoff (up to 15 min). Always use `max_retries=0` and wrap with `@with_retry` instead.
- **`time.monotonic()` cannot go backward**: The rate limiter uses `time.monotonic()` precisely because it's guaranteed non-decreasing. Don't add clock-skew handling for monotonic timestamps.

## Testing

Always use `python3 -m pytest` instead of bare `pytest` to avoid module import errors (Python's stdlib `code` module shadows the `code.shukketsu` package; `python3 -m pytest` adds CWD to `sys.path`).

- `asyncio_mode = auto` in `pyproject.toml` — async tests run automatically
- Markers: `@pytest.mark.integration`, `@pytest.mark.e2e`
- `conftest.py` provides two key fixtures:
  - `test_db(tmp_path)` — fresh SQLite DB per test with full schema, WAL mode, sqlite-vec, foreign keys (same `get_connection()` + `init_db()` as production)
  - `_reset_breakers()` (autouse) — resets all circuit breakers before/after each test to prevent state leakage between tests
- Unit tests must have zero external dependencies (no network, no running services)
- When running parallel sessions, check for orphaned pytest processes: `pkill -f pytest || true`

## Pre-Commit Checks

After implementing changes, always run the full check suite before committing:

```bash
ruff check . --fix && ruff format . && python3 -m mypy . && python3 -m pytest
```

## Git Workflow

- When pushing to GitHub, prefer SSH URLs (`git@github.com:...`) over HTTPS. If HTTPS fails, immediately switch to SSH without prompting.
- When committing, stage only the files relevant to the current task. Use `git add <specific-files>` rather than `git add .` to avoid accidentally staging unrelated changes.

## Development Phases

Phases 1-5 are complete (1,461 unit tests). Planning docs live in `docs/plans/`. Key references:
- `2026-02-09-shukketsu-design.md` — original comprehensive design spec
- `shukketsu-architecture.md` — architecture reference and design rationale
- `phase-roadmap.md` — lightweight outline of all phases

### Phase 1: Agent Core — COMPLETE (245 tests)

Chat UI, database, structured output, ingest pipeline, hybrid search, multi-model router, web search, resilience, observability.

### Phase 2: Multi-Agent + Agentic RAG — COMPLETE (741 tests)

Structured Task Protocol, knowledge graph + entity extraction, graph search + reranker, Researcher/Writer/Editor/Orchestrator agents, wiki UI, content freshness, automated backups, phase gate evaluation.

### Phase 3: Memory, Reflection, Performance — COMPLETE (778 tests)

Cross-encoder reranker, parallel subtask execution, streaming, context compaction, reflection passes, cross-session memory, strategy hints, trust scoring.

### Post-Phase 3 Additions

- Batch ingest engine with YAML manifest, concurrent/browser modes, entity extraction pass
- Fast 7B model tier, GB10 timeout tuning, WCL API integration (168 tests)

### Phase 4: DPS Simulation Engine — COMPLETE (421 tests)

Full discrete-event TBC 2.4.3 Rogue DPS simulation: combat mechanics, 18 abilities + poisons, 33 talents, 50 items, buff system, rotation engine, combat loop, SimRunner API (sim_run/compare/optimize), Analyst agent + 3 sim tools, web UI (Chart.js, HTMX), 9 validation profiles.

### Phase 5: Evaluation + Observability Polish — COMPLETE (53 tests)

Langfuse-primary eval pipeline: three-tier 60-question dataset (retrieval/reasoning/simulation), LLM-as-judge scoring (faithfulness, relevancy, trajectory precision, domain accuracy + sim accuracy), EvalRunner with same routing path as chat, user feedback via WebSocket → Langfuse scores, lean HTMX dashboard with Chart.js history, fine-tuning JSONL export (ShareGPT + function-calling formats).

### Future Phases

- **Phase 6**: UI polish + growth (talent trees, sim builder, charts, PvP)
- **Phase 7**: Advanced evaluation (A/B testing, regression detection, automated CI eval)
