# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Shukketsu (出血) is a local AI-powered multi-agent research system for the WoW TBC Rogue class. It runs on an NVIDIA DGX Spark inside an NVIDIA AI Workbench container (PyTorch 2.6, CUDA 12.6.3, Ubuntu 24.04, ARM64). Primary language is Python 3.12 with full type hints on all functions, using ruff for linting/formatting and mypy for type checking.

Phases 1 (Agent Core), 2 (Multi-Agent + Agentic RAG), and 3 (Memory, Reflection, Performance) are complete (813 tests). Post-Phase 3 additions include a batch ingest engine, fast 7B model tier, and GB10 timeout tuning. The `sim/` directory remains a stub (Phase 4). All other modules contain real implementation code — see Project Layout for the full listing.

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

All accessed via OpenAI-compatible HTTP APIs (`/v1`) using `instructor` + `openai` clients for structured output (Pydantic models). The 7B tier handles moderate queries that are too complex for a direct 4B answer but don't need full 70B reasoning.

### Multi-Agent System (no external framework)

Plain Python classes with ReAct loops, no external framework (LangChain, CrewAI, etc.). Phase 1 has a single **General Agent** that answers questions using tools. Phase 2 adds four specialists via a **Structured Task Protocol** (typed Pydantic Task/Result objects, all communication flows through the Orchestrator):

- **Orchestrator** — decomposes complex tasks, dispatches to specialists, synthesizes results
- **Researcher** — information gathering (hybrid search, graph search, web search); promoted from General Agent
- **Writer** — wiki article creation from research findings
- **Editor** — fact-checking articles against knowledge base and graph
- **Analyst** — deferred to Phase 4 (requires simulation engine)

The Orchestrator routes queries through Qwen 4B first (trivial → answered directly, moderate → Researcher solo via 7B, complex → multi-agent plan via 70B). Phase 3 added cross-session memory (recall/extraction), reflection passes, context compaction, parallel subtask execution, and evidence-based trust scoring.

### Storage

Single SQLite file (`data/shukketsu.db`) with three extensions:
- **sqlite-vec** for vector similarity search (cosine distance)
- **FTS5** for keyword search (BM25)
- Results fused via Reciprocal Rank Fusion (RRF)

The database also stores a **knowledge graph** (schema v4): `entity_types`, `entities` (deduplicated by canonical name + type), `relationships` (deduplicated by source + target + type), plus Phase 3 tables: `session_memories`, `strategy_memories`, `trust_events`. Entity names are normalized and resolved through an alias table for WoW abbreviations (e.g., DST → Dragonspine Trophy). Entity extraction is performed via Llama 70B structured output during ingest (best-effort, non-blocking).

Wiki articles are git-tracked Markdown files in `knowledge/` with YAML frontmatter (confidence scores, sources, tags).

### Key Data Flow

User query → Qwen 4B router classifies complexity → trivial: direct answer → moderate: Researcher via 7B → complex: Orchestrator decomposes into sub-tasks → specialist agents execute ReAct loops with tools (parallel where independent) → results synthesized → Editor verifies → response returned via WebSocket chat. Memory recall injects relevant context from prior sessions; memory extraction stores key facts after each response.

## Project Layout

```
code/shukketsu/          # Main Python package (import as code.shukketsu)
  config.py              # All env vars, model names, paths, agent defaults
  routing/               # Qwen 4B query classification (models.py, router.py)
  agents/                # BaseAgent, Orchestrator, Researcher, Writer, Editor, guardrails
  llm/                   # Ollama clients, Instructor integration, prompts/
  tools/                 # Agent tools: knowledge/ (search, graph_search), research/ (web_search, web_ingest)
  rag/                   # Agentic RAG: entities, graph (GraphStore), fusion, search, reranker
  ingest/                # pipeline.py (chunk+embed+extract), batch.py (concurrent/browser ingest), manifest.py
  scraping/              # Rate limiter, robots.txt compliance, httpx fetcher
  memory/                # Cross-session memory: models.py, manager.py (recall/extract/strategy)
  sim/                   # TBC Rogue DPS simulation engine — STUB (Phase 4)
  db/                    # SQLite connection factory, schema.sql v4 (WAL, sqlite-vec, FTS5, graph, memory)
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

## Testing

Always use `python3 -m pytest` instead of bare `pytest` to avoid module import errors (Python's stdlib `code` module shadows the `code.shukketsu` package; `python3 -m pytest` adds CWD to `sys.path`).

- `asyncio_mode = auto` in `pyproject.toml` — async tests run automatically
- Markers: `@pytest.mark.integration`, `@pytest.mark.e2e`
- `conftest.py` provides a `test_db` fixture via `db/connection.py` (WAL mode, sqlite-vec, foreign keys, full schema)
- Unit tests must have zero external dependencies (no network, no running services)

## Pre-Commit Checks

After implementing changes, always run the full check suite before committing:

```bash
ruff check . --fix && ruff format . && python3 -m mypy . && python3 -m pytest
```

## Git Workflow

- When pushing to GitHub, prefer SSH URLs (`git@github.com:...`) over HTTPS. If HTTPS fails, immediately switch to SSH without prompting.
- When committing, stage only the files relevant to the current task. Use `git add <specific-files>` rather than `git add .` to avoid accidentally staging unrelated changes.

## Development Phases

Phases 1-3 are complete (813 unit tests). Post-Phase 3 work: batch ingest engine, 7B model tier, GB10 timeout tuning. Planning docs live in `docs/plans/`:

| Document | Purpose |
|----------|---------|
| `2026-02-09-shukketsu-design.md` | Original comprehensive design spec (full system vision, schema, all phases) |
| `shukketsu-architecture.md` | Architecture reference (design rationale, not implementation steps) |
| `phase-1-agent-core.md` | Phase 1 implementation guide (complete) |
| `phase-2-multi-agent-rag.md` | Phase 2 implementation guide (complete) |
| `2026-02-10-phase2-step2-knowledge-graph.md` | Step 2 detailed plan (complete) |
| `2026-02-10-phase2-step3-graph-search-reranker.md` | Step 3 design doc (complete) |
| `2026-02-10-phase2-step3-implementation.md` | Step 3 implementation plan (complete) |
| `2026-02-10-deployment-setup.md` | Deployment setup (complete) |
| `2026-02-11-phase2-step4-researcher-agent.md` | Step 4 design doc (Researcher agent, structuring pass, prompts) |
| `2026-02-11-phase2-step4-implementation.md` | Step 4 implementation plan (complete) |
| `2026-02-11-phase2-step5-writer-wiki.md` | Step 5 design doc (Writer agent, KnowledgeManager, schema v3) |
| `2026-02-11-phase2-step5-implementation.md` | Step 5 implementation plan (complete) |
| `2026-02-11-phase2-step6-editor-agent.md` | Step 6 design doc (Editor agent, claim verification, confidence scoring) |
| `2026-02-11-phase2-step7-orchestrator-agent.md` | Step 7 design doc (Orchestrator agent, task decomposition, dispatch, synthesis) |
| `2026-02-11-phase2-step7-implementation.md` | Step 7 implementation plan (complete) |
| `2026-02-11-phase2-step8-wiki-ui.md` | Step 8 design doc (Wiki UI, article browser, review/approve flow, HTMX) |
| `2026-02-11-phase2-step8-implementation.md` | Step 8 implementation plan (complete) |
| `2026-02-11-phase2-step9-freshness-backups.md` | Step 9 design doc (freshness checker, backup manager, API endpoints) |
| `2026-02-11-phase2-step9-implementation.md` | Step 9 implementation plan (complete) |
| `2026-02-12-phase2-step10-integration-eval.md` | Step 10 design doc (integration + phase gate evaluation) |
| `2026-02-12-phase2-step10-implementation.md` | Step 10 implementation plan (complete) |
| `2026-02-12-phase3-memory-performance.md` | Phase 3 design doc (complete) |
| `2026-02-12-phase3-step1-cross-encoder.md` | Step 1 implementation plan (cross-encoder reranker, complete) |
| `2026-02-12-phase3-steps2-5-implementation.md` | Steps 2-5 implementation plan (parallel, streaming, compaction, reflection, complete) |
| `2026-02-12-phase3-steps6-9-implementation.md` | Steps 6-9 implementation plan (memory, integration, strategy, trust, complete) |
| `phase-roadmap.md` | Lightweight outline of Phases 2-5 (detailed specs written per-phase) |

### Phase 1: Agent Core — COMPLETE

All 10 steps done. 245 unit tests passing. System deployed and accessible in browser.

### Phase 2: Multi-Agent + Agentic RAG (10 steps) — COMPLETE

The active implementation plan (`phase-2-multi-agent-rag.md`) builds on Phase 1:

1. ~~Structured Task Protocol + Agent Framework~~ — COMPLETE (302 tests: tasks.py, base.py refactor, factory.py)
2. ~~Knowledge Graph Schema + Entity Extraction~~ — COMPLETE (381 tests: schema v2, entity types/aliases, GraphStore, extraction pipeline)
3. ~~Graph Traversal Tool + Qwen 4B Reranking~~ — COMPLETE (438 tests: graph_search tool, reranker, rag_search reranking)
4. ~~Researcher Agent~~ — COMPLETE (473 tests: Researcher subclass, structuring pass, prompts, factory registry)
5. ~~Writer Agent + Wiki Backend~~ — COMPLETE (527 tests: Writer agent, KnowledgeManager, schema v3, YAML frontmatter, two-pass generation)
6. ~~Editor Agent~~ — COMPLETE (570 tests: Editor subclass, claim verification, confidence scoring, VerificationStatus enum, ClaimJudgment/ClaimVerification/EditResult models, entity matching, frontmatter update)
7. ~~Orchestrator Agent~~ — COMPLETE (620 tests: Orchestrator subclass, 3-phase execute, plan validation, topological sort, task building, research merge, synthesis, factory registration, chat routing by complexity)
8. ~~Wiki UI~~ — COMPLETE (652 tests: wiki routes, markdown render, KM extensions, article browser, review/approve, HTMX fragments)
9. ~~Content Freshness + Automated Backups~~ — COMPLETE (690 tests: freshness checker, backup manager, API routes, ingest integration, wiki stale badge)
10. ~~Integration + Phase Gate Evaluation~~ — COMPLETE (741 tests: ToolCallRecord trajectory, eval metrics/judge/runner, 30-question dataset, seed content, tier 1-3 integration tests)

**Phase gate**: Complex multi-part question → Orchestrator decomposes → specialists cooperate → wiki articles produced and verified → traces in Langfuse. RAG faithfulness >= 0.8, trajectory precision >= 0.7, domain accuracy >= 0.7.

### Phase 3: Memory, Reflection, and Performance (9 steps) — COMPLETE

All 9 steps done. 778 tests at completion, now 813 with post-phase additions (batch ingest, manifest, 7B routing).

**Post-Phase 3 additions** (not part of a numbered phase):
- Batch ingest engine (`ingest/batch.py`) with YAML manifest, concurrent/browser modes, entity extraction pass
- Fast 7B model tier (`FAST_MODEL = qwen2.5:7b`) for moderate-complexity queries
- GB10 timeout tuning (`LLM_TIMEOUT_SECONDS = 300s`, `RESEARCHER_MAX_ITERATIONS = 6`)
- Source manifest at `data/sources/manifest.yaml` with 20+ TBC Rogue content URLs

### Future Phases

- **Phase 4**: DPS simulation engine (TBC combat mechanics, validation vs WoWSims)
- **Phase 5**: Evaluation + observability polish (Ragas, trajectory eval, feedback)
- **Phase 6**: UI polish + growth (talent trees, sim builder, charts, PvP)
