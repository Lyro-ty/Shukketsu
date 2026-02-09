# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Shukketsu (出血) is a local AI-powered multi-agent research system for the WoW TBC Rogue class. It runs on an NVIDIA DGX Spark inside an NVIDIA AI Workbench container (PyTorch 2.6, CUDA 12.6.3, Ubuntu 24.04, ARM64).

The project is in early development — directory structure and `__init__.py` files are scaffolded, but most modules are stubs. Key files with real code: `config.py`, `web/app.py`, `routing/models.py`, `resilience/errors.py`, `trust/scoring.py`, `tests/conftest.py`.

## Commands

```bash
# Run the web server (registered as Workbench app on port 9000)
python -m uvicorn code.shukketsu.web.app:app --host 0.0.0.0 --port 9000

# Tests
pytest tests/unit/ -v                        # Unit tests (no external deps)
pytest tests/integration/ -v -m integration  # Needs Ollama + SQLite
pytest tests/e2e/ -v -m e2e                  # Needs all services
pytest --cov=shukketsu --cov-report=html     # Full suite with coverage

# Linting
ruff check code/ tests/
ruff format code/ tests/
mypy code/shukketsu/
```

## Architecture

### Three-Model System

- **Llama 3.3 70B AWQ INT4** via vLLM (port 8000) — all substantive reasoning, tool calling, article writing
- **Qwen3 4B** via Ollama (port 11434) — fast query classification/routing, trivial answers
- **nomic-embed-text-v2** via Ollama (port 11434) — 768-dim embeddings for RAG

Both model servers are accessed via OpenAI-compatible HTTP APIs from Python using `instructor` + `openai` clients for structured output (Pydantic models).

### Multi-Agent System (no external framework)

Plain Python classes with ReAct loops, no external framework (LangChain, CrewAI, etc.). Phase 1 starts with a single **General Agent** that answers questions using tools. In Phase 2, this grows into five specialists:
- **Orchestrator** — decomposes complex tasks, coordinates other agents
- **Researcher** — information gathering (web search, APIs, RAG); promoted from General Agent
- **Analyst** — simulations, log analysis, quantitative work
- **Writer** — wiki article creation
- **Editor** — fact-checking, verification

Agents communicate via an in-memory async `MessageBus`. The Orchestrator routes queries through Qwen 4B first (trivial → answered directly, moderate → single agent, complex → multi-agent plan).

### Storage

Single SQLite file (`data/shukketsu.db`) with three extensions:
- **sqlite-vec** for vector similarity search (cosine distance)
- **FTS5** for keyword search (BM25)
- Results fused via Reciprocal Rank Fusion (RRF)

Wiki articles are git-tracked Markdown files in `knowledge/` with YAML frontmatter (confidence scores, sources, tags).

### Key Data Flow

User query → Qwen 4B router classifies complexity → if complex: Orchestrator decomposes into sub-tasks → specialist agents execute ReAct loops with tools → results synthesized → Editor verifies → response returned via WebSocket chat.

## Project Layout

```
code/shukketsu/          # Main Python package (import as code.shukketsu)
  config.py              # All env vars, model names, paths, agent defaults
  routing/               # Qwen 4B query classification
  agents/                # BaseAgent, Orchestrator, Researcher, Analyst, Writer, Editor
  llm/                   # vLLM + Ollama clients, Instructor integration, prompts
  tools/                 # Agent tools: research/, analysis/, knowledge/
  rag/                   # Agentic RAG: decomposer, iterative retrieval, self-RAG, corrective
  ingest/                # Chunking (semantic + WoW-specific) and embedding pipeline
  scraping/              # Rate limiter, robots.txt compliance, httpx fetcher
  sim/                   # TBC Rogue DPS simulation engine (discrete event)
  db/                    # SQLite connection, schema.sql, migrations
  web/                   # FastAPI app, Jinja2 templates, HTMX, static assets
  trust/                 # Source trust scoring with time-based decay
  freshness/             # Content staleness checking and re-ingestion
  resilience/            # Circuit breakers, retries, error taxonomy
  observability/         # Langfuse tracing integration
  evals/                 # RAG quality (Ragas), trajectory analysis, domain accuracy
  backup/                # SQLite online backup + integrity checks
tests/                   # pytest: unit/ (pure logic), integration/ (needs services), e2e/
knowledge/               # Git-tracked Markdown wiki articles
data/                    # Git-lfs: database, backups; scratch/ is gitignored
infra/                   # Docker Compose for Langfuse, vLLM/Ollama start scripts
```

## NVIDIA AI Workbench Conventions

- `apt.txt` — system packages installed during build
- `requirements.txt` — pip dependencies
- `variables.env` — environment variables (sourced into container; restart required on change)
- `preBuild.bash` — installs Node.js + Ollama before container build
- `postBuild.bash` — installs Claude Code CLI after build
- `.project/spec.yaml` — Workbench project config (apps, mounts, base image)
- `code/` is git-tracked, `models/` and `data/` use git-lfs, `data/scratch/` is gitignored

## Configuration

All config is read from environment variables in `code/shukketsu/config.py`. API keys for Brave Search, Warcraft Logs, and Blizzard go in `variables.env`. Model URLs default to localhost (vLLM :8000, Ollama :11434, Langfuse :3000).

## Testing Conventions

- `asyncio_mode = auto` in pytest.ini — async tests run automatically
- Markers: `@pytest.mark.integration`, `@pytest.mark.e2e`
- `conftest.py` provides a `test_db` fixture that creates a fresh SQLite DB from `db/schema.sql`
- Unit tests must have zero external dependencies (no network, no running services)

## Development Phases

Currently at **Phase 1** (agent core). The project planning is split across four documents in `docs/plans/`:

| Document | Purpose |
|----------|---------|
| `2026-02-09-shukketsu-design.md` | Original comprehensive design spec (full system vision, schema, all phases) |
| `shukketsu-architecture.md` | Architecture reference (design rationale, not implementation steps) |
| `phase-1-agent-core.md` | **Active plan** — 10-step implementation guide for Phase 1 |
| `phase-roadmap.md` | Lightweight outline of Phases 2-5 (detailed specs written per-phase) |

### Phase 1: Agent Core (10 steps)

The active implementation plan (`phase-1-agent-core.md`) builds the system incrementally:

1. Chat UI + streaming LLM (WebSocket + vLLM)
2. Database foundation (SQLite + schema + WAL mode)
3. Structured output (Instructor + Pydantic)
4. First tool + ReAct loop (BaseAgent + rag_search)
5. Ingest pipeline (chunking + embedding + storage)
6. Hybrid search (vector + FTS5 + RRF)
7. Multi-model router (Qwen 4B classification)
8. Web search + ingest tools (Brave API + scraping)
9. Resilience (circuit breakers, loop detection, retries)
10. Observability (Langfuse tracing)

**Phase gate**: Chat with agent in browser. It classifies queries, routes to correct model, calls tools, answers from knowledge base. Traces visible in Langfuse.

### Future Phases

- **Phase 2**: Multi-agent + knowledge building (Orchestrator, specialist agents, agentic RAG, API integrations, wiki)
- **Phase 3**: DPS simulation engine (TBC combat mechanics, validation vs WoWSims)
- **Phase 4**: Evaluation + observability polish (Ragas, trajectory eval, feedback)
- **Phase 5**: UI polish + growth (talent trees, sim builder, charts, PvP)
