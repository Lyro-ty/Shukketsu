# Shukketsu — TBC Rogue Research Agent

> The premier World of Warcraft TBC Anniversary Rogue application.
> A local AI-powered multi-agent research system that searches, analyzes, simulates, and synthesizes
> everything about the Rogue class in The Burning Crusade.

## Context

- **Hardware**: NVIDIA DGX Spark (GB10 Grace Blackwell, 128GB unified LPDDR5x memory, ARM64, 6144 CUDA cores, 192 Tensor cores)
- **Environment**: NVIDIA AI Workbench container (PyTorch 2.6, CUDA 12.6.3, Ubuntu 24.04)
- **Game**: TBC Classic Anniversary Edition (launched February 5, 2026)
- **Current Phase**: Phase 1 (Karazhan, Gruul, Magtheridon — raids opened Feb 19)
- **TBC Anniversary Changes**: Dual Spec available early, no GDKP, Guild Banks from start

## Goals

1. **Comprehensive coverage** of all 3 Rogue specs (Combat, Assassination, Subtlety) across all TBC phases, PvE and PvP
2. **Multi-agent research** — specialized agents discover, ingest, analyze, and synthesize information collaboratively
3. **Empirical verification** — claims backed by simulation data and real combat logs
4. **Living knowledgebase** — a browsable wiki + searchable knowledge store + chat interface
5. **Always current** — monitors sources for new data, re-evaluates conclusions as the meta evolves
6. **Observable and measurable** — every agent decision traced, every output evaluated, continuous improvement
7. **Educational** — every component designed to teach modern AI engineering, web dev, simulation, and data engineering

---

## 1. Architecture Overview

```
                           User Request
                                │
                     ┌──────────┴──────────┐
                     │    FastAPI Gateway    │
                     │  (web, chat, API)     │
                     └──────────┬──────────┘
                                │
                     ┌──────────┴──────────┐
                     │   Model Router       │
                     │   Qwen3 4B (Ollama)  │─── Classify complexity + category
                     └──────────┬──────────┘
                         │              │
                    TRIVIAL        MODERATE/COMPLEX
                    Qwen 4B            │
                    answers       ┌────┴────┐
                    directly      │Orchestrator│
                                  └────┬────┘
                         ┌─────────┬───┴───┬──────────┐
                         │         │       │          │
                    ┌────┴───┐ ┌──┴───┐ ┌─┴────┐ ┌──┴────┐
                    │Research│ │Analyst│ │Writer│ │Editor │
                    │ Agent  │ │ Agent │ │Agent │ │ Agent │
                    └───┬────┘ └──┬───┘ └──┬───┘ └──┬───┘
                        │         │        │        │
              All agents use Llama 3.3 70B (vLLM) for reasoning
                        │         │        │        │
                    ┌───┴─────────┴────────┴────────┴───┐
                    │          Tool Layer                 │
                    │  web_search · web_ingest · rag_search │
                    │  db_query · sim_run · log_analyze   │
                    │  wiki_write · wiki_read · wcl_query │
                    │  blizzard_api · stat_weights        │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────┴──────────────────┐
                    │         Storage Layer            │
                    │  SQLite (structured + vec + FTS) │
                    │  Git Markdown (wiki articles)    │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┼───────────────────────┐
              │                    │                       │
    ┌─────────┴────────┐  ┌──────┴──────┐  ┌────────────┴───────────┐
    │   Observability   │  │  Evaluation  │  │  Web Knowledgebase     │
    │   Langfuse        │  │  Ragas +     │  │  FastAPI + Jinja2 +    │
    │   (traces, spans, │  │  Trajectory  │  │  HTMX                  │
    │    scores)        │  │  + Domain    │  │  Wiki · Search · Chat  │
    └──────────────────┘  └─────────────┘  │  Logs · Sim · Evals    │
                                            └────────────────────────┘
```

---

## 2. Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM (reasoning) | Llama 3.3 70B Instruct AWQ INT4 via vLLM | 35GB weights, strong instruction following, native function calling, OpenAI-compatible API. Fits DGX Spark with ~56GB headroom. |
| LLM (routing) | Qwen3 4B Instruct via Ollama | ~2.5GB, sub-200ms classification. Handles trivial queries directly, routes complex ones to 70B. |
| LLM (embeddings) | nomic-embed-text-v2 via Ollama | ~500MB, 768-dim embeddings for RAG. Runs alongside router on Ollama. |
| Structured output | Instructor + Pydantic | Typed tool calling with automatic retry on validation failure. Works with both vLLM and Ollama. |
| Web framework | FastAPI + Uvicorn | Async, WebSocket support, production Python framework |
| Templating | Jinja2 + HTMX | Server-rendered HTML with interactivity. No JS build chain. |
| Styling | Custom WoW-themed CSS | Dark theme, rogue class colors, item rarity colors, WoW-style tooltips |
| Charts | Chart.js (~60KB CDN) | DPS breakdowns, stat comparisons, timeline visualizations |
| Database | SQLite + sqlite-vec + FTS5 | One file for structured data + vector search + full-text search |
| Wiki storage | Git-tracked Markdown | Version-controlled articles with revision history |
| Web scraping | httpx + BeautifulSoup4 | Async HTTP + HTML parsing |
| Web search | Brave Search API | Independent index, free tier (2,000/month), no tracking |
| Task scheduling | APScheduler | Periodic research tasks, freshness checks, backups |
| Observability | Langfuse (self-hosted) | MIT license, OpenTelemetry-native LLM tracing, built-in UI |
| RAG evaluation | Ragas | Faithfulness, context precision/recall, answer relevancy |
| SQL validation | sqlparse | Prevent injection in agent-generated SQL |
| Simulation | Custom Python engine + NumPy | TBC rogue combat mechanics modeled from first principles |
| Testing | pytest + pytest-asyncio + pytest-cov | Full test pyramid |
| Linting | Ruff + mypy | Fast linting + type checking |

### Memory Budget (DGX Spark, 128GB unified)

```
Total memory:                          128 GB
─ OS + CUDA runtime + drivers:          -6 GB
─ Application (Python, FastAPI, SQLite): -2 GB
= Available for models:                120 GB

Model allocations:
─ Llama 3.3 70B AWQ INT4 weights:      35 GB
─ Llama 3.3 70B KV cache (32K ctx):    25 GB
─ Qwen3 4B Q4_K_M weights + KV:         3.5 GB
─ nomic-embed-text-v2 FP16:             0.5 GB
= Total model memory:                  64 GB

Remaining headroom:                     56 GB
  → vLLM CUDA graphs + paged attention: ~4 GB
  → SQLite in-memory cache:             ~2 GB
  → Python heap (scraping, parsing):    ~4 GB
  → Spike capacity (longer contexts):  ~46 GB free
```

---

## 3. NVIDIA AI Workbench Integration

The project runs inside an AI Workbench container. All configuration respects the Workbench conventions.

### Workbench Configuration Files

**`apt.txt`** — System packages installed during container build:
```
build-essential
libsqlite3-dev
```

**`requirements.txt`** — Python dependencies installed via pip:
```
jupyterlab>3.0
fastapi>=0.115
uvicorn[standard]>=0.34
jinja2>=3.1
httpx>=0.28
beautifulsoup4>=4.12
markdown>=3.7
apscheduler>=3.10
instructor>=1.7
openai>=1.60
ollama>=0.5
sqlite-vec>=0.1
sqlparse>=0.5
ragas>=0.2
langfuse>=3.0
numpy>=2.0
pytest>=8.0
pytest-asyncio>=0.24
pytest-cov>=6.0
ruff>=0.9
mypy>=1.14
pydantic>=2.10
```

**`variables.env`** — Environment variables (API keys, service URLs):
```
TENSORBOARD_LOGS_DIRECTORY=/data/tensorboard/logs/

# Model serving
VLLM_BASE_URL=http://localhost:8000/v1
OLLAMA_BASE_URL=http://localhost:11434

# API credentials (fill in your values)
BRAVE_SEARCH_API_KEY=
WCL_CLIENT_ID=
WCL_CLIENT_SECRET=
BLIZZARD_CLIENT_ID=
BLIZZARD_CLIENT_SECRET=

# Langfuse (self-hosted)
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=local-pk
LANGFUSE_SECRET_KEY=local-sk

# Database
SHUKKETSU_DB_PATH=/project/data/shukketsu.db
SHUKKETSU_WIKI_PATH=/project/knowledge/
SHUKKETSU_CACHE_PATH=/project/data/scratch/cache/
SHUKKETSU_BACKUP_PATH=/project/data/backups/
```

**`preBuild.bash`** — Runs before container build. Installs Node.js + Ollama:
```bash
#!/usr/bin/env bash
set -euo pipefail

# --- Node.js LTS (for Claude Code CLI) ---
NODE_VER="v20.19.6"
# ... (existing Node.js installation code) ...

# --- Ollama (for router + embedding models) ---
curl -fsSL https://ollama.ai/install.sh | sh
```

**`postBuild.bash`** — Runs after container build. Installs Claude Code, pulls models:
```bash
#!/usr/bin/env bash
set -euo pipefail

# --- Claude Code CLI ---
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HOME/.npm-global"
npm config set prefix "$HOME/.npm-global"
export PATH="$HOME/.npm-global/bin:$PATH"
npm install -g @anthropic-ai/claude-code

# --- Pull Ollama models ---
ollama pull qwen3:4b
ollama pull nomic-embed-text

# --- vLLM is installed via requirements.txt (pip) ---
# Model weights for Llama 70B AWQ are pulled on first vLLM start
```

**`spec.yaml` apps** — Register Shukketsu web UI as a Workbench app:
```yaml
- name: shukketsu
  type: custom
  class: webapp
  start_command: >-
    cd /project && python -m uvicorn
    code.shukketsu.web.app:app
    --host 0.0.0.0 --port 9000
  health_check_command: 'curl -sf http://localhost:9000/health'
  stop_command: pkill -f "uvicorn code.shukketsu"
  user_msg: "Shukketsu web knowledgebase"
  webapp_options:
    autolaunch: false
    port: "9000"
    proxy:
      trim_prefix: false
```

### Data Layout (Workbench-enforced)

```
/project/
├── code/           → git tracked (source code)
├── models/         → git-lfs tracked (model weights/configs)
├── data/           → git-lfs tracked (database, backups)
│   ├── scratch/    → gitignored (caches, temp files)
│   └── backups/    → automated SQLite backups
└── knowledge/      → git tracked (wiki markdown articles)
```

---

## 4. Project Structure

```
/project/
├── code/
│   └── shukketsu/                      # Main Python package
│       ├── __init__.py
│       ├── config.py                   # Centralized config (reads variables.env)
│       │
│       ├── routing/                    # Multi-model routing
│       │   ├── __init__.py
│       │   ├── router.py              # Query classification via Qwen3 4B
│       │   └── models.py              # RoutingDecision, TaskComplexity, TaskCategory
│       │
│       ├── agents/                     # Multi-agent system
│       │   ├── __init__.py
│       │   ├── base.py                # BaseAgent (ReAct loop, reflection, structured output)
│       │   ├── orchestrator.py        # Task decomposition, agent coordination
│       │   ├── researcher.py          # Information gathering specialist
│       │   ├── analyst.py             # Simulation and data analysis specialist
│       │   ├── writer.py              # Wiki article creation specialist
│       │   ├── editor.py              # Verification and fact-checking specialist
│       │   ├── message_bus.py         # In-process async message passing
│       │   └── guardrails.py          # Loop detection, token budgets, safety limits
│       │
│       ├── llm/                        # LLM interaction layer
│       │   ├── __init__.py
│       │   ├── clients.py             # vLLM client (70B), Ollama client (4B + embed)
│       │   ├── structured.py          # Instructor integration, all Pydantic schemas
│       │   ├── prompts.py             # System prompts for each agent role
│       │   └── context.py             # Context assembly, token budget management
│       │
│       ├── tools/                      # Agent tools
│       │   ├── __init__.py
│       │   ├── registry.py            # Registration, dispatch, circuit breakers
│       │   ├── schemas.py             # Pydantic input/output schemas for all tools
│       │   ├── research/
│       │   │   ├── __init__.py
│       │   │   ├── web_search.py      # Brave Search API client
│       │   │   ├── web_ingest.py      # Fetch -> chunk -> embed -> store
│       │   │   ├── warcraft_logs.py   # WCL v2 GraphQL (OAuth)
│       │   │   ├── blizzard_api.py    # Battle.net Game Data (OAuth)
│       │   │   ├── wowsims_import.py  # WoWSims rogue data from GitHub
│       │   │   └── log_parser.py      # WoW combat log file parser
│       │   ├── analysis/
│       │   │   ├── __init__.py
│       │   │   ├── sim_runner.py      # Run DPS simulations
│       │   │   ├── log_analyzer.py    # Compare logs vs optimal play
│       │   │   └── stat_weights.py    # Calculate stat priorities
│       │   └── knowledge/
│       │       ├── __init__.py
│       │       ├── search.py          # Hybrid RAG (vector + FTS5 + RRF)
│       │       ├── wiki_writer.py     # Create/update markdown articles
│       │       └── db_query.py        # Validated read-only SQL queries
│       │
│       ├── rag/                        # Agentic RAG subsystem
│       │   ├── __init__.py
│       │   ├── decomposer.py          # Query decomposition into sub-queries
│       │   ├── iterative.py           # Iterative retrieval with coverage eval
│       │   ├── self_rag.py            # Decide when to retrieve vs answer directly
│       │   └── corrective.py          # Filter irrelevant retrieved chunks
│       │
│       ├── ingest/                     # Content ingestion pipeline
│       │   ├── __init__.py
│       │   ├── chunker.py             # Semantic chunking (sentence + overlap)
│       │   ├── wow_chunker.py         # WoW-specific chunking rules
│       │   ├── embedder.py            # nomic-embed-text-v2 via Ollama
│       │   └── pipeline.py            # Full ingest pipeline orchestration
│       │
│       ├── scraping/                   # Web scraping infrastructure
│       │   ├── __init__.py
│       │   ├── rate_limiter.py        # Per-domain rate limiting
│       │   ├── robots.py              # robots.txt compliance
│       │   └── fetcher.py             # httpx page fetcher with retries
│       │
│       ├── freshness/                  # Content staleness management
│       │   ├── __init__.py
│       │   ├── policy.py              # Freshness policies per source type
│       │   └── checker.py             # Scheduled staleness re-checker
│       │
│       ├── resilience/                 # Error handling and recovery
│       │   ├── __init__.py
│       │   ├── circuit_breaker.py     # Circuit breaker pattern
│       │   ├── retry.py               # Exponential backoff retries
│       │   └── errors.py              # Error taxonomy + handlers
│       │
│       ├── observability/              # Tracing and monitoring
│       │   ├── __init__.py
│       │   ├── tracer.py              # Langfuse integration
│       │   ├── metrics.py             # Application metrics collection
│       │   └── dashboard.py           # Metrics API for web dashboard
│       │
│       ├── evals/                      # Evaluation framework
│       │   ├── __init__.py
│       │   ├── rag_eval.py            # Ragas-based RAG quality metrics
│       │   ├── trajectory_eval.py     # Agent trajectory analysis
│       │   ├── domain_eval.py         # TBC rogue domain accuracy
│       │   ├── runner.py              # Eval pipeline orchestrator
│       │   └── datasets/              # Curated eval datasets
│       │       ├── rag_eval_set.json
│       │       ├── trajectory_eval_set.json
│       │       └── domain_eval_set.json
│       │
│       ├── backup/                     # Data integrity
│       │   ├── __init__.py
│       │   ├── sqlite_backup.py       # SQLite online backup API
│       │   └── integrity.py           # Periodic integrity checks
│       │
│       ├── trust/                      # Source trust system
│       │   ├── __init__.py
│       │   ├── scoring.py             # Trust score computation + decay
│       │   └── conflicts.py           # Conflict detection and resolution
│       │
│       ├── sim/                        # TBC Rogue DPS simulation engine
│       │   ├── __init__.py
│       │   ├── mechanics.py           # Core combat formulas (TBC 2.4.3)
│       │   ├── abilities.py           # Rogue ability database
│       │   ├── talents.py             # Talent tree modeling
│       │   ├── items.py               # Gear stats, set bonuses, procs
│       │   ├── buffs.py               # Raid buffs, consumables, boss debuffs
│       │   ├── rotation.py            # Per-spec priority rotation logic
│       │   ├── combat.py              # Fight simulation loop (discrete event)
│       │   └── runner.py              # Public API: SimConfig -> SimResult
│       │
│       ├── db/                         # Database layer
│       │   ├── __init__.py
│       │   ├── schema.sql             # Complete DDL
│       │   ├── connection.py          # Connection pool, WAL mode, pragmas
│       │   └── migrations.py          # Schema versioning
│       │
│       └── web/                        # Web knowledgebase
│           ├── __init__.py
│           ├── app.py                 # FastAPI application + middleware
│           ├── routers/
│           │   ├── wiki.py            # Wiki browsing + markdown rendering
│           │   ├── search.py          # Hybrid search endpoint
│           │   ├── chat.py            # Chat interface (WebSocket)
│           │   ├── logs.py            # Combat log upload + analysis
│           │   ├── sim.py             # Sim runner from browser
│           │   ├── evals.py           # Eval results dashboard
│           │   └── traces.py          # Agent trace viewer
│           ├── templates/
│           │   ├── base.html          # Shared layout (nav, sidebar, theme)
│           │   ├── wiki/
│           │   │   ├── article.html
│           │   │   ├── index.html
│           │   │   └── history.html
│           │   ├── search.html
│           │   ├── chat.html
│           │   ├── logs/
│           │   │   ├── upload.html
│           │   │   └── report.html
│           │   ├── sim/
│           │   │   └── builder.html
│           │   ├── evals/
│           │   │   └── dashboard.html
│           │   └── traces/
│           │       └── viewer.html
│           └── static/
│               ├── css/
│               │   ├── theme.css       # Dark WoW-inspired base theme
│               │   ├── tooltips.css    # WoW-style item tooltip hovers
│               │   ├── talent-tree.css # In-game style talent grid
│               │   └── colors.css      # Item rarity + class color system
│               ├── js/
│               │   ├── chat.js         # WebSocket chat client
│               │   ├── tooltips.js     # Item tooltip hover engine
│               │   ├── talent-tree.js  # Interactive talent picker
│               │   ├── charts.js       # DPS breakdown charts
│               │   └── trace-viewer.js # Agent trace visualization
│               └── img/
│                   ├── ability-icons/
│                   ├── item-slots/
│                   └── ui/
│
├── tests/                              # Test suite
│   ├── conftest.py                    # Shared fixtures
│   ├── unit/                          # No external dependencies
│   │   ├── test_chunker.py
│   │   ├── test_wow_chunker.py
│   │   ├── test_rate_limiter.py
│   │   ├── test_sql_validation.py
│   │   ├── test_circuit_breaker.py
│   │   ├── test_loop_detector.py
│   │   ├── test_trust_scoring.py
│   │   ├── test_freshness.py
│   │   ├── test_routing_models.py
│   │   ├── test_rrf.py
│   │   └── test_sim_mechanics.py
│   ├── integration/                   # Needs Ollama + SQLite
│   │   ├── test_rag_pipeline.py
│   │   ├── test_structured_output.py
│   │   ├── test_db_operations.py
│   │   ├── test_embedding.py
│   │   └── test_wcl_api.py
│   └── e2e/                           # Needs all services
│       └── test_agent_workflows.py
│
├── data/
│   ├── shukketsu.db                   # SQLite database
│   ├── backups/                       # Automated backups
│   └── scratch/                       # Gitignored: HTML cache, temp files
│       └── cache/
│
├── knowledge/                         # Git-tracked Markdown wiki
│   ├── specs/
│   │   ├── combat/
│   │   ├── assassination/
│   │   └── subtlety/
│   ├── gearing/
│   ├── encounters/
│   ├── pvp/
│   └── fundamentals/
│
├── models/                            # Git-lfs: model configs, download scripts
│   └── README.md                      # Model download instructions
│
├── infra/                             # Infrastructure scripts
│   ├── docker-compose.langfuse.yml    # Langfuse self-hosted stack
│   ├── vllm-start.sh                 # vLLM server launch with model path
│   ├── ollama-setup.sh               # Ollama model pull script
│   └── backup-cron.sh                # Backup scheduling
│
├── docs/
│   └── plans/
│       └── 2026-02-09-shukketsu-design.md  # This document
│
├── .project/spec.yaml                 # NVIDIA Workbench project config
├── apt.txt                            # System packages
├── requirements.txt                   # Python dependencies
├── variables.env                      # Environment variables
├── preBuild.bash                      # Pre-build container setup
├── postBuild.bash                     # Post-build container setup
├── pyproject.toml                     # Project metadata
├── pytest.ini                         # Test configuration
└── .gitignore
```

---

## 5. Multi-Model Routing System

### Three Models, Three Roles

```
User Query
    │
    ▼
┌──────────────────────────────┐
│  Qwen3 4B (Ollama :11434)   │  ← ROUTER: classify in <200ms
│  Role: classification,       │
│        trivial answers,      │
│        output validation     │
└──────────┬───────────────────┘
           │
    ┌──────┴──────┐
    │             │
 TRIVIAL      MODERATE/COMPLEX
    │             │
 Qwen 4B     ┌───┴───────────────────┐
 answers     │ Llama 3.3 70B (vLLM)  │  ← REASONER: all substantive work
 directly    │ Role: planning, tool   │
             │   calling, synthesis,  │
             │   SQL/GraphQL gen,     │
             │   article writing      │
             └────────────────────────┘

┌──────────────────────────────────┐
│ nomic-embed-text-v2 (Ollama)     │  ← EMBEDDER: called by RAG pipeline
│ Role: embed queries + chunks     │
└──────────────────────────────────┘
```

### Routing Logic

```python
class TaskComplexity(str, Enum):
    TRIVIAL = "trivial"       # Direct lookup, simple fact recall
    MODERATE = "moderate"     # Single-step reasoning, one tool call
    COMPLEX = "complex"       # Multi-step research, planning, synthesis

class TaskCategory(str, Enum):
    RETRIEVAL = "retrieval"
    ANALYSIS = "analysis"
    RESEARCH = "research"
    WRITING = "writing"
    CONVERSATION = "conversation"

class RoutingDecision(BaseModel):
    complexity: TaskComplexity
    category: TaskCategory
    needs_tools: bool
    suggested_agent: str  # "researcher", "analyst", "writer", "orchestrator"
```

### Routing Rules

| Complexity | Category | Handler |
|-----------|----------|---------|
| TRIVIAL | any | Qwen 4B answers directly from cached knowledge |
| MODERATE | RETRIEVAL | Researcher agent, single ReAct loop |
| MODERATE | ANALYSIS | Analyst agent, single ReAct loop |
| MODERATE | CONVERSATION | Llama 70B direct response |
| COMPLEX | RESEARCH | Orchestrator decomposes into sub-tasks for multiple agents |
| COMPLEX | WRITING | Writer agent + Editor review |
| COMPLEX | ANALYSIS | Analyst agent with sim + Orchestrator synthesis |

### Model Serving

- **vLLM** serves Llama 70B on port 8000 with OpenAI-compatible API
- **Ollama** serves Qwen3 4B + nomic-embed-text-v2 on port 11434
- Both run as persistent services inside the Workbench container
- Application communicates with both via HTTP (localhost)

---

## 6. Multi-Agent Architecture

### Five Specialist Agents

Each agent is a Python class with its own system prompt, tool access, and behavior. No external frameworks — just classes, structured messages, and an orchestrator loop.

| Agent | Model | Tools | Responsibility |
|-------|-------|-------|----------------|
| **Orchestrator** | Qwen 4B (routing) + Llama 70B (planning) | None (delegates) | Task decomposition, agent coordination, result aggregation |
| **Researcher** | Llama 70B | web_search, web_ingest, rag_search, wcl_query, blizzard_api, wowsims_import | Information gathering from all sources |
| **Analyst** | Llama 70B | sim_run, db_query, log_analyze, stat_weights, rag_search | Simulation, log analysis, quantitative work |
| **Writer** | Llama 70B | wiki_write, wiki_read, rag_search, db_query | Article creation, editing, formatting |
| **Editor** | Llama 70B | wiki_read, rag_search, db_query, sim_run | Verification, fact-checking, confidence scoring |

### Base Agent Class

```python
class BaseAgent:
    """Base class for all Shukketsu agents."""

    def __init__(self, config: AgentConfig, tool_registry, llm_client, tracer):
        self.config = config
        self.tools = tool_registry.get_tools(config.tools)
        self.llm = llm_client
        self.tracer = tracer
        self.scratchpad: list[dict] = []

    async def run(self, task: AgentMessage) -> AgentMessage:
        """Execute the agent's ReAct loop."""
        for iteration in range(self.config.max_iterations):
            # THINK: Get next step via structured output
            step = await self._think(self._build_context(task))

            if step.action == "final_answer":
                # REFLECT: Evaluate own output quality
                assessment = await self._reflect(self.scratchpad)
                if assessment.score >= 0.7 or iteration >= self.config.max_iterations - 1:
                    return self._make_result(step.answer, assessment)
                # Below threshold: add feedback to scratchpad, continue
                self.scratchpad.append({"type": "reflection", "feedback": assessment.feedback})
                continue

            # ACT: Execute the tool
            observation = await self._act(step.tool_call)

            # OBSERVE: Add result to scratchpad
            self.scratchpad.append({
                "type": "observation",
                "tool": step.tool_call.tool_name,
                "input": step.tool_call.tool_input,
                "output": observation
            })
```

### Orchestrator Flow

```python
class Orchestrator:
    async def handle_request(self, user_query: str, trace_id: str) -> str:
        # Step 1: Route via Qwen 4B
        routing = await self.router.classify(user_query)

        if routing.complexity == TaskComplexity.TRIVIAL:
            return await self.quick_answer(user_query)

        if routing.complexity == TaskComplexity.MODERATE:
            agent = self._select_agent(routing.category)
            result = await agent.run(self._make_task(user_query, trace_id))
            return result.content["answer"]

        # COMPLEX: decompose into sub-tasks
        plan = await self._create_plan(user_query, trace_id)
        results = {}

        for step in plan.steps:
            agent = self._select_agent(step.category)
            task = self._make_task(step.instruction, trace_id, context=results)
            result = await agent.run(task)
            results[step.id] = result

            # Check if plan needs revision based on intermediate results
            if step.checkpoint:
                plan = await self._maybe_revise_plan(plan, results)

        # Synthesize final answer from all step results
        synthesis = await self._synthesize(user_query, results, trace_id)

        # Editor review for wiki-bound content
        if routing.category in (TaskCategory.RESEARCH, TaskCategory.WRITING):
            reviewed = await self.editor.run(self._make_review_task(synthesis, trace_id))
            return reviewed.content["final"]

        return synthesis
```

### Agent Communication

Agents communicate via an in-memory async message bus. All messages are structured dataclasses.

```python
@dataclass
class AgentMessage:
    from_agent: AgentRole
    to_agent: AgentRole
    task_id: str
    message_type: str  # "task_assignment", "result", "clarification", "error"
    content: dict[str, Any]
    trace_id: str
    parent_span_id: str | None = None

class MessageBus:
    def __init__(self):
        self._queues: dict[AgentRole, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._history: list[AgentMessage] = []  # for tracing

    async def send(self, message: AgentMessage): ...
    async def receive(self, agent: AgentRole, timeout: float = 300.0) -> AgentMessage: ...
    def get_trace(self, trace_id: str) -> list[AgentMessage]: ...
```

---

## 7. Structured Output & Tool Calling

### Instructor Integration

All LLM output is typed via Instructor + Pydantic. This eliminates the "how does the LLM generate tool calls" problem entirely.

```python
import instructor
from openai import AsyncOpenAI

# vLLM client for 70B reasoning model
vllm_client = instructor.from_openai(
    AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed"),
    mode=instructor.Mode.JSON
)

# Ollama client for 4B router
ollama_client = instructor.from_provider(
    "ollama/qwen3-4b",
    mode=instructor.Mode.JSON
)
```

### Tool Call Schema

```python
class ToolCall(BaseModel):
    thought: str = Field(description="Chain-of-thought reasoning")
    tool_name: str = Field(description="Name of the tool to call")
    tool_input: dict = Field(description="Arguments for the tool")

class AgentStep(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning about current state")
    action: Literal["tool_call", "final_answer"]
    tool_call: ToolCall | None = None
    answer: str | None = None
```

### Per-Tool Input Schemas

Every tool has a Pydantic input schema. This gives the LLM clear structure and enables automatic validation:

```python
class WebSearchInput(BaseModel):
    query: str
    max_results: int = Field(default=10, ge=1, le=20)

class RagSearchInput(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=50)
    min_trust: float = Field(default=0.0, ge=0.0, le=1.0)

class DbQueryInput(BaseModel):
    sql: str = Field(description="Read-only SQL query")
    params: list = Field(default_factory=list)

class SimRunInput(BaseModel):
    spec: Literal["combat_swords", "combat_fists", "combat_daggers",
                   "assassination_mutilate", "assassination_backstab", "subtlety"]
    talents: str
    gear_ids: dict[str, int]
    buffs: list[str] = Field(default_factory=list)
    fight_length: int = 300
    iterations: int = 10000

class WikiWriteInput(BaseModel):
    path: str
    content: str
    commit_message: str

class WclQueryInput(BaseModel):
    query_type: Literal["rankings", "parses", "gear", "talents", "fights"]
    zone_id: int | None = None
    encounter_id: int | None = None
    spec: str | None = None
    metric: str = "dps"
    limit: int = 100
```

### Error Recovery

Instructor's `max_retries` handles the most common failure: malformed output. When JSON fails Pydantic validation, Instructor re-prompts with the validation error, giving the model a chance to self-correct.

```python
async def safe_structured_call(
    llm_client, response_model: type[BaseModel],
    messages: list[dict], max_retries: int = 3
) -> BaseModel:
    """Get structured output with automatic retry on validation failure."""
    try:
        return await llm_client.chat.completions.create(
            response_model=response_model,
            messages=messages,
            max_retries=max_retries,
            temperature=0.1
        )
    except Exception as e:
        # All retries exhausted — try with temperature=0
        return await llm_client.chat.completions.create(
            response_model=response_model,
            messages=messages,
            max_retries=1,
            temperature=0.0
        )
```

### SQL Injection Prevention

Agent-generated SQL is validated before execution:

```python
BLOCKED_KEYWORDS = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
                    "TRUNCATE", "EXEC", "EXECUTE", "GRANT", "REVOKE"}

def validate_sql(sql: str) -> bool:
    parsed = sqlparse.parse(sql)
    for statement in parsed:
        stmt_type = statement.get_type()
        if stmt_type and stmt_type.upper() not in ("SELECT", "UNKNOWN"):
            return False
        if any(kw in sql.upper() for kw in BLOCKED_KEYWORDS):
            return False
    return True
```

---

## 8. Agentic RAG

Beyond basic retrieve-and-generate. The RAG subsystem actively decides what to search, evaluates if results are sufficient, and iterates.

### Query Decomposition

Complex questions are broken into atomic sub-queries before retrieval:

```python
class SubQuery(BaseModel):
    query: str
    intent: Literal["factual", "comparative", "temporal", "procedural"]
    priority: int = Field(ge=1, le=5)

class DecomposedQuery(BaseModel):
    original: str
    sub_queries: list[SubQuery]
    requires_synthesis: bool = True

# Example:
# "What's the best combat rogue build for Gruul, and how does it compare to assassination?"
# Decomposes to:
#   1. "Best combat rogue talent build for Gruul" (factual, priority=1)
#   2. "Best assassination rogue talent build for Gruul" (factual, priority=1)
#   3. "Combat vs assassination DPS comparison on Gruul" (comparative, priority=2)
#   4. "Gruul fight mechanics relevant to rogue spec choice" (factual, priority=1)
```

### Iterative Retrieval

Retrieve, evaluate coverage, re-query if gaps exist:

```python
async def iterative_retrieve(
    query: str, search_fn, llm_client,
    max_iterations: int = 3, min_coverage: float = 0.8
) -> list[dict]:
    all_chunks = []
    gaps = [query]

    for i in range(max_iterations):
        for gap_query in gaps:
            chunks = await search_fn(gap_query, top_k=5)
            all_chunks.extend(chunks)

        # Deduplicate by chunk ID
        unique_chunks = deduplicate(all_chunks)

        # Ask LLM: do these chunks answer the query?
        state = await llm_client.chat.completions.create(
            response_model=RetrievalState,
            messages=[{"role": "user", "content":
                f"Evaluate if these chunks answer: '{query}'\n"
                f"Chunks: {format_chunks(unique_chunks)}\n"
                f"Rate coverage 0-1. List information gaps."}]
        )

        if state.coverage_score >= min_coverage:
            break
        gaps = state.gaps  # Refined queries for next iteration

    return unique_chunks
```

### Self-RAG

The agent decides whether retrieval is even needed:

```python
class RetrievalDecision(BaseModel):
    needs_retrieval: bool
    reason: str
    confidence_without_retrieval: float = Field(ge=0.0, le=1.0)

# High confidence without retrieval → answer directly (saves latency)
# Low confidence → retrieve first
```

### Corrective RAG

After retrieval, verify chunks are actually relevant before passing to generation:

```python
class FilteredRetrieval(BaseModel):
    relevant_chunks: list[ChunkRelevance]
    should_web_search: bool  # If local knowledge insufficient
    web_search_query: str | None = None

# Filters out noise, triggers web search if local KB lacks coverage
```

### Hybrid Search (Vector + FTS5 + Reciprocal Rank Fusion)

```python
async def hybrid_search(
    query: str, db, embedding_client,
    top_k: int = 10, vector_weight: float = 0.7, fts_weight: float = 0.3,
    min_trust: float = 0.0
) -> list[dict]:
    # 1. Get query embedding via nomic-embed-text-v2
    query_embedding = await embedding_client.embed(query)

    # 2. Vector search via sqlite-vec (cosine distance)
    vector_results = db.execute("""
        SELECT c.id, c.content, s.trust_score,
               vec_distance_cosine(v.embedding, ?) as distance
        FROM chunks_vec v
        JOIN chunks c ON c.id = v.id
        JOIN sources s ON s.id = c.source_id
        WHERE s.trust_score >= ?
        ORDER BY distance ASC LIMIT ?
    """, [query_embedding, min_trust, top_k * 2])

    # 3. FTS5 keyword search (BM25)
    fts_results = db.execute("""
        SELECT c.id, c.content, s.trust_score, rank as bm25_score
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.rowid
        JOIN sources s ON s.id = c.source_id
        WHERE chunks_fts MATCH ? AND s.trust_score >= ?
        ORDER BY rank LIMIT ?
    """, [query, min_trust, top_k * 2])

    # 4. Reciprocal Rank Fusion
    rrf_scores = compute_rrf(vector_results, fts_results,
                             vector_weight, fts_weight, k=60)

    # 5. Return top_k by combined score
    return sorted_by_rrf(rrf_scores, top_k)
```

---

## 9. Data Collection Layer

### Philosophy

Two general-purpose tools (web search + web ingest) for most sources, with dedicated adapters only for structured APIs that require special handling.

### Dedicated Adapters (4 total)

| Adapter | Source | Why dedicated |
|---------|--------|---------------|
| `warcraft_logs.py` | Warcraft Logs v2 | GraphQL API with OAuth, complex query structure for parses/rankings/gear |
| `blizzard_api.py` | Battle.net API | REST with OAuth, live authoritative game data (items, spells, characters) |
| `wowsims_import.py` | WoWSims GitHub | Go source code containing validated spell coefficients + combat models |
| `log_parser.py` | WoW combat log files | Unique binary/text format specific to WoW |

### General-Purpose Tools

**`web_search.py`** — Brave Search API. Agent searches the web, discovers sources organically.

**`web_ingest.py`** — Agent feeds any URL. Pipeline:
1. Check robots.txt compliance
2. Acquire rate limiter slot for domain
3. Fetch page content (httpx)
4. Extract text (BeautifulSoup)
5. Semantic chunk with WoW-specific rules (~400 tokens, 50 token overlap)
6. Embed each chunk (nomic-embed-text-v2 via Ollama)
7. Store chunks + embeddings in SQLite (chunks + chunks_vec + chunks_fts)
8. Store source metadata for deduplication + trust scoring

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

### Source Watcher (scheduled)

| Source Type | Check Frequency |
|------------|----------------|
| Warcraft Logs (new top parses) | Every 6 hours |
| Blizzard news / hotfixes | Daily |
| Reddit / forum threads | Every 12 hours |
| Guide sites | Weekly |
| WoWSims repo (updates) | Weekly |

---

## 10. Storage Layer

### SQLite (`data/shukketsu.db`)

One database file containing structured data, vector embeddings, full-text search, and operational data.

**Database pragmas (set on connection):**
```sql
PRAGMA journal_mode=WAL;          -- Write-Ahead Logging for concurrent access
PRAGMA synchronous=NORMAL;        -- Balance safety vs speed
PRAGMA busy_timeout=5000;         -- Wait up to 5s for locks
PRAGMA foreign_keys=ON;
```

### Schema

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
    talents_json TEXT,
    source TEXT,
    usage_pct REAL,
    context TEXT            -- "pve_raid", "pvp_arena_2v2", etc.
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
    chunk_count INTEGER,
    -- Freshness tracking
    last_checked TIMESTAMP,
    check_interval_hours INTEGER DEFAULT 168,
    content_hash_previous TEXT,
    change_count INTEGER DEFAULT 0,
    is_stale BOOLEAN DEFAULT FALSE
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
    confidence_score REAL,
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
-- EVALUATION & OBSERVABILITY
-- ============================================

CREATE TABLE eval_reports (
    id INTEGER PRIMARY KEY,
    run_at TIMESTAMP NOT NULL,
    rag_faithfulness REAL,
    rag_answer_relevancy REAL,
    rag_context_precision REAL,
    rag_context_recall REAL,
    trajectory_precision REAL,
    trajectory_recall REAL,
    trajectory_exact_match REAL,
    domain_accuracy REAL,
    report_json TEXT,          -- Full detailed report
    models_json TEXT           -- Which models were used
);

CREATE TABLE feedback (
    id INTEGER PRIMARY KEY,
    trace_id TEXT NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    rating INTEGER,            -- 1-5 or thumbs up/down
    correction TEXT,           -- Human-provided correct answer
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE agent_tasks (
    id INTEGER PRIMARY KEY,
    trace_id TEXT NOT NULL,
    agent_role TEXT NOT NULL,
    task_description TEXT,
    tools_used TEXT,           -- JSON array of tool names
    iterations INTEGER,
    total_tokens INTEGER,
    latency_ms INTEGER,
    success BOOLEAN,
    error_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Source Trust Levels

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

Conflict resolution:
1. Higher-trust source wins by default
2. Newer data preferred for meta-dependent questions
3. If simulatable, run sim to settle empirically
4. If still ambiguous, present both views + flag for human review

### Git Markdown Wiki (`knowledge/`)

Agent-written articles stored as Markdown, tracked in git. Each includes YAML frontmatter:

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

```
knowledge/
├── specs/
│   ├── combat/
│   │   ├── overview.md
│   │   ├── rotation.md
│   │   ├── talents.md
│   │   └── tips.md
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
│   ├── stat-weights.md
│   ├── gems-enchants.md
│   └── profession-choices.md
├── encounters/
│   ├── karazhan/
│   │   ├── attumen.md ... prince-malchezaar.md
│   ├── gruul/
│   └── magtheridon/
├── pvp/
│   ├── arena/
│   │   ├── rmp.md
│   │   ├── rogue-druid.md
│   │   └── season-1-meta.md
│   └── honor-gear.md
└── fundamentals/
    ├── hit-cap.md
    ├── expertise.md
    ├── crit-and-agi.md
    ├── haste-breakpoints.md
    ├── weapon-speed.md
    ├── energy-system.md
    ├── poisons.md
    └── stealth-mechanics.md
```

---

## 11. Resilience & Error Handling

### Failure Mode Taxonomy

```python
class FailureMode(str, Enum):
    # LLM failures
    LLM_TIMEOUT = "llm_timeout"
    LLM_LOOP = "llm_loop"
    LLM_MALFORMED_OUTPUT = "llm_malformed"
    LLM_REFUSAL = "llm_refusal"

    # Tool failures
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_EXECUTION_ERROR = "tool_error"
    TOOL_TIMEOUT = "tool_timeout"

    # Network failures
    NETWORK_TIMEOUT = "network_timeout"
    RATE_LIMITED = "rate_limited"
    HTTP_ERROR = "http_error"

    # Data failures
    DB_ERROR = "db_error"
    EMBEDDING_ERROR = "embedding_error"

    # System failures
    OOM = "out_of_memory"
    MODEL_UNAVAILABLE = "model_unavailable"
```

### Circuit Breaker Pattern

Each tool and external service gets a circuit breaker:

```python
class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5,
                 recovery_timeout: float = 60.0):
        self.state = CircuitState.CLOSED  # CLOSED → OPEN → HALF_OPEN
        self.failure_count = 0
        ...

    def can_execute(self) -> bool: ...
    def record_success(self): ...
    def record_failure(self): ...
```

States: **CLOSED** (normal) → **OPEN** (failing, reject requests) → **HALF_OPEN** (testing recovery)

### Agent Loop Protection

```python
class LoopDetector:
    MAX_ITERATIONS = 15
    MAX_CONSECUTIVE_SAME_TOOL = 3
    MAX_TOTAL_TOKENS = 100_000

    def check(self, tool_name, tool_input, tokens_used) -> str | None:
        """Returns error message if loop detected, None if OK."""
        # Detect exact duplicate tool calls
        # Check token budget exhaustion
        # Check iteration count
```

### Graceful Degradation

| Failure | Recovery |
|---------|----------|
| 70B model timeout (>30s) | Retry once, then return partial answer with warning |
| 70B model OOM | Reduce context window, retry with fewer RAG chunks |
| Agent loop detected | Force final_answer with partial results + explanation |
| Tool execution error | Log error, add "tool unavailable" to scratchpad, continue |
| Web search fails | Fall back to RAG-only (existing knowledge base) |
| Rate limited (web) | Queue request, return "researching in background" |
| SQLite locked | Retry with exponential backoff (up to 5 attempts) |
| Embedding model down | Skip embedding, use FTS5 keyword search only |
| Router model down | Default all requests to 70B model (skip routing) |

---

## 12. Observability & Tracing

### Langfuse (Self-Hosted)

MIT license, runs in Docker Compose, purpose-built for LLM tracing with built-in UI. Memory footprint ~2GB.

```yaml
# infra/docker-compose.langfuse.yml
services:
  langfuse-server:
    image: langfuse/langfuse:latest
    ports: ["3000:3000"]
    environment:
      DATABASE_URL: postgresql://...
      CLICKHOUSE_URL: http://langfuse-clickhouse:8123
  langfuse-clickhouse:
    image: clickhouse/clickhouse-server:latest
  langfuse-postgres:
    image: postgres:16
```

### What Gets Traced

Every operation gets a trace with nested spans:

```
Trace: "user_query_abc123"
├── Span: "routing" (model=qwen-4b, latency=180ms, tokens=50)
│   └── Generation: routing_decision {complexity: "complex", category: "research"}
├── Span: "orchestrator.plan" (model=llama-70b, latency=2100ms, tokens=800)
│   └── Generation: plan {steps: [...]}
├── Span: "researcher.execute"
│   ├── Span: "react_step_1"
│   │   ├── Generation: thought {reasoning: "...", action: "tool_call"}
│   │   └── Span: "tool.web_search" (latency=1200ms)
│   ├── Span: "react_step_2"
│   │   ├── Generation: thought {reasoning: "...", action: "tool_call"}
│   │   └── Span: "tool.rag_search" (latency=45ms)
│   └── Span: "react_step_3"
│       └── Generation: final_answer
├── Span: "editor.verify"
│   ├── Generation: fact_check_results
│   └── Score: {name: "faithfulness", value: 0.92}
└── Score: {name: "end_to_end_latency", value: 8.4}
```

### Tracer Implementation

```python
class ShukketsuTracer:
    def __init__(self):
        self.langfuse = Langfuse(
            host="http://localhost:3000",
            public_key="local-pk",
            secret_key="local-sk"
        )

    @asynccontextmanager
    async def trace(self, name, user_id=None, metadata=None):
        trace = self.langfuse.trace(name=name, user_id=user_id, metadata=metadata)
        try: yield trace
        finally: self.langfuse.flush()

    @asynccontextmanager
    async def span(self, trace, name, **kwargs):
        span = trace.span(name=name, **kwargs)
        try: yield span
        finally: span.end()
```

### Metrics Exposed to Web UI

| Metric | Source |
|--------|--------|
| Query latency (p50, p95, p99) | Langfuse traces |
| Token usage per query | Langfuse generations |
| Tool call frequency | Langfuse spans |
| RAG quality scores | Langfuse scores |
| Agent iteration count | Langfuse spans |
| Error rate by type | Langfuse events |
| Model routing distribution | Langfuse traces |

---

## 13. Evaluation Framework

### Three Dimensions

**Dimension 1: RAG Quality (Ragas)**

Metrics: faithfulness, answer relevancy, context precision, context recall.

Evaluation dataset: 50+ curated question-answer-context triples covering all major rogue topics.

```python
EVAL_DATASET = [
    {
        "question": "What is the hit cap for a combat rogue in TBC?",
        "ground_truth": "The special attack hit cap is 9% (142 hit rating). "
                       "With 5/5 Precision, you need 6% (95 hit rating) from gear.",
    },
    {
        "question": "Is Dragonspine Trophy or Bloodlust Brooch better for combat?",
        "ground_truth": "Dragonspine Trophy is significantly better. "
                       "The haste proc has high uptime and scales with combat potency.",
    },
    # ... 48 more
]
```

**Dimension 2: Agent Trajectory Analysis**

Measures whether agents call the right tools in the right order:

```python
@dataclass
class ExpectedTrajectory:
    question: str
    expected_tools: list[str]
    must_include_tools: set[str]
    must_not_include_tools: set[str]

# Metrics: trajectory_precision, trajectory_recall, exact_match
```

**Dimension 3: Domain Accuracy**

50+ questions across categories (factual, calculation, judgment, comparative):

```python
DOMAIN_EVAL = [
    {"q": "Energy cost of max-rank Sinister Strike?", "a": "45 energy", "type": "factual"},
    {"q": "Agility per 1% crit at level 70?", "a": "40 agility", "type": "calculation"},
    {"q": "Should combat rogues use Expose Armor in 25-man?",
     "a": "Generally no — warrior Sunder Armor stacks maintained by tank.",
     "type": "judgment"},
    {"q": "Rank: hit, expertise, agility, haste for P1 combat swords",
     "a": "Hit (to cap) > Expertise (to cap) > Haste > Agility",
     "type": "comparative"},
]
```

### Evaluation Pipeline

```bash
# Run full eval suite
python -m shukketsu.evals.runner

# Results stored in eval_reports table, viewable at /evals/ in web UI
```

Eval reports track metrics over time for trend analysis. Each phase gate requires minimum scores before proceeding.

---

## 14. Content Freshness Management

### Freshness Policies

```python
FRESHNESS_POLICIES = {
    "blizzard_api": FreshnessPolicy(
        max_age=timedelta(days=1),
        check_interval=timedelta(hours=6),
        decay_factor=0.99,
    ),
    "warcraft_logs": FreshnessPolicy(
        max_age=timedelta(days=7),
        check_interval=timedelta(hours=6),
        decay_factor=0.95,
    ),
    "guide_site": FreshnessPolicy(
        max_age=timedelta(days=30),
        check_interval=timedelta(days=7),
        decay_factor=0.90,
    ),
    "forum_post": FreshnessPolicy(
        max_age=timedelta(days=14),
        check_interval=timedelta(days=14),
        decay_factor=0.80,
    ),
    "archived_theory": FreshnessPolicy(
        max_age=timedelta(days=365),
        check_interval=timedelta(days=30),
        decay_factor=0.98,
    ),
}
```

### Trust Decay

When content ages past its max_age, effective trust score decays:

```python
def effective_trust(base_trust, fetched_at, policy):
    age = datetime.utcnow() - fetched_at
    if age <= policy.max_age:
        return base_trust
    periods_past = (age - policy.max_age) / policy.max_age
    return base_trust * (policy.decay_factor ** periods_past)
```

### Staleness Checker (Scheduled)

APScheduler runs periodically: finds sources due for re-check, re-fetches, compares content hash. If content changed, re-ingests. If source unreachable, marks stale (doesn't delete).

---

## 15. Web Scraping Infrastructure

### Per-Domain Rate Limiter

```python
DOMAIN_POLICIES = {
    "warcraftlogs.com": 2.0,     # seconds between requests
    "wowhead.com": 3.0,
    "silentshadows.net": 5.0,    # small site, be gentle
    "shadowpanther.net": 5.0,
    "reddit.com": 2.0,
    "icy-veins.com": 1.5,
    "web.archive.org": 1.0,
    "github.com": 1.0,
    "__default__": 3.0,
}
```

### robots.txt Compliance

Every domain's robots.txt is fetched and cached before scraping. If a URL is disallowed, it's skipped.

### Request Headers

```python
SCRAPING_HEADERS = {
    "User-Agent": "Shukketsu/1.0 (TBC Rogue Research Agent; "
                  "personal project; github.com/user/shukketsu)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
```

---

## 16. Semantic Chunking

### Strategy

Split by semantic boundaries (headers, paragraphs) with sentence-level fallback and overlap:

```python
class SemanticChunker:
    target_tokens: int = 400
    max_tokens: int = 600
    min_tokens: int = 100
    overlap_tokens: int = 50

    def chunk(self, text, metadata=None) -> list[Chunk]:
        # 1. Split by structural boundaries (## headers, blank lines, <hr>)
        # 2. If section too long, split by sentences
        # 3. If sentence too long (code/tables), split by line
        # 4. Merge adjacent undersized chunks
        # 5. Add overlap from previous chunk's last N tokens
```

### WoW-Specific Rules

The `WoWChunker` extends `SemanticChunker` with domain-specific protections:

- **Item tooltip blocks** — never split mid-tooltip
- **Talent tree tables** — kept intact
- **Rotation priority lists** — kept intact
- **Forum post boundaries** — each post is a natural chunk boundary
- **Code/formula blocks** — kept intact

---

## 17. Backup & Data Integrity

### SQLite Backup

Uses SQLite's online backup API (safe during concurrent reads/writes):

```python
class BackupManager:
    def backup(self) -> Path:
        source = sqlite3.connect(db_path)
        dest = sqlite3.connect(backup_path)
        with dest:
            source.backup(dest)
        # Prune old backups (keep last N)

    def verify_backup(self, path) -> bool:
        return conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
```

### Schedule

| What | Frequency | Retention |
|------|-----------|-----------|
| SQLite database | Every 6 hours | Last 10 (2.5 days) |
| SQLite database | Daily | Last 30 daily |
| Wiki markdown | On every write | Full git history |

### Integrity Checks (Periodic)

- SQLite `PRAGMA integrity_check`
- Vector table row count matches chunks table
- FTS5 row count matches chunks table
- Wiki files exist for all articles in DB
- Source URL spot-check reachability

---

## 18. DPS Simulation Engine

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

1. **CMaNGOS TBC Database** (GitHub SQL) — raw spell coefficients, proc rates
2. **WoWSims TBC Rogue** (GitHub Go source) — validated combat model
3. **Warcraft Logs data** — real-world validation of sim output

### Implementation Strategy

```
Phase 3a: Wrap WoWSims as a callable tool (WASM or web API)
          → Immediate sim capability

Phase 3b: Build Python sim referencing WoWSims + CMaNGOS
          → Deeper understanding, customizable

Phase 3c: Validate our sim against WoWSims + real WCL parses
          → Ensure accuracy within 1-2% of real data
```

### Sim Runner API

```python
config = SimConfig(
    spec="combat_swords",
    talents="20/41/0",
    gear={...},
    buffs=["blessing_of_kings", "windfury_totem", "battle_shout"],
    consumables=["haste_potion", "elixir_of_major_agility", "roasted_clefthoof"],
    boss=BossConfig(name="Prince Malchezaar", armor=7700, level=73),
    fight_length=300,
    iterations=10000,
)

result = sim_run(config)
# result.dps_mean = 1847.3
# result.dps_std = 42.1
# result.breakdown = {"Sinister Strike": 28.3%, "Melee (MH)": 24.1%, ...}
# result.stat_weights = {"agility": 1.0, "hit_rating": 1.21, "haste_rating": 0.87, ...}
```

---

## 19. Web Knowledgebase

### Design Philosophy

Custom WoW Rogue-themed UI. Server-rendered HTML (FastAPI + Jinja2 + HTMX) with vanilla JS for interactive components.

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

### Seven Interfaces

**1. Wiki Browser (`/wiki/`)**
- Renders git-tracked Markdown as themed HTML
- Confidence badge per article (verified / needs review / unverified)
- Revision history via git log
- Cross-linked related articles
- Item names colored by rarity, hoverable for tooltips

**2. Search (`/search/`)**
- Single search bar, hybrid results: FTS5 keyword + sqlite-vec semantic
- Results grouped: wiki articles first, then ingested sources, then structured data
- Trust badges on source results

**3. Chat (`/chat/`)**
- WebSocket real-time conversation with agent
- Streamed token-by-token responses
- Rich rendering: item tooltips, inline sim results, bar charts
- Conversation history preserved per session

**4. Log Analysis (`/logs/`)**
- Upload WoW combat log or paste Warcraft Logs URL
- Report: DPS breakdown, rotation analysis, uptime tracking
- Comparison vs top percentile players
- Specific improvement suggestions

**5. Sim Builder (`/sim/`)** *(Phase 4)*
- Visual spec selector + talent tree grid
- Gear slot dropdowns with item quality colors
- Buff/consumable checkboxes
- Run sim, see DPS result + stat weights + breakdown chart

**6. Eval Dashboard (`/evals/`)** *(New)*
- RAG quality metrics over time (line charts)
- Agent trajectory analysis results
- Domain accuracy scores
- Historical comparison between eval runs

**7. Agent Trace Viewer (`/traces/`)** *(New)*
- Browse recent agent traces
- Expand to see full span tree (routing → planning → tool calls → reflection)
- Token usage, latency, model used per span
- Filter by agent role, tool, error status

### WoW-Style Item Tooltips

Any item name on the site is hoverable:

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
│ 92% usage in top parses          │
└──────────────────────────────────┘
```

---

## 20. Testing Strategy

### Test Pyramid

```
          /\
         /  \        5-10 E2E tests (full agent workflows)
        /────\
       /      \      30-50 integration tests (tool + DB + LLM)
      /────────\
     /          \    200+ unit tests (pure logic, no I/O)
    /────────────\
```

### Unit Tests (`tests/unit/`)

No external dependencies. Test pure logic:
- Chunking (semantic splitter, WoW-specific rules)
- SQL validation (injection prevention)
- Rate limiter timing
- Trust scoring and decay
- Circuit breaker state transitions
- Loop detector logic
- Pydantic schema validation
- RRF score calculation
- Sim mechanics formulas

### Integration Tests (`tests/integration/`)

Need Ollama + SQLite running:
- Ingest a document → retrieve via RAG search
- Structured output parsing from 70B model
- Vector similarity search end-to-end
- Embedding generation
- WCL/Blizzard API calls (with credentials)

### E2E Tests (`tests/e2e/`)

Full agent workflows:
- Simple question → agent answers from knowledge base
- Research task → agent uses multiple tools, produces synthesis
- Sim comparison → agent runs sims, compares results

### Running Tests

```bash
pytest tests/unit/ -v                          # Fast, no deps
pytest tests/integration/ -v -m integration    # Needs services
pytest tests/e2e/ -v -m e2e                    # Needs everything
pytest --cov=shukketsu --cov-report=html       # Full + coverage
```

---

## 21. Development Phases

### Phase 0: Infrastructure (1-2 days)

**Gate: All models respond to API calls, database created, web server serves a page.**

```
[ ] Install Ollama in preBuild.bash, pull qwen3:4b + nomic-embed-text
[ ] Set up vLLM with Llama 3.3 70B AWQ INT4 on port 8000
[ ] Verify both models respond from Python (instructor + openai clients)
[ ] Benchmark: tokens/sec for 100-token and 1000-token generations
[ ] Initialize SQLite database with complete schema (WAL mode, FTS5, sqlite-vec)
[ ] Initialize project directory structure under code/shukketsu/
[ ] FastAPI skeleton with health check at localhost:9000
[ ] Register shukketsu app in spec.yaml
[ ] Docker Compose for Langfuse, verify UI at localhost:3000
[ ] Git init knowledge/ directory
[ ] Update requirements.txt, apt.txt, variables.env
```

### Phase 1: Agent Core + Routing (2-3 weeks)

**Gate: Chat with agent in browser. It classifies queries, routes to correct model, calls tools, answers from knowledge base. Traces visible in Langfuse.**

```
[ ] Router: Qwen 4B query classification (routing/router.py)
[ ] Base agent class: ReAct loop + reflection + structured output (agents/base.py)
[ ] LLM clients: vLLM (70B) + Ollama (4B + embed) via Instructor (llm/clients.py, structured.py)
[ ] Tool registry with circuit breakers (tools/registry.py)
[ ] Tools: web_search (Brave API), web_ingest, rag_search, db_query
[ ] Semantic chunker + WoW chunker (ingest/chunker.py, wow_chunker.py)
[ ] Rate limiter + robots.txt checker (scraping/)
[ ] Hybrid search: vector + FTS5 + RRF (tools/knowledge/search.py)
[ ] Langfuse tracer integration (observability/tracer.py)
[ ] Error handling: circuit breakers, loop detector, token budget (resilience/)
[ ] WebSocket chat endpoint (web/routers/chat.py)
[ ] Chat UI with WoW dark theme (templates/chat.html, static/css/theme.css)
[ ] Context assembly with token budgeting (llm/context.py)
[ ] 50+ unit tests
[ ] 10+ integration tests
```

**Eval gate:** Run domain eval suite. Agent answers 50%+ of factual questions correctly.

### Phase 2: Multi-Agent + Knowledge Building (3-4 weeks)

**Gate: Specialist agents cooperate on complex tasks. Wiki articles written and verified. Agentic RAG produces high-quality retrievals.**

```
[ ] Orchestrator: task decomposition, agent coordination (agents/orchestrator.py)
[ ] Researcher agent (agents/researcher.py)
[ ] Analyst agent (agents/analyst.py)
[ ] Writer agent (agents/writer.py)
[ ] Editor agent (agents/editor.py)
[ ] Message bus (agents/message_bus.py)
[ ] Agentic RAG: decomposer, iterative retrieval, self-RAG, corrective (rag/)
[ ] Warcraft Logs API client (OAuth + GraphQL)
[ ] Blizzard Battle.net API client (OAuth + REST)
[ ] WoWSims data importer
[ ] CMaNGOS game data importer
[ ] Source trust scoring + conflict resolution (trust/)
[ ] Content freshness checker (freshness/)
[ ] Backup system: SQLite online backup + wiki git bundles (backup/)
[ ] Ingest first wave of high-value sources
[ ] Agent writes first set of wiki articles
[ ] Wiki browser UI (templates/wiki/, markdown rendering, confidence badges)
[ ] Search UI (templates/search.html)
[ ] WoW-style item tooltips (static/js/tooltips.js, static/css/tooltips.css)
[ ] 30+ additional unit tests
[ ] 15+ integration tests
[ ] 3+ E2E workflow tests
```

**Eval gate:** RAG faithfulness > 0.8. Trajectory precision > 0.7. Domain accuracy > 70%.

### Phase 3: Simulation Engine (3-4 weeks)

**Gate: Sim produces DPS within 2% of WoWSims. Agent uses sim to verify claims.**

```
[ ] Wrap WoWSims as callable tool (immediate sim capability)
[ ] Core combat mechanics (sim/mechanics.py)
[ ] Rogue abilities (sim/abilities.py)
[ ] Talent system (sim/talents.py)
[ ] Combat spec rotation
[ ] Assassination spec rotation
[ ] Subtlety spec rotation
[ ] Buff/debuff/consumable system (sim/buffs.py)
[ ] Item proc modeling (sim/items.py)
[ ] Sim runner tool integration with analyst agent
[ ] Log analysis tool
[ ] Agent re-validates wiki articles using sim data
[ ] Sim API endpoint (/sim/)
[ ] Log upload + analysis report UI (/logs/)
[ ] Validate sim vs WoWSims + real WCL parses
[ ] 50+ sim-specific unit tests
```

**Eval gate:** Sim accuracy: P1 BiS combat swords on Patchwerk-style fight within 2% of WoWSims.

### Phase 4: Evaluation & Observability Polish (2-3 weeks)

**Gate: Full eval pipeline automated. Results tracked over time. Human feedback collected.**

```
[ ] Ragas integration for RAG quality metrics (evals/rag_eval.py)
[ ] Trajectory evaluation automation (evals/trajectory_eval.py)
[ ] Domain accuracy evaluation with 50+ questions (evals/domain_eval.py)
[ ] Eval dashboard UI (web/routers/evals.py, templates/evals/)
[ ] Agent trace viewer UI (web/routers/traces.py, templates/traces/)
[ ] Historical eval comparison (line charts over time)
[ ] Human feedback mechanism (thumbs up/down on answers)
[ ] Fine-tuning dataset curation:
    - 500+ Q&A pairs from verified wiki content
    - 200+ tool-calling examples from successful traces
    - 100+ routing classification examples
[ ] LoRA fine-tuning pipeline (documented)
```

### Phase 5: Polish & Growth (Ongoing)

```
[ ] Interactive talent tree UI
[ ] Sim builder visual interface
[ ] Fight timeline visualization
[ ] DPS breakdown charts (Chart.js)
[ ] Gear comparison tool
[ ] Agent confidence dashboard
[ ] PvP/Arena section build-out
[ ] Agent proactive gap research (scheduled wiki expansion)
[ ] Export wiki as shareable static site
[ ] LoRA fine-tuning execution
[ ] Phase 2-5 TBC content updates as raids release
```

---

## 22. Skills Learned

| Skill | Phase |
|-------|-------|
| Local LLM serving (vLLM + Ollama) | 0 |
| Multi-model routing and orchestration | 0-1 |
| Python web development (FastAPI) | 0-1 |
| Database design (SQLite + extensions) | 0-2 |
| Structured output engineering (Instructor + Pydantic) | 1 |
| Custom multi-agent architecture | 1-2 |
| LLM agent patterns (ReAct, reflection) | 1-2 |
| Agentic RAG (iterative, self-RAG, corrective) | 1-2 |
| WebSocket real-time communication | 1 |
| LLM observability (Langfuse tracing) | 1-4 |
| Circuit breaker and resilience patterns | 1 |
| API integration (OAuth, GraphQL, REST) | 2 |
| Web scraping + rate limiting + robots.txt | 2 |
| AI planning + task decomposition | 2 |
| Source evaluation + trust scoring + freshness | 2 |
| RAG evaluation (Ragas metrics) | 2-4 |
| Agent trajectory evaluation | 4 |
| Discrete event simulation | 3 |
| TBC game mechanic modeling | 3 |
| Data validation (sim vs real) | 3 |
| Custom UI/UX design | 2-4 |
| Data visualization | 4 |
| LoRA fine-tuning (domain adaptation) | 4-5 |
| Full-stack AI application design | All |

---

## Appendix: API Credentials Needed

| Service | Auth Type | How to Get |
|---------|----------|------------|
| Brave Search | API key | Register at brave.com/search/api |
| Warcraft Logs v2 | OAuth2 (client credentials) | Create client at warcraftlogs.com/api/clients |
| Blizzard Battle.net | OAuth2 (client credentials) | Register at develop.battle.net |
| Langfuse | Local keys | Self-hosted, keys in docker-compose config |
| Ollama | None (local) | N/A |
| vLLM | None (local) | N/A |

---

*Shukketsu (出血) — bleeding. Every rogue's favorite debuff.*
