# Shukketsu Architecture

> Reference document. Describes the system design and rationale.
> Not an implementation plan — see `phase-1-agent-core.md` for that.

## Context

- **Hardware**: NVIDIA DGX Spark (GB10 Grace Blackwell, 128GB unified memory, ARM64, 6144 CUDA cores)
- **Environment**: NVIDIA AI Workbench container (PyTorch 2.6, CUDA 12.6.3, Ubuntu 24.04)
- **Game**: TBC Classic Anniversary Edition (launched February 5, 2026)
- **Current Phase**: Phase 1 (Karazhan, Gruul, Magtheridon)

## What This System Does

Shukketsu is a local AI-powered research agent for WoW TBC Rogues. You ask it questions ("What's the hit cap for combat swords?", "Compare DST vs Bloodlust Brooch", "Analyze this Warcraft Logs parse"), and it:

1. Searches its knowledge base and the web
2. Runs simulations when needed
3. Synthesizes an answer with cited sources
4. Writes and maintains wiki articles for future reference

Everything runs locally on the DGX Spark. No cloud LLM APIs.

---

## Three-Model Architecture

Three models serve three distinct roles. This matters because a 70B model is overkill for "is this question simple?" and too slow for embedding 500 chunks.

```
User Query
    |
    v
Qwen3 4B (Ollama :11434)       ~2.5 GB, <200ms
  Role: classify query complexity, answer trivial questions
    |
    +-- TRIVIAL --> Qwen answers directly
    +-- MODERATE/COMPLEX --> forwards to:
    v
Llama 3.3 70B AWQ INT4 (vLLM :8000)    ~35 GB weights + ~25 GB KV cache
  Role: all substantive reasoning, tool calling, article writing

nomic-embed-text-v2 (Ollama :11434)    ~500 MB
  Role: embed queries and document chunks for vector search
```

**Why three models instead of one?**
- The 70B model takes 2-5 seconds per response. For "What's Sinister Strike's energy cost?" that's wasteful — Qwen answers in <200ms.
- Embedding models are architecturally different from generative models. nomic-embed-text-v2 produces 768-dimensional vectors; the generative models can't do this efficiently.
- Total memory: ~64 GB of 128 GB. Plenty of headroom.

**Why these specific models?**
- Llama 3.3 70B: strong instruction following, native function calling, AWQ INT4 fits in ~35 GB.
- Qwen3 4B: fast enough for classification, small enough to coexist with 70B.
- nomic-embed-text-v2: good quality at 768 dimensions, runs on Ollama alongside Qwen.

> **Design flag — ARM64 compatibility**: The DGX Spark uses an ARM64 (Grace) CPU.
> vLLM's ARM64 support should be verified during Phase 0. If there are issues,
> the fallback is running the 70B model through Ollama as well (slower but works).

---

## Agent Architecture

### The ReAct Pattern

Each agent follows a **ReAct loop** (Reason + Act). This is the core pattern:

```
1. THINK: "I need to find the hit cap. Let me search the knowledge base."
2. ACT:   call rag_search(query="TBC rogue hit cap")
3. OBSERVE: [results about hit rating, 9% cap, Precision talent...]
4. THINK: "I have the answer. The hit cap is 9% (142 rating), 6% with Precision."
5. ACT:   final_answer("The hit cap is 9%...")
```

The agent loops through think-act-observe until it has enough information, then delivers a final answer. A **reflection** step optionally checks the answer quality before returning it.

### Why No Framework?

LangChain, LlamaIndex, CrewAI, etc. all add abstraction layers that hide what's happening. Since this project is about learning, the agents are plain Python classes. You'll understand every line because you wrote it.

### Agent Roles (Target State)

The system grows into five specialists, but **starts with one general agent**:

| Agent | Role | When It's Built |
|-------|------|-----------------|
| **General Agent** | Answers questions using tools | Phase 1, Step 4 |
| **Orchestrator** | Decomposes complex tasks, coordinates agents | Phase 2 |
| **Researcher** | Information gathering specialist | Phase 2 |
| **Analyst** | Simulation, log analysis, quantitative work | Phase 2-3 |
| **Writer** | Wiki article creation | Phase 2 |
| **Editor** | Fact-checking, verification | Phase 2 |

The General Agent is promoted to Researcher when we add more specialists. The Orchestrator routes complex queries to the right specialist.

### Structured Output

All LLM responses are typed via **Instructor + Pydantic**. Instead of parsing free-text, the model returns validated Python objects:

```python
class AgentStep(BaseModel):
    reasoning: str                           # Chain-of-thought
    action: Literal["tool_call", "final_answer"]
    tool_call: ToolCall | None = None
    answer: str | None = None
```

If the model returns invalid JSON, Instructor re-prompts with the validation error automatically. This eliminates most "LLM returned garbage" failures.

---

## Storage Architecture

### SQLite + Extensions

One database file (`data/shukketsu.db`) serves three purposes:

| Purpose | Technology | What It Does |
|---------|-----------|--------------|
| Structured data | SQLite tables | Game data (spells, items, talents, bosses), sources, articles, agent logs |
| Vector search | sqlite-vec extension | Cosine similarity search over 768-dim embeddings |
| Keyword search | FTS5 extension | BM25-ranked full-text search over chunk content |

**Why SQLite instead of Postgres?**
- Single file, zero configuration, zero maintenance
- WAL mode gives concurrent read/write
- sqlite-vec and FTS5 eliminate the need for separate vector DB and search engine
- This is a single-user local application — SQLite's concurrency limits don't matter

**Hybrid search** combines both approaches using Reciprocal Rank Fusion (RRF):
- Vector search finds semantically similar content ("what's the best trinket" matches "Dragonspine Trophy is BiS")
- FTS5 finds exact keyword matches ("DST proc rate" matches documents containing those exact terms)
- RRF merges both ranked lists into one, getting the benefits of both

### Wiki Storage

Agent-written articles are Markdown files in `knowledge/`, tracked in git. Each has YAML frontmatter with confidence scores, sources, and tags. Git gives free version history.

```
knowledge/
  specs/combat/         # Combat Rogue guides
  specs/assassination/  # Assassination guides
  specs/subtlety/       # Subtlety guides
  gearing/              # BiS lists, stat weights, gems/enchants
  encounters/           # Boss-specific guides
  pvp/                  # Arena and PvP guides
  fundamentals/         # Hit cap, expertise, crit, poisons, etc.
```

---

## Web Architecture

**FastAPI + Jinja2 + HTMX** — server-rendered HTML with interactive sprinkles.

**Why not React/Vue/Next.js?**
- No JavaScript build chain to maintain
- HTMX gives AJAX-like interactivity with HTML attributes
- Server-rendered pages are simpler to reason about
- The only JS-heavy features (chat WebSocket, tooltips, charts) use vanilla JS

### Interfaces

| Route | Purpose |
|-------|---------|
| `/chat/` | WebSocket chat with the agent |
| `/wiki/` | Browse wiki articles (rendered Markdown) |
| `/search/` | Hybrid search across all content |
| `/logs/` | Combat log upload and analysis (Phase 3) |
| `/sim/` | DPS simulation builder (Phase 3) |
| `/evals/` | Evaluation dashboard (Phase 4) |
| `/traces/` | Agent trace viewer (Phase 4) |

### Theme

Dark WoW-inspired UI: rogue class yellow (#FFF468), item quality colors (grey/white/green/blue/purple/orange), tooltip-style hover cards. CSS custom properties, no framework.

---

## Data Flow

### Simple Query (Trivial)
```
Browser --> WebSocket --> Qwen 4B classifies as TRIVIAL
                          Qwen 4B answers directly --> Browser
```

### Knowledge Query (Moderate)
```
Browser --> WebSocket --> Qwen 4B classifies as MODERATE
                          --> Agent ReAct loop:
                              THINK: need to search KB
                              ACT: rag_search("hit cap combat rogue")
                              OBSERVE: [chunks about hit rating]
                              THINK: have enough info
                              ACT: final_answer(...)
                          --> Browser
```

### Research Query (Complex, Phase 2+)
```
Browser --> WebSocket --> Qwen 4B classifies as COMPLEX
                          --> Orchestrator decomposes into sub-tasks
                          --> Researcher gathers information
                          --> Analyst runs simulations
                          --> Writer synthesizes article
                          --> Editor fact-checks
                          --> Browser
```

---

## Source Trust System

Not all information is equally reliable. Each source gets a base trust score:

| Source Type | Trust | Examples |
|------------|-------|---------|
| Game data | 1.0 | Blizzard API, CMaNGOS spell database |
| Simulation | 0.9 | WoWSims, our sim engine |
| Combat logs | 0.85 | Warcraft Logs real parses |
| Expert guides | 0.75 | Icy Veins, ShadowPanther, Sno |
| Archived theory | 0.7 | Elitist Jerks (correct but possibly outdated) |
| Community | 0.5 | Reddit, forums |
| Unknown | 0.3 | Unclassified web content |

Trust **decays over time** — a guide from 6 months ago is less trustworthy than one from last week. The decay has a **floor of 0.1** to prevent old-but-valid content from becoming invisible (TBC game mechanics don't change).

When sources conflict: higher trust wins by default. If simulatable, run a sim to settle empirically. If ambiguous, present both views.

---

## Project Structure

```
code/shukketsu/              # Main package (import as code.shukketsu)
  config.py                  # All env vars, model URLs, agent defaults
  routing/                   # Qwen 4B query classification
  agents/                    # BaseAgent, specialists (Phase 2)
  llm/                       # vLLM + Ollama clients, Instructor, prompts
  tools/                     # Agent tools: research/, analysis/, knowledge/
  rag/                       # Agentic RAG: decompose, iterate, self-RAG, correct
  ingest/                    # Chunking + embedding pipeline
  scraping/                  # Rate limiter, robots.txt, httpx fetcher
  sim/                       # DPS simulation engine (Phase 3)
  db/                        # SQLite connection, schema.sql, migrations
  web/                       # FastAPI app, templates, static assets
  trust/                     # Source trust scoring + decay
  freshness/                 # Content staleness management
  resilience/                # Circuit breakers, retries, error taxonomy
  observability/             # Langfuse tracing (Phase 1 Step 10)
  evals/                     # Quality evaluation framework (Phase 4)
  backup/                    # SQLite backup + integrity (Phase 2)
tests/unit/                  # Pure logic, no external deps
tests/integration/           # Needs Ollama + SQLite
tests/e2e/                   # Needs all services
knowledge/                   # Git-tracked Markdown wiki
data/                        # Database, backups (git-lfs); scratch/ gitignored
infra/                       # Docker Compose (Langfuse), model start scripts
```

---

## Workbench Integration

| File | Purpose |
|------|---------|
| `apt.txt` | System packages (build-essential, libsqlite3-dev) |
| `requirements.txt` | Python dependencies (pip install) |
| `variables.env` | Environment variables (model URLs, API keys, paths) |
| `preBuild.bash` | Pre-build: installs Node.js + Ollama |
| `postBuild.bash` | Post-build: installs Claude Code CLI |
| `.project/spec.yaml` | Workbench config (apps, mounts, base image) |

The Shukketsu web UI is registered as a Workbench app on port 9000.

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM (reasoning) | Llama 3.3 70B AWQ INT4 via vLLM | Fits in ~35 GB, strong tool calling |
| LLM (routing) | Qwen3 4B via Ollama | Fast classification, ~2.5 GB |
| LLM (embeddings) | nomic-embed-text-v2 via Ollama | 768-dim, ~500 MB |
| Structured output | Instructor + Pydantic | Typed tool calling with retry |
| Web framework | FastAPI + Uvicorn | Async, WebSocket, production-grade |
| Templating | Jinja2 + HTMX | Server-rendered, no JS build chain |
| Database | SQLite + sqlite-vec + FTS5 | One file, three capabilities |
| Web scraping | httpx + BeautifulSoup4 | Async HTTP + HTML parsing |
| Web search | Brave Search API | Free tier (2,000/month) |
| Observability | Langfuse (self-hosted) | LLM-specific tracing with UI |
| Simulation | Custom Python + NumPy | TBC combat mechanics from first principles |
| Testing | pytest + pytest-asyncio + pytest-cov | Full test pyramid |
| Linting | Ruff + mypy | Fast linting + type checking |
