# Phase 1: Agent Core

> Implementation plan. Each step produces a working system that does something
> visibly better than the previous one. Steps are ordered by dependency — each
> builds on what came before.

## Prerequisites

- NVIDIA AI Workbench container running (PyTorch 2.6, CUDA 12.6.3)
- vLLM serving Llama 3.3 70B on port 8000
- Ollama running on port 11434 (models pulled in Steps 5 and 7)
- Python dependencies installed from requirements.txt

## Phase Gate

Phase 1 is complete when: you can ask a question in the browser, the system
classifies it, routes it to the right model, searches its knowledge base,
and returns a cited answer — with traces visible in Langfuse.

---

## Step 1: Chat UI + Direct LLM

**Goal**: Type a question in the browser, get a streamed response from Llama 70B.

**What you learn**: FastAPI WebSockets, the OpenAI-compatible API that vLLM
exposes, server-sent streaming, basic HTML/CSS.

### What to build

- `web/app.py` — Add WebSocket endpoint at `/ws/chat`
- `web/routers/chat.py` — Chat router with WebSocket handler
- `web/templates/base.html` — Shared layout (dark theme, nav)
- `web/templates/chat.html` — Chat interface (message list, input box)
- `web/static/css/theme.css` — Dark WoW-inspired base theme
- `web/static/js/chat.js` — WebSocket client, renders streamed tokens
- `llm/clients.py` — AsyncOpenAI client pointing at vLLM

### How it works

```
Browser opens WebSocket to /ws/chat
  --> user sends {"message": "What is Sinister Strike?"}
  --> FastAPI handler calls vLLM via OpenAI client (stream=True)
  --> each token chunk sent back over WebSocket
  --> browser JS appends tokens to the page
```

The vLLM server exposes an OpenAI-compatible API. That means you use the
standard `openai` Python library, just pointed at localhost:8000 instead of
api.openai.com. This is a pattern you'll see everywhere in local LLM work.

### Key decisions

- **Streaming vs. wait-for-complete**: Stream. A 70B model takes 2-5 seconds
  for a full response. Streaming shows tokens as they arrive, which feels
  much faster. vLLM supports this natively.
- **WebSocket vs. SSE**: WebSocket, because we need bidirectional communication
  (user sends messages, server sends tokens). SSE is one-directional.
- **Chat history**: In-memory per WebSocket connection for now. No database
  persistence yet — that's Step 2.

### Tests

- Unit: test that the WebSocket endpoint accepts connections
- Unit: test that malformed messages get an error response
- Manual: open browser, type a question, see streamed response

### Gate

Open http://localhost:9000/chat/, type "What is a rogue?", see tokens appear
one by one.

---

## Step 2: Database Foundation

**Goal**: SQLite database created with schema, connection management, test
fixtures working.

**What you learn**: SQLite WAL mode (concurrent reads during writes), FTS5
(full-text search built into SQLite), sqlite-vec (vector search in SQLite),
schema design for a RAG system.

### What to build

- `db/schema.sql` — DDL for all Phase 1 tables
- `db/connection.py` — Connection factory with WAL mode + pragmas
- Update `tests/conftest.py` — Verify test_db fixture works with real schema

### Schema (Phase 1 subset)

Only the tables needed for Phase 1. Game data tables (spells, items, talents,
bosses) come in Phase 2-3 when we build the sim engine.

```sql
-- Sources we've ingested content from
CREATE TABLE sources (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE,
    title TEXT,
    source_type TEXT,
    trust_score REAL DEFAULT 0.3,
    fetched_at TIMESTAMP,
    content_hash TEXT,
    chunk_count INTEGER DEFAULT 0
);

-- Text chunks extracted from sources
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    source_id INTEGER REFERENCES sources(id),
    content TEXT NOT NULL,
    chunk_index INTEGER,
    metadata_json TEXT
);

-- Vector embeddings for semantic search
CREATE VIRTUAL TABLE chunks_vec USING vec0(
    id INTEGER PRIMARY KEY,
    embedding FLOAT[768]
);

-- Full-text search index
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    content=chunks,
    content_rowid=id
);

-- Wiki article metadata (articles themselves are Markdown files)
CREATE TABLE articles (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,
    title TEXT,
    last_updated TIMESTAMP,
    confidence_score REAL,
    needs_review BOOLEAN DEFAULT FALSE
);
```

### Connection management

```python
# Key pragmas set on every connection:
PRAGMA journal_mode=WAL;       # Write-Ahead Logging: readers don't block writers
PRAGMA synchronous=NORMAL;     # Fsync on checkpoint only (safe with WAL)
PRAGMA busy_timeout=5000;      # Wait up to 5s if another writer has the lock
PRAGMA foreign_keys=ON;        # Enforce referential integrity
```

WAL mode is important to understand: without it, SQLite locks the entire
database during writes. With WAL, readers and one writer can operate
concurrently. This matters because our agent might be writing ingested
chunks while a search query reads them.

### Key decisions

- **sqlite-vec on ARM64**: May need compilation from source. If unavailable,
  fall back to storing embeddings as BLOBs and computing distance in Python.
  Slower but functional.
- **Minimal schema**: Only RAG tables now. Adding tables later is trivial
  with SQLite (just run CREATE TABLE).

### Tests

- Unit: test connection factory returns properly configured connection
- Unit: test schema creates all tables without errors
- Unit: test WAL mode is active
- Unit: test foreign key enforcement works

### Gate

`pytest tests/unit/test_db.py -v` passes. Test fixture creates a real
database from schema.sql.

---

## Step 3: Structured Output

**Goal**: LLM returns validated Pydantic models instead of free text.

**What you learn**: Instructor library, Pydantic model design for LLM output,
automatic retry on validation failure, the difference between hoping the LLM
formats correctly vs. guaranteeing it.

### What to build

- `llm/structured.py` — Instructor-wrapped vLLM client + helper function
- `llm/schemas.py` — Pydantic response models (AgentStep, ToolCall, etc.)

### How Instructor works

Without Instructor, you prompt the LLM to return JSON and hope it does.
With Instructor, you define a Pydantic model and the library:

1. Adds the schema to the prompt as a JSON schema
2. Parses the LLM's response as JSON
3. Validates against the Pydantic model
4. If validation fails, re-prompts with the error message
5. Repeats up to `max_retries` times

```python
import instructor
from openai import AsyncOpenAI

client = instructor.from_openai(
    AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed"),
    mode=instructor.Mode.JSON
)

# Now every call returns a validated Pydantic model:
result = await client.chat.completions.create(
    model="llama-3.3-70b-instruct-awq",
    response_model=AgentStep,
    messages=[{"role": "user", "content": "What is the hit cap?"}],
    max_retries=3,
    temperature=0.1
)
# result is guaranteed to be a valid AgentStep instance
```

### Key schemas

```python
class ToolCall(BaseModel):
    thought: str       # Why the agent wants to call this tool
    tool_name: str     # Which tool
    tool_input: dict   # Arguments

class AgentStep(BaseModel):
    reasoning: str
    action: Literal["tool_call", "final_answer"]
    tool_call: ToolCall | None = None
    answer: str | None = None
```

### Key decisions

- **Mode.JSON vs Mode.TOOLS**: JSON mode. vLLM's guided decoding constrains
  token generation to valid JSON, making it more reliable than function-calling
  mode for structured output.
- **Temperature 0.1**: Low creativity, high consistency. We want the same
  question to produce structurally similar responses.
- **max_retries=3**: If the model fails 3 times to produce valid output,
  something is wrong with the schema or prompt, not bad luck.

### Tests

- Unit: test Pydantic schemas validate correct input
- Unit: test schemas reject invalid input (missing fields, wrong types)
- Integration: test Instructor client returns validated model from vLLM

### Gate

A Python script calls the Instructor client with a question, gets back a
valid `AgentStep` object. Print it, see the typed fields.

---

## Step 4: First Tool + ReAct Loop

**Goal**: Agent uses a `rag_search` tool to answer questions from the knowledge
base. This is the core agent pattern everything else builds on.

**What you learn**: The ReAct agent loop, tool registration and dispatch, how
an LLM decides which tool to call, the observation cycle.

### What to build

- `agents/base.py` — BaseAgent with ReAct loop
- `tools/registry.py` — Tool registration and dispatch
- `tools/schemas.py` — Tool input/output Pydantic schemas
- `tools/knowledge/search.py` — rag_search tool (vector-only for now, hybrid in Step 6)

### The ReAct loop

```python
class BaseAgent:
    async def run(self, query: str) -> str:
        scratchpad = []

        for i in range(self.max_iterations):
            # Build context: system prompt + query + scratchpad history
            messages = self._build_messages(query, scratchpad)

            # Ask LLM for next step (structured output from Step 3)
            step: AgentStep = await self.llm.create(
                response_model=AgentStep,
                messages=messages
            )

            if step.action == "final_answer":
                return step.answer

            # Execute the tool
            tool = self.tools[step.tool_call.tool_name]
            observation = await tool.execute(step.tool_call.tool_input)

            # Add to scratchpad so LLM sees what happened
            scratchpad.append({
                "thought": step.reasoning,
                "tool": step.tool_call.tool_name,
                "input": step.tool_call.tool_input,
                "observation": observation
            })

        return "I wasn't able to find a complete answer."
```

The scratchpad is the key insight: each iteration, the LLM sees all previous
thoughts and observations. It's building up context about what it's tried and
what it's learned, letting it decide what to do next.

### Tool registration

```python
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get_tool_descriptions(self) -> str:
        """Format tool descriptions for the system prompt."""
        ...

    async def execute(self, name: str, input: dict) -> str:
        """Validate input, execute tool, return observation."""
        ...
```

### Key decisions

- **Max iterations = 5** for Phase 1 (not 15). The agent has one tool. If it
  can't answer in 5 tries, more iterations won't help.
- **Graceful failure**: If max iterations reached, return "I don't know" rather
  than crashing. Users prefer an honest "I don't know" over silence.
- **No reflection step yet**: The original design has a self-evaluation step.
  Add it in Phase 2 when there's enough agent behavior to evaluate.

### Tests

- Unit: test ReAct loop calls tool when LLM returns tool_call action
- Unit: test ReAct loop returns answer when LLM returns final_answer
- Unit: test max_iterations limit is enforced
- Unit: test tool registry dispatches to correct tool
- Unit: test unknown tool name returns error observation

### Gate

In the chat UI, type "What is the hit cap for combat rogues?" The agent calls
rag_search, gets relevant chunks, synthesizes an answer. (Requires Step 5 to
have ingested some content first — these two steps work together.)

---

## Step 5: Ingest Pipeline

**Goal**: Feed a web page (or text) into the system: chunk it, embed it, store
it in SQLite. Then retrieve it via the agent's rag_search tool.

**What you learn**: Semantic chunking (splitting text into meaningful segments),
text embeddings (turning text into numbers that capture meaning), vector storage.

### What to build

- `ingest/chunker.py` — Semantic text chunker
- `ingest/embedder.py` — Embedding via nomic-embed-text-v2 (Ollama)
- `ingest/pipeline.py` — Orchestrates: text -> chunks -> embeddings -> store

### How chunking works

You can't embed an entire 5,000-word article as one vector — the meaning gets
diluted. Instead, split into ~400-token chunks with 50-token overlap.

**Splitting strategy** (in priority order):
1. Split on `##` headers (strongest boundary)
2. Split on blank lines (paragraph boundary)
3. Split on sentences (fallback)
4. If a sentence exceeds max_tokens, split on newlines

The overlap matters: if a relevant fact spans a chunk boundary, the overlap
ensures at least one chunk contains the full context.

### How embedding works

nomic-embed-text-v2 takes text in, returns a 768-dimensional vector out. Texts
with similar meaning produce similar vectors (high cosine similarity).

```python
# Important: nomic-embed-text requires task prefixes
# For documents being stored:
"search_document: The hit cap for TBC rogues is 9% (142 hit rating)..."

# For queries at search time:
"search_query: what is the hit cap for combat rogues"
```

These prefixes tell the model whether it's embedding a document (for storage)
or a query (for retrieval). This asymmetry improves retrieval quality.

### Key decisions

- **Chunk size 400 tokens**: Small enough to be specific, large enough to have
  context. This is an industry standard starting point.
- **Token counting**: Use character approximation (1 token ~ 4 chars) for
  chunking. Exact token counts aren't critical for chunk boundaries.
- **WoW-specific chunker deferred**: The original design has special rules for
  item tooltips, talent tables, etc. Build the basic chunker first; add WoW
  rules in Phase 2 when we see real data and know which rules actually matter.

### Tests

- Unit: test chunker splits on headers
- Unit: test chunker respects max_tokens
- Unit: test chunker adds overlap
- Unit: test chunks below min_tokens get merged with neighbors
- Integration: test embedder returns 768-dim vector
- Integration: test full pipeline stores chunks + vectors in SQLite

### Gate

Run the pipeline on a test document (e.g., paste the hit cap section from a
rogue guide). Then call rag_search — the relevant chunks come back.

---

## Step 6: Hybrid Search

**Goal**: Search combines vector similarity (semantic) with FTS5 keyword
matching, merged via Reciprocal Rank Fusion.

**What you learn**: Two fundamentally different search paradigms and why
combining them beats either alone.

### What to build

- `tools/knowledge/search.py` — Upgrade from vector-only to hybrid search
- Utility function for RRF score computation

### Why hybrid?

**Vector search** understands meaning: "best trinket for combat" matches
"Dragonspine Trophy is BiS" even though they share no words. But it can miss
exact terms: searching for "DST" might not match "Dragonspine Trophy".

**FTS5 keyword search** matches exact words: "DST proc rate" finds documents
containing those exact terms. But it misses synonyms and paraphrases.

**Combined**: you get semantic understanding AND exact matching.

### Reciprocal Rank Fusion (RRF)

Two ranked lists (vector results and FTS5 results) need to be merged. RRF is
the simplest effective method:

```python
def compute_rrf(vector_ranks: dict, fts_ranks: dict, k: int = 60) -> dict:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    For each document, score = sum(1 / (k + rank)) across all lists
    where it appears. k=60 is the standard constant that prevents
    top-ranked items from dominating too aggressively.
    """
    scores = defaultdict(float)
    for doc_id, rank in vector_ranks.items():
        scores[doc_id] += 1.0 / (k + rank)
    for doc_id, rank in fts_ranks.items():
        scores[doc_id] += 1.0 / (k + rank)
    return scores
```

### Tests

- Unit: test RRF merges two disjoint lists correctly
- Unit: test RRF with overlapping results ranks overlap higher
- Unit: test RRF with one empty list returns other list's order
- Integration: test hybrid search returns results from both vector and FTS5

### Gate

Ingest a document about Dragonspine Trophy. Search for "DST" (keyword match)
and "best haste trinket" (semantic match). Both return the DST chunk.

---

## Step 7: Multi-Model Router

**Goal**: Simple questions go to Qwen 4B (<200ms), complex ones go to the
agent (Llama 70B). First time using two models together.

**What you learn**: Multi-model orchestration, query classification, the
tradeoff between routing overhead and compute savings.

### What to build

- `routing/router.py` — Qwen 4B classification via Ollama + Instructor
- Update `web/routers/chat.py` — Route based on classification result

### How routing works

```
User: "What's Sinister Strike's energy cost?"
  --> Qwen 4B classifies: TRIVIAL / RETRIEVAL
  --> Qwen answers directly: "45 energy"
  --> Total latency: ~200ms

User: "Compare combat swords vs combat fists for Gruul"
  --> Qwen 4B classifies: COMPLEX / ANALYSIS
  --> Forwards to Llama 70B agent with tools
  --> Total latency: 5-15 seconds (but worth it for quality)
```

### Ollama client

```python
# Ollama also exposes an OpenAI-compatible API
ollama_client = instructor.from_openai(
    AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="not-needed"),
    mode=instructor.Mode.JSON
)
```

### Key decisions

- **Fallback if Ollama/Qwen is down**: Send everything to Llama 70B. The
  system works fine without routing — it's just slower.
- **Don't over-classify**: Start with just TRIVIAL vs. everything else. The
  full TRIVIAL/MODERATE/COMPLEX split can come later when we have enough
  query examples to see where the boundaries should be.
- **Routing adds latency to every query**: ~200ms. For complex queries that
  take 10 seconds, this is negligible. For trivial queries that Qwen answers
  in 200ms, it's the entire latency. Worth it.

### Tests

- Unit: test RoutingDecision model validates correctly (already exists)
- Integration: test Qwen classifies simple factual question as TRIVIAL
- Integration: test Qwen classifies comparison question as COMPLEX
- Unit: test fallback routes to 70B when Ollama unreachable

### Gate

Ask "What's Sinister Strike's energy cost?" — see fast response (routed to
Qwen). Ask "Compare combat vs assassination for Karazhan" — see slower but
thorough response (routed to agent).

---

## Step 8: Web Search + Ingest Tools

**Goal**: Agent can discover new information from the web and permanently
ingest it into the knowledge base.

**What you learn**: External API integration (Brave Search), ethical web
scraping (rate limiting, robots.txt), how an agent builds its own knowledge.

### What to build

- `tools/research/web_search.py` — Brave Search API client
- `tools/research/web_ingest.py` — Fetch URL, extract text, run ingest pipeline
- `scraping/rate_limiter.py` — Per-domain rate limiting
- `scraping/robots.py` — robots.txt fetching and compliance
- `scraping/fetcher.py` — httpx page fetcher with retries

### How the agent uses these tools

```
User: "What's the best off-hand weapon for combat in Phase 1?"

Agent THINK: "I don't have enough info in the KB. Let me search the web."
Agent ACT:   web_search("TBC combat rogue phase 1 offhand BiS")
Agent OBSERVE: [list of URLs and snippets from Brave Search]

Agent THINK: "The ShadowPanther link looks authoritative. Let me ingest it."
Agent ACT:   web_ingest("https://shadowpanther.net/tbc-combat-offhand.html")
Agent OBSERVE: "Ingested 12 chunks from shadowpanther.net"

Agent THINK: "Now I can search for the answer in the KB."
Agent ACT:   rag_search("combat rogue phase 1 offhand weapon")
Agent OBSERVE: [chunks about Latro's Shifting Sword, Felsteel Whisper Knives...]

Agent ACT:   final_answer("The best off-hand options for Phase 1 combat are...")
```

This is a powerful pattern: the agent is **building its own knowledge base** as
it answers questions. The next time someone asks about combat off-hands, the
information is already ingested.

### Rate limiting

```python
DOMAIN_POLICIES = {
    "warcraftlogs.com": 2.0,     # seconds between requests
    "wowhead.com": 3.0,
    "silentshadows.net": 5.0,    # small site, be gentle
    "__default__": 3.0,
}
```

Be respectful of small community sites. They're serving us valuable content
for free.

### Key decisions

- **Brave Search free tier**: 2,000 queries/month. Enough for development.
  Track usage to avoid surprises.
- **robots.txt compliance**: Always check before scraping. Cache per domain
  for 24 hours.
- **User-Agent honesty**: Identify ourselves, don't pretend to be a browser.

### Tests

- Unit: test rate limiter enforces delay between requests
- Unit: test robots.txt parser allows/disallows correctly
- Unit: test Brave Search response parsing
- Integration: test web_search returns results for known query
- Integration: test web_ingest stores chunks in database

### Gate

Ask the agent a question it can't answer from the KB. Watch it search the web,
ingest a source, then answer using the newly ingested content.

---

## Step 9: Resilience

**Goal**: System handles failures gracefully instead of crashing.

**What you learn**: Circuit breaker pattern, exponential backoff, loop
detection — production reliability patterns used in real distributed systems.

### What to build

- `resilience/circuit_breaker.py` — Circuit breaker for tools and services
- `resilience/retry.py` — Exponential backoff retry wrapper
- `agents/guardrails.py` — Loop detector, token budget enforcement
- Update tools to use circuit breakers

### Circuit breaker pattern

A circuit breaker tracks failures for a specific service (e.g., Brave Search
API). Three states:

```
CLOSED (normal)     -- requests pass through
  --> 5 consecutive failures
OPEN (failing)      -- requests rejected immediately (don't waste time)
  --> wait 60 seconds
HALF_OPEN (testing) -- allow ONE request through
  --> success: back to CLOSED
  --> failure: back to OPEN
```

This prevents cascading failures. If Brave Search is down, the circuit opens
and the agent immediately falls back to KB-only answers instead of waiting for
timeouts on every query.

### Loop detection

Agents can get stuck: calling the same tool with the same input repeatedly.
The loop detector watches for this:

```python
class LoopDetector:
    MAX_ITERATIONS = 15
    MAX_CONSECUTIVE_SAME_TOOL = 3
    MAX_TOTAL_TOKENS = 100_000

    def check(self, history: list[dict]) -> str | None:
        """Returns error message if loop detected, None if OK."""
```

### Graceful degradation table

| Failure | What happens |
|---------|-------------|
| vLLM timeout | Retry once, then return partial answer |
| Ollama down | Skip routing, send everything to vLLM |
| Web search fails | Fall back to KB-only search |
| Tool execution error | Log it, tell agent "tool unavailable", continue |
| Agent loop detected | Force final_answer with partial results |
| SQLite locked | Retry with exponential backoff (up to 5 attempts) |
| Embedding model down | Use FTS5 keyword search only |

### Tests

- Unit: test circuit breaker state transitions (CLOSED -> OPEN -> HALF_OPEN)
- Unit: test circuit breaker resets on success
- Unit: test loop detector catches repeated tool calls
- Unit: test loop detector catches token budget exhaustion
- Unit: test exponential backoff timing

### Gate

Kill Ollama while chatting. System should continue working (routes everything
to vLLM). Restart Ollama — system recovers automatically.

---

## Step 10: Observability

**Goal**: See what the agent is doing internally — every routing decision,
tool call, and LLM generation traced and viewable.

**What you learn**: LLM-specific observability (different from traditional
APM), trace/span/generation hierarchy, how to debug agent behavior.

### What to build

- `observability/tracer.py` — Langfuse client wrapper
- `infra/docker-compose.langfuse.yml` — Langfuse stack (Postgres + ClickHouse)
- Instrument: router, agent loop, tool calls, LLM generations
- `web/routers/traces.py` — Link to Langfuse UI from Shukketsu

### What gets traced

Every user query creates a **trace** with nested **spans**:

```
Trace: "user_query_abc123"
+-- Span: "routing" (model=qwen-4b, 180ms, 50 tokens)
|   +-- Generation: {complexity: "moderate", category: "retrieval"}
+-- Span: "agent.react_loop"
|   +-- Span: "step_1"
|   |   +-- Generation: thought="need to search KB"
|   |   +-- Span: "tool.rag_search" (45ms)
|   +-- Span: "step_2"
|       +-- Generation: final_answer="The hit cap is..."
+-- Score: {latency: 3.2s, tokens: 850}
```

### Why Langfuse?

- Purpose-built for LLM tracing (vs. generic APM like Datadog/Jaeger)
- Self-hosted, MIT license, no cloud dependency
- Built-in UI for browsing traces, viewing token usage, comparing runs
- Scores system lets you track quality metrics alongside traces

### Key decisions

- **This is Step 10, not Step 1**: Traditional advice says "add observability
  first." For LLM apps, the tracing infrastructure (Langfuse needs Postgres +
  ClickHouse) is heavy. Python's `logging` module works fine for debugging
  during Steps 1-9. Langfuse is worth it once you have agent behavior to
  analyze.
- **Memory budget**: Langfuse stack uses ~2-4 GB RAM. Fine on the DGX Spark
  with 128 GB.

### Tests

- Integration: test tracer creates trace in Langfuse
- Integration: test nested spans appear correctly
- Manual: browse Langfuse UI, find a trace, expand the span tree

### Gate

Ask a question, then open Langfuse UI at localhost:3000. Find the trace,
see the full routing → agent → tool → response span tree.

---

## Step Summary

| Step | Adds | External Deps | Approx Files |
|------|------|--------------|-------------|
| 1 | Chat UI + streaming LLM | vLLM | 7 |
| 2 | Database + schema | SQLite | 3 |
| 3 | Structured output | Instructor | 2 |
| 4 | Agent + ReAct + first tool | — | 4 |
| 5 | Chunking + embedding pipeline | Ollama (nomic-embed) | 3 |
| 6 | Hybrid search (vector + FTS5 + RRF) | — | 1 (upgrade) |
| 7 | Multi-model router | Ollama (Qwen 4B) | 2 |
| 8 | Web search + ingest tools | Brave API, httpx | 5 |
| 9 | Circuit breakers + resilience | — | 3 |
| 10 | Langfuse tracing | Docker (Postgres, ClickHouse) | 3 |

Total: ~33 files, each building on the last.
