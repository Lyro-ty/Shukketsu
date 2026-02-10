# Phase 2: Multi-Agent + Agentic RAG

> Implementation plan. Each step produces a working system that does something
> visibly better than the previous one. Steps are ordered by dependency — each
> builds on what came before.

## Prerequisites

- Phase 1 complete (245 tests passing, all 10 steps done)
- Ollama running on port 11434 (Llama 3.3 70B, Qwen3 4B, nomic-embed-text-v2)
- Langfuse stack running (Docker Compose)
- SQLite database initialized with Phase 1 schema

## Phase Gate

Phase 2 is complete when: you can ask a complex multi-part question in the
browser, the Orchestrator decomposes it into sub-tasks, specialist agents
(Researcher, Writer, Editor) cooperate to answer it, wiki articles are produced
and verified, and the knowledge graph enables multi-hop reasoning — with all
traces visible in Langfuse.

**Metrics:**
- RAG faithfulness > 0.8
- Agent trajectory precision > 0.7
- Domain accuracy > 70%
- Wiki has published articles covering all three rogue specs

---

## Scope & Architecture Overview

Phase 2 delivers four agents (Orchestrator, Researcher, Writer, Editor)
communicating via a Structured Task Protocol, backed by a GraphRAG-enhanced
retrieval system with Qwen 4B reranking, producing a human-approved wiki.

The Analyst agent is deferred to Phase 3 (requires simulation engine). API
integrations (Warcraft Logs, Blizzard) are deferred to Phase 2b.

### System Flow

```
User query
  → Qwen 4B router (existing, classifies complexity)
  → If TRIVIAL: direct answer (existing)
  → If MODERATE: Researcher agent solo
  → If COMPLEX: Orchestrator decomposes into sub-tasks
      → Researcher(s) gather information
      → Writer drafts article if enough coverage
      → Editor fact-checks against KB
      → Orchestrator synthesizes final response
```

### Key Architectural Decisions

- **Structured Task Protocol** — Typed Pydantic Task/Result objects. Agents are
  pure async functions. All communication flows through the Orchestrator. No
  message bus, no queues. Maximizes testability and traceability.

- **GraphRAG via SQLite** — `entities` and `relationships` tables with a
  predefined WoW TBC entity schema. Extracted during ingest alongside
  chunks/embeddings. No Neo4j — SQLite handles our corpus scale (100-1000 docs).

- **Unified Agentic Retrieval** — The Researcher agent's ReAct loop IS the
  retrieval system. Three search tools (hybrid, graph, web) chosen per-query.
  Self-RAG and Corrective RAG are natural agent behaviors, not separate modules:
  - Self-RAG = agent evaluates search results before using them
  - Corrective RAG = agent decides to web search when KB is insufficient
  - Query decomposition = agent breaks complex questions into sub-searches

- **Qwen 4B Reranking** — After retrieval, the fast router model scores results
  for relevance before the reasoning model synthesizes. +33-47% accuracy in
  production benchmarks, ~200ms added latency.

- **Agent-initiated wiki articles** — When research reveals comprehensive
  coverage of a topic, the Writer auto-drafts an article. The Editor verifies
  claims. A human approves via the wiki UI before publishing.

### Unified Agentic Retrieval System

```
Researcher Agent (ReAct loop)
  │
  ├─ rag_search (existing)     — hybrid vector + FTS5 + RRF
  ├─ graph_search (new)        — knowledge graph traversal
  └─ web_search (existing)     — Brave API + ingest
  │
  ├─ Qwen 4B reranks results before synthesis
  │
  └─ Agent reasoning naturally handles:
       • Query decomposition (breaks complex Q into sub-searches)
       • Self-RAG (evaluates "does this answer my question?")
       • Corrective RAG (decides to web search when KB is insufficient)
       • Iterative retrieval (refines query and searches again)
```

---

## Step Summary

| Step | Adds | Depends on | Approx Files |
|------|------|------------|-------------|
| 1 | Structured Task Protocol + Agent Framework | Phase 1 base | 3 |
| 2 | Knowledge Graph Schema + Entity Extraction | Step 1 | 4 |
| 3 | Graph Traversal Tool + Qwen 4B Reranking | Step 2 | 3 |
| 4 | Researcher Agent | Steps 1, 3 | 3 |
| 5 | Writer Agent + Wiki Backend | Steps 1, 4 | 4 |
| 6 | Editor Agent | Steps 1, 5 | 3 |
| 7 | Orchestrator Agent | Steps 4, 5, 6 | 4 |
| 8 | Wiki UI (basic browser) | Step 5 | 6 |
| 9 | Content Freshness + Automated Backups | Step 2 | 5 |
| 10 | Integration + Phase Gate Evaluation | All steps | 5 |

Total: ~40 files, each building on the last.

---

## Step 1: Structured Task Protocol + Agent Framework

**Goal**: Define the typed Task/Result contracts that all agents use, refactor
BaseAgent to support specialization, and create the agent factory.

**What you learn**: The Structured Task Protocol pattern used in production
multi-agent systems (OpenAI Swarm, Anthropic multi-agent cookbook, AutoGen v0.4).
Agents as pure functions with typed inputs and outputs.

### What to build

- `agents/tasks.py` — Pydantic Task and Result models for each agent role
- `agents/base.py` — Refactor to support task-based execution alongside query-based
- `agents/factory.py` — Creates and configures agent instances with their tools

### Task/Result Types

```python
class AgentRole(str, Enum):
    RESEARCHER = "researcher"
    WRITER = "writer"
    EDITOR = "editor"
    ORCHESTRATOR = "orchestrator"

class TaskStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"

# Base types — all tasks and results inherit from these
class AgentTask(BaseModel):
    task_id: str                    # UUID for tracing
    trace_id: str                   # Langfuse trace propagation
    query: str                      # The core question/instruction
    context: dict[str, Any] = {}    # Shared context from Orchestrator

class AgentResult(BaseModel):
    task_id: str
    agent_role: AgentRole
    status: TaskStatus
    output: str                     # Main text output
    evidence: list[str] = []        # Source URLs / chunk IDs used
    metadata: dict[str, Any] = {}   # Role-specific extras

# Specialist task types
class ResearchTask(AgentTask):
    search_strategy: SearchStrategy | None = None  # HYBRID | GRAPH | WEB | AUTO
    max_sources: int = 10

class WriteTask(AgentTask):
    research: AgentResult           # Output from Researcher
    article_type: ArticleType       # GUIDE | REFERENCE | ANALYSIS

class EditTask(AgentTask):
    article_path: str               # Path to draft article
    claims: list[str]               # Claims to verify

class OrchestratorPlan(BaseModel):
    subtasks: list[AgentTask]       # Ordered list of specialist tasks
    parallel_groups: list[list[int]]  # Which subtasks can run concurrently
```

### BaseAgent Refactor

The current `BaseAgent.run(query: str) -> str` becomes:

```python
class BaseAgent:
    role: AgentRole
    tools: ToolRegistry

    # Existing interface preserved for backward compat
    async def run(self, query: str, ...) -> str: ...

    # New task-based interface
    async def execute(self, task: AgentTask) -> AgentResult: ...
```

Both methods use the same ReAct loop internally. `execute()` wraps the result in
a typed `AgentResult` with evidence tracking and status.

### Agent Factory

```python
class AgentFactory:
    def create(self, role: AgentRole, db_conn, ...) -> BaseAgent:
        tools = self._build_tool_registry(role)
        system_prompt = self._load_prompt(role)
        return BaseAgent(role=role, tools=tools, system_prompt=system_prompt, ...)
```

Each role gets a different tool set and system prompt, but they all share the
same ReAct loop. No subclassing needed unless a specialist needs custom loop
behavior.

### Tests

- Unit: Task/Result models validate correctly, reject invalid states
- Unit: `execute()` returns typed `AgentResult` with proper status
- Unit: Factory creates agents with correct tool sets per role
- Unit: Backward compat — `run()` still works as before
- Unit: Trace ID propagates from task to agent spans

### Gate

All existing 245 tests still pass (no regressions). New tests cover task
protocol. `AgentFactory.create(AgentRole.RESEARCHER)` returns a working agent
that can execute a `ResearchTask`.

---

## Step 2: Knowledge Graph Schema + Entity Extraction

**Goal**: Define the WoW TBC entity schema, add graph tables to SQLite, and
extract entities/relationships during ingest. After this step, every ingested
document populates both the existing chunk/embedding pipeline and the knowledge
graph.

**What you learn**: Knowledge graph construction for domain-specific RAG
(GraphRAG). Entity extraction via structured LLM output. How a well-defined
domain schema dramatically improves extraction quality vs. generic approaches.

### What to build

- `db/schema.sql` — Add `entities`, `relationships`, `entity_types` tables
- `rag/entities.py` — WoW TBC entity type definitions and extraction prompts
- `rag/graph.py` — Graph storage: insert/query entities and relationships
- `ingest/pipeline.py` — Upgrade to extract entities after chunking

### WoW TBC Entity Schema

TBC Rogue knowledge revolves around these entity types and their relationships:

```python
class EntityType(str, Enum):
    ITEM = "item"               # Dragonspine Trophy, Warglaive of Azzinoth
    SPELL = "spell"             # Sinister Strike, Slice and Dice
    TALENT = "talent"           # Combat Potency, Surprise Attacks
    TALENT_TREE = "talent_tree" # Combat, Assassination, Subtlety
    SPEC = "spec"               # Combat Swords, Combat Fists, Mutilate
    BOSS = "boss"               # Gruul, Illidan, Prince Malchezaar
    INSTANCE = "instance"       # Karazhan, Gruul's Lair, Black Temple
    PHASE = "phase"             # Phase 1, Phase 2, Phase 3, Phase 4, Phase 5
    STAT = "stat"               # Hit Rating, Crit, AP, Haste, Expertise
    CONSUMABLE = "consumable"   # Haste Potion, Scorpid Surprise
    ENCHANT = "enchant"         # Mongoose, Executioner
    GEM = "gem"                 # Delicate Living Ruby, Rigid Star of Elune
    PROFESSION = "profession"   # Leatherworking, Engineering (for BoP crafts)
    BUFF = "buff"               # Windfury Totem, Blessing of Might
    DEBUFF = "debuff"           # Expose Armor, Sunder Armor
    MECHANIC = "mechanic"       # Hit table, Dual wield penalty, Poison proc
    SLOT = "slot"               # Main hand, Off hand, Trinket 1, Head
```

### Relationship Types

```python
class RelationType(str, Enum):
    # Item relationships
    DROPS_FROM = "drops_from"           # item → boss
    AVAILABLE_IN = "available_in"       # item/instance → phase
    EQUIPS_IN = "equips_in"             # item → slot
    HAS_STAT = "has_stat"               # item/enchant/gem → stat
    CRAFTED_BY = "crafted_by"           # item → profession
    BEST_IN_SLOT = "best_in_slot"       # item → spec (contextual)

    # Spec/talent relationships
    BELONGS_TO = "belongs_to"           # talent → talent_tree
    SPEC_USES = "spec_uses"             # spec → talent_tree (primary)
    BENEFITS_FROM = "benefits_from"     # spec → stat/buff/item
    SYNERGIZES_WITH = "synergizes_with" # talent ↔ talent, spell ↔ spell

    # Combat mechanics
    AFFECTED_BY = "affected_by"         # spell → stat/mechanic
    THRESHOLD_AT = "threshold_at"       # stat → value (e.g., hit cap = 142)
    COUNTERS = "counters"               # debuff → boss mechanic
    APPLIES = "applies"                 # spell → buff/debuff

    # Instance/boss
    CONTAINS = "contains"               # instance → boss
    HAS_MECHANIC = "has_mechanic"       # boss → mechanic
    REQUIRES = "requires"               # instance → attunement/gear level
```

### Database Tables

```sql
CREATE TABLE entity_types (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL
);

CREATE TABLE entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type_id INTEGER REFERENCES entity_types(id),
    canonical_name TEXT NOT NULL,
    properties_json TEXT,
    source_chunk_id INTEGER REFERENCES chunks(id),
    confidence REAL DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(canonical_name, entity_type_id)
);

CREATE TABLE relationships (
    id INTEGER PRIMARY KEY,
    source_entity_id INTEGER REFERENCES entities(id),
    target_entity_id INTEGER REFERENCES entities(id),
    relation_type TEXT NOT NULL,
    properties_json TEXT,
    source_chunk_id INTEGER REFERENCES chunks(id),
    confidence REAL DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_entity_id, target_entity_id, relation_type)
);

CREATE INDEX idx_entities_type ON entities(entity_type_id);
CREATE INDEX idx_entities_canonical ON entities(canonical_name);
CREATE INDEX idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX idx_relationships_target ON relationships(target_entity_id);
CREATE INDEX idx_relationships_type ON relationships(relation_type);
```

### Entity Extraction Pipeline

Extraction happens during ingest, after chunking but before embedding:

```
URL → fetch → chunk → EXTRACT ENTITIES → embed → store
                           │
                           ├─ LLM extracts entities + relationships from each chunk
                           ├─ Deduplicate against existing entities (canonical_name match)
                           └─ Insert new entities/relationships with provenance
```

The extraction uses structured output from Llama 70B:

```python
class ExtractedEntity(BaseModel):
    name: str
    entity_type: EntityType
    properties: dict[str, Any] = {}

class ExtractedRelationship(BaseModel):
    source: str
    target: str
    relation_type: RelationType
    properties: dict[str, Any] = {}

class ChunkExtraction(BaseModel):
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]
```

### Entity Deduplication

Entity names in WoW content vary: "DST", "Dragonspine Trophy", "Dragonspine".
The dedup strategy:
1. Normalize to lowercase, strip articles ("the")
2. Check for exact canonical_name match
3. If no match, check for known abbreviations (a configurable alias table)
4. If still no match, create new entity

### Tests

- Unit: Entity/Relationship Pydantic models validate correctly
- Unit: Canonical name normalization handles edge cases
- Unit: Deduplication matches "DST" → "Dragonspine Trophy" via alias table
- Unit: Graph storage insert + query round-trips correctly
- Unit: Extraction prompt produces valid `ChunkExtraction` from sample text
- Unit: Duplicate relationships are upserted, not duplicated
- Unit: schema.sql creates graph tables without errors

### Gate

Ingest a real TBC Rogue guide excerpt. Verify entities (items, spells, stats)
and relationships (drops_from, has_stat, available_in) appear in the graph
tables with correct provenance back to source chunks.

---

## Step 3: Graph Traversal Tool + Qwen 4B Reranking

**Goal**: The agent can now search the knowledge graph and get results reranked
for relevance. After this step, the unified agentic retrieval system has all
three search strategies operational.

**What you learn**: Graph query patterns for knowledge retrieval, LLM-based
reranking as a precision improvement technique.

### What to build

- `tools/knowledge/graph_search.py` — Graph traversal tool for the agent
- `rag/reranker.py` — Qwen 4B relevance scoring of search results
- `rag/search.py` — Upgrade to support multi-strategy retrieval with reranking

### Graph Search Tool

```python
class GraphSearchTool(Tool):
    name = "graph_search"
    description = """Search the WoW TBC knowledge graph for entity relationships.
    Use this when you need to find connections between game concepts:
    - "What items drop from [boss]?"
    - "What stats does [item] have?"
    - "What's BiS for [spec] in [phase]?"
    - "What talents synergize with [spell]?"
    """

    parameters_schema = {
        "entity": "str - the entity name to start from",
        "relation_types": "list[str] | None - filter by relationship type",
        "target_type": "str | None - filter target entity type",
        "depth": "int - traversal depth (1=direct, 2=two hops), default 1",
    }
```

### Traversal Query Patterns

**1. Direct neighbors** — "What does Dragonspine Trophy relate to?"
```sql
SELECT e2.name, e2.entity_type_id, r.relation_type, r.properties_json
FROM relationships r
JOIN entities e1 ON r.source_entity_id = e1.id
JOIN entities e2 ON r.target_entity_id = e2.id
WHERE e1.canonical_name = ?
  AND (r.relation_type IN (?, ?, ...) OR ? IS NULL)
  AND (e2.entity_type_id = ? OR ? IS NULL)
```

**2. Reverse lookup** — "What drops from Gruul?" (query target → sources)
```sql
SELECT e1.name, e1.entity_type_id, r.relation_type, r.properties_json
FROM relationships r
JOIN entities e1 ON r.source_entity_id = e1.id
JOIN entities e2 ON r.target_entity_id = e2.id
WHERE e2.canonical_name = ?
  AND r.relation_type = ?
```

**3. Two-hop path** — "What items from Phase 1 bosses have hit rating?"

For two-hop queries, the tool builds the SQL dynamically based on the entity
types and relationship types involved, rather than hardcoding every possible
path.

### Tool Output Format

```
Found 4 relationships for "Dragonspine Trophy":
- drops_from → Gruul the Dragonkiller (confidence: 0.9)
- has_stat → haste proc (confidence: 0.85)
- available_in → Phase 1 (confidence: 0.95)
- best_in_slot → Combat Swords (confidence: 0.7)

Related chunks: [chunk_id: 42, chunk_id: 78]
```

The linked chunk IDs let the agent follow up with `rag_search` to get full text
context — graph finds the structure, hybrid search fills in the detail.

### Qwen 4B Reranking

After any search returns results, the reranker scores each for relevance:

```python
class RerankerResult(BaseModel):
    relevance: Literal["HIGH", "MEDIUM", "LOW"]
    reason: str

async def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int = 5,
) -> list[SearchResult]:
    """Rerank results using Qwen 4B, return top_k most relevant."""
```

The reranker batches all results into a single Qwen 4B call (not one per result)
to keep latency under ~200ms. Over-retrieve 3x (fetch 15, rerank to 5).

Results scored HIGH are kept. MEDIUM kept if needed to fill top_k. LOW dropped.
Circuit breaker wraps the reranker — falls back to unreranked on Qwen failure.

### Unified Search Entry Point

```python
async def search_with_rerank(
    query: str,
    strategy: SearchStrategy,  # HYBRID | GRAPH | AUTO
    db_conn: Connection,
    top_k: int = 5,
) -> list[RankedResult]:
    if strategy == SearchStrategy.AUTO:
        strategy = _classify_search_strategy(query)

    if strategy == SearchStrategy.HYBRID:
        raw = await hybrid_search(query, db_conn, top_k=top_k * 3)
    elif strategy == SearchStrategy.GRAPH:
        raw = await graph_search(query, db_conn)

    return await rerank(query, raw, top_k=top_k)
```

### Tests

- Unit: Graph traversal returns correct neighbors for test entities
- Unit: Reverse lookup finds sources given a target
- Unit: Two-hop traversal follows the correct path
- Unit: Empty graph returns empty results (no crash)
- Unit: Reranker filters LOW results and preserves ordering
- Unit: Reranker handles empty result list
- Unit: `search_with_rerank` calls correct strategy based on enum
- Unit: Graph search tool formats observation string correctly
- Unit: Circuit breaker wraps reranker (falls back to unreranked on Qwen failure)

### Gate

Ingest a TBC Rogue guide. Ask `graph_search("Dragonspine Trophy")` — get back
boss, stats, phase relationships. Ask `rag_search` for the same topic — results
come back reranked by Qwen 4B with HIGH/MEDIUM/LOW scores.

---

## Step 4: Researcher Agent

**Goal**: The first specialist agent. The Researcher uses the unified agentic
retrieval system — choosing between hybrid search, graph traversal, and web
search — with query decomposition, iterative refinement, and self-correction as
natural ReAct loop behaviors.

**What you learn**: How agentic RAG patterns (query decomposition, self-RAG,
corrective RAG) emerge from good prompting rather than separate infrastructure.

### What to build

- `agents/researcher.py` — Researcher agent with specialized prompts and retrieval logic
- `llm/prompts/researcher.py` — System prompt + few-shot examples for research behavior
- `agents/tasks.py` — Add `ResearchResult` with structured findings

### How the Researcher Differs from BaseAgent

The Researcher doesn't need a different ReAct loop — it uses the same `execute()`
from Step 1. What makes it a specialist is:

1. **Its system prompt** teaches agentic retrieval patterns
2. **Its tool set** is research-focused (rag_search, graph_search, web_search, web_ingest)
3. **Its result type** captures structured findings with evidence

### System Prompt Design

The Researcher's system prompt encodes retrieval strategies as reasoning
patterns rather than hardcoded logic:

```
You are a Research Specialist for WoW TBC Rogue content. Your job is to
gather comprehensive, accurate information using your search tools.

## Search Strategy Selection
- Use `rag_search` for factual questions about specific topics
- Use `graph_search` when you need relationships between game concepts
- Use `web_search` when your knowledge base doesn't have enough information
- Use `web_ingest` to permanently add valuable web sources to the KB

## Research Patterns

DECOMPOSE complex questions into sub-queries:
  "Compare combat swords vs mutilate for Gruul"
  → search BiS gear for combat swords in Phase 1
  → search BiS gear for mutilate in Phase 1
  → search Gruul boss mechanics
  → graph_search for stat priorities per spec

EVALUATE your results before answering:
  - Do the retrieved chunks actually answer the question?
  - Are sources trustworthy (check trust scores)?
  - Do multiple sources agree, or is there conflict?
  - If results are insufficient, refine your query or try web_search

ITERATE when needed:
  - First search too broad? Narrow with more specific terms
  - First search too narrow? Broaden or try different strategy
  - Found partial info? Search for the missing pieces

NEVER guess. If you can't find solid evidence, say what you found
and what's still missing.
```

### ResearchResult

```python
class Finding(BaseModel):
    claim: str                          # "The hit cap for combat is 142 rating"
    evidence: list[str]                 # chunk IDs or URLs that support this
    confidence: float                   # 0.0-1.0 based on source trust + agreement
    entity_refs: list[str] = []         # entity names mentioned

class ResearchResult(AgentResult):
    findings: list[Finding]
    sources_used: list[str]
    strategies_used: list[str]
    gaps: list[str] = []                # What the Researcher couldn't find
    sufficient: bool                    # Does the Researcher think it has enough?
```

The `findings` list is what the Writer agent consumes to produce articles. The
`gaps` list tells the Orchestrator what's still missing. The `sufficient` flag
drives the Orchestrator's decision to request more research or move forward.

### Agent Iteration Limits

```python
RESEARCHER_MAX_ITERATIONS = 10    # up from 5 — multi-step research expected
RESEARCHER_MAX_TOKENS = 150_000   # up from 100k — research produces more context
```

### Fallback Behavior

- If `graph_search` fails → falls back to `rag_search`
- If `rag_search` returns low-confidence results → tries `web_search`
- If `web_search` circuit breaker is open → returns partial results with `sufficient=False`
- If max iterations reached → forces final answer with `gaps` listing what's missing

### Tests

- Unit: Researcher system prompt includes all three search strategies
- Unit: ResearchResult/Finding models validate correctly
- Unit: Researcher execute() returns ResearchResult (not plain string)
- Unit: Mock LLM that decomposes a two-part question calls tools twice
- Unit: Mock LLM that gets LOW-relevance results triggers a second search
- Unit: Researcher respects RESEARCHER_MAX_ITERATIONS limit
- Unit: Gaps populated when Researcher can't find sufficient evidence
- Unit: `sufficient=False` when all search strategies return empty
- Integration: Researcher with real tools answers a factual question from ingested content

### Gate

Give the Researcher "What are the best trinkets for a combat rogue in Phase 1
and why?" It decomposes (search trinkets, search stat priorities, check graph
for Phase 1 availability), reranks results, and returns structured findings
with evidence and confidence scores.

---

## Step 5: Writer Agent + Wiki Backend

**Goal**: The Writer agent takes research findings and produces wiki articles as
git-tracked Markdown files with YAML frontmatter. Articles are stored in
`knowledge/`, tracked in the database, and require human approval before
publishing.

**What you learn**: LLM-driven content generation with structured output,
Markdown + YAML frontmatter as a knowledge representation, article lifecycle
management.

### What to build

- `agents/writer.py` — Writer agent with article generation prompts
- `llm/prompts/writer.py` — System prompt + article structure templates
- `knowledge/manager.py` — Article CRUD: create drafts, read, update, list, publish
- `agents/tasks.py` — Add `WriteResult` with article metadata

### Article Format

```markdown
---
title: "Combat Swords Rogue: Phase 1 BiS Gear Guide"
spec: combat
category: gear
phase: 1
status: draft
confidence: 0.78
created_at: 2026-02-15T14:30:00
updated_at: 2026-02-15T14:30:00
sources:
  - url: https://shadowpanther.net/tbc-combat.html
    trust: 0.7
  - url: https://wowhead.com/tbc/guide/rogue-dps
    trust: 0.6
claims:
  - text: "Dragonspine Trophy is BiS trinket for combat"
    verified: false
    evidence: [chunk:42, chunk:78]
  - text: "Hit cap is 142 rating (9%) for dual wield"
    verified: true
    evidence: [chunk:15, chunk:91, chunk:203]
entity_refs: [dragonspine trophy, combat swords, gruul, phase 1]
tags: [combat, gear, bis, phase-1, trinkets]
---

# Combat Swords Rogue: Phase 1 BiS Gear Guide

## Overview

Combat Swords is the dominant PvE rogue spec in Phase 1 of TBC...
```

### Knowledge Directory Structure

```
knowledge/
  combat/
    gear/
      phase-1-bis.md
      trinkets.md
    rotation/
      priority-list.md
  assassination/
    ...
  subtlety/
    ...
  general/
    stat-caps.md
    hit-table.md
    consumables.md
```

The path is derived from spec + category: `knowledge/{spec}/{category}/{slug}.md`

### Knowledge Manager

```python
class ArticleStatus(str, Enum):
    DRAFT = "draft"
    REVIEW = "review"
    PUBLISHED = "published"

class KnowledgeManager:
    async def create_draft(
        self, title: str, spec: str, category: str,
        content: str, frontmatter: dict
    ) -> str:
        """Write markdown file, insert DB record. Returns article path."""

    async def read_article(self, path: str) -> Article: ...

    async def update_article(
        self, path: str, content: str | None, frontmatter: dict | None
    ) -> None:
        """Update file + DB record. Bumps updated_at."""

    async def list_articles(
        self, status: ArticleStatus | None = None,
        spec: str | None = None
    ) -> list[ArticleSummary]: ...

    async def set_status(self, path: str, status: ArticleStatus) -> None:
        """Transition article status (draft→review→published)."""
```

The manager writes both the Markdown file and the `articles` table row. The file
is the source of truth; the DB row is for querying and status tracking.

### Writer System Prompt

```
You are a Wiki Writer for WoW TBC Rogue content. You produce clear,
accurate, well-structured Markdown articles from research findings.

## Writing Rules

STRUCTURE every article with:
  - A concise overview paragraph (2-3 sentences)
  - Logical sections with ## headers
  - Specific numbers and values (never vague)
  - Source attribution for key claims

TONE: Authoritative but approachable. Like an experienced raider
explaining to a guild member.

CLAIMS: Every factual claim must trace back to a research finding.
Never invent facts the research didn't provide.

COMPLETENESS: If the research has gaps, note them explicitly with a
<!-- NEEDS RESEARCH: [topic] --> comment rather than filling in guesses.

FORMAT:
  - Use tables for gear comparisons, stat breakdowns
  - Use bold for item names, spell names, stat values
  - Use ordered lists for priorities/rankings
  - Keep paragraphs short (3-5 sentences max)
```

### WriteResult

```python
class WriteResult(AgentResult):
    article_path: str
    title: str
    claims: list[str]               # Factual claims for Editor to verify
    research_gaps: list[str]        # Gaps inherited from research
    word_count: int
    entity_refs: list[str]          # Entities mentioned
```

### Writer Execution Flow

```
WriteTask received (contains ResearchResult from Researcher)
  │
  ├─ Writer reads research findings + evidence
  ├─ Determines article path from spec/category/topic
  ├─ Checks if article already exists (update vs create)
  │     ├─ Exists: reads current content, merges new findings
  │     └─ New: generates from scratch
  ├─ Generates article via LLM (Llama 70B)
  ├─ Extracts claims list from generated content
  ├─ Writes draft via KnowledgeManager.create_draft()
  └─ Returns WriteResult with article_path + claims
```

### Tests

- Unit: KnowledgeManager creates file + DB record in correct path
- Unit: KnowledgeManager read/update round-trips correctly
- Unit: Article status transitions enforce valid order (draft→review→published)
- Unit: Invalid transitions rejected (draft→published, published→draft)
- Unit: YAML frontmatter parsed and serialized correctly
- Unit: Writer system prompt produces well-structured markdown from sample findings
- Unit: WriteResult extracts claims list from article content
- Unit: Existing article triggers update path, not duplicate creation
- Unit: Research gaps carry through to article as `<!-- NEEDS RESEARCH -->` comments
- Unit: list_articles filters by status and spec correctly

### Gate

Feed the Writer a `ResearchResult` about Phase 1 combat trinkets. It produces
`knowledge/combat/gear/trinkets.md` with proper frontmatter, status=draft, and
a readable article. The article row appears in the `articles` table.

---

## Step 6: Editor Agent

**Goal**: The Editor fact-checks articles against the knowledge base and graph,
verifies claims, assigns confidence scores, and transitions articles from draft
to review status for human approval.

**What you learn**: Automated fact-checking against a knowledge graph,
confidence scoring for generated content, the verification pattern that
separates trustworthy from unreliable AI-generated text.

### What to build

- `agents/editor.py` — Editor agent with verification logic
- `llm/prompts/editor.py` — System prompt for fact-checking behavior
- `agents/tasks.py` — Add `EditResult` with verification details

### Claim Verification

```python
class VerificationStatus(str, Enum):
    VERIFIED = "verified"           # Multiple sources agree
    UNCERTAIN = "uncertain"         # Some evidence but not conclusive
    CONTRADICTED = "contradicted"   # Evidence directly conflicts
    UNSUPPORTED = "unsupported"     # No evidence found in KB

class ClaimVerification(BaseModel):
    claim: str
    status: VerificationStatus
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    confidence: float
    note: str = ""

class EditResult(AgentResult):
    article_path: str
    claim_results: list[ClaimVerification]
    internal_consistency: bool
    overall_confidence: float
    approved_for_review: bool       # Met threshold for human review?
    corrections: list[str]          # Suggested text changes
    needs_more_research: list[str]  # Topics that need Researcher follow-up
```

### Confidence Scoring

```python
def compute_article_confidence(claims: list[ClaimVerification]) -> float:
    if not claims:
        return 0.0
    total = sum(c.confidence for c in claims)
    base = total / len(claims)
    contradicted = sum(1 for c in claims if c.status == VerificationStatus.CONTRADICTED)
    penalty = contradicted * 0.15
    return max(0.0, min(1.0, base - penalty))
```

An article needs `overall_confidence >= 0.6` to be approved for human review.
Below that, the Editor returns `needs_more_research` for the Orchestrator to
feed back to the Researcher.

### Editor System Prompt

```
You are a Fact-Checking Editor for WoW TBC Rogue content. Your job is to
verify claims in draft articles against the knowledge base.

## Verification Process

For EACH claim in the article:
1. Search the knowledge base for supporting evidence (rag_search)
2. Check the knowledge graph for relationship consistency (graph_search)
3. Assign a verification status:
   - VERIFIED: 2+ independent sources agree
   - UNCERTAIN: 1 source supports, no contradictions
   - CONTRADICTED: sources disagree — flag for human review
   - UNSUPPORTED: no evidence found in KB

## What to Check

NUMBERS: Stat caps, DPS values, proc rates, item stats
RELATIONSHIPS: "X drops from Y", "X is BiS for Z" — verify via graph
RANKINGS: "A is better than B for C" — look for comparative evidence
MECHANICS: Combat formulas, hit tables, proc behavior
PHASE ACCURACY: "Available in Phase X" — verify against phase data

## You Do NOT:
- Rewrite the article (that's the Writer's job)
- Research new topics (that's the Researcher's job)
- Make judgment calls on contradictions (that's the human's job)
```

### Graph Verification Examples

```
Claim: "Dragonspine Trophy drops from Gruul"
→ graph_search(entity="dragonspine trophy", relation_types=["drops_from"])
→ Found: drops_from → Gruul the Dragonkiller (confidence: 0.9)
→ VERIFIED

Claim: "Dragonspine Trophy is available in Phase 2"
→ graph_search(entity="dragonspine trophy", relation_types=["available_in"])
→ Found: available_in → Phase 1 (confidence: 0.95)
→ CONTRADICTED (article says Phase 2, graph says Phase 1)
```

### Tests

- Unit: ClaimVerification model validates all status types
- Unit: `compute_article_confidence` averages correctly
- Unit: Contradiction penalty reduces confidence
- Unit: Zero claims returns 0.0 confidence
- Unit: Editor tool set is correct (rag_search + graph_search only, no web_search)
- Unit: Article below threshold returns `approved_for_review=False`
- Unit: Article above threshold returns `approved_for_review=True`
- Unit: CONTRADICTED claims populate `contradicting_evidence`
- Unit: `needs_more_research` populated for UNSUPPORTED claims
- Unit: Frontmatter correctly updated after verification pass

### Gate

Give the Editor a draft article with 5 claims (mix of correct and one wrong).
It marks correct claims VERIFIED/UNCERTAIN, catches the wrong one as
CONTRADICTED, computes a reasonable confidence score, and updates the
frontmatter.

---

## Step 7: Orchestrator Agent

**Goal**: The Orchestrator decomposes complex queries into sub-tasks, dispatches
them to specialists (Researcher, Writer, Editor), and synthesizes results into a
final response. This is where the Structured Task Protocol comes together.

**What you learn**: Task decomposition with dependency graphs, topological
execution ordering, multi-agent coordination as a planning problem.

### What to build

- `agents/orchestrator.py` — Task decomposition, dispatch, synthesis
- `llm/prompts/orchestrator.py` — System prompt for planning and coordination
- `agents/tasks.py` — Add `OrchestratorPlan` and `OrchestratorResult`
- `routing/router.py` — Upgrade to route COMPLEX queries to Orchestrator

### Three-Phase Execution

The Orchestrator doesn't use the standard ReAct loop. It has its own flow:

```python
class Orchestrator:
    async def execute(self, task: AgentTask) -> OrchestratorResult:
        plan = await self._decompose(task)       # Phase 1: Plan
        results = await self._dispatch(plan)      # Phase 2: Execute
        return await self._synthesize(task, results)  # Phase 3: Synthesize
```

### Task Decomposition

```python
class SubTask(BaseModel):
    agent_role: AgentRole
    description: str
    depends_on: list[int] = []          # Indices of dependent tasks
    task_params: dict[str, Any] = {}

class OrchestratorPlan(BaseModel):
    reasoning: str
    subtasks: list[SubTask]
    can_answer_directly: bool = False
    direct_answer: str | None = None
```

Example:

```
User: "Write a guide about Phase 1 BiS trinkets for combat rogues"

Plan:
  [0] RESEARCHER: "Find all Phase 1 trinkets relevant for combat rogues"
  [1] RESEARCHER: "Find combat rogue stat priorities" (depends_on: [])
  [2] WRITER: "Write a BiS trinket guide" (depends_on: [0, 1])
  [3] EDITOR: "Verify claims in the guide" (depends_on: [2])
```

### Task Dispatch

The Orchestrator walks the dependency graph, executing tasks whose dependencies
are satisfied:

```python
async def _dispatch(self, plan: OrchestratorPlan) -> list[AgentResult]:
    results: list[AgentResult | None] = [None] * len(plan.subtasks)

    for group in self._topological_groups(plan.subtasks):
        group_tasks = []
        for idx in group:
            subtask = plan.subtasks[idx]
            agent = self.factory.create(subtask.agent_role)
            typed_task = self._build_task(subtask, results)
            group_tasks.append((idx, agent.execute(typed_task)))

        for idx, coro in group_tasks:
            results[idx] = await coro

    return [r for r in results if r is not None]
```

`_topological_groups` returns tasks in dependency order, grouping independent
tasks for concurrent execution. In the trinket guide example: [0] and [1] run
concurrently, then [2], then [3].

### Router Integration

```
TRIVIAL  → Qwen 4B direct answer (unchanged)
MODERATE → Researcher agent solo (new)
COMPLEX  → Orchestrator (new)
```

### Orchestrator System Prompt

```
You are the Orchestrator for a WoW TBC Rogue knowledge system.

## Available Specialists
- RESEARCHER: Gathers information from knowledge base, graph, and web
- WRITER: Produces wiki articles from research findings
- EDITOR: Fact-checks articles against the knowledge base

## Planning Rules

DECOMPOSE queries into the smallest useful sub-tasks.
PARALLELIZE when possible.
DO NOT over-decompose simple questions.

TRIGGER article writing when:
- The user explicitly asks for a guide/article
- Research reveals comprehensive coverage of a topic not yet in the wiki

## When Things Go Wrong
- Researcher returns sufficient=False: plan additional research
- Editor returns contradictions: include them in the response
- Specialist fails entirely: answer with partial results
```

### OrchestratorResult

```python
class OrchestratorResult(AgentResult):
    plan: OrchestratorPlan
    specialist_results: list[AgentResult]
    article_path: str | None = None
    needs_human_review: bool = False
```

### Guardrails

```python
ORCHESTRATOR_MAX_SUBTASKS = 6       # Prevent over-decomposition
ORCHESTRATOR_MAX_DEPTH = 1          # No recursive orchestration
ORCHESTRATOR_TIMEOUT_SECONDS = 120  # Total wall-clock budget
```

### Tests

- Unit: OrchestratorPlan validates subtask dependencies (no cycles)
- Unit: `_topological_groups` returns correct execution order
- Unit: Independent subtasks grouped for concurrent execution
- Unit: Dependent subtasks execute sequentially
- Unit: `can_answer_directly=True` skips specialist dispatch
- Unit: Specialist failure produces partial result (not crash)
- Unit: MAX_SUBTASKS enforced — plan with 7 tasks triggers re-plan
- Unit: Recursive orchestration blocked
- Unit: Router sends COMPLEX to Orchestrator, MODERATE to Researcher
- Unit: Synthesis combines multiple ResearchResults into coherent response
- Unit: Article workflow (research → write → edit) chains correctly
- Integration: Full pipeline — complex question → plan → research → answer

### Gate

Ask "Write a comprehensive guide about combat rogue trinkets in Phase 1" via
chat. The Orchestrator plans research → write → edit. A draft article appears
in `knowledge/combat/gear/trinkets.md` with status=review.

---

## Step 8: Wiki UI (Basic Browser)

**Goal**: A web interface where users can browse wiki articles, see confidence
badges, view source attribution, and approve drafts for publishing.

**What you learn**: HTMX-powered server-rendered UI, Markdown rendering,
article lifecycle management through a web interface.

### What to build

- `web/routers/wiki.py` — Wiki routes: list, view, approve, search
- `web/templates/wiki/list.html` — Article index with filters
- `web/templates/wiki/article.html` — Article rendering with metadata sidebar
- `web/templates/wiki/review.html` — Draft review + approve/reject UI
- `web/static/css/wiki.css` — Wiki-specific styles
- `web/app.py` — Mount wiki router

### Routes

```python
GET  /wiki/                         # Article index (filterable)
GET  /wiki/{spec}/{category}/{slug} # View single article
GET  /wiki/review/                  # List articles pending review
POST /wiki/review/{article_id}      # Approve or reject a draft
GET  /wiki/search?q=...             # Basic keyword search across articles
```

### Article Index (`/wiki/`)

Lists all published articles, grouped by spec. HTMX-powered filtering without
full page reloads. Confidence shown as a colored bar: green (>0.8), yellow
(0.6-0.8), red (<0.6).

### Article View (`/wiki/{spec}/{category}/{slug}`)

Renders Markdown article with a metadata sidebar showing:
- Confidence score with colored bar
- Article status and last updated date
- Source URLs with trust scores
- Claim verification summary (verified / uncertain / contradicted counts)
- Entity reference tags (clickable, link to graph search)

### Review Page (`/wiki/review/`)

Lists articles in `review` status with Editor's verification results.
Approve transitions `review` → `published`. Reject transitions back to `draft`
with optional rejection reason stored in frontmatter.

### Wiki Search (`/wiki/search?q=...`)

Basic keyword search across published article titles and content via simple
LIKE query. Full FTS5 on articles is overkill for Phase 2 — few articles
expected.

### Tests

- Unit: Wiki routes return correct HTTP status codes
- Unit: Article list filters by spec and status correctly
- Unit: Article view renders markdown to HTML
- Unit: YAML frontmatter parsed into sidebar data
- Unit: Approve transitions review → published
- Unit: Reject transitions review → draft with reason
- Unit: Search returns matching articles by title
- Unit: Empty wiki renders gracefully (no articles yet)
- Unit: Non-existent article returns 404

### Gate

Browse to `/wiki/`. See article index. Click an article, see rendered Markdown
with confidence sidebar. Go to `/wiki/review/`, approve a draft, see it appear
in the published list.

---

## Step 9: Content Freshness + Automated Backups

**Goal**: Detect stale content that needs re-checking and automate SQLite
database backups. The system starts maintaining its own knowledge quality
without manual intervention.

**What you learn**: Content lifecycle management, change detection via content
hashing, SQLite online backup API, scheduled job patterns.

### What to build

- `freshness/checker.py` — Staleness detection logic
- `freshness/scheduler.py` — Periodic freshness check jobs
- `backup/manager.py` — SQLite online backup + integrity verification
- `backup/scheduler.py` — Periodic backup jobs
- `config.py` — Add freshness and backup config values

### Freshness Detection

A source becomes stale based on time since last check and source type volatility:

```python
class SourceVolatility(str, Enum):
    STATIC = "static"           # Game mechanics, spell data (patch-locked)
    SLOW = "slow"               # Guides, wiki pages (monthly updates)
    MODERATE = "moderate"       # Forum discussions, tier lists (weekly)
    FAST = "fast"               # Rankings, logs, meta reports (daily)

VOLATILITY_MAX_AGE: dict[SourceVolatility, timedelta] = {
    SourceVolatility.STATIC: timedelta(days=90),
    SourceVolatility.SLOW: timedelta(days=30),
    SourceVolatility.MODERATE: timedelta(days=7),
    SourceVolatility.FAST: timedelta(days=1),
}
```

### Re-check Flow

```
Scheduler triggers freshness check
  │
  ├─ find_stale_sources() → sources past max_age
  ├─ For each stale source:
  │     ├─ Fetch URL headers (ETag / Last-Modified) — cheap check
  │     ├─ If headers suggest change: fetch full content
  │     ├─ Compare content_hash to stored hash
  │     ├─ If unchanged: update last_checked, done
  │     └─ If changed:
  │           ├─ Flag source as needs_reingest in DB
  │           ├─ Flag affected articles as needs_review
  │           └─ Log the change for Orchestrator awareness
  └─ Done
```

The checker does NOT re-ingest automatically. It flags sources and articles,
then the Orchestrator picks them up during normal operation.

### Volatility Assignment

During ingest, the pipeline assigns volatility based on domain heuristics:

```python
DOMAIN_VOLATILITY: dict[str, SourceVolatility] = {
    "wowhead.com": SourceVolatility.SLOW,
    "icy-veins.com": SourceVolatility.SLOW,
    "shadowpanther.net": SourceVolatility.STATIC,
    "warcraftlogs.com": SourceVolatility.FAST,
    "silentshadows.net": SourceVolatility.STATIC,
    "tbcdb.com": SourceVolatility.STATIC,
    "__default__": SourceVolatility.MODERATE,
}
```

### Schema Additions

```sql
ALTER TABLE sources ADD COLUMN volatility TEXT DEFAULT 'slow';
ALTER TABLE sources ADD COLUMN last_checked TIMESTAMP;
ALTER TABLE sources ADD COLUMN needs_reingest BOOLEAN DEFAULT FALSE;

CREATE TABLE freshness_log (
    id INTEGER PRIMARY KEY,
    source_id INTEGER REFERENCES sources(id),
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    changed BOOLEAN,
    old_hash TEXT,
    new_hash TEXT
);
```

### Automated Backups

SQLite's online backup API copies the database while it's in use:

```python
class BackupManager:
    async def create_backup(self) -> BackupResult:
        """Online backup to backup directory."""

    async def verify_integrity(self, path: Path) -> bool:
        """Run PRAGMA integrity_check on a backup."""

    async def prune_old_backups(self, keep: int = 7) -> int:
        """Delete backups older than the most recent `keep` count."""
```

### Scheduler

Both freshness checks and backups run via APScheduler:

```python
def register_jobs(scheduler: AsyncIOScheduler):
    scheduler.add_job(run_freshness_check, trigger="interval",
                      hours=FRESHNESS_CHECK_INTERVAL_HOURS)
    scheduler.add_job(run_backup, trigger="interval",
                      hours=BACKUP_INTERVAL_HOURS)
    scheduler.add_job(run_backup_prune, trigger="interval", days=1)
```

### Config Additions

```python
FRESHNESS_CHECK_INTERVAL_HOURS = int(os.getenv("FRESHNESS_CHECK_INTERVAL_HOURS", "6"))
FRESHNESS_HTTP_TIMEOUT = int(os.getenv("FRESHNESS_HTTP_TIMEOUT", "10"))
BACKUP_INTERVAL_HOURS = int(os.getenv("BACKUP_INTERVAL_HOURS", "12"))
BACKUP_KEEP_COUNT = int(os.getenv("BACKUP_KEEP_COUNT", "7"))
```

### Tests

- Unit: `find_stale_sources` returns sources past max_age, ignores fresh ones
- Unit: Volatility assignment maps domains correctly, falls back to default
- Unit: Content hash comparison detects changed vs unchanged
- Unit: ETag/Last-Modified header check short-circuits full fetch
- Unit: Changed source flags `needs_reingest` and affected articles `needs_review`
- Unit: Unchanged source only updates `last_checked`
- Unit: Freshness log records check history
- Unit: Backup creates valid SQLite file at correct path
- Unit: `verify_integrity` returns True for good backup, False for corrupted
- Unit: `prune_old_backups` keeps correct count, deletes oldest first
- Unit: Scheduler registers jobs at correct intervals

### Gate

Insert a test source with `last_checked` 31 days ago and volatility `slow`.
Run the freshness checker — it detects the stale source. Run a backup — valid
`.db` file appears, passes integrity check. Old backups get pruned.

---

## Step 10: Integration + Phase Gate Evaluation

**Goal**: Wire everything together end-to-end, run the full system through
realistic scenarios, and measure against the phase gate metrics. This step
produces no new features — it's about making the pieces work as a whole and
proving it.

**What you learn**: End-to-end system integration, LLM-as-judge evaluation
methodology, building evaluation datasets for domain-specific AI systems.

### What to build

- `web/routers/chat.py` — Upgrade to use Orchestrator, Researcher, article awareness
- `agents/factory.py` — Final wiring with all agents, tools, and DB connections
- `evals/phase2_gate.py` — Evaluation harness for phase gate metrics
- `evals/datasets/phase2_questions.json` — Curated evaluation questions
- `tests/integration/test_phase2_e2e.py` — End-to-end integration tests

### Chat Handler Upgrade

```python
async def handle_message(message: str, session: ChatSession):
    decision = await classify_query(message)

    if decision.complexity == TaskComplexity.TRIVIAL:
        response = decision.direct_answer

    elif decision.complexity == TaskComplexity.MODERATE:
        researcher = factory.create(AgentRole.RESEARCHER)
        result = await researcher.execute(ResearchTask(query=message))
        response = result.output

    else:  # COMPLEX
        orchestrator = Orchestrator(factory=factory)
        result = await orchestrator.execute(AgentTask(query=message))
        response = result.output

        if result.article_path:
            response += f"\n\n---\n*Draft article created: {result.article_path}*"
        if result.needs_human_review:
            response += "\n*Article pending your review in Wiki Review*"
```

### Enhanced Status Callbacks

```
"Routing query..."                              # existing
"Planning: decomposing into 3 sub-tasks..."     # Orchestrator
"Researcher: searching knowledge graph..."       # agent role label
"Researcher: evaluating 12 results..."           # reranking
"Writer: drafting article..."                    # article creation
"Editor: verifying 5 claims..."                  # fact-checking
"Synthesizing results..."                        # Orchestrator
```

### Phase Gate Evaluation

Three metrics, measured against a curated question set of 50+ questions:

**1. RAG Faithfulness > 0.8** — Is every claim in the answer supported by
retrieved evidence? LLM-as-judge evaluation.

**2. Agent Trajectory Precision > 0.7** — Did the agent take reasonable actions?
Correct tool selection, no wasted calls, efficient path to answer.

**3. Domain Accuracy > 70%** — Are the answers factually correct about WoW TBC
Rogue content? Compared against curated ground truth.

### Evaluation Dataset

`evals/datasets/phase2_questions.json` — 50+ questions across difficulty levels
covering mechanics, gear comparisons, rotation advice, BiS lists, boss
strategies, and article generation requests. Each has ground truth for accuracy
scoring.

### Wiki Coverage Check

The phase gate also verifies wiki articles exist for all three specs:
combat, assassination, subtlety.

### Integration Tests

```python
async def test_trivial_query_routes_to_qwen(): ...
async def test_moderate_query_uses_researcher(): ...
async def test_complex_query_uses_orchestrator(): ...
async def test_article_workflow_end_to_end(): ...
async def test_stale_source_flagged_by_freshness(): ...
async def test_backup_creates_and_verifies(): ...
async def test_wiki_ui_renders_article(): ...
async def test_wiki_review_approve_flow(): ...
```

### Gate

All tests pass. Phase gate evaluation meets thresholds. Wiki has at least one
published article per spec. Open the browser — ask a complex question, watch the
Orchestrator coordinate agents, see traces in Langfuse, browse the wiki.

---

## Future: Autonomous Knowledge Building

The Phase 2 architecture enables a progression toward fully autonomous research:

1. **Phase 2 (this plan)**: Agent-initiated articles triggered by user queries.
   Human approves before publishing.

2. **Auto-publish**: Once Editor verification quality is trusted (Phase 4 evals),
   articles above confidence threshold publish automatically.

3. **Proactive research**: A scheduler periodically asks the Orchestrator to find
   wiki gaps and stale content, dispatching research → write → edit autonomously.

4. **Self-healing knowledge**: User questions that reveal wrong info trigger
   Editor flagging → Researcher update → Writer revision automatically.

The Structured Task Protocol enables all of this — the Orchestrator doesn't care
whether a task was triggered by a user message or a scheduled job.

## Future: Phase 2b — API Integrations

After Phase 2 core is stable:

- **Warcraft Logs API** (OAuth + GraphQL) — parse rankings, log data
- **Blizzard Battle.net API** (OAuth + REST) — item/spell databases
- **WoWSims data importer** — simulation baseline data
- **CMaNGOS data importer** — server-side game mechanics data

These feed into the knowledge graph and chunk store, making the Researcher's
knowledge base far richer.

## Future: Phase 3 — Analyst Agent

The Analyst agent is deferred to Phase 3 alongside the DPS simulation engine.
It will use tools: `sim_run`, `db_query`, `log_analyze`, `stat_weights`,
`rag_search`. The Structured Task Protocol and Orchestrator dispatch patterns
built in Phase 2 will support it without architectural changes.
