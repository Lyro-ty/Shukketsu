# Phase 4: Analyst Agent — Sim-Powered Quantitative Analysis

## Overview

The Analyst is the fifth specialist agent, completing the multi-agent system. It uses the DPS simulation engine to answer quantitative questions, verify claims, and optimize gear. It's the agent that makes our sim better than WoWSims — it can reason about results, not just produce numbers.

## Agent Design

### Class: `AnalystAgent(BaseAgent)`

```python
# agents/analyst.py

class AnalystAgent(BaseAgent):
    ROLE = "analyst"
    DESCRIPTION = "Quantitative DPS analysis using simulation"

    TOOLS = [
        "sim_run",        # Run a full DPS simulation
        "sim_compare",    # Compare two setups
        "sim_optimize",   # Find best item for a slot
        "rag_search",     # Knowledge base for context
        "graph_search",   # Item/stat relationships
    ]
```

### System Prompt

```
You are a TBC Rogue DPS analyst with access to a simulation engine. Your job is
to provide accurate, data-driven answers to quantitative questions about gear,
talents, rotations, and optimization.

Guidelines:
1. ALWAYS sim before making DPS claims — never guess.
2. Explain WHY, not just WHAT. "DST gains 47 DPS because the haste proc increases
   auto-attack frequency, generating more Combat Potency energy."
3. Reference specific mechanics: hit cap (142 rating / 9%), crit suppression (4.8%),
   Seal Fate scaling, energy economy.
4. Show numbers: DPS delta, stat weights, proc uptimes, ability breakdown changes.
5. Consider the player's CURRENT gear, not just theoretical BiS.
6. When optimizing, explain trade-offs: "DST is better for sustained, Brooch is
   better for short burst fights under 2 minutes."
7. Flag stat caps: "You're 47 hit below yellow cap — hit is worth 2.4 EP right now
   but drops to 0.8 after cap."
```

### Task Categories

The Analyst handles tasks categorized as `ANALYSIS` by the router/orchestrator:

| Query Pattern | Tool Flow | Example |
|---------------|-----------|---------|
| "What DPS should I expect?" | sim_run → interpret | "With your P3 gear, expect ~1,847 DPS" |
| "Is X better than Y?" | sim_compare → explain | "DST > Brooch by 47 DPS because..." |
| "What should I upgrade?" | sim_optimize → recommend | "Best MH: Warglaive (+112 DPS)" |
| "What are my stat weights?" | sim_run (with weights) → present | "Hit: 2.4, Agi: 2.1, Crit: 1.8" |
| "How much DPS do I lose without WF?" | sim_compare (±WF) → quantify | "No WF costs 89 DPS (4.3%)" |
| "Should I use Expose Armor?" | sim_compare (±EA) → evaluate | "EA gains 23 raid DPS but costs you 67 personal" |

### ReAct Loop

```python
async def execute(self, task: AgentTask) -> AgentResult:
    """Standard ReAct loop with sim-specific reasoning."""

    # Phase 1: Understand the question
    # - Parse what's being asked (comparison, optimization, raw sim)
    # - Identify the current gear context (from task.context or session)
    # - Determine which tool(s) to call

    # Phase 2: Gather data
    # - Run appropriate sim tool(s)
    # - Optionally search knowledge base for context (stat caps, etc.)
    # - Optionally search graph for item relationships

    # Phase 3: Interpret results
    # - Analyze ability breakdowns, stat weights, proc uptimes
    # - Identify the primary driver of any DPS change
    # - Check for stat cap implications
    # - Consider practical factors (item availability, raid comp)

    # Phase 4: Synthesize answer
    # - Lead with the answer ("Yes, DST is better by 47 DPS")
    # - Explain why (mechanics-level reasoning)
    # - Show supporting data (tables, key numbers)
    # - Note caveats (fight length sensitivity, buff assumptions)
```

## Tool Schemas

### sim_run (registered in `tools/analysis/sim_run.py`)

```python
class SimRunInput(BaseModel):
    """Input for the sim_run tool."""
    import_string: str | None = None
    spec: RogueSpec | None = None
    talents: str | None = None
    gear_overrides: dict[str, int] = {}
    buff_preset: str = "full_25man"
    boss_armor: int = 7700
    fight_length: int = 300
    iterations: int = 10000
    compute_stat_weights: bool = False

class SimRunOutput(BaseModel):
    """Output from the sim_run tool."""
    dps_mean: float
    dps_std: float
    ability_breakdown: list[AbilityBreakdown]
    stat_weights: list[StatWeight] | None
    proc_uptimes: list[ProcUptime]
    resource_stats: ResourceStats
    # Full SimResult available but agent sees summary
```

### sim_compare (registered in `tools/analysis/sim_compare.py`)

```python
class SimCompareInput(BaseModel):
    """Input for comparing two setups."""
    base_import: str | None = None        # or use session config
    swap_slot: str                         # "trinket_1", "main_hand", etc.
    swap_item: str                         # item name or ID
    # OR for non-gear comparisons:
    change_type: str | None = None         # "buff", "talent", "boss"
    change_value: str | None = None        # buff ID, talent string, armor value

class SimCompareOutput(BaseModel):
    """Structured comparison result."""
    dps_before: float
    dps_after: float
    dps_delta: float
    dps_delta_pct: float
    stat_changes: list[StatDiff]
    ability_changes: list[AbilityDiff]
    summary: str                           # natural language explanation
```

### sim_optimize (registered in `tools/analysis/sim_optimize.py`)

```python
class SimOptimizeInput(BaseModel):
    """Input for finding best item in a slot."""
    base_import: str | None = None
    slot: str                              # "trinket_1", "main_hand", etc.
    phase: int = 5                         # content phase filter
    top_n: int = 5

class SimOptimizeOutput(BaseModel):
    """Ranked item recommendations."""
    current_item: str
    current_dps: float
    recommendations: list[ItemRecommendation]

class ItemRecommendation(BaseModel):
    item_name: str
    item_id: int
    dps: float
    dps_delta: float
    source: str                            # "Gruul", "Kael'thas", "Badge vendor"
    phase: int
```

## Orchestrator Integration

### Router Addition

```python
# routing/models.py
class QueryCategory(StrEnum):
    TRIVIAL = "trivial"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ANALYSIS = "analysis"     # NEW

# routing/router.py — Qwen 4B classification prompt addition
ANALYSIS_EXAMPLES = [
    "What DPS should I expect?",
    "Is DST better than Brooch?",
    "What should I upgrade next?",
    "Sim me with full P5 BiS",
    "What are my stat weights?",
    "Compare combat vs mutilate for my gear",
    "How much DPS do I lose without Windfury?",
]
```

### Orchestrator Dispatch

```python
# agents/orchestrator.py
async def _dispatch_task(self, task: AgentTask) -> AgentResult:
    match task.category:
        case QueryCategory.ANALYSIS:
            return await self.analyst.execute(task)
        case QueryCategory.MODERATE:
            return await self.researcher.execute(task)
        case QueryCategory.COMPLEX:
            return await self._multi_agent_plan(task)
```

### Factory Registration

```python
# agents/factory.py
AGENT_REGISTRY["analyst"] = AnalystAgent
```

## Chat Handler Integration

### Sim Profile Session State

```python
# web/routers/chat.py
class ChatSession:
    ...
    sim_profile: SimConfig | None = None  # current sim profile

    async def handle_message(self, message: str):
        # If message is sim-related and no profile exists:
        #   - Check if user has imported on /sim page
        #   - Prompt to import if not
        # If profile exists:
        #   - Pass as context to Analyst agent
        ...
```

### Rendering Sim Results in Chat

Sim results returned from the Analyst are rendered as structured chat messages:

```python
class SimResultMessage:
    """Chat message containing sim data + visualization hints."""
    text: str                       # natural language answer
    dps_summary: dict | None        # {mean, std, median}
    ability_table: list[dict] | None  # for markdown table
    chart_data: dict | None          # JSON for client-side Chart.js
```

The frontend renders `chart_data` using the same Chart.js components as the `/sim` page.

## Interpretation Patterns

### Pattern 1: Stat Cap Awareness

```
The Analyst checks stat weights for `is_capped` flags:

"Hit rating is your most valuable stat at 2.4 EP — but you only need 47 more
to reach the yellow cap (142 total). After that it drops to 0.8 EP for white
hit improvement only. Don't gem pure hit beyond cap."
```

### Pattern 2: Proc Item Reasoning

```
Uses proc_uptimes from SimResult:

"DST has 23% uptime on its +325 haste proc. During that uptime, your MH swings
every 1.1s instead of 1.7s, which means 55% more Combat Potency procs and 55%
more Sword Spec procs. That's where the 47 DPS comes from — it's not just the
raw haste, it's the proc cascade."
```

### Pattern 3: Spec Comparison

```
Runs sim_compare with different talent strings:

"With your current P3 gear, Combat Swords does 1,847 DPS vs Mutilate at 1,723 DPS.
Mutilate catches up around P5 ilvl when crit scaling with Seal Fate overtakes
Combat Potency energy generation. Your crossover point is approximately 900 agility."
```

### Pattern 4: Raid Utility Trade-offs

```
Compares personal DPS vs raid DPS contribution:

"Expose Armor (Improved 2/2) removes 3,075 armor from the boss vs Sunder Armor's
2,600. That extra 475 armor reduction gains ~1.2% damage for ALL physical DPS in
the raid. For a 10-person physical group, that's roughly 200 raid DPS gained, but
you lose 67 personal DPS from using EA finishers instead of Eviscerate. Net gain
for the raid: +133 DPS. Worth it if no warrior is Sundering consistently."
```

## Error Handling

```python
class SimError(ShukketsuError):
    """Base error for simulation failures."""
    failure_mode = FailureMode.INTERNAL

class InvalidConfigError(SimError):
    """Config validation failed (missing gear, invalid talents, etc.)."""
    failure_mode = FailureMode.VALIDATION

class ItemNotFoundError(SimError):
    """Item ID not in database."""
    failure_mode = FailureMode.NOT_FOUND

class SimTimeoutError(SimError):
    """Sim took too long (stat weights on large iteration count)."""
    failure_mode = FailureMode.TIMEOUT
```

## Testing Strategy

### Unit Tests
- Tool schema validation (valid/invalid inputs)
- Agent prompt construction
- Stat cap detection logic
- Config builder from import strings

### Integration Tests
- Analyst agent executes sim_run tool end-to-end
- Analyst agent handles "compare X vs Y" queries
- Router correctly classifies ANALYSIS queries
- Orchestrator dispatches to Analyst

### Agent Behavior Tests
- Given a gear set below hit cap, Analyst mentions hit cap in response
- Given a comparison request, Analyst explains WHY not just WHAT
- Given an optimization request, Analyst returns ranked recommendations
- Analyst refuses to make claims without simming first
