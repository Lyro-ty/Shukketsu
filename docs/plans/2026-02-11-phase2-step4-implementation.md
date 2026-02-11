# Phase 2 Step 4: Researcher Agent — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the Researcher specialist agent — a light BaseAgent subclass with a Llama 70B structuring pass that converts free-text ReAct output into typed `ResearchResult` with findings, evidence, gaps, and sufficiency.

**Architecture:** `Researcher(BaseAgent)` overrides `execute()` → calls `_run_loop()` for the ReAct loop → runs a second `get_structured_output()` call to structure findings into `ResearchResult`. System prompt lives in `llm/prompts/researcher.py`. Factory uses a `_ROLE_CLASSES` registry to return the correct subclass.

**Tech Stack:** Python 3.12, Pydantic v2, instructor, Langfuse `@observe`, pytest with `unittest.mock`

**Design doc:** `docs/plans/2026-02-11-phase2-step4-researcher-agent.md`

**Starting test count:** 438. **Target:** ~469.

---

## Task 1: Add scratchpad to _RunOutcome

Extend the internal NamedTuple so the Researcher subclass can access tool observations.

**Files:**
- Modify: `code/shukketsu/agents/base.py:32-40` (_RunOutcome), `:142`, `:165-168`, `:171`
- Test: `tests/unit/test_base_agent.py` (add 2 tests to existing TestRunLoop class)

**Step 1: Write failing tests in `test_base_agent.py`**

Add these two tests to the end of the `TestRunLoop` class (after line 247):

```python
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_run_outcome_has_scratchpad(self, mock_llm: AsyncMock) -> None:
        """_RunOutcome includes the scratchpad field."""
        mock_llm.return_value = _final_answer("answer")
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        outcome = await agent._run_loop("query", on_status=None)
        assert hasattr(outcome, "scratchpad")
        assert isinstance(outcome.scratchpad, list)

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_run_loop_returns_scratchpad_with_entries(self, mock_llm: AsyncMock) -> None:
        """_run_loop scratchpad contains tool call entries."""
        mock_llm.side_effect = [
            _tool_call("echo", {"text": "hello"}),
            _final_answer("Done"),
        ]
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        outcome = await agent._run_loop("query", on_status=None)
        assert len(outcome.scratchpad) == 1
        assert outcome.scratchpad[0]["tool_name"] == "echo"
        assert "Echo: hello" in outcome.scratchpad[0]["observation"]
```

**Step 2: Run tests, confirm failure**

```bash
python3 -m pytest tests/unit/test_base_agent.py::TestRunLoop::test_run_outcome_has_scratchpad tests/unit/test_base_agent.py::TestRunLoop::test_run_loop_returns_scratchpad_with_entries -v
```

Expected: FAIL — `_RunOutcome` has no `scratchpad` field.

**Step 3: Implement — modify `_RunOutcome` and `_run_loop`**

In `code/shukketsu/agents/base.py`:

1. Add `scratchpad` field to `_RunOutcome` (line 32-40):

```python
class _RunOutcome(NamedTuple):
    """Internal return type from the ReAct loop.

    Carries the output string, execution status, and the full scratchpad
    so that specialist agents (e.g. Researcher) can access tool observations
    for post-processing.
    """

    output: str
    status: TaskStatus
    scratchpad: list[dict[str, Any]]
```

2. Update all three return sites in `_run_loop()`:

Line 142 (FINAL_ANSWER):
```python
return _RunOutcome(output=step.answer, status=TaskStatus.SUCCESS, scratchpad=scratchpad)  # type: ignore[arg-type]
```

Lines 165-168 (loop detected):
```python
return _RunOutcome(
    output=self._synthesize_partial_answer(scratchpad),
    status=TaskStatus.PARTIAL,
    scratchpad=scratchpad,
)
```

Line 171 (max iterations):
```python
return _RunOutcome(output=config.AGENT_GRACEFUL_FAILURE, status=TaskStatus.FAILED, scratchpad=scratchpad)
```

**Step 4: Run ALL base agent tests**

```bash
python3 -m pytest tests/unit/test_base_agent.py -v
```

Expected: ALL PASS (30 tests). Existing tests use `.output` and `.status` attribute access — the new field is invisible to them.

**Step 5: Commit**

```bash
git add code/shukketsu/agents/base.py tests/unit/test_base_agent.py
git commit -m "feat(agents): add scratchpad field to _RunOutcome"
```

---

## Task 2: Add Finding and ResearchResult models

Add the Pydantic models for structured research output to the task protocol.

**Files:**
- Modify: `code/shukketsu/agents/tasks.py` (add after line 84, before WriteTask)
- Modify: `code/shukketsu/agents/__init__.py` (export new models)
- Test: `tests/unit/test_researcher.py` (new file, first test class only)

**Step 1: Create test file with model tests**

Create `tests/unit/test_researcher.py`:

```python
"""Tests for the Researcher agent and its models."""

import pytest
from pydantic import ValidationError

from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    Finding,
    ResearchResult,
    TaskStatus,
)


class TestFindingModel:
    def test_valid_finding(self) -> None:
        f = Finding(
            claim="The hit cap is 142 rating",
            evidence=["chunk:15", "chunk:91"],
            confidence=0.9,
            entity_refs=["hit rating", "combat swords"],
        )
        assert f.claim == "The hit cap is 142 rating"
        assert len(f.evidence) == 2
        assert f.confidence == 0.9

    def test_confidence_rejects_above_one(self) -> None:
        with pytest.raises(ValidationError):
            Finding(claim="test", confidence=1.5)

    def test_confidence_rejects_below_zero(self) -> None:
        with pytest.raises(ValidationError):
            Finding(claim="test", confidence=-0.1)

    def test_confidence_boundary_zero(self) -> None:
        f = Finding(claim="test", confidence=0.0)
        assert f.confidence == 0.0

    def test_confidence_boundary_one(self) -> None:
        f = Finding(claim="test", confidence=1.0)
        assert f.confidence == 1.0

    def test_defaults_empty_lists(self) -> None:
        f = Finding(claim="test", confidence=0.5)
        assert f.evidence == []
        assert f.entity_refs == []


class TestResearchResultModel:
    def test_valid_research_result(self) -> None:
        r = ResearchResult(
            task_id="abc-123",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="The hit cap is 142.",
            findings=[Finding(claim="Hit cap is 142", confidence=0.9)],
            sources_used=["https://example.com"],
            strategies_used=["rag_search"],
            sufficient=True,
        )
        assert len(r.findings) == 1
        assert r.sufficient is True

    def test_defaults(self) -> None:
        r = ResearchResult(
            task_id="abc",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="text",
        )
        assert r.findings == []
        assert r.sources_used == []
        assert r.strategies_used == []
        assert r.gaps == []
        assert r.sufficient is False

    def test_inherits_agent_result(self) -> None:
        assert issubclass(ResearchResult, AgentResult)

    def test_has_agent_result_fields(self) -> None:
        """ResearchResult includes all base AgentResult fields."""
        r = ResearchResult(
            task_id="abc",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="text",
            evidence=["src1"],
            metadata={"key": "val"},
        )
        assert r.evidence == ["src1"]
        assert r.metadata == {"key": "val"}
```

**Step 2: Run tests, confirm failure**

```bash
python3 -m pytest tests/unit/test_researcher.py::TestFindingModel tests/unit/test_researcher.py::TestResearchResultModel -v
```

Expected: FAIL — `Finding` and `ResearchResult` not importable from `agents.tasks`.

**Step 3: Implement models in `agents/tasks.py`**

Add after `ResearchTask` (after line 84), before `WriteTask`:

```python
class Finding(BaseModel):
    """A single research finding with evidence and confidence."""

    claim: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    entity_refs: list[str] = Field(default_factory=list)


class ResearchResult(AgentResult):
    """Structured output from the Researcher agent.

    Extends AgentResult with research-specific fields that the Writer
    and Orchestrator consume.
    """

    findings: list[Finding] = Field(default_factory=list)
    sources_used: list[str] = Field(default_factory=list)
    strategies_used: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    sufficient: bool = False
```

**Step 4: Update `agents/__init__.py` exports**

Add to the import block:

```python
from code.shukketsu.agents.tasks import (
    ...
    Finding,
    ResearchResult,
    ...
)
```

Add to `__all__`:
```python
    "Finding",
    "ResearchResult",
```

**Step 5: Run tests**

```bash
python3 -m pytest tests/unit/test_researcher.py::TestFindingModel tests/unit/test_researcher.py::TestResearchResultModel -v
```

Expected: ALL PASS (10 tests).

**Step 6: Run full suite to check for regressions**

```bash
python3 -m pytest tests/unit/ -v --tb=short
```

Expected: 450 tests pass (438 existing + 2 base_agent + 10 researcher models).

**Step 7: Commit**

```bash
git add code/shukketsu/agents/tasks.py code/shukketsu/agents/__init__.py tests/unit/test_researcher.py
git commit -m "feat(agents): add Finding and ResearchResult models"
```

---

## Task 3: Create the Researcher system prompt

Write the full system prompt with 3 worked examples in a dedicated prompts module.

**Files:**
- Create: `code/shukketsu/llm/prompts/__init__.py`
- Create: `code/shukketsu/llm/prompts/researcher.py`
- Modify: `code/shukketsu/config.py:116-120` (replace placeholder with import)
- Test: `tests/unit/test_researcher.py` (add TestResearcherPrompt class)

**Step 1: Write failing prompt tests**

Append to `tests/unit/test_researcher.py`:

```python
from code.shukketsu.llm.prompts.researcher import RESEARCHER_SYSTEM_PROMPT


class TestResearcherPrompt:
    def test_prompt_contains_all_tool_names(self) -> None:
        for tool in ["rag_search", "graph_search", "web_search", "web_ingest"]:
            assert tool in RESEARCHER_SYSTEM_PROMPT, f"Missing tool: {tool}"

    def test_prompt_contains_strategy_selection(self) -> None:
        assert "strategy" in RESEARCHER_SYSTEM_PROMPT.lower()

    def test_prompt_contains_worked_examples(self) -> None:
        assert "Example 1" in RESEARCHER_SYSTEM_PROMPT
        assert "Example 2" in RESEARCHER_SYSTEM_PROMPT
        assert "Example 3" in RESEARCHER_SYSTEM_PROMPT

    def test_prompt_contains_evaluation_criteria(self) -> None:
        prompt_lower = RESEARCHER_SYSTEM_PROMPT.lower()
        assert "evaluate" in prompt_lower or "assess" in prompt_lower
```

**Step 2: Run tests, confirm failure**

```bash
python3 -m pytest tests/unit/test_researcher.py::TestResearcherPrompt -v
```

Expected: FAIL — `llm.prompts.researcher` module does not exist.

**Step 3: Create prompts package**

Create `code/shukketsu/llm/prompts/__init__.py`:

```python
"""System prompts for specialist agents."""
```

**Step 4: Create the Researcher prompt**

Create `code/shukketsu/llm/prompts/researcher.py`:

```python
"""Researcher agent system prompt.

This module is a pure string constant with zero imports.
config.py imports from here — never the reverse (prevents circular imports).
"""

RESEARCHER_SYSTEM_PROMPT = """\
You are a Research Specialist for WoW: The Burning Crusade (TBC) Rogue content. \
Your job is to gather comprehensive, accurate information using your search tools, \
then present your findings clearly with evidence.

## Your Tools

- **rag_search**: Hybrid vector + keyword search over the knowledge base. Use for \
factual questions, stat lookups, gear comparisons, and any topic likely covered by \
existing guides. Results are ranked by relevance with trust scores.

- **graph_search**: Knowledge graph traversal for entity relationships. Use when you \
need connections between game concepts: "What drops from [boss]?", "What stats does \
[item] have?", "What's BiS for [spec] in [phase]?", "What talents synergize with \
[spell]?" Pass an entity name and optionally filter by relation_types or target_type.

- **web_search**: Brave Search API for information not in the knowledge base. Use when \
rag_search and graph_search return insufficient results. Returns titles, URLs, and \
snippets — NOT full page content.

- **web_ingest**: Fetch a web page and store it permanently in the knowledge base. Use \
after web_search finds a promising URL. After ingesting, re-run rag_search to find the \
newly stored content.

## Search Strategy Selection

Choose your tool based on the question type:

| Question Type | First Tool | Why |
|--------------|-----------|-----|
| Factual lookup ("What is the hit cap?") | rag_search | Direct answer likely in KB |
| Relationship query ("What drops from Gruul?") | graph_search | Structured entity data |
| Comparison ("Combat vs Mutilate for Phase 1") | rag_search (×2) | Search each side separately |
| Missing from KB ("Latest sim results for...") | web_search | Not in local KB |
| Stat/mechanic deep-dive | graph_search → rag_search | Graph finds structure, rag fills detail |

## Research Patterns

### 1. Decompose Complex Questions

Break multi-part questions into sub-queries. Each sub-query gets its own tool call.

Bad: One broad search for "compare combat swords vs mutilate for Gruul"
Good: Three focused searches:
  - rag_search("combat swords rogue stat priority Phase 1")
  - rag_search("mutilate rogue stat priority Phase 1")
  - graph_search(entity="gruul", relation_types=["has_mechanic"])

### 2. Evaluate Results Before Answering

After each search, ask yourself:
- Do these results actually answer my question?
- Are the sources trustworthy (check trust scores)?
- Do multiple sources agree, or is there conflict?
- Is there enough detail, or do I need a follow-up search?

If results are insufficient, refine your query or try a different tool.

### 3. Iterate When Needed

- First search too broad? Narrow with more specific terms.
- First search too narrow? Broaden or try different keywords.
- Found partial info? Search for the missing pieces.
- Low-trust results only? Try a different source via web_search.

### 4. Gather Evidence

For every claim you make:
- Note which source(s) support it (URL, trust score).
- If multiple sources agree, your confidence should be higher.
- If sources disagree, explicitly note the disagreement.

### 5. Identify Gaps

When you cannot find solid evidence for part of the question:
- State clearly what you found and what is still missing.
- Do NOT guess or fabricate information.
- If you tried multiple strategies and still have gaps, say so.

### 6. Know When to Stop

You have enough when:
- Each part of the question has at least one supporting source.
- Key claims are supported by 2+ sources where possible.
- You have checked both the knowledge base and graph for relevant data.

Stop searching when additional queries return redundant information.

## Example Research Traces

### Example 1: Multi-Strategy Decomposition

Query: "What are the best trinkets for a combat rogue in Phase 1 and why?"

Step 1: graph_search(entity="combat swords", target_type="item")
  → Found relationships to several items including trinkets
Step 2: rag_search("combat rogue trinket phase 1 BiS ranking")
  → 3 results: DST analysis, trinket comparison guide, Phase 1 gear guide (trust: 0.7, 0.6, 0.8)
Step 3: Evaluate — graph gave entity relationships, rag confirmed details with stat analysis.
  Missing: specific proc rate for Dragonspine Trophy haste effect.
Step 4: rag_search("dragonspine trophy proc rate haste internal cooldown")
  → 1 result with detailed proc math (trust: 0.7)
Step 5: Final answer — 4 findings with evidence from 3 sources, 0 gaps.

### Example 2: KB Insufficient → Web Fallback

Query: "What is the optimal poison setup for mutilate rogues on Illidan?"

Step 1: rag_search("mutilate rogue poison setup Illidan fight")
  → 1 result, low trust (0.4), vague on specifics
Step 2: graph_search(entity="illidan", relation_types=["has_mechanic"])
  → Boss mechanics found (shear, flames, demon form), but no poison-specific data
Step 3: Evaluate — know boss mechanics but need poison-specific advice. KB insufficient.
Step 4: web_search("TBC mutilate rogue poison Illidan Black Temple guide")
  → Found 2 promising URLs from Elitist Jerks and WoWhead
Step 5: web_ingest(url="https://example.com/mutilate-guide")
  → Page stored in KB
Step 6: rag_search("mutilate poison setup Illidan")
  → Now 3 results with specific poison recommendations (trust: 0.7)
Step 7: Final answer — 3 findings, gap on exact poison proc math vs. Illidan phases

### Example 3: Graph-First Relationship Query

Query: "What items drop from Gruul that rogues can use?"

Step 1: graph_search(entity="gruul", relation_types=["drops_from"])
  → Found 5 items linked to Gruul via drops_from relationships
Step 2: Evaluate — have item names and types. Need stat details and rogue relevance.
Step 3: rag_search("Gruul loot table rogue leather melee DPS")
  → 2 results with item stat breakdowns and rogue recommendations (trust: 0.7, 0.8)
Step 4: Final answer — item list with stats, which are rogue-relevant, evidence from graph + search

## Important Rules

- NEVER guess. If you cannot find evidence, state what is missing.
- ALWAYS cite your sources when making claims.
- If multiple sources disagree, note the disagreement — do not pick a side silently.
- Prefer knowledge base results over web search when both have relevant information.
- Trust scores matter — weight higher-trust sources more heavily in your reasoning.
- When you are done researching, provide a comprehensive final answer that covers all \
parts of the original question.\
"""

STRUCTURING_PROMPT = """\
You are a research structuring assistant. Given a research query, the researcher's \
final answer, and the tool observations collected during research, extract structured \
findings.

For each distinct factual claim in the answer:
1. Extract the claim text (one clear sentence).
2. List evidence references from the observations (source URLs, "chunk:N" IDs, or \
page titles — whatever identifies the source).
3. Assign confidence (0.0-1.0):
   - 0.9-1.0: Multiple independent sources agree
   - 0.6-0.8: Single reliable source supports it
   - 0.3-0.5: Weak or indirect evidence
   - 0.0-0.2: Mentioned but essentially unsupported
4. List entity references — game entities mentioned (item names, spell names, boss \
names, stat names, spec names).

Also extract:
- GAPS: Topics the research could not adequately cover (questions without answers, \
areas where results were empty or low-confidence). Empty list if research was thorough.
- SUFFICIENT: True if the research comprehensively answers the original query with \
solid evidence. False if significant gaps remain or evidence is weak.

Be precise. Only extract claims that have evidence in the observations. Do not invent \
findings. If the researcher found nothing useful, return empty findings with \
sufficient=false.\
"""
```

**Step 5: Update config.py — replace placeholder with import**

In `code/shukketsu/config.py`, replace lines 116-120:

Old:
```python
# Role-specific system prompts (placeholder — specialist prompts added in later steps)
RESEARCHER_SYSTEM_PROMPT = (
    "You are a Research Specialist for WoW TBC Rogue content. "
    "Your job is to gather comprehensive, accurate information using your search tools."
)
```

New:
```python
# Role-specific system prompts
from code.shukketsu.llm.prompts.researcher import RESEARCHER_SYSTEM_PROMPT  # noqa: E402
```

**Important**: This import must go at the BOTTOM of config.py (after all `os.getenv` calls) because `llm/prompts/researcher.py` has zero imports and config.py's existing content has no dependency on the prompt. Add the import on the line where the old constant was, replacing the 4-line placeholder.

**Step 6: Run prompt tests**

```bash
python3 -m pytest tests/unit/test_researcher.py::TestResearcherPrompt -v
```

Expected: ALL PASS (4 tests).

**Step 7: Run full suite**

```bash
python3 -m pytest tests/unit/ -v --tb=short
```

Expected: 454 tests pass. Verify that `test_agent_factory.py::test_researcher_prompt_mentions_search` still passes (the new prompt still contains "search").

**Step 8: Commit**

```bash
git add code/shukketsu/llm/prompts/__init__.py code/shukketsu/llm/prompts/researcher.py code/shukketsu/config.py tests/unit/test_researcher.py
git commit -m "feat(llm): add Researcher system prompt with worked examples"
```

---

## Task 4: Implement the Researcher subclass

Build the core Researcher class with `execute()` override, structuring pass, and error handling.

**Files:**
- Create: `code/shukketsu/agents/researcher.py`
- Modify: `code/shukketsu/agents/__init__.py` (export Researcher)
- Test: `tests/unit/test_researcher.py` (add TestResearcherExecute class)

**Step 1: Write failing execute tests**

Append to `tests/unit/test_researcher.py`:

```python
from typing import Any
from unittest.mock import AsyncMock, patch

from code.shukketsu.agents.researcher import Researcher, StructuredFindings
from code.shukketsu.agents.tasks import AgentTask, ResearchTask
from code.shukketsu.llm.schemas import ActionType, AgentStep, ToolCall
from code.shukketsu.resilience.errors import StructuredOutputError
from code.shukketsu.tools.registry import ToolRegistry
from code.shukketsu.tools.schemas import Tool


class _EchoTool(Tool):
    name = "rag_search"
    description = "Search the knowledge base."
    parameters_schema = {"query": {"type": "string", "description": "Search query"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        q = tool_input.get("query", "")
        return f"Found 1 result:\n[1] Source: Test Guide (https://example.com)\nTrust: 0.8\nContent: Info about {q}\n"


class _GraphTool(Tool):
    name = "graph_search"
    description = "Search the knowledge graph."
    parameters_schema = {"entity": {"type": "string", "description": "Entity name"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        e = tool_input.get("entity", "")
        return f'Found 1 relationship for "{e}":\n- drops_from → Gruul (confidence: 0.9)\n'


def _registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _final_answer(answer: str) -> AgentStep:
    return AgentStep(reasoning="I have the answer.", action=ActionType.FINAL_ANSWER, answer=answer)


def _tool_call(tool_name: str, tool_input: dict[str, Any]) -> AgentStep:
    return AgentStep(
        reasoning="Need info.",
        action=ActionType.TOOL_CALL,
        tool_call=ToolCall(thought="Searching", tool_name=tool_name, tool_input=tool_input),
    )


def _mock_structured_findings(**kwargs: Any) -> StructuredFindings:
    """Build a StructuredFindings with sensible defaults."""
    defaults: dict[str, Any] = {
        "findings": [Finding(claim="Test claim", evidence=["https://example.com"], confidence=0.8)],
        "gaps": [],
        "sufficient": True,
    }
    defaults.update(kwargs)
    return StructuredFindings(**defaults)


class TestResearcherExecute:
    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_returns_research_result(self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock) -> None:
        """execute() returns ResearchResult, not plain AgentResult."""
        mock_loop_llm.return_value = _final_answer("The hit cap is 142.")
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await researcher.execute(AgentTask(query="What is the hit cap?"))

        assert isinstance(result, ResearchResult)
        assert result.agent_role == AgentRole.RESEARCHER
        assert result.status == TaskStatus.SUCCESS

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_structuring_pass_called(self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock) -> None:
        """A second LLM call is made for structuring after the ReAct loop."""
        mock_loop_llm.return_value = _final_answer("Answer text.")
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        await researcher.execute(AgentTask(query="q"))

        mock_struct_llm.assert_called_once()
        # Verify it was called with StructuredFindings as response_model
        call_kwargs = mock_struct_llm.call_args.kwargs
        assert call_kwargs["response_model"] is StructuredFindings

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_scratchpad_passed_to_structuring(self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock) -> None:
        """Structuring pass receives tool observations from the scratchpad."""
        mock_loop_llm.side_effect = [
            _tool_call("rag_search", {"query": "hit cap"}),
            _final_answer("The hit cap is 142."),
        ]
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        await researcher.execute(AgentTask(query="hit cap"))

        # The structuring call's user message should contain the observation
        call_kwargs = mock_struct_llm.call_args.kwargs
        user_msg = call_kwargs["messages"][1]["content"]
        assert "rag_search" in user_msg
        assert "example.com" in user_msg

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_sources_derived_from_findings(self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock) -> None:
        """sources_used is flattened from Finding.evidence, not parsed from observations."""
        mock_loop_llm.return_value = _final_answer("Answer.")
        mock_struct_llm.return_value = _mock_structured_findings(
            findings=[
                Finding(claim="Claim 1", evidence=["src_a", "src_b"], confidence=0.9),
                Finding(claim="Claim 2", evidence=["src_b", "src_c"], confidence=0.7),
            ]
        )

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await researcher.execute(AgentTask(query="q"))

        assert result.sources_used == ["src_a", "src_b", "src_c"]  # deduplicated, order-preserving

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_strategies_extracted_from_scratchpad(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """strategies_used contains deduplicated tool names from scratchpad."""
        mock_loop_llm.side_effect = [
            _tool_call("rag_search", {"query": "q1"}),
            _tool_call("graph_search", {"entity": "e1"}),
            _tool_call("rag_search", {"query": "q2"}),
            _final_answer("Done"),
        ]
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(
            tool_registry=_registry(_EchoTool(), _GraphTool()), role=AgentRole.RESEARCHER
        )
        result = await researcher.execute(AgentTask(query="q"))

        assert result.strategies_used == ["rag_search", "graph_search"]

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_structuring_failure_graceful_fallback(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """StructuredOutputError from structuring pass → fallback with empty findings."""
        mock_loop_llm.return_value = _final_answer("Some answer text.")
        mock_struct_llm.side_effect = StructuredOutputError("Validation failed")

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await researcher.execute(AgentTask(query="q"))

        assert isinstance(result, ResearchResult)
        assert result.output == "Some answer text."
        assert result.findings == []
        assert result.sufficient is False

    def test_execute_has_langfuse_observe(self) -> None:
        """Researcher.execute() has @observe decorator (not inherited from base)."""
        # The @observe decorator wraps the method. We can check the method has
        # langfuse wrapper attributes or is different from BaseAgent.execute.
        from code.shukketsu.agents.base import BaseAgent

        # Researcher overrides execute — it should not be the same method object
        assert Researcher.execute is not BaseAgent.execute
```

**Step 2: Run tests, confirm failure**

```bash
python3 -m pytest tests/unit/test_researcher.py::TestResearcherExecute -v
```

Expected: FAIL — `agents.researcher` module does not exist.

**Step 3: Implement `agents/researcher.py`**

Create `code/shukketsu/agents/researcher.py`:

```python
"""Researcher specialist agent.

Overrides execute() to run the standard ReAct loop and then structure
the output into a typed ResearchResult via a Llama 70B structuring pass.
"""

import logging
from typing import Any

from langfuse import observe
from pydantic import BaseModel, Field

from code.shukketsu import config
from code.shukketsu.agents.base import BaseAgent, StatusCallback
from code.shukketsu.agents.tasks import AgentRole, AgentTask, Finding, ResearchResult, TaskStatus
from code.shukketsu.llm.prompts.researcher import STRUCTURING_PROMPT
from code.shukketsu.llm.structured import get_structured_output
from code.shukketsu.resilience.errors import StructuredOutputError

logger = logging.getLogger(__name__)


class StructuredFindings(BaseModel):
    """Schema for the Llama 70B structuring pass.

    Parsed from the agent's output + scratchpad, then mapped
    to ResearchResult fields.
    """

    findings: list[Finding] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    sufficient: bool = False


class Researcher(BaseAgent):
    """Research specialist agent.

    Uses the same ReAct loop as BaseAgent, then runs a second Llama 70B
    call to structure the free-text output into a typed ResearchResult
    with findings, evidence, gaps, and sufficiency assessment.
    """

    @observe(as_type="agent")
    async def execute(
        self, task: AgentTask, *, on_status: StatusCallback | None = None
    ) -> ResearchResult:
        """Execute a research task and return structured findings.

        Runs the ReAct loop, then structures the output via a second
        LLM call. Falls back to a basic result if structuring fails.

        Args:
            task: The research task to execute.
            on_status: Optional async callback for progress updates.

        Returns:
            ResearchResult with structured findings and evidence.

        Raises:
            ValueError: If no role is set on this agent.
            LLMUnavailableError: If the LLM backend is unreachable.
        """
        if self.role is None:
            raise ValueError("Researcher requires a role. Use AgentFactory or set role in constructor.")

        outcome = await self._run_loop(task.query, on_status=on_status)

        # Skip structuring for failed loops — no useful output to parse
        if outcome.status == TaskStatus.FAILED:
            logger.info("ReAct loop failed; skipping structuring pass")
            return ResearchResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=outcome.status,
                output=outcome.output,
                sufficient=False,
            )

        # Structure the output via Llama 70B
        if on_status:
            await on_status("structuring findings...")

        try:
            structured = await self._structure_findings(
                query=task.query,
                output=outcome.output,
                scratchpad=outcome.scratchpad,
            )
        except (StructuredOutputError, Exception) as exc:
            logger.warning("Structuring pass failed, returning basic result: %s", exc)
            return ResearchResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=outcome.status,
                output=outcome.output,
                strategies_used=self._extract_strategies(outcome.scratchpad),
                sufficient=False,
            )

        # Derive metadata from structuring output + scratchpad
        sources_used = list(dict.fromkeys(e for f in structured.findings for e in f.evidence))
        strategies_used = self._extract_strategies(outcome.scratchpad)

        return ResearchResult(
            task_id=task.task_id,
            agent_role=self.role,
            status=outcome.status,
            output=outcome.output,
            evidence=sources_used,
            findings=structured.findings,
            sources_used=sources_used,
            strategies_used=strategies_used,
            gaps=structured.gaps,
            sufficient=structured.sufficient,
        )

    async def _structure_findings(
        self,
        query: str,
        output: str,
        scratchpad: list[dict[str, Any]],
    ) -> StructuredFindings:
        """Run the Llama 70B structuring pass.

        Takes the agent's free-text output and tool observations,
        returns structured findings with evidence and confidence.
        """
        observations_text = self._format_scratchpad(scratchpad)

        messages = [
            {"role": "system", "content": STRUCTURING_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Researcher's answer:\n{output}\n\n"
                    f"Tool observations:\n{observations_text}"
                ),
            },
        ]

        return await get_structured_output(
            response_model=StructuredFindings,
            messages=messages,
        )

    def _extract_strategies(self, scratchpad: list[dict[str, Any]]) -> list[str]:
        """Extract unique tool names used during research."""
        return list(dict.fromkeys(e["tool_name"] for e in scratchpad))

    def _format_scratchpad(self, scratchpad: list[dict[str, Any]]) -> str:
        """Format scratchpad entries for the structuring prompt.

        Truncates each observation to 500 chars to fit within the
        structuring call's token budget.
        """
        if not scratchpad:
            return "(no tool calls made)"

        parts: list[str] = []
        for i, entry in enumerate(scratchpad, 1):
            observation = str(entry.get("observation", ""))
            truncated = observation[:500] + "..." if len(observation) > 500 else observation
            parts.append(
                f"[{i}] Tool: {entry['tool_name']}\n"
                f"    Input: {entry['tool_input']}\n"
                f"    Result: {truncated}"
            )
        return "\n\n".join(parts)
```

**Step 4: Update `agents/__init__.py`**

Add the Researcher import:

```python
from code.shukketsu.agents.researcher import Researcher
```

And add `"Researcher"` to `__all__`.

**Step 5: Run execute tests**

```bash
python3 -m pytest tests/unit/test_researcher.py::TestResearcherExecute -v
```

Expected: ALL PASS (7 tests).

**Step 6: Run full suite**

```bash
python3 -m pytest tests/unit/ -v --tb=short
```

Expected: 461 tests pass.

**Step 7: Commit**

```bash
git add code/shukketsu/agents/researcher.py code/shukketsu/agents/__init__.py tests/unit/test_researcher.py
git commit -m "feat(agents): add Researcher subclass with structuring pass"
```

---

## Task 5: Add Researcher behavior tests

Test the multi-step research patterns: decomposition, strategy selection, web fallback, iteration limits, gap detection.

**Files:**
- Test: `tests/unit/test_researcher.py` (add TestResearcherBehavior class)

**Step 1: Write behavior tests**

Append to `tests/unit/test_researcher.py`:

```python
class TestResearcherBehavior:
    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_decompose_two_part_question(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """Agent decomposes a multi-part question and calls tools multiple times."""
        mock_loop_llm.side_effect = [
            _tool_call("rag_search", {"query": "combat trinkets phase 1"}),
            _tool_call("rag_search", {"query": "combat stat priority"}),
            _final_answer("Trinkets ranked by stat priority..."),
        ]
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await researcher.execute(AgentTask(query="Best trinkets and why?"))

        # ReAct loop made 3 calls (2 tool + 1 final), structuring made 1
        assert mock_loop_llm.call_count == 3
        assert result.status == TaskStatus.SUCCESS

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_multiple_strategies_used(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """Agent uses both rag_search and graph_search tools."""
        mock_loop_llm.side_effect = [
            _tool_call("graph_search", {"entity": "combat swords"}),
            _tool_call("rag_search", {"query": "combat swords gear"}),
            _final_answer("Combined results."),
        ]
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(
            tool_registry=_registry(_EchoTool(), _GraphTool()), role=AgentRole.RESEARCHER
        )
        result = await researcher.execute(AgentTask(query="Combat gear"))

        assert "graph_search" in result.strategies_used
        assert "rag_search" in result.strategies_used

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_web_fallback_on_empty_kb(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """Agent falls back to web_search when rag_search returns nothing useful."""

        class _WebTool(Tool):
            name = "web_search"
            description = "Web search."
            parameters_schema = {"query": {"type": "string", "description": "Query"}}

            async def execute(self, tool_input: dict[str, Any]) -> str:
                return "Found 1 web result for ...: [1] Guide Title\n    https://example.com/guide\n"

        mock_loop_llm.side_effect = [
            _tool_call("rag_search", {"query": "obscure topic"}),
            _tool_call("web_search", {"query": "obscure topic TBC rogue"}),
            _final_answer("Found info via web."),
        ]
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(
            tool_registry=_registry(_EchoTool(), _WebTool()), role=AgentRole.RESEARCHER
        )
        result = await researcher.execute(AgentTask(query="obscure topic"))

        assert "web_search" in result.strategies_used

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_respects_max_iterations(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """Agent stops after max_iterations even if no final answer."""
        mock_loop_llm.return_value = _tool_call("rag_search", {"query": "endless"})
        # structuring should NOT be called — FAILED status skips it
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(
            tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER, max_iterations=2
        )
        result = await researcher.execute(AgentTask(query="q"))

        assert result.status == TaskStatus.FAILED
        assert mock_loop_llm.call_count == 2
        mock_struct_llm.assert_not_called()

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_gaps_populated_on_partial_results(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """gaps list is non-empty when structuring reports incomplete research."""
        mock_loop_llm.return_value = _final_answer("Partial info only.")
        mock_struct_llm.return_value = _mock_structured_findings(
            gaps=["Could not find proc rate data", "No Phase 2 comparison available"],
            sufficient=False,
        )

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await researcher.execute(AgentTask(query="detailed analysis"))

        assert len(result.gaps) == 2
        assert result.sufficient is False

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_sufficient_false_on_empty_results(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """All tools return empty → sufficient=False."""
        mock_loop_llm.return_value = _final_answer("Could not find any information.")
        mock_struct_llm.return_value = _mock_structured_findings(
            findings=[], gaps=["No data found"], sufficient=False
        )

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await researcher.execute(AgentTask(query="nonexistent topic"))

        assert result.findings == []
        assert result.sufficient is False

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_failed_loop_skips_structuring(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """FAILED status from loop → no structuring pass, minimal result."""
        mock_loop_llm.return_value = _tool_call("rag_search", {"query": "loop"})
        mock_struct_llm.return_value = _mock_structured_findings()

        researcher = Researcher(
            tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER, max_iterations=1
        )
        result = await researcher.execute(AgentTask(query="q"))

        assert result.status == TaskStatus.FAILED
        assert result.findings == []
        assert result.sufficient is False
        mock_struct_llm.assert_not_called()
```

**Step 2: Run behavior tests**

```bash
python3 -m pytest tests/unit/test_researcher.py::TestResearcherBehavior -v
```

Expected: ALL PASS (7 tests).

**Step 3: Run full suite**

```bash
python3 -m pytest tests/unit/ -v --tb=short
```

Expected: 468 tests pass.

**Step 4: Commit**

```bash
git add tests/unit/test_researcher.py
git commit -m "test(agents): add Researcher behavior tests"
```

---

## Task 6: Update the factory with class registry

Wire the factory to return `Researcher` for `AgentRole.RESEARCHER` using a `_ROLE_CLASSES` dict.

**Files:**
- Modify: `code/shukketsu/agents/factory.py`
- Test: `tests/unit/test_researcher.py` (add TestResearcherFactory class)
- Test: `tests/unit/test_agent_factory.py` (add registry test, verify existing pass)

**Step 1: Write failing factory tests**

Append to `tests/unit/test_researcher.py`:

```python
from code.shukketsu.agents.factory import AgentFactory
from code.shukketsu.llm.prompts.researcher import RESEARCHER_SYSTEM_PROMPT as FULL_PROMPT


class TestResearcherFactory:
    def test_factory_returns_researcher_type(self) -> None:
        """factory.create(RESEARCHER) returns a Researcher instance."""
        factory = AgentFactory()
        agent = factory.create(AgentRole.RESEARCHER)
        assert isinstance(agent, Researcher)

    def test_factory_other_roles_return_base_agent(self) -> None:
        """Non-researcher roles still return BaseAgent (not Researcher)."""
        from code.shukketsu.agents.base import BaseAgent

        factory = AgentFactory()
        writer = factory.create(AgentRole.WRITER)
        assert type(writer) is BaseAgent

    def test_researcher_has_correct_prompt(self) -> None:
        """Researcher gets the full prompt from llm/prompts/researcher.py."""
        factory = AgentFactory()
        agent = factory.create(AgentRole.RESEARCHER)
        assert agent._system_prompt is FULL_PROMPT

    def test_researcher_has_correct_max_iter(self) -> None:
        from code.shukketsu import config

        factory = AgentFactory()
        agent = factory.create(AgentRole.RESEARCHER)
        assert agent.max_iterations == config.RESEARCHER_MAX_ITERATIONS
```

Add one test to `tests/unit/test_agent_factory.py` — append to `TestAgentFactoryCreate`:

```python
    def test_role_classes_registry_exists(self) -> None:
        from code.shukketsu.agents.factory import _ROLE_CLASSES
        from code.shukketsu.agents.tasks import AgentRole

        assert AgentRole.RESEARCHER in _ROLE_CLASSES
```

**Step 2: Run tests, confirm failure**

```bash
python3 -m pytest tests/unit/test_researcher.py::TestResearcherFactory::test_factory_returns_researcher_type -v
```

Expected: FAIL — factory still returns `BaseAgent`.

**Step 3: Update factory**

In `code/shukketsu/agents/factory.py`:

Replace the entire file with:

```python
"""Factory for creating configured agent instances.

Each agent role gets a different system prompt, iteration limit, and
potentially a different class. Tool registries are injected by the caller
(or default to empty).
"""

import logging

from code.shukketsu import config
from code.shukketsu.agents.base import BaseAgent
from code.shukketsu.agents.researcher import Researcher
from code.shukketsu.agents.tasks import AgentRole
from code.shukketsu.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_ROLE_PROMPTS: dict[AgentRole, str] = {
    AgentRole.RESEARCHER: config.RESEARCHER_SYSTEM_PROMPT,
    AgentRole.WRITER: config.WRITER_SYSTEM_PROMPT,
    AgentRole.EDITOR: config.EDITOR_SYSTEM_PROMPT,
    AgentRole.ORCHESTRATOR: config.ORCHESTRATOR_SYSTEM_PROMPT,
}

_ROLE_MAX_ITERATIONS: dict[AgentRole, int] = {
    AgentRole.RESEARCHER: config.RESEARCHER_MAX_ITERATIONS,
}

_ROLE_CLASSES: dict[AgentRole, type[BaseAgent]] = {
    AgentRole.RESEARCHER: Researcher,
}


class AgentFactory:
    """Creates configured agent instances for each specialist role.

    Each role gets a different system prompt and may have different
    iteration limits and agent classes. Tool registries are either
    provided by the caller or default to empty.
    """

    def create(
        self,
        role: AgentRole,
        *,
        tool_registry: ToolRegistry | None = None,
    ) -> BaseAgent:
        """Create an agent configured for the given role.

        Args:
            role: The specialist role to configure.
            tool_registry: Optional pre-configured tool registry.
                If None, a new empty registry is created.

        Returns:
            An agent configured with the role's prompt, limits, and class.
        """
        registry = tool_registry if tool_registry is not None else ToolRegistry()
        prompt = _ROLE_PROMPTS.get(role, config.SYSTEM_PROMPT)
        max_iter = _ROLE_MAX_ITERATIONS.get(role, config.AGENT_MAX_ITERATIONS)
        cls = _ROLE_CLASSES.get(role, BaseAgent)

        agent = cls(
            tool_registry=registry,
            role=role,
            max_iterations=max_iter,
            system_prompt=prompt,
        )

        logger.info("Created %s agent (%s, max_iter=%d)", role.value, cls.__name__, max_iter)
        return agent
```

**Step 4: Run factory tests**

```bash
python3 -m pytest tests/unit/test_researcher.py::TestResearcherFactory tests/unit/test_agent_factory.py -v
```

Expected: ALL PASS. Existing factory tests still pass because `isinstance(Researcher(...), BaseAgent)` is `True`.

**Step 5: Run full suite**

```bash
python3 -m pytest tests/unit/ -v --tb=short
```

Expected: 473 tests pass (468 + 4 researcher factory + 1 agent_factory registry).

**Step 6: Commit**

```bash
git add code/shukketsu/agents/factory.py tests/unit/test_researcher.py tests/unit/test_agent_factory.py
git commit -m "feat(agents): add _ROLE_CLASSES registry to factory"
```

---

## Task 7: Lint, format, type-check, and final verification

Run the full pre-commit check suite. Fix any issues.

**Files:**
- All modified/created files

**Step 1: Ruff check + fix**

```bash
ruff check code/shukketsu/agents/researcher.py code/shukketsu/agents/base.py code/shukketsu/agents/tasks.py code/shukketsu/agents/factory.py code/shukketsu/agents/__init__.py code/shukketsu/config.py code/shukketsu/llm/prompts/researcher.py tests/unit/test_researcher.py tests/unit/test_base_agent.py tests/unit/test_agent_factory.py --fix
```

**Step 2: Ruff format**

```bash
ruff format code/shukketsu/agents/researcher.py code/shukketsu/agents/base.py code/shukketsu/agents/tasks.py code/shukketsu/agents/factory.py code/shukketsu/agents/__init__.py code/shukketsu/config.py code/shukketsu/llm/prompts/researcher.py tests/unit/test_researcher.py tests/unit/test_base_agent.py tests/unit/test_agent_factory.py
```

**Step 3: Mypy**

```bash
python3 -m mypy code/shukketsu/agents/researcher.py code/shukketsu/agents/base.py code/shukketsu/agents/tasks.py code/shukketsu/agents/factory.py code/shukketsu/llm/prompts/researcher.py code/shukketsu/config.py
```

Fix any type errors. Common issues to watch for:
- `_RunOutcome` scratchpad type annotation
- `StructuredFindings.findings` field type vs `Finding` import
- `Researcher.execute()` return type annotation

**Step 4: Full test suite**

```bash
python3 -m pytest tests/unit/ -v
```

Expected: ~473 tests pass, 0 failures.

**Step 5: Verify test count**

```bash
python3 -m pytest tests/unit/ --co -q 2>/dev/null | tail -3
```

Record the exact count.

**Step 6: Commit lint/format fixes (if any)**

```bash
git add -u
git commit -m "style: apply ruff formatting to Phase 2 Step 4 files"
```

---

## Task 8: Update CLAUDE.md and commit final state

Update project documentation to reflect Step 4 completion.

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

Update the step 4 status line from `DESIGN COMPLETE` to `COMPLETE` with test count:

```
4. ~~Researcher Agent~~ — COMPLETE (N tests: Researcher subclass, structuring pass, prompts, factory registry)
```

Update the "Key files with real code" list to add: `agents/researcher.py`, `llm/prompts/researcher.py`.

Update the step 5 line to mark it as `NEXT`:

```
5. Writer Agent + Wiki Backend (article generation, KnowledgeManager, YAML frontmatter) — **NEXT**
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Phase 2 Step 4 completion (N tests)"
```

Replace `N` with the actual test count from Task 7 Step 5.

---

## Summary

| Task | What it does | New tests | Running total |
|------|-------------|-----------|---------------|
| 1 | `_RunOutcome.scratchpad` | +2 | 440 |
| 2 | `Finding` + `ResearchResult` models | +10 | 450 |
| 3 | Researcher system prompt | +4 | 454 |
| 4 | `Researcher` subclass | +7 | 461 |
| 5 | Behavior tests | +7 | 468 |
| 6 | Factory `_ROLE_CLASSES` registry | +5 | 473 |
| 7 | Lint + format + type-check | 0 | 473 |
| 8 | CLAUDE.md update | 0 | 473 |

**Total: 8 tasks, ~35 new tests, 438 → ~473.**
