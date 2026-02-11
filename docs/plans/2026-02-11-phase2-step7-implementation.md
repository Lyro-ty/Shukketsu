# Phase 2 Step 7: Orchestrator Agent — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the Orchestrator agent that decomposes complex queries into specialist sub-tasks, dispatches them in dependency order, and synthesizes results. Wire the chat handler to route by complexity.

**Architecture:** Three-phase deterministic flow (decompose → dispatch → synthesize) overriding `execute()`. Decomposition via single Llama 70B structured output call. Sequential execution in topological order. Conditional synthesis: LLM for research, template for articles. Chat handler routes TRIVIAL → direct, MODERATE → Researcher, COMPLEX → Orchestrator.

**Tech Stack:** Pydantic v2 models, Instructor structured output, Kahn's algorithm (topological sort), pytest with AsyncMock/patch

**Design doc:** `docs/plans/2026-02-11-phase2-step7-orchestrator-agent.md`

---

## Task 1: OrchestratorResult Model

**Files:**
- Modify: `code/shukketsu/agents/tasks.py` (append after `OrchestratorPlan`)
- Test: `tests/unit/test_tasks.py` (append new test class)

### Step 1: Write the failing tests

Append to `tests/unit/test_tasks.py`:

```python
class TestOrchestratorResult:
    def test_validates_with_all_fields(self) -> None:
        """OrchestratorResult round-trips with all fields populated."""
        plan = OrchestratorPlan(
            reasoning="Simple research",
            subtasks=[SubTask(agent_role=AgentRole.RESEARCHER, description="Find info")],
        )
        result = OrchestratorResult(
            task_id="t1",
            agent_role=AgentRole.ORCHESTRATOR,
            status=TaskStatus.SUCCESS,
            output="Here are the findings.",
            plan=plan,
            specialist_results=[
                AgentResult(
                    task_id="r1",
                    agent_role=AgentRole.RESEARCHER,
                    status=TaskStatus.SUCCESS,
                    output="Found it.",
                ),
            ],
            article_path="combat/gear/trinkets.md",
            needs_human_review=True,
            skipped_tasks=["skipped task"],
        )
        assert result.plan.reasoning == "Simple research"
        assert len(result.specialist_results) == 1
        assert result.article_path == "combat/gear/trinkets.md"
        assert result.needs_human_review is True
        assert result.skipped_tasks == ["skipped task"]

    def test_defaults(self) -> None:
        """Default values for optional fields."""
        plan = OrchestratorPlan(reasoning="test", subtasks=[])
        result = OrchestratorResult(
            task_id="t1",
            agent_role=AgentRole.ORCHESTRATOR,
            status=TaskStatus.SUCCESS,
            output="answer",
            plan=plan,
        )
        assert result.specialist_results == []
        assert result.article_path is None
        assert result.needs_human_review is False
        assert result.skipped_tasks == []

    def test_with_skipped_tasks(self) -> None:
        """skipped_tasks stores descriptions of tasks skipped due to failed deps."""
        plan = OrchestratorPlan(reasoning="test", subtasks=[])
        result = OrchestratorResult(
            task_id="t1",
            agent_role=AgentRole.ORCHESTRATOR,
            status=TaskStatus.PARTIAL,
            output="partial",
            plan=plan,
            skipped_tasks=["Write article", "Verify claims"],
        )
        assert len(result.skipped_tasks) == 2

    def test_inherits_agent_result(self) -> None:
        """OrchestratorResult has all AgentResult fields."""
        plan = OrchestratorPlan(reasoning="test", subtasks=[])
        result = OrchestratorResult(
            task_id="t1",
            agent_role=AgentRole.ORCHESTRATOR,
            status=TaskStatus.SUCCESS,
            output="answer",
            evidence=["src1", "src2"],
            plan=plan,
        )
        assert isinstance(result, AgentResult)
        assert result.evidence == ["src1", "src2"]
        assert result.metadata == {}
```

Also add `OrchestratorResult` to the import block at the top of `test_tasks.py`:

```python
from code.shukketsu.agents.tasks import (
    ...
    OrchestratorResult,
    ...
)
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/unit/test_tasks.py::TestOrchestratorResult -v
```

Expected: `ImportError: cannot import name 'OrchestratorResult'`

### Step 3: Implement the model

Append to `code/shukketsu/agents/tasks.py` after the `OrchestratorPlan` class:

```python
class OrchestratorResult(AgentResult):
    """Structured output from the Orchestrator agent."""

    plan: OrchestratorPlan
    specialist_results: list[AgentResult] = Field(default_factory=list)
    article_path: str | None = None
    needs_human_review: bool = False
    skipped_tasks: list[str] = Field(default_factory=list)
```

### Step 4: Run tests to verify they pass

```bash
python3 -m pytest tests/unit/test_tasks.py::TestOrchestratorResult -v
```

Expected: 4 passed

### Step 5: Run full suite to check for regressions

```bash
python3 -m pytest tests/unit/ -v --tb=short
```

Expected: 574 passed (570 existing + 4 new)

### Step 6: Commit

```bash
git add code/shukketsu/agents/tasks.py tests/unit/test_tasks.py
git commit -m "feat(tasks): add OrchestratorResult model"
```

---

## Task 2: Orchestrator Prompts + Config Update

**Files:**
- Create: `code/shukketsu/llm/prompts/orchestrator.py`
- Modify: `code/shukketsu/config.py` (replace stub with import)
- Test: `tests/unit/test_orchestrator_prompt.py`

### Step 1: Write the failing tests

Create `tests/unit/test_orchestrator_prompt.py`:

```python
"""Tests for Orchestrator prompt content and templates."""


class TestOrchestratorSystemPrompt:
    def test_mentions_all_specialist_roles(self) -> None:
        """Prompt describes all available specialist agents."""
        from code.shukketsu.llm.prompts.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT

        for role in ["RESEARCHER", "WRITER", "EDITOR"]:
            assert role in ORCHESTRATOR_SYSTEM_PROMPT, f"Missing role: {role}"

    def test_mentions_planning_rules(self) -> None:
        """Prompt includes guidance on decomposition and dependencies."""
        from code.shukketsu.llm.prompts.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT

        assert "depends_on" in ORCHESTRATOR_SYSTEM_PROMPT
        assert "task_params" in ORCHESTRATOR_SYSTEM_PROMPT
        assert "Maximum" in ORCHESTRATOR_SYSTEM_PROMPT or "maximum" in ORCHESTRATOR_SYSTEM_PROMPT


class TestDecompositionPrompt:
    def test_template_fills_query(self) -> None:
        """DECOMPOSITION_PROMPT includes the user query."""
        from code.shukketsu.llm.prompts.orchestrator import DECOMPOSITION_PROMPT

        result = DECOMPOSITION_PROMPT.format(query="What is the hit cap?")
        assert "What is the hit cap?" in result


class TestSynthesisPrompt:
    def test_synthesis_prompt_exists(self) -> None:
        """SYNTHESIS_PROMPT is a non-empty string."""
        from code.shukketsu.llm.prompts.orchestrator import SYNTHESIS_PROMPT

        assert isinstance(SYNTHESIS_PROMPT, str)
        assert len(SYNTHESIS_PROMPT) > 50
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/unit/test_orchestrator_prompt.py -v
```

Expected: `ModuleNotFoundError: No module named 'code.shukketsu.llm.prompts.orchestrator'`

### Step 3: Create the prompts module

Create `code/shukketsu/llm/prompts/orchestrator.py`:

```python
"""System and task prompts for the Orchestrator agent.

The Orchestrator uses these prompts for:
- Decomposing queries into specialist sub-task plans
- Synthesizing research results into coherent answers
"""

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the Orchestrator for a WoW TBC Rogue knowledge system. Your job is to \
decompose complex queries into sub-tasks for specialist agents.

## Available Specialists

- RESEARCHER: Searches the knowledge base (hybrid vector + keyword search), \
knowledge graph (entity relationships), and web. Returns structured findings \
with evidence and confidence scores. Use for any information gathering.

- WRITER: Produces wiki articles from research findings. Requires research \
results as input — always schedule a RESEARCHER task first. Use when the user \
asks for a guide, article, or comprehensive write-up.

- EDITOR: Fact-checks a draft article against the knowledge base. Requires a \
written article — always schedule after WRITER. Use when an article has been \
produced and needs verification.

## Planning Rules

1. DECOMPOSE into the smallest useful sub-tasks. Each sub-task should have a \
clear, specific description.
2. Use depends_on to express ordering. A WRITER task must depend on the \
RESEARCHER task(s) that feed it. An EDITOR task must depend on WRITER.
3. DO NOT over-decompose. A simple factual question needs one RESEARCHER, not \
three. Reserve multi-step plans for genuinely multi-part questions.
4. If you can answer the question directly without specialists, set \
can_answer_directly=True and provide direct_answer.
5. Maximum 6 sub-tasks. If the question needs more, simplify.

## task_params

For WRITER sub-tasks, you MUST include these fields in task_params:
- "spec": one of "combat", "assassination", "subtlety", "general"
- "category": a short topic category like "gear", "rotation", "talents", \
"consumables", "general"
- "article_type": one of "guide", "reference", "analysis"

For RESEARCHER and EDITOR sub-tasks, task_params can be empty.

## Examples

Query: "What are the best trinkets for combat rogues in Phase 1?"
→ 1 RESEARCHER task (straightforward retrieval)

Query: "Write a guide about Phase 1 BiS trinkets for combat rogues"
→ RESEARCHER (find trinkets + stat priorities) → WRITER (draft guide, \
task_params: {"spec": "combat", "category": "gear", "article_type": "guide"}) \
→ EDITOR (verify)

Query: "Compare combat swords vs mutilate for Gruul"
→ RESEARCHER (combat swords gear + rotation for Gruul)
→ RESEARCHER (mutilate gear + rotation for Gruul) [parallel, no dependency]
→ These feed into the synthesis step

Query: "Hello, what can you do?"
→ can_answer_directly=True
"""

DECOMPOSITION_PROMPT = "Decompose this query into a plan:\n\n{query}"

SYNTHESIS_PROMPT = """\
You are synthesizing research results into a coherent answer about \
WoW TBC Rogue content.

Combine the specialist findings into a clear, well-structured response. \
Include specific numbers, item names, and evidence where available. \
If findings conflict, note the disagreement. \
If there are gaps, acknowledge what couldn't be determined.
"""
```

### Step 4: Update config.py to import from prompts module

In `code/shukketsu/config.py`, replace the stub (lines 130-133):

```python
# BEFORE:
ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are the Orchestrator for a WoW TBC Rogue knowledge system. "
    "You decompose complex queries into sub-tasks for specialist agents."
)

# AFTER:
from code.shukketsu.llm.prompts.orchestrator import (  # noqa: E402
    ORCHESTRATOR_SYSTEM_PROMPT as ORCHESTRATOR_SYSTEM_PROMPT,
)
```

### Step 5: Run tests to verify they pass

```bash
python3 -m pytest tests/unit/test_orchestrator_prompt.py -v
```

Expected: 4 passed

### Step 6: Run full suite

```bash
python3 -m pytest tests/unit/ -v --tb=short
```

Expected: 578 passed (574 + 4 new)

### Step 7: Commit

```bash
git add code/shukketsu/llm/prompts/orchestrator.py code/shukketsu/config.py tests/unit/test_orchestrator_prompt.py
git commit -m "feat(prompts): add Orchestrator system/decomposition/synthesis prompts"
```

---

## Task 3: Plan Validation + Topological Sort

**Files:**
- Create: `code/shukketsu/agents/orchestrator.py` (skeleton with validation + topo sort)
- Test: `tests/unit/test_orchestrator_validate.py`
- Test: `tests/unit/test_orchestrator_topo.py`

### Step 1: Write the validation tests

Create `tests/unit/test_orchestrator_validate.py`:

```python
"""Tests for OrchestratorPlan validation."""

from code.shukketsu.agents.tasks import AgentRole, OrchestratorPlan, SubTask


def _plan(*subtasks: SubTask) -> OrchestratorPlan:
    """Build an OrchestratorPlan from subtasks."""
    return OrchestratorPlan(reasoning="test", subtasks=list(subtasks))


def _sub(role: AgentRole = AgentRole.RESEARCHER, desc: str = "test", depends_on: list[int] | None = None, **kw: object) -> SubTask:
    """Build a SubTask with defaults."""
    return SubTask(agent_role=role, description=desc, depends_on=depends_on or [], **kw)


class TestValidatePlan:
    def _validate(self, plan: OrchestratorPlan) -> list[str]:
        from code.shukketsu.agents.orchestrator import Orchestrator

        # Orchestrator._validate_plan is a pure method; create minimal instance
        orch = object.__new__(Orchestrator)
        return orch._validate_plan(plan)

    def test_valid_plan_passes(self) -> None:
        """Well-formed plan with correct dependencies returns no errors."""
        plan = _plan(
            _sub(AgentRole.RESEARCHER, "find trinkets"),
            _sub(AgentRole.WRITER, "write guide", depends_on=[0]),
            _sub(AgentRole.EDITOR, "verify", depends_on=[1]),
        )
        assert self._validate(plan) == []

    def test_cycle_detection(self) -> None:
        """Circular dependency is caught."""
        plan = _plan(
            _sub(AgentRole.RESEARCHER, "a", depends_on=[1]),
            _sub(AgentRole.RESEARCHER, "b", depends_on=[0]),
        )
        errors = self._validate(plan)
        assert any("cycle" in e.lower() for e in errors)

    def test_self_dependency_rejected(self) -> None:
        """A subtask depending on itself is caught."""
        plan = _plan(_sub(AgentRole.RESEARCHER, "loop", depends_on=[0]))
        errors = self._validate(plan)
        assert any("self-dependency" in e.lower() for e in errors)

    def test_out_of_range_dependency(self) -> None:
        """Dependency index beyond subtask count is caught."""
        plan = _plan(_sub(AgentRole.RESEARCHER, "test", depends_on=[99]))
        errors = self._validate(plan)
        assert any("out of range" in e.lower() for e in errors)

    def test_too_many_subtasks(self) -> None:
        """More than ORCHESTRATOR_MAX_SUBTASKS is rejected."""
        plan = _plan(*[_sub(AgentRole.RESEARCHER, f"task {i}") for i in range(7)])
        errors = self._validate(plan)
        assert any("too many" in e.lower() for e in errors)

    def test_orchestrator_role_rejected(self) -> None:
        """ORCHESTRATOR role in subtasks is rejected (no recursion)."""
        plan = _plan(_sub(AgentRole.ORCHESTRATOR, "recurse"))
        errors = self._validate(plan)
        assert any("orchestrator" in e.lower() and "not allowed" in e.lower() for e in errors)

    def test_writer_without_researcher_dependency(self) -> None:
        """WRITER not depending on a RESEARCHER is caught."""
        plan = _plan(
            _sub(AgentRole.RESEARCHER, "research"),
            _sub(AgentRole.WRITER, "write"),  # no depends_on
        )
        errors = self._validate(plan)
        assert any("writer" in e.lower() and "researcher" in e.lower() for e in errors)

    def test_editor_without_writer_dependency(self) -> None:
        """EDITOR not depending on a WRITER is caught."""
        plan = _plan(
            _sub(AgentRole.RESEARCHER, "research"),
            _sub(AgentRole.EDITOR, "edit", depends_on=[0]),
        )
        errors = self._validate(plan)
        assert any("editor" in e.lower() and "writer" in e.lower() for e in errors)
```

### Step 2: Write the topological sort tests

Create `tests/unit/test_orchestrator_topo.py`:

```python
"""Tests for topological sort of OrchestratorPlan subtasks."""

import pytest

from code.shukketsu.agents.tasks import AgentRole, SubTask


def _sub(depends_on: list[int] | None = None) -> SubTask:
    """Build a minimal SubTask for topo sort tests."""
    return SubTask(agent_role=AgentRole.RESEARCHER, description="test", depends_on=depends_on or [])


class TestTopologicalSort:
    def _sort(self, subtasks: list[SubTask]) -> list[int]:
        from code.shukketsu.agents.orchestrator import Orchestrator

        orch = object.__new__(Orchestrator)
        return orch._topological_sort(subtasks)

    def test_linear_chain(self) -> None:
        """A→B→C returns [0, 1, 2]."""
        subtasks = [
            _sub(),                   # 0: no deps
            _sub(depends_on=[0]),     # 1: depends on 0
            _sub(depends_on=[1]),     # 2: depends on 1
        ]
        assert self._sort(subtasks) == [0, 1, 2]

    def test_independent_tasks(self) -> None:
        """Tasks with no dependencies all appear (stable order)."""
        subtasks = [_sub(), _sub(), _sub()]
        result = self._sort(subtasks)
        assert set(result) == {0, 1, 2}
        assert len(result) == 3

    def test_diamond_dependency(self) -> None:
        """A→C, B→C, C→D returns valid topological order."""
        subtasks = [
            _sub(),                       # 0: A
            _sub(),                       # 1: B
            _sub(depends_on=[0, 1]),      # 2: C
            _sub(depends_on=[2]),         # 3: D
        ]
        result = self._sort(subtasks)
        assert result.index(0) < result.index(2)
        assert result.index(1) < result.index(2)
        assert result.index(2) < result.index(3)

    def test_empty_plan(self) -> None:
        """Empty subtask list returns empty order."""
        assert self._sort([]) == []

    def test_single_task(self) -> None:
        """Single subtask returns [0]."""
        assert self._sort([_sub()]) == [0]

    def test_cycle_raises_value_error(self) -> None:
        """Cycle in dependency graph raises ValueError."""
        subtasks = [
            _sub(depends_on=[1]),
            _sub(depends_on=[0]),
        ]
        with pytest.raises(ValueError, match="cycle"):
            self._sort(subtasks)
```

### Step 3: Run tests to verify they fail

```bash
python3 -m pytest tests/unit/test_orchestrator_validate.py tests/unit/test_orchestrator_topo.py -v
```

Expected: `ModuleNotFoundError: No module named 'code.shukketsu.agents.orchestrator'`

### Step 4: Create the Orchestrator skeleton with validation + topo sort

Create `code/shukketsu/agents/orchestrator.py`:

```python
"""Orchestrator specialist agent.

Overrides execute() with a three-phase flow: decompose → dispatch → synthesize.
Does not use the ReAct loop. Creates an execution plan via LLM, dispatches
typed tasks to specialist agents, and synthesizes results.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

from code.shukketsu.agents.base import BaseAgent, StatusCallback
from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    OrchestratorPlan,
    OrchestratorResult,
    SubTask,
    TaskStatus,
)
from code.shukketsu.knowledge.manager import KnowledgeManager
from code.shukketsu.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from code.shukketsu.agents.factory import AgentFactory

logger = logging.getLogger(__name__)

# Roles that require knowledge_manager kwarg
_KM_ROLES = frozenset({AgentRole.WRITER, AgentRole.EDITOR})


class Orchestrator(BaseAgent):
    """Multi-agent coordinator that decomposes complex queries into specialist tasks."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        role: AgentRole | None = None,
        max_iterations: int | None = None,
        system_prompt: str | None = None,
        factory: AgentFactory,
        knowledge_manager: KnowledgeManager | None = None,
    ) -> None:
        from code.shukketsu import config

        super().__init__(
            tool_registry=tool_registry,
            role=role,
            max_iterations=max_iterations or config.AGENT_MAX_ITERATIONS,
            system_prompt=system_prompt or config.SYSTEM_PROMPT,
        )
        self._factory = factory
        self._km = knowledge_manager

    async def execute(
        self, task: AgentTask, *, on_status: StatusCallback | None = None,
    ) -> OrchestratorResult:
        """Execute a complex task by decomposing, dispatching, and synthesizing.

        Placeholder — dispatch and synthesis added in later tasks.
        """
        raise NotImplementedError("execute() implemented in Task 5")

    def _validate_plan(self, plan: OrchestratorPlan) -> list[str]:
        """Validate an OrchestratorPlan for structural issues.

        Returns a list of error strings. Empty list means valid.
        """
        from code.shukketsu import config

        errors: list[str] = []

        if len(plan.subtasks) > config.ORCHESTRATOR_MAX_SUBTASKS:
            errors.append(
                f"Too many subtasks: {len(plan.subtasks)} "
                f"(max {config.ORCHESTRATOR_MAX_SUBTASKS})"
            )

        for i, st in enumerate(plan.subtasks):
            if st.agent_role == AgentRole.ORCHESTRATOR:
                errors.append(
                    f"Subtask {i}: ORCHESTRATOR role not allowed (no recursion)"
                )

            for dep in st.depends_on:
                if dep < 0 or dep >= len(plan.subtasks):
                    errors.append(
                        f"Subtask {i}: dependency index {dep} out of range"
                    )

            if i in st.depends_on:
                errors.append(f"Subtask {i}: self-dependency")

            valid_deps = {
                d for d in st.depends_on if 0 <= d < len(plan.subtasks)
            }
            dep_roles = {plan.subtasks[d].agent_role for d in valid_deps}

            if st.agent_role == AgentRole.WRITER:
                if AgentRole.RESEARCHER not in dep_roles:
                    errors.append(
                        f"Subtask {i}: WRITER must depend on a RESEARCHER"
                    )

            if st.agent_role == AgentRole.EDITOR:
                if AgentRole.WRITER not in dep_roles:
                    errors.append(
                        f"Subtask {i}: EDITOR must depend on a WRITER"
                    )

        try:
            self._topological_sort(plan.subtasks)
        except ValueError:
            errors.append("Dependency cycle detected")

        return errors

    def _topological_sort(self, subtasks: list[SubTask]) -> list[int]:
        """Return subtask indices in topological order.

        Uses Kahn's algorithm. Raises ValueError if a cycle exists.
        """
        n = len(subtasks)
        if n == 0:
            return []

        adj: dict[int, list[int]] = defaultdict(list)
        in_degree = [0] * n

        for i, st in enumerate(subtasks):
            for dep in st.depends_on:
                if 0 <= dep < n:
                    adj[dep].append(i)
                    in_degree[i] += 1

        queue: deque[int] = deque(i for i in range(n) if in_degree[i] == 0)
        order: list[int] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != n:
            raise ValueError("Dependency cycle detected")

        return order
```

### Step 5: Run tests to verify they pass

```bash
python3 -m pytest tests/unit/test_orchestrator_validate.py tests/unit/test_orchestrator_topo.py -v
```

Expected: 14 passed (8 validate + 6 topo)

### Step 6: Run full suite

```bash
python3 -m pytest tests/unit/ -v --tb=short
```

Expected: 592 passed (578 + 14)

### Step 7: Lint check

```bash
ruff check code/shukketsu/agents/orchestrator.py tests/unit/test_orchestrator_validate.py tests/unit/test_orchestrator_topo.py
```

### Step 8: Commit

```bash
git add code/shukketsu/agents/orchestrator.py tests/unit/test_orchestrator_validate.py tests/unit/test_orchestrator_topo.py
git commit -m "feat(orchestrator): add plan validation and topological sort"
```

---

## Task 4: Task Building + Research Merging

**Files:**
- Modify: `code/shukketsu/agents/orchestrator.py` (add `_build_task`, `_merge_research`)
- Test: `tests/unit/test_orchestrator_build_task.py`

### Step 1: Write the failing tests

Create `tests/unit/test_orchestrator_build_task.py`:

```python
"""Tests for Orchestrator._build_task and _merge_research."""

import pytest

from code.shukketsu.agents.tasks import (
    AgentRole,
    AgentTask,
    ArticleType,
    EditTask,
    Finding,
    ResearchResult,
    ResearchTask,
    SubTask,
    TaskStatus,
    WriteResult,
    WriteTask,
)


def _sub(
    role: AgentRole = AgentRole.RESEARCHER,
    desc: str = "test",
    depends_on: list[int] | None = None,
    task_params: dict | None = None,
) -> SubTask:
    """Build a SubTask with defaults."""
    return SubTask(
        agent_role=role,
        description=desc,
        depends_on=depends_on or [],
        task_params=task_params or {},
    )


def _research_result(**overrides: object) -> ResearchResult:
    """Build a ResearchResult with defaults."""
    defaults: dict[str, object] = dict(
        task_id="r1",
        agent_role=AgentRole.RESEARCHER,
        status=TaskStatus.SUCCESS,
        output="Research done",
        findings=[Finding(claim="Hit cap is 142", confidence=0.9, evidence=["src1"])],
        sources_used=["src1"],
        strategies_used=["rag_search"],
        gaps=[],
        sufficient=True,
    )
    defaults.update(overrides)
    return ResearchResult(**defaults)


def _write_result(**overrides: object) -> WriteResult:
    """Build a WriteResult with defaults."""
    defaults: dict[str, object] = dict(
        task_id="w1",
        agent_role=AgentRole.WRITER,
        status=TaskStatus.SUCCESS,
        output="Article written",
        article_path="combat/gear/trinkets.md",
        title="Trinkets Guide",
        claims=["Hit cap is 142", "DST is BiS"],
    )
    defaults.update(overrides)
    return WriteResult(**defaults)


def _build(subtask: SubTask, results: list, trace_id: str = "trace-1") -> AgentTask:
    """Call Orchestrator._build_task on a minimal instance."""
    from code.shukketsu.agents.orchestrator import Orchestrator

    orch = object.__new__(Orchestrator)
    return orch._build_task(subtask, results, trace_id)


class TestBuildResearchTask:
    def test_researcher_creates_research_task(self) -> None:
        """RESEARCHER subtask builds a ResearchTask with description as query."""
        task = _build(_sub(AgentRole.RESEARCHER, "Find trinkets"), [])
        assert isinstance(task, ResearchTask)
        assert task.query == "Find trinkets"

    def test_propagates_trace_id(self) -> None:
        """trace_id flows from parent task to built task."""
        task = _build(_sub(AgentRole.RESEARCHER, "test"), [], trace_id="my-trace")
        assert task.trace_id == "my-trace"


class TestBuildWriteTask:
    def test_pulls_research_from_dependency(self) -> None:
        """WRITER subtask extracts ResearchResult from completed dependency."""
        results = [_research_result(), None]
        subtask = _sub(AgentRole.WRITER, "write guide", depends_on=[0],
                       task_params={"spec": "combat", "category": "gear"})
        task = _build(subtask, results)
        assert isinstance(task, WriteTask)
        assert task.research.task_id == "r1"

    def test_reads_task_params(self) -> None:
        """spec, category, article_type read from task_params."""
        results = [_research_result()]
        subtask = _sub(
            AgentRole.WRITER, "write",
            depends_on=[0],
            task_params={"spec": "assassination", "category": "rotation", "article_type": "analysis"},
        )
        task = _build(subtask, results)
        assert isinstance(task, WriteTask)
        assert task.spec == "assassination"
        assert task.category == "rotation"
        assert task.article_type == ArticleType.ANALYSIS

    def test_defaults_on_missing_params(self) -> None:
        """Missing task_params fall back to general/general/GUIDE."""
        results = [_research_result()]
        subtask = _sub(AgentRole.WRITER, "write", depends_on=[0])
        task = _build(subtask, results)
        assert isinstance(task, WriteTask)
        assert task.spec == "general"
        assert task.category == "general"
        assert task.article_type == ArticleType.GUIDE

    def test_merges_multiple_research(self) -> None:
        """Two Researcher deps are merged into one ResearchResult."""
        r1 = _research_result(
            task_id="r1",
            findings=[Finding(claim="Claim A", confidence=0.9, evidence=["s1"])],
            sources_used=["s1"],
            strategies_used=["rag_search"],
            gaps=["gap1"],
        )
        r2 = _research_result(
            task_id="r2",
            output="Second research",
            findings=[Finding(claim="Claim B", confidence=0.8, evidence=["s2"])],
            sources_used=["s2"],
            strategies_used=["graph_search"],
            gaps=["gap2"],
        )
        results = [r1, r2]
        subtask = _sub(AgentRole.WRITER, "write", depends_on=[0, 1])
        task = _build(subtask, results)
        assert isinstance(task, WriteTask)
        assert len(task.research.findings) == 2
        assert set(task.research.sources_used) == {"s1", "s2"}
        assert set(task.research.strategies_used) == {"rag_search", "graph_search"}
        assert set(task.research.gaps) == {"gap1", "gap2"}

    def test_no_research_dependency_raises(self) -> None:
        """WRITER with no ResearchResult dependency raises ValueError."""
        subtask = _sub(AgentRole.WRITER, "write", depends_on=[0])
        results: list = [None]
        with pytest.raises(ValueError, match="no ResearchResult"):
            _build(subtask, results)


class TestBuildEditTask:
    def test_pulls_write_result(self) -> None:
        """EDITOR subtask extracts article_path and claims from WriteResult."""
        results = [None, _write_result()]
        subtask = _sub(AgentRole.EDITOR, "verify", depends_on=[1])
        task = _build(subtask, results)
        assert isinstance(task, EditTask)
        assert task.article_path == "combat/gear/trinkets.md"
        assert task.claims == ["Hit cap is 142", "DST is BiS"]

    def test_no_write_dependency_raises(self) -> None:
        """EDITOR with no WriteResult dependency raises ValueError."""
        subtask = _sub(AgentRole.EDITOR, "verify", depends_on=[0])
        results: list = [_research_result()]
        with pytest.raises(ValueError, match="no WriteResult"):
            _build(subtask, results)
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/unit/test_orchestrator_build_task.py -v
```

Expected: `AttributeError: 'Orchestrator' object has no attribute '_build_task'`

### Step 3: Add _build_task and _merge_research to orchestrator.py

Add these imports to the top of `code/shukketsu/agents/orchestrator.py`:

```python
from code.shukketsu.agents.tasks import (
    ...  # existing imports
    ArticleType,
    EditTask,
    Finding,
    ResearchResult,
    ResearchTask,
    WriteResult,
    WriteTask,
)
```

Add these methods to the `Orchestrator` class, after `_topological_sort`:

```python
    def _build_task(
        self,
        subtask: SubTask,
        results: list[AgentResult | None],
        trace_id: str,
    ) -> AgentTask:
        """Build a typed task from a SubTask and completed dependency results."""
        if subtask.agent_role == AgentRole.RESEARCHER:
            return ResearchTask(query=subtask.description, trace_id=trace_id)

        if subtask.agent_role == AgentRole.WRITER:
            research_results = [
                results[d]
                for d in subtask.depends_on
                if results[d] is not None
                and isinstance(results[d], ResearchResult)
            ]
            if not research_results:
                raise ValueError(
                    "WRITER subtask has no ResearchResult dependencies"
                )

            research = (
                self._merge_research(research_results)
                if len(research_results) > 1
                else research_results[0]
            )

            spec = subtask.task_params.get("spec", "general")
            category = subtask.task_params.get("category", "general")
            article_type_str = subtask.task_params.get("article_type", "guide")
            try:
                article_type = ArticleType(article_type_str)
            except ValueError:
                article_type = ArticleType.GUIDE

            return WriteTask(
                query=subtask.description,
                trace_id=trace_id,
                research=research,
                article_type=article_type,
                spec=spec,
                category=category,
            )

        if subtask.agent_role == AgentRole.EDITOR:
            write_result = next(
                (
                    results[d]
                    for d in subtask.depends_on
                    if results[d] is not None
                    and isinstance(results[d], WriteResult)
                ),
                None,
            )
            if write_result is None:
                raise ValueError(
                    "EDITOR subtask has no WriteResult dependency"
                )

            return EditTask(
                query=subtask.description,
                trace_id=trace_id,
                article_path=write_result.article_path,
                claims=write_result.claims,
            )

        raise ValueError(f"Unsupported agent role: {subtask.agent_role}")

    def _merge_research(
        self, results: list[ResearchResult],
    ) -> ResearchResult:
        """Merge multiple ResearchResults into one for the Writer."""
        seen_claims: set[str] = set()
        merged_findings: list[Finding] = []
        for r in results:
            for f in r.findings:
                if f.claim not in seen_claims:
                    seen_claims.add(f.claim)
                    merged_findings.append(f)

        return ResearchResult(
            task_id=results[0].task_id,
            agent_role=AgentRole.RESEARCHER,
            status=(
                TaskStatus.SUCCESS
                if all(r.status == TaskStatus.SUCCESS for r in results)
                else TaskStatus.PARTIAL
            ),
            output="\n\n---\n\n".join(r.output for r in results),
            findings=merged_findings,
            sources_used=list(
                dict.fromkeys(s for r in results for s in r.sources_used)
            ),
            strategies_used=list(
                dict.fromkeys(s for r in results for s in r.strategies_used)
            ),
            gaps=list(dict.fromkeys(g for r in results for g in r.gaps)),
            sufficient=all(r.sufficient for r in results),
        )
```

### Step 4: Run tests to verify they pass

```bash
python3 -m pytest tests/unit/test_orchestrator_build_task.py -v
```

Expected: 9 passed

### Step 5: Run full suite

```bash
python3 -m pytest tests/unit/ -v --tb=short
```

Expected: 601 passed (592 + 9)

### Step 6: Commit

```bash
git add code/shukketsu/agents/orchestrator.py tests/unit/test_orchestrator_build_task.py
git commit -m "feat(orchestrator): add task building and research merging"
```

---

## Task 5: Full Orchestrator Execute + Synthesis

**Files:**
- Modify: `code/shukketsu/agents/orchestrator.py` (implement `execute`, `_decompose`, `_dispatch`, `_synthesize`)
- Test: `tests/unit/test_orchestrator_execute.py`

### Step 1: Write the failing tests

Create `tests/unit/test_orchestrator_execute.py`:

```python
"""Tests for Orchestrator.execute — decompose, dispatch, synthesize."""

from unittest.mock import AsyncMock, MagicMock, patch

from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    EditResult,
    Finding,
    OrchestratorPlan,
    OrchestratorResult,
    ResearchResult,
    SubTask,
    TaskStatus,
    WriteResult,
)
from code.shukketsu.tools.registry import ToolRegistry


def _plan(
    subtasks: list[SubTask] | None = None,
    can_answer_directly: bool = False,
    direct_answer: str | None = None,
) -> OrchestratorPlan:
    return OrchestratorPlan(
        reasoning="test plan",
        subtasks=subtasks or [],
        can_answer_directly=can_answer_directly,
        direct_answer=direct_answer,
    )


def _research_result(**overrides: object) -> ResearchResult:
    defaults: dict[str, object] = dict(
        task_id="r1",
        agent_role=AgentRole.RESEARCHER,
        status=TaskStatus.SUCCESS,
        output="Research findings here.",
        findings=[Finding(claim="Hit cap is 142", confidence=0.9, evidence=["src1"])],
        sources_used=["src1"],
        strategies_used=["rag_search"],
        gaps=[],
        sufficient=True,
    )
    defaults.update(overrides)
    return ResearchResult(**defaults)


def _write_result(**overrides: object) -> WriteResult:
    defaults: dict[str, object] = dict(
        task_id="w1",
        agent_role=AgentRole.WRITER,
        status=TaskStatus.SUCCESS,
        output="Article written",
        article_path="combat/gear/trinkets.md",
        title="Trinkets Guide",
        claims=["Hit cap is 142"],
    )
    defaults.update(overrides)
    return WriteResult(**defaults)


def _edit_result(**overrides: object) -> EditResult:
    defaults: dict[str, object] = dict(
        task_id="e1",
        agent_role=AgentRole.EDITOR,
        status=TaskStatus.SUCCESS,
        output="Verified 1 claim. Confidence: 0.85",
        article_path="combat/gear/trinkets.md",
        overall_confidence=0.85,
        approved_for_review=True,
    )
    defaults.update(overrides)
    return EditResult(**defaults)


def _mock_factory(role_results: dict[AgentRole, AgentResult]) -> MagicMock:
    """Create a mock AgentFactory that returns mock agents with preset results."""
    factory = MagicMock()

    def _create(role, **kwargs):
        agent = MagicMock()
        result = role_results.get(role)
        agent.execute = AsyncMock(return_value=result)
        return agent

    factory.create.side_effect = _create
    return factory


def _orchestrator(
    factory: MagicMock | None = None,
    km: MagicMock | None = None,
) -> "Orchestrator":
    from code.shukketsu.agents.orchestrator import Orchestrator

    return Orchestrator(
        tool_registry=ToolRegistry(),
        role=AgentRole.ORCHESTRATOR,
        factory=factory or _mock_factory({}),
        knowledge_manager=km,
    )


class TestDirectAnswer:
    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_direct_answer_skips_dispatch(self, mock_llm: AsyncMock) -> None:
        """can_answer_directly=True returns immediately without specialists."""
        mock_llm.return_value = _plan(
            can_answer_directly=True,
            direct_answer="Shukketsu is a WoW TBC Rogue research system.",
        )
        orch = _orchestrator()
        result = await orch.execute(AgentTask(query="What can you do?"))

        assert isinstance(result, OrchestratorResult)
        assert result.status == TaskStatus.SUCCESS
        assert result.output == "Shukketsu is a WoW TBC Rogue research system."
        assert result.specialist_results == []


class TestSingleResearcher:
    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_single_researcher_plan(self, mock_llm: AsyncMock) -> None:
        """Single RESEARCHER subtask dispatches and returns synthesized output."""
        plan = _plan(subtasks=[
            SubTask(agent_role=AgentRole.RESEARCHER, description="Find trinkets"),
        ])

        call_count = 0

        async def _mock_structured(response_model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if response_model is OrchestratorPlan:
                return plan
            # Synthesis call
            return response_model(response="The best trinkets are DST and Bloodlust Brooch.")

        mock_llm.side_effect = _mock_structured

        factory = _mock_factory({AgentRole.RESEARCHER: _research_result()})
        orch = _orchestrator(factory=factory)
        result = await orch.execute(AgentTask(query="Best trinkets?"))

        assert result.status == TaskStatus.SUCCESS
        assert len(result.specialist_results) == 1
        factory.create.assert_called_once()


class TestRoleConditionalKwargs:
    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_writer_receives_km_researcher_does_not(self, mock_llm: AsyncMock) -> None:
        """knowledge_manager passed to Writer/Editor but NOT to Researcher."""
        plan = _plan(subtasks=[
            SubTask(agent_role=AgentRole.RESEARCHER, description="research"),
            SubTask(agent_role=AgentRole.WRITER, description="write",
                    depends_on=[0], task_params={"spec": "combat", "category": "gear"}),
        ])
        mock_llm.return_value = plan

        km = MagicMock()
        factory = MagicMock()
        agents_created: list[tuple[AgentRole, dict]] = []

        def _create(role, **kwargs):
            agents_created.append((role, kwargs))
            agent = MagicMock()
            if role == AgentRole.RESEARCHER:
                agent.execute = AsyncMock(return_value=_research_result())
            else:
                agent.execute = AsyncMock(return_value=_write_result())
            return agent

        factory.create.side_effect = _create

        orch = _orchestrator(factory=factory, km=km)
        await orch.execute(AgentTask(query="Write a guide"))

        # Researcher should NOT get knowledge_manager
        researcher_kwargs = agents_created[0][1]
        assert "knowledge_manager" not in researcher_kwargs

        # Writer SHOULD get knowledge_manager
        writer_kwargs = agents_created[1][1]
        assert writer_kwargs["knowledge_manager"] is km


class TestDependencyChain:
    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_research_write_edit_chain(self, mock_llm: AsyncMock) -> None:
        """Full pipeline: research → write → edit chains correctly."""
        plan = _plan(subtasks=[
            SubTask(agent_role=AgentRole.RESEARCHER, description="research"),
            SubTask(agent_role=AgentRole.WRITER, description="write",
                    depends_on=[0], task_params={"spec": "combat", "category": "gear"}),
            SubTask(agent_role=AgentRole.EDITOR, description="edit", depends_on=[1]),
        ])
        mock_llm.return_value = plan

        km = MagicMock()
        factory = MagicMock()

        def _create(role, **kwargs):
            agent = MagicMock()
            if role == AgentRole.RESEARCHER:
                agent.execute = AsyncMock(return_value=_research_result())
            elif role == AgentRole.WRITER:
                agent.execute = AsyncMock(return_value=_write_result())
            else:
                agent.execute = AsyncMock(return_value=_edit_result())
            return agent

        factory.create.side_effect = _create

        orch = _orchestrator(factory=factory, km=km)
        result = await orch.execute(AgentTask(query="Write a guide"))

        assert result.status == TaskStatus.SUCCESS
        assert result.article_path == "combat/gear/trinkets.md"
        assert result.needs_human_review is True
        assert len(result.specialist_results) == 3

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_failed_dependency_skips_dependents(self, mock_llm: AsyncMock) -> None:
        """Failed Researcher → Writer and Editor skipped."""
        plan = _plan(subtasks=[
            SubTask(agent_role=AgentRole.RESEARCHER, description="research"),
            SubTask(agent_role=AgentRole.WRITER, description="write",
                    depends_on=[0], task_params={"spec": "combat", "category": "gear"}),
            SubTask(agent_role=AgentRole.EDITOR, description="edit", depends_on=[1]),
        ])
        mock_llm.return_value = plan

        factory = MagicMock()
        failed_research = _research_result(status=TaskStatus.FAILED, output="Failed")

        def _create(role, **kwargs):
            agent = MagicMock()
            agent.execute = AsyncMock(return_value=failed_research)
            return agent

        factory.create.side_effect = _create

        orch = _orchestrator(factory=factory)
        result = await orch.execute(AgentTask(query="Write a guide"))

        assert result.status == TaskStatus.FAILED
        assert len(result.skipped_tasks) == 2  # Writer + Editor skipped

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_partial_dependency_proceeds(self, mock_llm: AsyncMock) -> None:
        """Partial Researcher → Writer still runs."""
        plan = _plan(subtasks=[
            SubTask(agent_role=AgentRole.RESEARCHER, description="research"),
            SubTask(agent_role=AgentRole.WRITER, description="write",
                    depends_on=[0], task_params={"spec": "combat", "category": "gear"}),
        ])
        mock_llm.return_value = plan

        km = MagicMock()
        factory = MagicMock()
        partial_research = _research_result(status=TaskStatus.PARTIAL)

        def _create(role, **kwargs):
            agent = MagicMock()
            if role == AgentRole.RESEARCHER:
                agent.execute = AsyncMock(return_value=partial_research)
            else:
                agent.execute = AsyncMock(return_value=_write_result())
            return agent

        factory.create.side_effect = _create

        orch = _orchestrator(factory=factory, km=km)
        result = await orch.execute(AgentTask(query="test"))

        assert len(result.specialist_results) == 2  # Both ran
        assert result.skipped_tasks == []


class TestKnowledgeManagerGuard:
    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_writer_skipped_without_km(self, mock_llm: AsyncMock) -> None:
        """Writer subtask skipped when Orchestrator has no knowledge_manager."""
        plan = _plan(subtasks=[
            SubTask(agent_role=AgentRole.RESEARCHER, description="research"),
            SubTask(agent_role=AgentRole.WRITER, description="write",
                    depends_on=[0], task_params={"spec": "combat", "category": "gear"}),
        ])
        mock_llm.return_value = plan

        factory = _mock_factory({AgentRole.RESEARCHER: _research_result()})
        orch = _orchestrator(factory=factory, km=None)  # No KM
        result = await orch.execute(AgentTask(query="test"))

        assert len(result.skipped_tasks) == 1
        assert "write" in result.skipped_tasks[0].lower()


class TestErrorHandling:
    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_all_failed_returns_failed(self, mock_llm: AsyncMock) -> None:
        """All specialists failing returns FAILED status."""
        plan = _plan(subtasks=[
            SubTask(agent_role=AgentRole.RESEARCHER, description="a"),
            SubTask(agent_role=AgentRole.RESEARCHER, description="b"),
        ])
        mock_llm.return_value = plan

        factory = MagicMock()
        failed = AgentResult(
            task_id="f", agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.FAILED, output="Failed",
        )

        def _create(role, **kwargs):
            agent = MagicMock()
            agent.execute = AsyncMock(return_value=failed)
            return agent

        factory.create.side_effect = _create

        orch = _orchestrator(factory=factory)
        result = await orch.execute(AgentTask(query="test"))

        assert result.status == TaskStatus.FAILED

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_decomposition_failure_returns_failed(self, mock_llm: AsyncMock) -> None:
        """LLM failure during decomposition returns FAILED."""
        from code.shukketsu.resilience.errors import StructuredOutputError

        mock_llm.side_effect = StructuredOutputError("LLM failed")
        orch = _orchestrator()
        result = await orch.execute(AgentTask(query="test"))

        assert result.status == TaskStatus.FAILED
        assert "decompose" in result.output.lower() or "failed" in result.output.lower()

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_validation_retry_on_structural_error(self, mock_llm: AsyncMock) -> None:
        """First plan invalid, second valid → succeeds."""
        bad_plan = _plan(subtasks=[
            SubTask(agent_role=AgentRole.ORCHESTRATOR, description="recurse"),
        ])
        good_plan = _plan(subtasks=[
            SubTask(agent_role=AgentRole.RESEARCHER, description="research"),
        ])
        mock_llm.side_effect = [bad_plan, good_plan]

        factory = _mock_factory({AgentRole.RESEARCHER: _research_result()})
        orch = _orchestrator(factory=factory)
        result = await orch.execute(AgentTask(query="test"))

        assert result.status == TaskStatus.SUCCESS
        assert mock_llm.call_count >= 2  # decompose + retry (synthesis may add more)

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_synthesis_llm_failure_falls_back(self, mock_llm: AsyncMock) -> None:
        """Synthesis LLM fail → falls back to concatenated output."""
        plan = _plan(subtasks=[
            SubTask(agent_role=AgentRole.RESEARCHER, description="research"),
        ])

        call_count = 0

        async def _side_effect(response_model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if response_model is OrchestratorPlan:
                return plan
            # Synthesis call fails
            raise Exception("Synthesis failed")

        mock_llm.side_effect = _side_effect

        factory = _mock_factory({AgentRole.RESEARCHER: _research_result()})
        orch = _orchestrator(factory=factory)
        result = await orch.execute(AgentTask(query="test"))

        assert result.status == TaskStatus.SUCCESS
        # Falls back to concatenated research output
        assert "Research findings" in result.output or "Research done" in result.output
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/unit/test_orchestrator_execute.py -v
```

Expected: `NotImplementedError: execute() implemented in Task 5`

### Step 3: Implement the full execute flow

Replace the `execute` method stub and add `_decompose`, `_dispatch`, `_synthesize` methods in `code/shukketsu/agents/orchestrator.py`.

Add to imports at top:

```python
from pydantic import BaseModel

from code.shukketsu.llm.prompts.orchestrator import (
    DECOMPOSITION_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    SYNTHESIS_PROMPT,
)
from code.shukketsu.llm.structured import get_structured_output
from code.shukketsu.resilience.errors import LLMUnavailableError, StructuredOutputError
```

Replace the `execute` stub with:

```python
    async def execute(
        self, task: AgentTask, *, on_status: StatusCallback | None = None,
    ) -> OrchestratorResult:
        """Execute a complex task by decomposing, dispatching, and synthesizing."""
        if self.role is None:
            raise ValueError(
                "Orchestrator requires a role. Use AgentFactory or set role in constructor."
            )

        # Phase 1: Decompose
        if on_status:
            await on_status("planning...")

        try:
            plan = await self._decompose(task.query)
        except (LLMUnavailableError, StructuredOutputError, ValueError, Exception) as exc:
            logger.warning("Decomposition failed: %s", exc)
            return OrchestratorResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=TaskStatus.FAILED,
                output=f"Failed to decompose query: {exc}",
                plan=OrchestratorPlan(
                    reasoning="Decomposition failed", subtasks=[],
                ),
            )

        # Handle direct answer
        if plan.can_answer_directly and plan.direct_answer:
            return OrchestratorResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=TaskStatus.SUCCESS,
                output=plan.direct_answer,
                plan=plan,
            )

        # Phase 2: Dispatch
        if on_status:
            await on_status(
                f"executing {len(plan.subtasks)} sub-tasks..."
            )

        results, skipped = await self._dispatch(plan, task, on_status)

        # Phase 3: Synthesize
        if on_status:
            await on_status("synthesizing results...")

        return await self._synthesize(task, plan, results, skipped)
```

Add the `_decompose` method:

```python
    async def _decompose(self, query: str) -> OrchestratorPlan:
        """Phase 1: Decompose query into an execution plan via Llama 70B."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": DECOMPOSITION_PROMPT.format(query=query)},
        ]

        plan: OrchestratorPlan = await get_structured_output(
            response_model=OrchestratorPlan, messages=messages,
        )

        errors = self._validate_plan(plan)
        if errors:
            logger.info(
                "Plan validation failed (%d errors), retrying", len(errors),
            )
            feedback = (
                "Your plan had errors:\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\n\nPlease fix and try again."
            )
            messages.append({"role": "assistant", "content": plan.model_dump_json()})
            messages.append({"role": "user", "content": feedback})

            plan = await get_structured_output(
                response_model=OrchestratorPlan, messages=messages,
            )

            errors = self._validate_plan(plan)
            if errors:
                raise ValueError(
                    f"Plan validation failed after retry: {'; '.join(errors)}"
                )

        return plan
```

Add the `_dispatch` method:

```python
    async def _dispatch(
        self,
        plan: OrchestratorPlan,
        task: AgentTask,
        on_status: StatusCallback | None = None,
    ) -> tuple[list[AgentResult | None], list[str]]:
        """Phase 2: Execute subtasks in topological order."""
        results: list[AgentResult | None] = [None] * len(plan.subtasks)
        skipped: list[str] = []

        order = self._topological_sort(plan.subtasks)

        for idx in order:
            subtask = plan.subtasks[idx]

            # Check failed dependencies
            failed_deps = [
                d for d in subtask.depends_on
                if results[d] is not None
                and results[d].status == TaskStatus.FAILED
            ]
            if failed_deps:
                skipped.append(subtask.description)
                logger.info(
                    "Skipping subtask %d (%s): failed dependencies",
                    idx, subtask.description,
                )
                continue

            # Check if Writer/Editor needs knowledge_manager
            if subtask.agent_role in _KM_ROLES and self._km is None:
                skipped.append(
                    f"{subtask.description} (no knowledge_manager)"
                )
                logger.warning(
                    "Skipping %s subtask: no knowledge_manager",
                    subtask.agent_role,
                )
                continue

            # Build typed task
            try:
                typed_task = self._build_task(
                    subtask, results, task.trace_id,
                )
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "Failed to build task for subtask %d: %s", idx, exc,
                )
                skipped.append(subtask.description)
                continue

            # Create specialist with role-conditional kwargs
            extra_kwargs: dict[str, Any] = {}
            if subtask.agent_role in _KM_ROLES:
                extra_kwargs["knowledge_manager"] = self._km

            agent = self._factory.create(
                subtask.agent_role,
                tool_registry=self.tool_registry,
                **extra_kwargs,
            )

            if on_status:
                await on_status(
                    f"{subtask.agent_role}: {subtask.description[:50]}..."
                )

            try:
                results[idx] = await agent.execute(
                    typed_task, on_status=on_status,
                )
            except Exception as exc:
                logger.warning("Subtask %d failed: %s", idx, exc)
                results[idx] = AgentResult(
                    task_id=typed_task.task_id,
                    agent_role=subtask.agent_role,
                    status=TaskStatus.FAILED,
                    output=f"Specialist failed: {exc}",
                )

        return results, skipped
```

Add the synthesis helper model and methods:

```python
class _SynthesisOutput(BaseModel):
    """Schema for the synthesis LLM call."""

    response: str
```

Place this **above** the `Orchestrator` class definition. Then add the synthesis methods to `Orchestrator`:

```python
    async def _synthesize(
        self,
        task: AgentTask,
        plan: OrchestratorPlan,
        results: list[AgentResult | None],
        skipped: list[str],
    ) -> OrchestratorResult:
        """Phase 3: Synthesize specialist results into a final response."""
        completed = [r for r in results if r is not None]

        # Extract article info
        article_path: str | None = None
        needs_human_review = False
        for r in completed:
            if isinstance(r, WriteResult) and r.article_path:
                article_path = r.article_path
            if isinstance(r, EditResult) and r.approved_for_review:
                needs_human_review = True

        # Determine overall status
        if not completed:
            status = TaskStatus.FAILED
        elif (
            all(r.status == TaskStatus.SUCCESS for r in completed)
            and not skipped
        ):
            status = TaskStatus.SUCCESS
        elif all(r.status == TaskStatus.FAILED for r in completed):
            status = TaskStatus.FAILED
        else:
            status = TaskStatus.PARTIAL

        # Synthesize output
        is_article_workflow = any(
            isinstance(r, WriteResult) for r in completed
        )

        if is_article_workflow:
            output = self._synthesize_article_template(completed, skipped)
        elif completed:
            output = await self._synthesize_research(task.query, completed)
        else:
            output = "No specialist results available."
            if skipped:
                output += f" Skipped tasks: {', '.join(skipped)}"

        # Aggregate evidence
        evidence = list(
            dict.fromkeys(e for r in completed for e in r.evidence)
        )

        return OrchestratorResult(
            task_id=task.task_id,
            agent_role=self.role,
            status=status,
            output=output,
            evidence=evidence,
            plan=plan,
            specialist_results=completed,
            article_path=article_path,
            needs_human_review=needs_human_review,
            skipped_tasks=skipped,
        )

    def _synthesize_article_template(
        self,
        results: list[AgentResult],
        skipped: list[str],
    ) -> str:
        """Template-based synthesis for article workflows."""
        parts: list[str] = []
        for r in results:
            if isinstance(r, WriteResult):
                parts.append(f"Article written: **{r.title}**")
                parts.append(f"Path: `{r.article_path}`")
                parts.append(f"Claims: {len(r.claims)}")
                if r.research_gaps:
                    parts.append(
                        f"Research gaps: {', '.join(r.research_gaps)}"
                    )
            elif isinstance(r, EditResult):
                parts.append(
                    f"Verification: confidence {r.overall_confidence:.0%}"
                )
                if r.approved_for_review:
                    parts.append("Status: approved for human review")
                if r.corrections:
                    parts.append(
                        f"Corrections needed: {len(r.corrections)}"
                    )
        if skipped:
            parts.append(f"Skipped: {', '.join(skipped)}")
        return "\n".join(parts)

    async def _synthesize_research(
        self, query: str, results: list[AgentResult],
    ) -> str:
        """LLM-based synthesis for research-only workflows."""
        findings_text = "\n\n".join(
            r.output for r in results if r.output
        )

        messages = [
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {
                "role": "user",
                "content": f"Original question: {query}\n\n"
                f"Research findings:\n{findings_text}",
            },
        ]

        try:
            result: _SynthesisOutput = await get_structured_output(
                response_model=_SynthesisOutput, messages=messages,
            )
            return result.response
        except Exception as exc:
            logger.warning(
                "Synthesis LLM call failed, falling back: %s", exc,
            )
            return findings_text
```

### Step 4: Run tests to verify they pass

```bash
python3 -m pytest tests/unit/test_orchestrator_execute.py -v
```

Expected: 11 passed

### Step 5: Run full suite

```bash
python3 -m pytest tests/unit/ -v --tb=short
```

Expected: 612 passed (601 + 11)

### Step 6: Lint check

```bash
ruff check code/shukketsu/agents/orchestrator.py
```

### Step 7: Commit

```bash
git add code/shukketsu/agents/orchestrator.py tests/unit/test_orchestrator_execute.py
git commit -m "feat(orchestrator): implement execute with decompose, dispatch, synthesize"
```

---

## Task 6: Factory Registration

**Files:**
- Modify: `code/shukketsu/agents/factory.py`
- Test: `tests/unit/test_agent_factory.py` (append new tests)

### Step 1: Write the failing tests

Append to `tests/unit/test_agent_factory.py`:

```python
class TestOrchestratorFactory:
    def test_factory_creates_orchestrator(self) -> None:
        """AgentFactory creates an Orchestrator instance for ORCHESTRATOR role."""
        from unittest.mock import MagicMock

        from code.shukketsu.agents.orchestrator import Orchestrator

        factory = AgentFactory()
        agent = factory.create(AgentRole.ORCHESTRATOR, factory=factory)
        assert isinstance(agent, Orchestrator)

    def test_orchestrator_has_correct_prompt(self) -> None:
        """Orchestrator gets ORCHESTRATOR_SYSTEM_PROMPT from config."""
        from code.shukketsu.llm.prompts.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT

        factory = AgentFactory()
        agent = factory.create(AgentRole.ORCHESTRATOR, factory=factory)
        assert agent._system_prompt == ORCHESTRATOR_SYSTEM_PROMPT

    def test_orchestrator_requires_factory_kwarg(self) -> None:
        """Orchestrator via factory without factory kwarg raises TypeError."""
        import pytest

        factory = AgentFactory()
        with pytest.raises(TypeError):
            factory.create(AgentRole.ORCHESTRATOR)
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/unit/test_agent_factory.py::TestOrchestratorFactory -v
```

Expected: FAIL — Orchestrator not registered in `_ROLE_CLASSES`, falls back to `BaseAgent`

### Step 3: Register Orchestrator in factory

In `code/shukketsu/agents/factory.py`, add the import and registry entry:

Add to imports:

```python
from code.shukketsu.agents.orchestrator import Orchestrator
```

Update `_ROLE_CLASSES`:

```python
_ROLE_CLASSES: dict[AgentRole, type[BaseAgent]] = {
    AgentRole.RESEARCHER: Researcher,
    AgentRole.WRITER: Writer,
    AgentRole.EDITOR: Editor,
    AgentRole.ORCHESTRATOR: Orchestrator,
}
```

### Step 4: Run tests to verify they pass

```bash
python3 -m pytest tests/unit/test_agent_factory.py::TestOrchestratorFactory -v
```

Expected: 3 passed

### Step 5: Run full suite

```bash
python3 -m pytest tests/unit/ -v --tb=short
```

Expected: 615 passed (612 + 3)

### Step 6: Commit

```bash
git add code/shukketsu/agents/factory.py tests/unit/test_agent_factory.py
git commit -m "feat(factory): register Orchestrator in agent factory"
```

---

## Task 7: Chat Handler Routing

**Files:**
- Modify: `code/shukketsu/web/routers/chat.py`
- Modify: `tests/unit/test_chat_handler.py` (update existing patches + add new tests)

### Step 1: Update chat.py

Replace the contents of `code/shukketsu/web/routers/chat.py`. Key changes:
- Replace `_get_agent()` singleton with `_get_agents()` returning (researcher, orchestrator)
- Route by complexity: TRIVIAL → direct, MODERATE → Researcher.execute(), COMPLEX → Orchestrator.execute()

Full replacement for `_agent_instance`, `_get_agent()`, and `_agent_response()`:

Replace lines 23-24 (`_agent_instance`):

```python
_researcher_instance: BaseAgent | None = None
_orchestrator_instance: BaseAgent | None = None
```

Replace the entire `_get_agent()` function with:

```python
def _get_agents() -> tuple[BaseAgent, BaseAgent]:
    """Get or create agent singletons (researcher + orchestrator).

    Lazy initialization avoids import-time side effects (DB connection,
    extension loading). Agents are created once and reused.
    """
    global _researcher_instance, _orchestrator_instance  # noqa: PLW0603

    if _researcher_instance is None:
        from code.shukketsu.agents.factory import AgentFactory
        from code.shukketsu.agents.tasks import AgentRole
        from code.shukketsu.db.connection import get_connection, init_db
        from code.shukketsu.ingest.embedder import get_embedder
        from code.shukketsu.ingest.pipeline import IngestPipeline
        from code.shukketsu.knowledge.manager import KnowledgeManager
        from code.shukketsu.scraping.fetcher import WebFetcher
        from code.shukketsu.scraping.rate_limiter import RateLimiter
        from code.shukketsu.scraping.robots import RobotsChecker
        from code.shukketsu.tools.knowledge.graph_search import GraphSearchTool
        from code.shukketsu.tools.knowledge.search import RagSearchTool
        from code.shukketsu.tools.registry import ToolRegistry
        from code.shukketsu.tools.research.web_ingest import WebIngestTool
        from code.shukketsu.tools.research.web_search import WebSearchTool

        conn = get_connection()
        init_db(conn)
        embedder = get_embedder()

        registry = ToolRegistry()
        registry.register(RagSearchTool(conn=conn, embed_fn=embedder.embed_query))
        registry.register(GraphSearchTool(conn=conn))

        # Web search + ingest tools
        fetcher = WebFetcher(rate_limiter=RateLimiter(), robots_checker=RobotsChecker())
        pipeline = IngestPipeline(conn=conn, embedder=embedder)
        registry.register(WebSearchTool())
        registry.register(WebIngestTool(fetcher=fetcher, pipeline=pipeline))

        factory = AgentFactory()
        km = KnowledgeManager(conn, config.WIKI_PATH)

        _researcher_instance = factory.create(
            AgentRole.RESEARCHER, tool_registry=registry,
        )
        _orchestrator_instance = factory.create(
            AgentRole.ORCHESTRATOR,
            tool_registry=registry,
            factory=factory,
            knowledge_manager=km,
        )

    return _researcher_instance, _orchestrator_instance
```

Replace the `_agent_response` function. Add these imports at the top of the file (with the existing ones):

```python
from code.shukketsu.agents.tasks import AgentTask, OrchestratorResult, ResearchTask
```

Replace `_agent_response`:

```python
@observe()
async def _agent_response(websocket: WebSocket, session: ChatSession, content: str) -> None:
    """Get an agent response for the given user message.

    Routes through Qwen 4B first: trivial queries get a direct answer,
    moderate queries go to the Researcher, complex queries go to the
    Orchestrator for multi-agent coordination.
    """
    session.is_streaming = True
    session.add_message("user", content)

    try:
        langfuse = get_client()
        langfuse.update_current_trace(
            session_id=str(id(session)),
            tags=["chat"],
            input=content,
        )
        await websocket.send_json({"type": "status", "content": "routing..."})
        decision = await classify_query(content)
        logger.info("Route: %s → %s", decision.complexity, decision.category)

        async def _send_status(msg: str) -> None:
            await websocket.send_json({"type": "status", "content": msg})

        if (
            decision.complexity == TaskComplexity.TRIVIAL
            and decision.direct_answer is not None
            and decision.direct_answer.strip()
        ):
            answer = decision.direct_answer
        elif decision.complexity == TaskComplexity.MODERATE:
            await websocket.send_json({"type": "status", "content": "researching..."})
            researcher, _ = _get_agents()
            result = await researcher.execute(
                ResearchTask(query=content), on_status=_send_status,
            )
            answer = result.output
        else:
            await websocket.send_json({"type": "status", "content": "planning..."})
            _, orchestrator = _get_agents()
            result = await orchestrator.execute(
                AgentTask(query=content), on_status=_send_status,
            )
            answer = result.output
            if isinstance(result, OrchestratorResult):
                if result.article_path:
                    answer += f"\n\n---\n*Draft article created: {result.article_path}*"
                if result.needs_human_review:
                    answer += "\n*Article pending review in Wiki*"

        session.add_message("assistant", answer)
        await websocket.send_json({"type": "done", "content": answer})
    except ShukketsuError as exc:
        if session.history and session.history[-1]["role"] == "user":
            session.history.pop()
        await websocket.send_json({"type": "error", "content": str(exc)})
    except Exception:
        logger.exception("Unexpected error in agent response")
        if session.history and session.history[-1]["role"] == "user":
            session.history.pop()
        await websocket.send_json(
            {"type": "error", "content": "An unexpected error occurred. Please try again."}
        )
    finally:
        session.is_streaming = False
```

### Step 2: Replace the entire test file

Replace the full contents of `tests/unit/test_chat_handler.py`. The key changes from the old version:
- `_get_agent` → `_get_agents` (returns tuple)
- `_mock_agent` → `_mock_agents` (returns `(researcher, orchestrator)`)
- `agent.run()` → `orchestrator.execute()` (new API)
- `"thinking..."` → `"planning..."` (COMPLEX path sends different status)
- `mock_get.return_value.run` → `orchestrator.execute` assertions
- New `_moderate_decision()`, `_drain_status()` helpers
- New `TestComplexityRouting` class

```python
"""Tests for the WebSocket chat handler."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from code.shukketsu.resilience.errors import LLMUnavailableError
from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity


def _get_app():
    from code.shukketsu.web.app import app

    return app


def _mock_agents(answer: str = "Test answer") -> tuple[MagicMock, MagicMock]:
    """Create mock researcher + orchestrator agents."""
    researcher = MagicMock()
    researcher.execute = AsyncMock(return_value=MagicMock(output=answer))

    orchestrator = MagicMock()
    orchestrator.execute = AsyncMock(return_value=MagicMock(output=answer))

    return researcher, orchestrator


def _trivial_decision(answer: str = "Direct answer.") -> RoutingDecision:
    return RoutingDecision(
        complexity=TaskComplexity.TRIVIAL,
        category=TaskCategory.CONVERSATION,
        needs_tools=False,
        suggested_agent="general",
        direct_answer=answer,
    )


def _moderate_decision() -> RoutingDecision:
    return RoutingDecision(
        complexity=TaskComplexity.MODERATE,
        category=TaskCategory.RETRIEVAL,
        needs_tools=True,
        suggested_agent="researcher",
    )


def _complex_decision() -> RoutingDecision:
    return RoutingDecision(
        complexity=TaskComplexity.COMPLEX,
        category=TaskCategory.RESEARCH,
        needs_tools=True,
        suggested_agent="orchestrator",
    )


def _drain_status(ws) -> dict:
    """Drain status messages, return the first non-status message."""
    while True:
        msg = ws.receive_json()
        if msg["type"] != "status":
            return msg


class TestWebSocketProtocol:
    """Tests for WebSocket connection and message protocol."""

    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
    def test_connects_and_sends_status(self, mock_get: MagicMock) -> None:
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            msg = ws.receive_json()
            assert msg == {"type": "status", "content": "connected"}

    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
    def test_rejects_empty_message(self, mock_get: MagicMock) -> None:
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": ""})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "empty" in msg["content"].lower()

    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
    def test_rejects_unknown_type(self, mock_get: MagicMock) -> None:
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "bogus"})
            msg = ws.receive_json()
            assert msg["type"] == "error"

    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
    def test_rejects_missing_type(self, mock_get: MagicMock) -> None:
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"content": "hello"})
            msg = ws.receive_json()
            assert msg["type"] == "error"

    @patch("code.shukketsu.web.routers.chat._get_agents", return_value=_mock_agents())
    def test_rejects_oversized_message(self, mock_get: MagicMock) -> None:
        """Messages exceeding CHAT_MAX_MESSAGE_LENGTH should be rejected."""
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            huge_message = "x" * 10_001
            ws.send_json({"type": "message", "content": huge_message})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "too long" in msg["content"].lower()


class TestAgentResponse:
    """Tests for the agent-based response path."""

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_sends_routing_then_planning_status(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        mock_classify.return_value = _complex_decision()
        mock_get.return_value = _mock_agents("Answer")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hi"})
            routing = ws.receive_json()
            assert routing == {"type": "status", "content": "routing..."}
            planning = ws.receive_json()
            assert planning == {"type": "status", "content": "planning..."}

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_sends_done_with_answer(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        mock_classify.return_value = _complex_decision()
        mock_get.return_value = _mock_agents("The hit cap is 9%.")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "What is hit cap?"})
            ws.receive_json()  # routing
            ws.receive_json()  # planning
            done = ws.receive_json()
            assert done == {"type": "done", "content": "The hit cap is 9%."}

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_error_sends_error_message(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        mock_classify.return_value = _complex_decision()
        researcher, orchestrator = _mock_agents()
        orchestrator.execute = AsyncMock(side_effect=LLMUnavailableError("Server down"))
        mock_get.return_value = (researcher, orchestrator)
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hello"})
            ws.receive_json()  # routing
            ws.receive_json()  # planning
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "Server down" in err["content"]

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_second_message_after_response(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        mock_classify.return_value = _complex_decision()
        mock_get.return_value = _mock_agents("Answer")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "First"})
            ws.receive_json()  # routing
            ws.receive_json()  # planning
            ws.receive_json()  # done
            ws.send_json({"type": "message", "content": "Second"})
            ws.receive_json()  # routing
            ws.receive_json()  # planning
            done = ws.receive_json()
            assert done["type"] == "done"

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_rejects_message_during_agent_run(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        mock_classify.return_value = _complex_decision()
        mock_get.return_value = _mock_agents("Answer")
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hello"})
            ws.receive_json()  # routing
            ws.receive_json()  # planning
            done = ws.receive_json()
            assert done["type"] == "done"


class TestQueryRouting:
    """Tests for the multi-model routing integration."""

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_trivial_query_returns_fast_answer(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """Trivial queries should return Qwen's direct answer without calling agents."""
        mock_classify.return_value = _trivial_decision("Sinister Strike costs 40 energy.")
        researcher, orchestrator = _mock_agents()
        mock_get.return_value = (researcher, orchestrator)
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "How much energy does SS cost?"})
            routing = ws.receive_json()
            assert routing == {"type": "status", "content": "routing..."}
            done = ws.receive_json()
            assert done == {"type": "done", "content": "Sinister Strike costs 40 energy."}
            researcher.execute.assert_not_called()
            orchestrator.execute.assert_not_called()

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_complex_query_routes_to_orchestrator(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """Complex queries should route to the Orchestrator after classification."""
        mock_classify.return_value = _complex_decision()
        researcher, orchestrator = _mock_agents("Detailed analysis here.")
        mock_get.return_value = (researcher, orchestrator)
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Compare combat vs assassination"})
            routing = ws.receive_json()
            assert routing == {"type": "status", "content": "routing..."}
            planning = ws.receive_json()
            assert planning == {"type": "status", "content": "planning..."}
            done = ws.receive_json()
            assert done == {"type": "done", "content": "Detailed analysis here."}
            orchestrator.execute.assert_called_once()

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_routing_failure_falls_through_to_orchestrator(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """If classify_query returns a fallback (COMPLEX), orchestrator should handle the query."""
        mock_classify.return_value = _complex_decision()
        researcher, orchestrator = _mock_agents("Orchestrator handled it.")
        mock_get.return_value = (researcher, orchestrator)
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Hello"})
            ws.receive_json()  # routing
            ws.receive_json()  # planning
            done = ws.receive_json()
            assert done == {"type": "done", "content": "Orchestrator handled it."}

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_trivial_with_empty_answer_falls_through(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """Trivial classification with empty direct_answer should fall through to orchestrator."""
        mock_classify.return_value = _trivial_decision("")
        researcher, orchestrator = _mock_agents("Orchestrator answer.")
        mock_get.return_value = (researcher, orchestrator)
        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "test"})
            ws.receive_json()  # routing
            ws.receive_json()  # planning
            done = ws.receive_json()
            assert done == {"type": "done", "content": "Orchestrator answer."}
            orchestrator.execute.assert_called_once()


class TestComplexityRouting:
    """Tests for complexity-based routing to specialists."""

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_trivial_uses_direct_answer(
        self, mock_classify: AsyncMock, mock_agents: MagicMock,
    ) -> None:
        """TRIVIAL complexity returns direct answer without calling agents."""
        mock_classify.return_value = _trivial_decision("Energy costs 40.")
        researcher, orchestrator = _mock_agents()
        mock_agents.return_value = (researcher, orchestrator)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "How much energy does SS cost?"})
            ws.receive_json()  # routing
            done = ws.receive_json()
            assert done["content"] == "Energy costs 40."

        researcher.execute.assert_not_called()
        orchestrator.execute.assert_not_called()

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_moderate_uses_researcher(
        self, mock_classify: AsyncMock, mock_agents: MagicMock,
    ) -> None:
        """MODERATE complexity routes to Researcher.execute()."""
        mock_classify.return_value = _moderate_decision()
        researcher, orchestrator = _mock_agents("Researcher answer")
        mock_agents.return_value = (researcher, orchestrator)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "What trinkets for combat?"})
            done = _drain_status(ws)
            assert done["content"] == "Researcher answer"

        researcher.execute.assert_called_once()
        orchestrator.execute.assert_not_called()

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_complex_uses_orchestrator(
        self, mock_classify: AsyncMock, mock_agents: MagicMock,
    ) -> None:
        """COMPLEX complexity routes to Orchestrator.execute()."""
        mock_classify.return_value = _complex_decision()
        researcher, orchestrator = _mock_agents("Orchestrator answer")
        mock_agents.return_value = (researcher, orchestrator)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Write a guide about trinkets"})
            done = _drain_status(ws)
            assert done["content"] == "Orchestrator answer"

        orchestrator.execute.assert_called_once()
        researcher.execute.assert_not_called()

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_article_path_appended_to_response(
        self, mock_classify: AsyncMock, mock_agents: MagicMock,
    ) -> None:
        """OrchestratorResult with article_path adds notification to response."""
        from code.shukketsu.agents.tasks import (
            AgentRole,
            OrchestratorPlan,
            OrchestratorResult,
            TaskStatus,
        )

        mock_classify.return_value = _complex_decision()
        orch_result = OrchestratorResult(
            task_id="t1",
            agent_role=AgentRole.ORCHESTRATOR,
            status=TaskStatus.SUCCESS,
            output="Guide written.",
            plan=OrchestratorPlan(reasoning="test", subtasks=[]),
            article_path="combat/gear/trinkets.md",
            needs_human_review=True,
        )

        researcher, orchestrator = _mock_agents()
        orchestrator.execute = AsyncMock(return_value=orch_result)
        mock_agents.return_value = (researcher, orchestrator)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Write a guide"})
            done = _drain_status(ws)
            assert "combat/gear/trinkets.md" in done["content"]
            assert "pending review" in done["content"].lower()

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_fallback_routes_to_orchestrator(
        self, mock_classify: AsyncMock, mock_agents: MagicMock,
    ) -> None:
        """Router failure (defaults to COMPLEX) routes to Orchestrator."""
        mock_classify.return_value = _complex_decision()
        researcher, orchestrator = _mock_agents("Fallback answer")
        mock_agents.return_value = (researcher, orchestrator)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "Something complex"})
            done = _drain_status(ws)
            assert done["content"] == "Fallback answer"

        orchestrator.execute.assert_called_once()
```

**Important notes for the implementer:**
- This is a **complete file replacement** for `test_chat_handler.py` — do not try to patch individual tests
- `_drain_status` returns the first non-status message (avoids consuming and discarding the done message)
- All `"thinking..."` assertions changed to `"planning..."` (COMPLEX path)
- All `agent.run()` assertions changed to `orchestrator.execute()`
- `_mock_agents()` returns `(researcher, orchestrator)` tuple matching `_get_agents()` return type
- `_complex_decision()` now uses `TaskCategory.RESEARCH` and `suggested_agent="orchestrator"`

### Step 3: Run new tests

```bash
python3 -m pytest tests/unit/test_chat_handler.py -v
```

Expected: All existing tests pass (with updated patches) + 5 new routing tests pass.

### Step 4: Run full suite

```bash
python3 -m pytest tests/unit/ -v --tb=short
```

Expected: ~620 passed (615 + 5 new routing tests)

### Step 5: Full pre-commit check

```bash
ruff check . --fix && ruff format . && python3 -m mypy code/shukketsu/ && python3 -m pytest tests/unit/ -v --tb=short
```

### Step 6: Commit

```bash
git add code/shukketsu/web/routers/chat.py tests/unit/test_chat_handler.py
git commit -m "feat(chat): route MODERATE to Researcher, COMPLEX to Orchestrator"
```

---

## Final: Full Verification + CLAUDE.md Update

### Step 1: Run complete test suite

```bash
python3 -m pytest tests/unit/ -v --tb=short 2>&1 | tail -5
```

Verify total count is ~620+ and all passing.

### Step 2: Run lint + type checks

```bash
ruff check . --fix && ruff format . && python3 -m mypy code/shukketsu/
```

### Step 3: Update CLAUDE.md

- Update "Key files with real code" to include `agents/orchestrator.py`, `llm/prompts/orchestrator.py`
- Update Step 7 line to show COMPLETE with test count
- Add implementation plan doc to the plans table

### Step 4: Final commit

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Phase 2 Step 7 completion"
```
