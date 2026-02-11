# Phase 2 Step 1: Structured Task Protocol + Agent Framework

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Define typed Task/Result contracts for multi-agent communication, refactor BaseAgent to support task-based execution, and create the AgentFactory that configures specialist agents.

**Architecture:** All agents share the same ReAct loop (BaseAgent). Specialization comes from different system prompts and tool sets, injected by the AgentFactory. The ReAct loop lives in a private `_run_loop()` method that returns a `_RunOutcome(output, status)` NamedTuple. Both `run()` (backward-compat string interface) and `execute()` (new task-based interface) are thin wrappers around `_run_loop()`. This avoids mutable state, keeps the design concurrency-safe for the Orchestrator's `asyncio.gather` in Step 7, and ensures both entry points share identical loop behavior. The Structured Task Protocol uses Pydantic models for all inter-agent communication — no message bus, no queues, maximum testability.

**Tech Stack:** Pydantic v2 BaseModel, Python StrEnum, uuid4, existing BaseAgent ReAct loop

---

## Context: Current State

Phase 1 has a single `BaseAgent` class in `code/shukketsu/agents/base.py` with:
- `run(query: str, *, on_status: StatusCallback | None = None) -> str` — ReAct loop
- `ToolRegistry` for tool dispatch
- `LoopDetector` guardrails
- Langfuse tracing via `@observe`

The agent is created directly in the chat handler (`web/routers/chat.py:_get_agent()`) with hardcoded tool setup. There's no concept of agent roles, typed tasks, or factory pattern.

**Existing tests:** 245 passing. All must continue to pass after this step.

**Files to modify:**
- `code/shukketsu/agents/tasks.py` (create)
- `code/shukketsu/agents/base.py` (modify)
- `code/shukketsu/agents/factory.py` (create)
- `code/shukketsu/agents/__init__.py` (modify — currently empty)
- `code/shukketsu/config.py` (modify — add Phase 2 constants)
- `tests/unit/test_tasks.py` (create)
- `tests/unit/test_agent_factory.py` (create)
- `tests/unit/test_base_agent.py` (modify — add execute() tests)

---

## Task 1: AgentRole, TaskStatus, and Base Task/Result Models

**Files:**
- Create: `tests/unit/test_tasks.py`
- Create: `code/shukketsu/agents/tasks.py`

### Step 1: Write failing tests for enums and base models

Create `tests/unit/test_tasks.py`:

```python
"""Tests for the Structured Task Protocol models."""

import uuid

import pytest
from pydantic import ValidationError

from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    TaskStatus,
)


class TestAgentRole:
    def test_all_roles_exist(self) -> None:
        assert AgentRole.RESEARCHER == "researcher"
        assert AgentRole.WRITER == "writer"
        assert AgentRole.EDITOR == "editor"
        assert AgentRole.ORCHESTRATOR == "orchestrator"

    def test_role_is_str(self) -> None:
        assert isinstance(AgentRole.RESEARCHER, str)


class TestTaskStatus:
    def test_all_statuses_exist(self) -> None:
        assert TaskStatus.SUCCESS == "success"
        assert TaskStatus.PARTIAL == "partial"
        assert TaskStatus.FAILED == "failed"


class TestAgentTask:
    def test_minimal_creation(self) -> None:
        task = AgentTask(query="What is the hit cap?")
        assert task.query == "What is the hit cap?"
        assert task.context == {}

    def test_auto_generates_task_id(self) -> None:
        task = AgentTask(query="test")
        uuid.UUID(task.task_id)  # raises if not valid UUID

    def test_auto_generates_trace_id(self) -> None:
        task = AgentTask(query="test")
        uuid.UUID(task.trace_id)  # raises if not valid UUID

    def test_unique_ids_per_instance(self) -> None:
        t1 = AgentTask(query="a")
        t2 = AgentTask(query="b")
        assert t1.task_id != t2.task_id
        assert t1.trace_id != t2.trace_id

    def test_explicit_ids(self) -> None:
        task = AgentTask(task_id="my-id", trace_id="my-trace", query="q")
        assert task.task_id == "my-id"
        assert task.trace_id == "my-trace"

    def test_context_dict(self) -> None:
        task = AgentTask(query="q", context={"phase": 1, "spec": "combat"})
        assert task.context["phase"] == 1

    def test_rejects_missing_query(self) -> None:
        with pytest.raises(ValidationError):
            AgentTask()  # type: ignore[call-arg]


class TestAgentResult:
    def test_minimal_creation(self) -> None:
        result = AgentResult(
            task_id="t1",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="The hit cap is 142 rating.",
        )
        assert result.output == "The hit cap is 142 rating."
        assert result.evidence == []
        assert result.metadata == {}

    def test_with_evidence(self) -> None:
        result = AgentResult(
            task_id="t1",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="answer",
            evidence=["chunk:42", "https://example.com"],
        )
        assert len(result.evidence) == 2

    def test_with_metadata(self) -> None:
        result = AgentResult(
            task_id="t1",
            agent_role=AgentRole.WRITER,
            status=TaskStatus.PARTIAL,
            output="draft",
            metadata={"word_count": 500},
        )
        assert result.metadata["word_count"] == 500

    def test_rejects_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            AgentResult(task_id="t1")  # type: ignore[call-arg]

    def test_all_statuses_valid(self) -> None:
        for status in TaskStatus:
            r = AgentResult(
                task_id="t",
                agent_role=AgentRole.EDITOR,
                status=status,
                output="x",
            )
            assert r.status == status
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/unit/test_tasks.py -v
```

Expected: `ModuleNotFoundError: No module named 'code.shukketsu.agents.tasks'`

### Step 3: Implement enums and base models

Create `code/shukketsu/agents/tasks.py`:

```python
"""Structured Task Protocol models for multi-agent communication.

Defines the typed Task/Result contracts that all agents use. Tasks describe
what a specialist should do; Results capture what they produced. All
inter-agent communication flows through these models.
"""

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentRole(StrEnum):
    """Specialist agent roles in the multi-agent system."""

    RESEARCHER = "researcher"
    WRITER = "writer"
    EDITOR = "editor"
    ORCHESTRATOR = "orchestrator"


class TaskStatus(StrEnum):
    """Outcome status of an agent task execution."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class AgentTask(BaseModel):
    """Base task that any agent can execute.

    All specialist task types inherit from this. The task_id and trace_id
    enable end-to-end tracing through Langfuse.
    """

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    context: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """Base result returned by any agent after executing a task.

    Specialist result types (ResearchResult, WriteResult, etc.) inherit
    from this and add role-specific fields.
    """

    task_id: str
    agent_role: AgentRole
    status: TaskStatus
    output: str
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### Step 4: Run tests to verify they pass

```bash
python3 -m pytest tests/unit/test_tasks.py -v
```

Expected: All 12 tests PASS.

### Step 5: Commit

```bash
git add code/shukketsu/agents/tasks.py tests/unit/test_tasks.py
git commit -m "feat(agents): add AgentRole, TaskStatus, AgentTask, AgentResult base models"
```

---

## Task 2: Specialist Task Types

**Files:**
- Modify: `tests/unit/test_tasks.py`
- Modify: `code/shukketsu/agents/tasks.py`

### Step 1: Write failing tests for specialist types

Append to `tests/unit/test_tasks.py`:

```python
from code.shukketsu.agents.tasks import (
    ArticleType,
    EditTask,
    OrchestratorPlan,
    ResearchTask,
    SearchStrategy,
    SubTask,
    WriteTask,
)


class TestSearchStrategy:
    def test_all_strategies(self) -> None:
        assert SearchStrategy.HYBRID == "hybrid"
        assert SearchStrategy.GRAPH == "graph"
        assert SearchStrategy.WEB == "web"
        assert SearchStrategy.AUTO == "auto"


class TestArticleType:
    def test_all_types(self) -> None:
        assert ArticleType.GUIDE == "guide"
        assert ArticleType.REFERENCE == "reference"
        assert ArticleType.ANALYSIS == "analysis"


class TestResearchTask:
    def test_inherits_agent_task(self) -> None:
        task = ResearchTask(query="What drops from Gruul?")
        assert isinstance(task, AgentTask)
        uuid.UUID(task.task_id)

    def test_defaults(self) -> None:
        task = ResearchTask(query="q")
        assert task.search_strategy is None
        assert task.max_sources == 10

    def test_explicit_strategy(self) -> None:
        task = ResearchTask(query="q", search_strategy=SearchStrategy.GRAPH)
        assert task.search_strategy == SearchStrategy.GRAPH

    def test_custom_max_sources(self) -> None:
        task = ResearchTask(query="q", max_sources=5)
        assert task.max_sources == 5


class TestWriteTask:
    def test_requires_research_result(self) -> None:
        research = AgentResult(
            task_id="r1",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="findings",
        )
        task = WriteTask(query="Write a gear guide", research=research, article_type=ArticleType.GUIDE)
        assert task.research.task_id == "r1"
        assert task.article_type == ArticleType.GUIDE

    def test_rejects_missing_research(self) -> None:
        with pytest.raises(ValidationError):
            WriteTask(query="Write", article_type=ArticleType.GUIDE)  # type: ignore[call-arg]

    def test_rejects_missing_article_type(self) -> None:
        research = AgentResult(
            task_id="r1", agent_role=AgentRole.RESEARCHER, status=TaskStatus.SUCCESS, output="x"
        )
        with pytest.raises(ValidationError):
            WriteTask(query="Write", research=research)  # type: ignore[call-arg]


class TestEditTask:
    def test_creation(self) -> None:
        task = EditTask(
            query="Verify claims",
            article_path="knowledge/combat/gear/trinkets.md",
            claims=["DST drops from Gruul", "Hit cap is 142"],
        )
        assert task.article_path == "knowledge/combat/gear/trinkets.md"
        assert len(task.claims) == 2

    def test_inherits_agent_task(self) -> None:
        task = EditTask(query="q", article_path="p", claims=[])
        assert isinstance(task, AgentTask)

    def test_rejects_missing_fields(self) -> None:
        with pytest.raises(ValidationError):
            EditTask(query="q")  # type: ignore[call-arg]


class TestSubTask:
    def test_creation(self) -> None:
        st = SubTask(
            agent_role=AgentRole.RESEARCHER,
            description="Find Phase 1 trinkets",
        )
        assert st.depends_on == []
        assert st.task_params == {}

    def test_with_dependencies(self) -> None:
        st = SubTask(
            agent_role=AgentRole.WRITER,
            description="Write article",
            depends_on=[0, 1],
        )
        assert st.depends_on == [0, 1]


class TestOrchestratorPlan:
    def test_creation(self) -> None:
        plan = OrchestratorPlan(
            reasoning="Need research then writing",
            subtasks=[
                SubTask(agent_role=AgentRole.RESEARCHER, description="research"),
                SubTask(agent_role=AgentRole.WRITER, description="write", depends_on=[0]),
            ],
        )
        assert len(plan.subtasks) == 2
        assert plan.can_answer_directly is False
        assert plan.direct_answer is None

    def test_direct_answer(self) -> None:
        plan = OrchestratorPlan(
            reasoning="Simple question",
            subtasks=[],
            can_answer_directly=True,
            direct_answer="Sinister Strike costs 40 energy.",
        )
        assert plan.can_answer_directly is True
        assert plan.direct_answer is not None
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/unit/test_tasks.py::TestSearchStrategy -v
```

Expected: `ImportError: cannot import name 'SearchStrategy'`

### Step 3: Implement specialist types

Add to `code/shukketsu/agents/tasks.py` (after the existing base models):

```python
class SearchStrategy(StrEnum):
    """Search strategy for the Researcher agent."""

    HYBRID = "hybrid"
    GRAPH = "graph"
    WEB = "web"
    AUTO = "auto"


class ArticleType(StrEnum):
    """Article types the Writer agent can produce."""

    GUIDE = "guide"
    REFERENCE = "reference"
    ANALYSIS = "analysis"


class ResearchTask(AgentTask):
    """Task for the Researcher agent.

    Optionally specifies a search strategy and max sources to consider.
    """

    search_strategy: SearchStrategy | None = None
    max_sources: int = 10


class WriteTask(AgentTask):
    """Task for the Writer agent.

    Requires research results and an article type.
    """

    research: AgentResult
    article_type: ArticleType


class EditTask(AgentTask):
    """Task for the Editor agent.

    Points to a draft article and lists claims to verify.
    """

    article_path: str
    claims: list[str]


class SubTask(BaseModel):
    """A single sub-task in an Orchestrator plan."""

    agent_role: AgentRole
    description: str
    depends_on: list[int] = Field(default_factory=list)
    task_params: dict[str, Any] = Field(default_factory=dict)


class OrchestratorPlan(BaseModel):
    """Execution plan produced by the Orchestrator.

    Contains an ordered list of sub-tasks with dependency information.
    The Orchestrator walks the dependency graph, executing tasks whose
    dependencies are satisfied, grouping independent tasks for concurrency.
    """

    reasoning: str
    subtasks: list[SubTask]
    can_answer_directly: bool = False
    direct_answer: str | None = None
```

### Step 4: Run tests to verify they pass

```bash
python3 -m pytest tests/unit/test_tasks.py -v
```

Expected: All 28 tests PASS (12 from Task 1 + 16 new).

### Step 5: Commit

```bash
git add code/shukketsu/agents/tasks.py tests/unit/test_tasks.py
git commit -m "feat(agents): add specialist task types — ResearchTask, WriteTask, EditTask, OrchestratorPlan"
```

---

## Task 3: Phase 2 Config Constants

**Files:**
- Modify: `code/shukketsu/config.py`

### Step 1: Add Phase 2 agent constants

Add to `code/shukketsu/config.py` after the existing `LOOP_MAX_TOTAL_REPEATS` line:

```python
# Phase 2 agent limits
RESEARCHER_MAX_ITERATIONS = 10  # Multi-step research needs more room
RESEARCHER_MAX_TOKENS = 150_000  # Research produces more context
ORCHESTRATOR_MAX_SUBTASKS = 6  # Prevent over-decomposition
ORCHESTRATOR_MAX_DEPTH = 1  # No recursive orchestration
ORCHESTRATOR_TIMEOUT_SECONDS = 120  # Total wall-clock budget

# Role-specific system prompts (placeholder — specialist prompts added in later steps)
RESEARCHER_SYSTEM_PROMPT = (
    "You are a Research Specialist for WoW TBC Rogue content. "
    "Your job is to gather comprehensive, accurate information using your search tools."
)
WRITER_SYSTEM_PROMPT = (
    "You are a Wiki Writer for WoW TBC Rogue content. "
    "You produce clear, accurate, well-structured Markdown articles from research findings."
)
EDITOR_SYSTEM_PROMPT = (
    "You are a Fact-Checking Editor for WoW TBC Rogue content. "
    "Your job is to verify claims in draft articles against the knowledge base."
)
ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are the Orchestrator for a WoW TBC Rogue knowledge system. "
    "You decompose complex queries into sub-tasks for specialist agents."
)
```

### Step 2: Run existing tests to verify no regressions

```bash
python3 -m pytest tests/unit/ -v --tb=short
```

Expected: All 245 + new tests PASS. Config is just constants — no behavior change.

### Step 3: Commit

```bash
git add code/shukketsu/config.py
git commit -m "feat(config): add Phase 2 agent limits and role-specific system prompts"
```

---

## Task 4: BaseAgent Refactor — Add Role, `_run_loop()`, and `execute()`

**Files:**
- Modify: `tests/unit/test_base_agent.py`
- Modify: `code/shukketsu/agents/base.py`

### Design: `_run_loop()` + `_RunOutcome` pattern

The ReAct loop currently lives in `run()`. We extract it to a private `_run_loop()` that
returns a `_RunOutcome(output, status)` NamedTuple. Both public entry points become thin
wrappers:

- `run()` — calls `_run_loop()`, returns `outcome.output` (backward compat, still `str`)
- `execute()` — calls `_run_loop()`, wraps in `AgentResult` using both `output` and `status`

**Why not `_last_run_status` instance attribute?** Mutable state between `run()` and `execute()`
is not concurrency-safe. The Orchestrator (Step 7) dispatches agents via `asyncio.gather`. If
two coroutines share an agent instance, the status from one overwrites the other at any `await`
inside the loop. `_RunOutcome` as a return value eliminates this class of bug.

**Why not have `execute()` call `run()` directly?** That creates nested `@observe` spans
(`execute` → `run`) adding noise to Langfuse traces. Both entry points should trace at the
same level, sharing the undecorated `_run_loop()` underneath.

**Note on `task.context`:** The base `execute()` only passes `task.query` to `_run_loop()`.
The `context` dict is intentionally ignored at this stage — specialist agents in Steps 4-7
will override `execute()` to incorporate context into the system prompt or tool configuration.

### Step 1: Write failing tests for role and execute()

Add these imports and test classes to `tests/unit/test_base_agent.py`:

```python
from code.shukketsu.agents.tasks import AgentResult, AgentRole, AgentTask, TaskStatus


class TestBaseAgentRole:
    def test_default_role_is_none(self) -> None:
        agent = BaseAgent(tool_registry=_registry())
        assert agent.role is None

    def test_explicit_role(self) -> None:
        agent = BaseAgent(tool_registry=_registry(), role=AgentRole.RESEARCHER)
        assert agent.role == AgentRole.RESEARCHER


class TestRunLoop:
    """Tests for _run_loop() returning _RunOutcome with status."""

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_success_outcome(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _final_answer("The answer.")
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        outcome = await agent._run_loop("query", on_status=None)
        assert outcome.output == "The answer."
        assert outcome.status == TaskStatus.SUCCESS

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_failed_outcome_on_max_iterations(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _tool_call("echo", {"text": "loop"})
        agent = BaseAgent(tool_registry=_registry(EchoTool()), max_iterations=1)
        outcome = await agent._run_loop("query", on_status=None)
        assert outcome.status == TaskStatus.FAILED

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_partial_outcome_on_loop_detection(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _tool_call("echo", {"text": "stuck"})
        agent = BaseAgent(tool_registry=_registry(EchoTool()), max_iterations=10)
        outcome = await agent._run_loop("query", on_status=None)
        assert outcome.status == TaskStatus.PARTIAL
        assert "partial" in outcome.output.lower()


class TestBaseAgentExecute:
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_returns_agent_result(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _final_answer("The answer.")
        agent = BaseAgent(tool_registry=_registry(), role=AgentRole.RESEARCHER)
        task = AgentTask(query="What is the hit cap?")
        result = await agent.execute(task)
        assert isinstance(result, AgentResult)
        assert result.task_id == task.task_id
        assert result.agent_role == AgentRole.RESEARCHER
        assert result.output == "The answer."

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_success_status(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _final_answer("Good answer.")
        agent = BaseAgent(tool_registry=_registry(), role=AgentRole.RESEARCHER)
        result = await agent.execute(AgentTask(query="q"))
        assert result.status == TaskStatus.SUCCESS

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_failed_status_on_graceful_failure(self, mock_llm: AsyncMock) -> None:
        """Max iterations reached → AGENT_GRACEFUL_FAILURE → FAILED status."""
        mock_llm.return_value = _tool_call("echo", {"text": "loop"})
        agent = BaseAgent(tool_registry=_registry(EchoTool()), role=AgentRole.RESEARCHER, max_iterations=1)
        result = await agent.execute(AgentTask(query="q"))
        assert result.status == TaskStatus.FAILED

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_partial_status_on_loop(self, mock_llm: AsyncMock) -> None:
        """Loop detected → partial answer → PARTIAL status."""
        mock_llm.return_value = _tool_call("echo", {"text": "stuck"})
        agent = BaseAgent(tool_registry=_registry(EchoTool()), role=AgentRole.RESEARCHER, max_iterations=10)
        result = await agent.execute(AgentTask(query="q"))
        assert result.status == TaskStatus.PARTIAL

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_execute_propagates_llm_errors(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = LLMUnavailableError("Server down")
        agent = BaseAgent(tool_registry=_registry(), role=AgentRole.RESEARCHER)
        with pytest.raises(LLMUnavailableError):
            await agent.execute(AgentTask(query="q"))

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_execute_requires_role(self, mock_llm: AsyncMock) -> None:
        """execute() requires a role to be set (needed for AgentResult)."""
        mock_llm.return_value = _final_answer("answer")
        agent = BaseAgent(tool_registry=_registry())  # no role
        with pytest.raises(ValueError, match="role"):
            await agent.execute(AgentTask(query="q"))

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_on_status_callback(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _final_answer("answer")
        agent = BaseAgent(tool_registry=_registry(), role=AgentRole.RESEARCHER)
        statuses: list[str] = []

        async def capture(msg: str) -> None:
            statuses.append(msg)

        await agent.execute(AgentTask(query="q"), on_status=capture)
        assert len(statuses) > 0


class TestBackwardCompat:
    """Verify that the run() interface is completely unchanged."""

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_run_still_returns_string(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _final_answer("42")
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        result = await agent.run("What?")
        assert isinstance(result, str)
        assert result == "42"

    def test_init_without_role(self) -> None:
        """Existing code that creates BaseAgent without role still works."""
        agent = BaseAgent(tool_registry=_registry())
        assert agent.role is None
        assert agent.max_iterations > 0
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/unit/test_base_agent.py::TestBaseAgentRole -v
```

Expected: `TypeError: BaseAgent.__init__() got an unexpected keyword argument 'role'`

### Step 3: Implement the refactored BaseAgent

Modify `code/shukketsu/agents/base.py`. The full file after changes:

```python
"""BaseAgent with ReAct (Reason + Act) loop."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any, NamedTuple

from langfuse import observe

from code.shukketsu import config
from code.shukketsu.agents.guardrails import LoopDetector
from code.shukketsu.agents.tasks import AgentResult, AgentRole, AgentTask, TaskStatus
from code.shukketsu.llm.schemas import ActionType, AgentStep
from code.shukketsu.llm.structured import get_structured_output
from code.shukketsu.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

StatusCallback = Callable[[str], Awaitable[None]]

_REACT_INSTRUCTIONS = """You have access to the following tools:

{tool_descriptions}

When you need information to answer the question, use a tool by responding with action "tool_call".
When you have enough information to answer, respond with action "final_answer".
Always think step by step about what you need to do.
If a tool returns no results or an error, try a DIFFERENT tool or different query — never repeat the same tool call.
If rag_search finds nothing, try web_search. If web_search finds relevant pages, use web_ingest to store them.
Once you have useful information from any source, provide your final_answer — do not keep searching."""


class _RunOutcome(NamedTuple):
    """Internal return type from the ReAct loop.

    Carries both the output string and the execution status, so that
    run() and execute() can consume the loop result without mutable state.
    """

    output: str
    status: TaskStatus


class BaseAgent:
    """Agent that uses a ReAct loop to answer questions with tools.

    Iterates: think -> act (call tool) -> observe (read result)
    until it reaches a final answer or hits the iteration limit.

    Two public entry points share the same loop:
    - run(query) -> str           — Phase 1 backward-compat interface
    - execute(task) -> AgentResult — Phase 2 task-based interface
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        *,
        role: AgentRole | None = None,
        max_iterations: int = config.AGENT_MAX_ITERATIONS,
        system_prompt: str = config.SYSTEM_PROMPT,
    ) -> None:
        self.role = role
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self._system_prompt = system_prompt
        self._loop_detector = LoopDetector()

    @observe(as_type="agent")
    async def run(self, query: str, *, on_status: StatusCallback | None = None) -> str:
        """Run the ReAct loop to answer a query.

        Args:
            query: The user's question.
            on_status: Optional async callback for progress updates.

        Returns:
            The agent's final answer, or a graceful failure message.

        Raises:
            LLMUnavailableError: If the LLM backend is unreachable.
            StructuredOutputError: If structured output validation fails.
        """
        outcome = await self._run_loop(query, on_status=on_status)
        return outcome.output

    @observe(as_type="agent")
    async def execute(self, task: AgentTask, *, on_status: StatusCallback | None = None) -> AgentResult:
        """Execute a typed task and return a structured result.

        This is the task-based interface used by the multi-agent system.
        Calls the same ReAct loop as run() but wraps the result in a
        typed AgentResult with status tracking.

        Note: task.context is intentionally ignored in the base implementation.
        Specialist agents (Steps 4-7) override execute() to incorporate context.

        Args:
            task: The task to execute.
            on_status: Optional async callback for progress updates.

        Returns:
            AgentResult with the output and execution status.

        Raises:
            ValueError: If no role is set on this agent.
            LLMUnavailableError: If the LLM backend is unreachable.
            StructuredOutputError: If structured output validation fails.
        """
        if self.role is None:
            raise ValueError("Cannot execute() without a role. Use AgentFactory or set role in constructor.")

        outcome = await self._run_loop(task.query, on_status=on_status)

        return AgentResult(
            task_id=task.task_id,
            agent_role=self.role,
            status=outcome.status,
            output=outcome.output,
        )

    async def _run_loop(self, query: str, *, on_status: StatusCallback | None = None) -> _RunOutcome:
        """The core ReAct loop. Shared by run() and execute().

        Returns a _RunOutcome with the output string and status, so callers
        can decide how to surface the result without relying on mutable state.
        """
        scratchpad: list[dict[str, Any]] = []

        for iteration in range(self.max_iterations):
            messages = self._build_messages(query, scratchpad)

            logger.info("Agent iteration %d/%d", iteration + 1, self.max_iterations)
            if on_status:
                await on_status(f"thinking ({iteration + 1}/{self.max_iterations})...")
            step: AgentStep = await get_structured_output(
                response_model=AgentStep,
                messages=messages,
            )

            if step.action == ActionType.FINAL_ANSWER:
                logger.info("Agent reached final answer after %d iteration(s)", iteration + 1)
                return _RunOutcome(output=step.answer, status=TaskStatus.SUCCESS)  # type: ignore[arg-type]

            tool_call = step.tool_call
            assert tool_call is not None  # guaranteed by AgentStep validator
            logger.info("Agent calling tool: %s", tool_call.tool_name)
            if on_status:
                await on_status(f"using {tool_call.tool_name}...")

            observation = await self.tool_registry.execute(tool_call.tool_name, tool_call.tool_input)

            scratchpad.append(
                {
                    "reasoning": step.reasoning,
                    "tool_name": tool_call.tool_name,
                    "tool_input": tool_call.tool_input,
                    "observation": observation,
                }
            )

            # Check for loops after each tool call
            loop_msg = self._loop_detector.check(scratchpad)
            if loop_msg:
                logger.warning("Loop detected: %s", loop_msg)
                return _RunOutcome(
                    output=self._synthesize_partial_answer(scratchpad),
                    status=TaskStatus.PARTIAL,
                )

        logger.warning("Agent reached max iterations (%d) without final answer", self.max_iterations)
        return _RunOutcome(output=config.AGENT_GRACEFUL_FAILURE, status=TaskStatus.FAILED)

    def _synthesize_partial_answer(self, scratchpad: list[dict[str, Any]]) -> str:
        """Build an answer from partial observations when a loop is detected."""
        observations = [e["observation"] for e in scratchpad if e.get("observation")]
        if observations:
            unique = list(dict.fromkeys(observations))
            joined = "\n\n".join(unique)
            return f"Based on partial results:\n\n{joined}"
        return config.AGENT_GRACEFUL_FAILURE

    def _build_messages(self, query: str, scratchpad: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Build the messages array for the LLM."""
        tool_descriptions = self.tool_registry.get_tool_descriptions()
        system_content = self._system_prompt + "\n\n" + _REACT_INSTRUCTIONS.format(tool_descriptions=tool_descriptions)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        for entry in scratchpad:
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Thought: {entry['reasoning']}\n"
                        f"Action: tool_call\n"
                        f"Tool: {entry['tool_name']}\n"
                        f"Input: {entry['tool_input']}"
                    ),
                }
            )
            messages.append({"role": "user", "content": f"Observation: {entry['observation']}"})

        return messages
```

**Key changes from Phase 1 `base.py`:**
1. Added `NamedTuple` import and `_RunOutcome` class
2. Added `AgentResult, AgentRole, AgentTask, TaskStatus` imports from `agents.tasks`
3. Added `role: AgentRole | None = None` to `__init__`
4. Extracted loop body from `run()` into `_run_loop()` returning `_RunOutcome`
5. `run()` is now a thin wrapper: calls `_run_loop()`, returns `outcome.output`
6. Added `execute()` with `@observe`: calls `_run_loop()`, wraps in `AgentResult`
7. `_synthesize_partial_answer` and `_build_messages` are unchanged

### Step 4: Run tests to verify they pass

```bash
python3 -m pytest tests/unit/test_base_agent.py -v
```

Expected: All existing tests PASS + 12 new tests PASS (2 role + 3 run_loop + 7 execute/compat).

### Step 5: Run full suite to verify no regressions

```bash
python3 -m pytest tests/unit/ -v --tb=short
```

Expected: All tests PASS (245 existing + 28 task tests + 12 base agent tests).

### Step 6: Commit

```bash
git add code/shukketsu/agents/base.py tests/unit/test_base_agent.py
git commit -m "feat(agents): extract _run_loop(), add role and execute() to BaseAgent"
```

---

## Task 5: AgentFactory

**Files:**
- Create: `tests/unit/test_agent_factory.py`
- Create: `code/shukketsu/agents/factory.py`

### Step 1: Write failing tests for AgentFactory

Create `tests/unit/test_agent_factory.py`:

```python
"""Tests for the AgentFactory."""

from typing import Any

import pytest

from code.shukketsu.agents.base import BaseAgent
from code.shukketsu.agents.factory import AgentFactory
from code.shukketsu.agents.tasks import AgentRole
from code.shukketsu.tools.registry import ToolRegistry
from code.shukketsu.tools.schemas import Tool


class StubTool(Tool):
    name = "stub"
    description = "A stub tool for testing."
    parameters_schema = {"input": {"type": "string", "description": "test"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        return "stub result"


class TestAgentFactoryCreate:
    def test_creates_base_agent(self) -> None:
        factory = AgentFactory()
        agent = factory.create(AgentRole.RESEARCHER)
        assert isinstance(agent, BaseAgent)

    def test_sets_role(self) -> None:
        factory = AgentFactory()
        agent = factory.create(AgentRole.WRITER)
        assert agent.role == AgentRole.WRITER

    def test_each_role_gets_different_prompt(self) -> None:
        factory = AgentFactory()
        researcher = factory.create(AgentRole.RESEARCHER)
        writer = factory.create(AgentRole.WRITER)
        assert researcher._system_prompt != writer._system_prompt

    def test_researcher_prompt_mentions_search(self) -> None:
        factory = AgentFactory()
        agent = factory.create(AgentRole.RESEARCHER)
        assert "search" in agent._system_prompt.lower()

    def test_writer_prompt_mentions_article(self) -> None:
        factory = AgentFactory()
        agent = factory.create(AgentRole.WRITER)
        prompt = agent._system_prompt.lower()
        assert "article" in prompt or "markdown" in prompt

    def test_editor_prompt_mentions_verify(self) -> None:
        factory = AgentFactory()
        agent = factory.create(AgentRole.EDITOR)
        assert "verify" in agent._system_prompt.lower()

    def test_orchestrator_prompt_mentions_decompose(self) -> None:
        factory = AgentFactory()
        agent = factory.create(AgentRole.ORCHESTRATOR)
        assert "decompose" in agent._system_prompt.lower()


class TestAgentFactoryToolRegistry:
    def test_default_empty_registry(self) -> None:
        factory = AgentFactory()
        agent = factory.create(AgentRole.RESEARCHER)
        assert len(agent.tool_registry) == 0

    def test_custom_registry(self) -> None:
        registry = ToolRegistry()
        registry.register(StubTool())
        factory = AgentFactory()
        agent = factory.create(AgentRole.RESEARCHER, tool_registry=registry)
        assert "stub" in agent.tool_registry

    def test_agents_get_independent_registries(self) -> None:
        """Two agents created without explicit registry should not share state."""
        factory = AgentFactory()
        a1 = factory.create(AgentRole.RESEARCHER)
        a2 = factory.create(AgentRole.WRITER)
        assert a1.tool_registry is not a2.tool_registry


class TestAgentFactoryIterationLimits:
    def test_researcher_gets_higher_limit(self) -> None:
        from code.shukketsu import config

        factory = AgentFactory()
        agent = factory.create(AgentRole.RESEARCHER)
        assert agent.max_iterations == config.RESEARCHER_MAX_ITERATIONS

    def test_non_researcher_gets_default_limit(self) -> None:
        from code.shukketsu import config

        factory = AgentFactory()
        for role in [AgentRole.WRITER, AgentRole.EDITOR, AgentRole.ORCHESTRATOR]:
            agent = factory.create(role)
            assert agent.max_iterations == config.AGENT_MAX_ITERATIONS
```

### Step 2: Run tests to verify they fail

```bash
python3 -m pytest tests/unit/test_agent_factory.py -v
```

Expected: `ModuleNotFoundError: No module named 'code.shukketsu.agents.factory'`

### Step 3: Implement AgentFactory

Create `code/shukketsu/agents/factory.py`:

```python
"""Factory for creating configured agent instances.

Each agent role gets a different system prompt and iteration limit.
Tool registries are injected by the caller (or default to empty).
Specialist tools are registered in later steps as they are implemented.
"""

import logging

from code.shukketsu import config
from code.shukketsu.agents.base import BaseAgent
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


class AgentFactory:
    """Creates configured agent instances for each specialist role.

    Each role gets a different system prompt and may have different
    iteration limits. Tool registries are either provided by the caller
    or default to empty (tools registered by later pipeline steps).
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
            A BaseAgent configured with the role's prompt and limits.
        """
        registry = tool_registry if tool_registry is not None else ToolRegistry()
        prompt = _ROLE_PROMPTS.get(role, config.SYSTEM_PROMPT)
        max_iter = _ROLE_MAX_ITERATIONS.get(role, config.AGENT_MAX_ITERATIONS)

        agent = BaseAgent(
            tool_registry=registry,
            role=role,
            max_iterations=max_iter,
            system_prompt=prompt,
        )

        logger.info("Created %s agent (max_iter=%d)", role.value, max_iter)
        return agent
```

### Step 4: Run tests to verify they pass

```bash
python3 -m pytest tests/unit/test_agent_factory.py -v
```

Expected: All 11 tests PASS.

### Step 5: Commit

```bash
git add code/shukketsu/agents/factory.py tests/unit/test_agent_factory.py
git commit -m "feat(agents): add AgentFactory for role-based agent creation"
```

---

## Task 6: Module Exports and Final Verification

**Files:**
- Modify: `code/shukketsu/agents/__init__.py`

### Step 1: Update module exports

Replace `code/shukketsu/agents/__init__.py` content with:

```python
"""Multi-agent system: BaseAgent, AgentFactory, and Structured Task Protocol."""

from code.shukketsu.agents.base import BaseAgent
from code.shukketsu.agents.factory import AgentFactory
from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    ArticleType,
    EditTask,
    OrchestratorPlan,
    ResearchTask,
    SearchStrategy,
    SubTask,
    TaskStatus,
    WriteTask,
)

__all__ = [
    "AgentFactory",
    "AgentResult",
    "AgentRole",
    "AgentTask",
    "ArticleType",
    "BaseAgent",
    "EditTask",
    "OrchestratorPlan",
    "ResearchTask",
    "SearchStrategy",
    "SubTask",
    "TaskStatus",
    "WriteTask",
]
```

### Step 2: Run full test suite

```bash
python3 -m pytest tests/unit/ -v --tb=short
```

Expected: All tests PASS — 245 existing + ~51 new = ~296 total. (Exact count depends on final test additions.)

### Step 3: Run linter

```bash
ruff check code/shukketsu/agents/ tests/unit/test_tasks.py tests/unit/test_agent_factory.py tests/unit/test_base_agent.py
ruff format code/shukketsu/agents/ tests/unit/test_tasks.py tests/unit/test_agent_factory.py tests/unit/test_base_agent.py
```

Expected: No errors (or auto-fixed formatting).

### Step 4: Commit

```bash
git add code/shukketsu/agents/__init__.py
git commit -m "feat(agents): export all task protocol types from agents package"
```

---

## Verification Checklist

After all tasks complete, verify:

- [ ] `python3 -m pytest tests/unit/ -v` — all tests pass, including 245 existing
- [ ] `ruff check code/ tests/` — no lint errors
- [ ] `ruff format --check code/ tests/` — properly formatted
- [ ] `AgentFactory().create(AgentRole.RESEARCHER)` — returns a BaseAgent with role set
- [ ] The chat handler (`web/routers/chat.py`) still works unchanged — it creates BaseAgent directly without role, which is fine (backward compat)
- [ ] New files: `agents/tasks.py`, `agents/factory.py`, `tests/unit/test_tasks.py`, `tests/unit/test_agent_factory.py`
- [ ] Modified files: `agents/base.py`, `agents/__init__.py`, `config.py`, `tests/unit/test_base_agent.py`

## Summary of New Test Count

| Test File | New Tests | What They Cover |
|-----------|-----------|----------------|
| `test_tasks.py` | ~28 | Enums, base models, specialist types, validation |
| `test_base_agent.py` | ~12 | Role, `_run_loop` outcomes, execute(), backward compat |
| `test_agent_factory.py` | ~11 | Factory creation, prompts, registries, limits |
| **Total new** | **~51** | |
| **Grand total** | **~296** | 245 existing + 51 new |
