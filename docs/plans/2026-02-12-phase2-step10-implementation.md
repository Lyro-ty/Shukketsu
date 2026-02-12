# Phase 2 Step 10: Integration + Phase Gate Evaluation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add trajectory tracking to all agents, build pure eval metrics + LLM-as-judge modules, create a 30-question eval dataset, write wiring/integration tests, and produce a phase gate runner — completing Phase 2.

**Architecture:** Two-layer eval system: pure scoring functions in `evals/metrics.py` (unit-testable, no LLM) and an eval runner in `evals/phase2_gate.py` that feeds LLM judge results into the scoring functions. Agent trajectory tracking via a `ToolCallRecord` model on `AgentResult`. Integration tests in three tiers: wiring smoke (mocked, CI-safe), live agent (real Ollama), and phase gate (formal eval).

**Tech Stack:** Python 3.12, Pydantic v2, pytest (async_mode=auto), unittest.mock (AsyncMock), instructor, sqlite-vec, Ollama (Llama 70B, Qwen 4B, nomic-embed-text)

---

## Task 1: ToolCallRecord Model + AgentResult Trajectory Field

**Files:**
- Modify: `code/shukketsu/agents/tasks.py` (add ToolCallRecord, modify AgentResult)
- Test: `tests/unit/test_trajectory.py` (new)

**Step 1: Write failing tests for ToolCallRecord and trajectory field**

Create `tests/unit/test_trajectory.py`:

```python
"""Tests for ToolCallRecord and AgentResult trajectory field."""

from pydantic import ValidationError

from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    TaskStatus,
    ToolCallRecord,
)


class TestToolCallRecord:
    def test_valid_record(self) -> None:
        r = ToolCallRecord(tool_name="rag_search", tool_input={"query": "hit cap"})
        assert r.tool_name == "rag_search"
        assert r.tool_input == {"query": "hit cap"}

    def test_empty_tool_input(self) -> None:
        r = ToolCallRecord(tool_name="graph_search")
        assert r.tool_input == {}

    def test_requires_tool_name(self) -> None:
        with pytest.raises(ValidationError):
            ToolCallRecord(tool_input={"query": "test"})  # type: ignore[call-arg]


class TestAgentResultTrajectory:
    def test_default_empty_trajectory(self) -> None:
        r = AgentResult(
            task_id="t1",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="answer",
        )
        assert r.trajectory == []

    def test_trajectory_with_records(self) -> None:
        records = [
            ToolCallRecord(tool_name="rag_search", tool_input={"query": "q1"}),
            ToolCallRecord(tool_name="graph_search", tool_input={"entity": "e1"}),
        ]
        r = AgentResult(
            task_id="t1",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="answer",
            trajectory=records,
        )
        assert len(r.trajectory) == 2
        assert r.trajectory[0].tool_name == "rag_search"

    def test_trajectory_serializes_to_json(self) -> None:
        r = AgentResult(
            task_id="t1",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="answer",
            trajectory=[ToolCallRecord(tool_name="rag_search", tool_input={"query": "q"})],
        )
        data = r.model_dump()
        assert len(data["trajectory"]) == 1
        assert data["trajectory"][0]["tool_name"] == "rag_search"
```

Add missing `import pytest` at the top of the file.

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/unit/test_trajectory.py -v
```

Expected: FAIL — `ToolCallRecord` not importable from `tasks.py`.

**Step 3: Implement ToolCallRecord and add trajectory to AgentResult**

In `code/shukketsu/agents/tasks.py`, add after the `TaskStatus` class (around line 30):

```python
class ToolCallRecord(BaseModel):
    """Record of a single tool call made during agent execution."""

    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
```

In `AgentResult` (around line 45), add the trajectory field:

```python
class AgentResult(BaseModel):
    """Base result returned by any agent after executing a task."""

    task_id: str
    agent_role: AgentRole
    status: TaskStatus
    output: str
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    trajectory: list[ToolCallRecord] = Field(default_factory=list)
```

**Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/unit/test_trajectory.py -v
```

Expected: 4 PASSED.

**Step 5: Run full suite to check for regressions**

```bash
python3 -m pytest tests/unit/ -v
```

Expected: All 690+ tests pass (new field has a default, so existing code is unaffected).

**Step 6: Commit**

```bash
git add code/shukketsu/agents/tasks.py tests/unit/test_trajectory.py
git commit -m "feat(tasks): add ToolCallRecord model and trajectory field to AgentResult"
```

---

## Task 2: Populate Trajectory in BaseAgent and Specialist Agents

**Files:**
- Modify: `code/shukketsu/agents/base.py:112-121` (execute method)
- Modify: `code/shukketsu/agents/researcher.py` (execute method, 3 return paths)
- Modify: `code/shukketsu/agents/writer.py` (no ReAct loop — trajectory stays empty)
- Modify: `code/shukketsu/agents/editor.py` (no ReAct loop — trajectory stays empty)
- Modify: `code/shukketsu/agents/orchestrator.py:83-132` (execute method)
- Test: `tests/unit/test_trajectory.py` (add tests)

**Step 1: Write failing tests for trajectory population**

Append to `tests/unit/test_trajectory.py`:

```python
from typing import Any
from unittest.mock import AsyncMock, patch

from code.shukketsu.agents.base import BaseAgent
from code.shukketsu.agents.tasks import AgentTask, AgentRole, ToolCallRecord
from code.shukketsu.llm.schemas import ActionType, AgentStep, ToolCall
from code.shukketsu.tools.registry import ToolRegistry
from code.shukketsu.tools.schemas import Tool


class _EchoTool(Tool):
    name = "echo"
    description = "Echo."
    parameters_schema = {"text": {"type": "string", "description": "Text"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        return f"Echo: {tool_input.get('text', '')}"


def _registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _final(answer: str) -> AgentStep:
    return AgentStep(reasoning="Done.", action=ActionType.FINAL_ANSWER, answer=answer)


def _call(name: str, inp: dict[str, Any]) -> AgentStep:
    return AgentStep(
        reasoning="Need.",
        action=ActionType.TOOL_CALL,
        tool_call=ToolCall(thought="go", tool_name=name, tool_input=inp),
    )


class TestBaseAgentTrajectory:
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_execute_populates_trajectory(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = [
            _call("echo", {"text": "hello"}),
            _final("Done"),
        ]
        agent = BaseAgent(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await agent.execute(AgentTask(query="q"))
        assert len(result.trajectory) == 1
        assert result.trajectory[0].tool_name == "echo"
        assert result.trajectory[0].tool_input == {"text": "hello"}

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_empty_trajectory_on_immediate_answer(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _final("Answer")
        agent = BaseAgent(tool_registry=_registry(), role=AgentRole.RESEARCHER)
        result = await agent.execute(AgentTask(query="q"))
        assert result.trajectory == []

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_multiple_tool_calls_in_trajectory(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = [
            _call("echo", {"text": "a"}),
            _call("echo", {"text": "b"}),
            _final("Done"),
        ]
        agent = BaseAgent(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await agent.execute(AgentTask(query="q"))
        assert len(result.trajectory) == 2


class TestResearcherTrajectory:
    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_researcher_populates_trajectory(self, mock_loop: AsyncMock, mock_struct: AsyncMock) -> None:
        from code.shukketsu.agents.researcher import Researcher, StructuredFindings
        from code.shukketsu.agents.tasks import Finding

        mock_loop.side_effect = [
            _call("echo", {"text": "search"}),
            _final("Found it"),
        ]
        mock_struct.return_value = StructuredFindings(
            findings=[Finding(claim="test", confidence=0.9)],
            sufficient=True,
        )
        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        result = await researcher.execute(AgentTask(query="q"))
        assert len(result.trajectory) == 1
        assert result.trajectory[0].tool_name == "echo"


class TestOrchestratorTrajectory:
    @patch("code.shukketsu.agents.orchestrator.get_structured_output")
    async def test_orchestrator_dispatch_trajectory(self, mock_llm: AsyncMock) -> None:
        from code.shukketsu.agents.factory import AgentFactory
        from code.shukketsu.agents.orchestrator import Orchestrator
        from code.shukketsu.agents.tasks import OrchestratorPlan, SubTask

        plan = OrchestratorPlan(
            reasoning="Research needed",
            subtasks=[SubTask(agent_role=AgentRole.RESEARCHER, description="Find hit cap info")],
        )

        # First call: decomposition returns plan
        # Second call: researcher ReAct loop returns final answer
        # Third call: researcher structuring pass
        mock_llm.side_effect = [
            plan,  # decompose
            _final("Hit cap is 142"),  # researcher loop
            # structuring pass for researcher
        ]

        # Patch the researcher's structuring separately
        with patch("code.shukketsu.agents.researcher.get_structured_output") as mock_struct:
            from code.shukketsu.agents.researcher import StructuredFindings
            from code.shukketsu.agents.tasks import Finding

            mock_struct.return_value = StructuredFindings(
                findings=[Finding(claim="Hit cap is 142", confidence=0.9)],
                sufficient=True,
            )

            factory = AgentFactory()
            orch = Orchestrator(
                tool_registry=_registry(),
                role=AgentRole.ORCHESTRATOR,
                factory=factory,
            )
            result = await orch.execute(AgentTask(query="What is the hit cap?"))

        assert len(result.trajectory) >= 1
        assert result.trajectory[0].tool_name == "dispatch:researcher"
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/unit/test_trajectory.py::TestBaseAgentTrajectory -v
python3 -m pytest tests/unit/test_trajectory.py::TestResearcherTrajectory -v
python3 -m pytest tests/unit/test_trajectory.py::TestOrchestratorTrajectory -v
```

Expected: FAIL — trajectory not populated yet.

**Step 3: Populate trajectory in BaseAgent.execute()**

In `code/shukketsu/agents/base.py`, modify the `execute()` method (lines 112-121). Replace the `return AgentResult(...)` block:

```python
        outcome = await self._run_loop(task.query, on_status=on_status)

        trajectory = [
            ToolCallRecord(tool_name=entry["tool_name"], tool_input=entry["tool_input"])
            for entry in outcome.scratchpad
        ]

        return AgentResult(
            task_id=task.task_id,
            agent_role=self.role,
            status=outcome.status,
            output=outcome.output,
            trajectory=trajectory,
        )
```

Add `ToolCallRecord` to the import from `code.shukketsu.agents.tasks`:

```python
from code.shukketsu.agents.tasks import AgentResult, AgentRole, AgentTask, TaskStatus, ToolCallRecord
```

**Step 4: Populate trajectory in Researcher.execute()**

In `code/shukketsu/agents/researcher.py`, add `ToolCallRecord` to the import:

```python
from code.shukketsu.agents.tasks import AgentTask, Finding, ResearchResult, TaskStatus, ToolCallRecord
```

In each of the three `return ResearchResult(...)` blocks, add the `trajectory` field:

For the FAILED path (line 68-74):
```python
        trajectory = [
            ToolCallRecord(tool_name=e["tool_name"], tool_input=e["tool_input"])
            for e in outcome.scratchpad
        ]
        return ResearchResult(
            ...existing fields...,
            trajectory=trajectory,
        )
```

For the structuring-failed fallback path (line 88-95):
```python
        trajectory = [
            ToolCallRecord(tool_name=e["tool_name"], tool_input=e["tool_input"])
            for e in outcome.scratchpad
        ]
        return ResearchResult(
            ...existing fields...,
            trajectory=trajectory,
        )
```

For the success path (line 101-112):
```python
        trajectory = [
            ToolCallRecord(tool_name=e["tool_name"], tool_input=e["tool_input"])
            for e in outcome.scratchpad
        ]
        return ResearchResult(
            ...existing fields...,
            trajectory=trajectory,
        )
```

The cleanest approach: extract trajectory once right after the `_run_loop` call (after line 63), and reference it in all three return paths:

```python
        outcome = await self._run_loop(task.query, on_status=on_status)
        trajectory = [
            ToolCallRecord(tool_name=e["tool_name"], tool_input=e["tool_input"])
            for e in outcome.scratchpad
        ]
```

Then add `trajectory=trajectory` to all three `return ResearchResult(...)` calls.

**Step 5: Populate trajectory in Orchestrator.execute()**

In `code/shukketsu/agents/orchestrator.py`, add `ToolCallRecord` to the import from tasks:

```python
from code.shukketsu.agents.tasks import (
    ...existing imports...,
    ToolCallRecord,
)
```

In the `execute()` method, after the dispatch phase (around line 126-132), build the trajectory from dispatched subtasks:

After `results, skipped = await self._dispatch(plan, task, on_status)`, add:

```python
        # Build trajectory from dispatched subtasks
        trajectory: list[ToolCallRecord] = []
        for i, subtask in enumerate(plan.subtasks):
            if results[i] is not None:
                trajectory.append(
                    ToolCallRecord(
                        tool_name=f"dispatch:{subtask.agent_role}",
                        tool_input={"description": subtask.description},
                    )
                )
```

Then in the `_synthesize()` method's return statement, pass the trajectory. However, `_synthesize` returns the `OrchestratorResult`. The cleanest approach: add trajectory as a parameter to `_synthesize()`, or build it in `execute()` and set it on the result after.

Simplest: build trajectory in `execute()` and set it on the returned result:

```python
        result = await self._synthesize(task, plan, results, skipped)
        result_with_trajectory = result.model_copy(update={"trajectory": trajectory})
        return result_with_trajectory  # type: ignore[return-value]
```

Actually, cleaner to just pass it through. In `execute()`, after getting the result from `_synthesize()`:

```python
        orch_result = await self._synthesize(task, plan, results, skipped)
        # Replace trajectory with dispatch records
        object.__setattr__(orch_result, 'trajectory', trajectory)
        return orch_result
```

Wait — Pydantic v2 models are not frozen by default, so simple attribute assignment works:

```python
        orch_result = await self._synthesize(task, plan, results, skipped)
        orch_result.trajectory = trajectory
        return orch_result
```

**Step 6: Run tests to verify they pass**

```bash
python3 -m pytest tests/unit/test_trajectory.py -v
```

Expected: All 8+ tests pass.

**Step 7: Run full suite to check for regressions**

```bash
python3 -m pytest tests/unit/ -v
```

Expected: All tests pass.

**Step 8: Commit**

```bash
git add code/shukketsu/agents/base.py code/shukketsu/agents/researcher.py code/shukketsu/agents/orchestrator.py tests/unit/test_trajectory.py
git commit -m "feat(agents): populate trajectory field in BaseAgent, Researcher, and Orchestrator"
```

---

## Task 3: Eval Metrics — Pure Scoring Functions

**Files:**
- Create: `code/shukketsu/evals/metrics.py`
- Test: `tests/unit/test_eval_metrics.py` (new)

**Step 1: Write failing tests**

Create `tests/unit/test_eval_metrics.py`:

```python
"""Tests for eval metrics — pure scoring functions."""

import pytest

from code.shukketsu.evals.metrics import (
    ACCURACY_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    TRAJECTORY_THRESHOLD,
    ClaimFaithfulness,
    EvalQuestionResult,
    FaithfulnessJudgment,
    PhaseGateReport,
    TrajectoryScore,
    compute_faithfulness,
    compute_phase_gate,
    compute_trajectory_precision,
)


class TestComputeFaithfulness:
    def test_all_supported(self) -> None:
        claims = [
            ClaimFaithfulness(claim="A", judgment=FaithfulnessJudgment.SUPPORTED),
            ClaimFaithfulness(claim="B", judgment=FaithfulnessJudgment.SUPPORTED),
        ]
        assert compute_faithfulness(claims) == 1.0

    def test_all_not_supported(self) -> None:
        claims = [
            ClaimFaithfulness(claim="A", judgment=FaithfulnessJudgment.NOT_SUPPORTED),
            ClaimFaithfulness(claim="B", judgment=FaithfulnessJudgment.NOT_SUPPORTED),
        ]
        assert compute_faithfulness(claims) == 0.0

    def test_mixed_claims(self) -> None:
        claims = [
            ClaimFaithfulness(claim="A", judgment=FaithfulnessJudgment.SUPPORTED),
            ClaimFaithfulness(claim="B", judgment=FaithfulnessJudgment.NOT_SUPPORTED),
            ClaimFaithfulness(claim="C", judgment=FaithfulnessJudgment.SUPPORTED),
        ]
        assert compute_faithfulness(claims) == pytest.approx(2 / 3)

    def test_all_unclear(self) -> None:
        claims = [
            ClaimFaithfulness(claim="A", judgment=FaithfulnessJudgment.UNCLEAR),
            ClaimFaithfulness(claim="B", judgment=FaithfulnessJudgment.UNCLEAR),
        ]
        assert compute_faithfulness(claims) == 1.0

    def test_unclear_excluded_from_denominator(self) -> None:
        claims = [
            ClaimFaithfulness(claim="A", judgment=FaithfulnessJudgment.SUPPORTED),
            ClaimFaithfulness(claim="B", judgment=FaithfulnessJudgment.UNCLEAR),
            ClaimFaithfulness(claim="C", judgment=FaithfulnessJudgment.NOT_SUPPORTED),
        ]
        # denominator = 2 (SUPPORTED + NOT_SUPPORTED), numerator = 1
        assert compute_faithfulness(claims) == pytest.approx(0.5)

    def test_empty_claims(self) -> None:
        assert compute_faithfulness([]) == 1.0

    def test_evidence_snippet_optional(self) -> None:
        c = ClaimFaithfulness(
            claim="X",
            judgment=FaithfulnessJudgment.SUPPORTED,
            evidence_snippet="some evidence",
        )
        assert c.evidence_snippet == "some evidence"


class TestComputeTrajectoryPrecision:
    def test_exact_match(self) -> None:
        actual = ["rag_search", "graph_search"]
        expected = ["rag_search", "graph_search"]
        assert compute_trajectory_precision(actual, expected) == 1.0

    def test_no_expected_tools(self) -> None:
        """Trivial query: no expected tools, no tool calls."""
        assert compute_trajectory_precision([], []) == 1.0

    def test_wasted_consecutive_calls(self) -> None:
        actual = ["rag_search", "rag_search", "rag_search"]
        expected = ["rag_search"]
        # First call is useful, 2nd and 3rd are wasted → 1/3
        assert compute_trajectory_precision(actual, expected) == pytest.approx(1 / 3)

    def test_unexpected_tool_lowers_precision(self) -> None:
        actual = ["rag_search", "web_search"]
        expected = ["rag_search"]
        # rag_search useful, web_search not expected → 1/2
        assert compute_trajectory_precision(actual, expected) == pytest.approx(0.5)

    def test_empty_actual_returns_one(self) -> None:
        assert compute_trajectory_precision([], ["rag_search"]) == 1.0

    def test_non_consecutive_repeats_not_penalized(self) -> None:
        actual = ["rag_search", "graph_search", "rag_search"]
        expected = ["rag_search", "graph_search"]
        # All calls match expected tools, no consecutive repeats → 3/3
        assert compute_trajectory_precision(actual, expected) == 1.0


class TestComputePhaseGate:
    def _make_result(
        self, *, faith: float = 0.9, traj: float = 0.8, acc: float = 0.8
    ) -> EvalQuestionResult:
        return EvalQuestionResult(
            question_id="q01",
            faithfulness=faith,
            trajectory_precision=traj,
            domain_accuracy=acc,
            claims=[],
            tool_calls=[],
        )

    def test_all_passing(self) -> None:
        results = [self._make_result() for _ in range(5)]
        report = compute_phase_gate(results)
        assert report.passed is True
        assert report.avg_faithfulness == pytest.approx(0.9)
        assert report.avg_trajectory_precision == pytest.approx(0.8)
        assert report.avg_domain_accuracy == pytest.approx(0.8)

    def test_one_metric_below_threshold(self) -> None:
        results = [self._make_result(faith=0.5) for _ in range(5)]
        report = compute_phase_gate(results)
        assert report.passed is False
        assert report.avg_faithfulness == pytest.approx(0.5)

    def test_empty_results(self) -> None:
        report = compute_phase_gate([])
        assert report.passed is False
        assert report.avg_faithfulness == 0.0
        assert report.avg_trajectory_precision == 0.0
        assert report.avg_domain_accuracy == 0.0

    def test_report_includes_question_results(self) -> None:
        results = [self._make_result()]
        report = compute_phase_gate(results)
        assert len(report.question_results) == 1

    def test_borderline_passes(self) -> None:
        """Exactly at threshold should pass."""
        results = [
            self._make_result(
                faith=FAITHFULNESS_THRESHOLD,
                traj=TRAJECTORY_THRESHOLD,
                acc=ACCURACY_THRESHOLD,
            )
        ]
        report = compute_phase_gate(results)
        assert report.passed is True


class TestModels:
    def test_trajectory_score_fields(self) -> None:
        ts = TrajectoryScore(precision=0.8, expected_tools_hit=3, wasted_calls=1)
        assert ts.precision == 0.8
        assert ts.expected_tools_hit == 3
        assert ts.wasted_calls == 1

    def test_phase_gate_report_wiki_coverage(self) -> None:
        report = PhaseGateReport(
            avg_faithfulness=0.9,
            avg_trajectory_precision=0.8,
            avg_domain_accuracy=0.8,
            passed=True,
            wiki_coverage={"combat": 3, "assassination": 1},
            question_results=[],
        )
        assert report.wiki_coverage["combat"] == 3
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/unit/test_eval_metrics.py -v
```

Expected: FAIL — module not found.

**Step 3: Implement metrics.py**

Create `code/shukketsu/evals/metrics.py`:

```python
"""Pure scoring functions for phase gate evaluation.

All functions are deterministic and unit-testable — no LLM calls.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

# --- Thresholds ---

FAITHFULNESS_THRESHOLD = 0.8
TRAJECTORY_THRESHOLD = 0.7
ACCURACY_THRESHOLD = 0.7


# --- Models ---


class FaithfulnessJudgment(StrEnum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    UNCLEAR = "unclear"


class ClaimFaithfulness(BaseModel):
    """Result of judging one claim against evidence."""

    claim: str
    judgment: FaithfulnessJudgment
    evidence_snippet: str | None = None


class TrajectoryScore(BaseModel):
    """Detailed trajectory analysis."""

    precision: float
    expected_tools_hit: int
    wasted_calls: int


class EvalQuestionResult(BaseModel):
    """Per-question evaluation scores."""

    question_id: str
    faithfulness: float
    trajectory_precision: float
    domain_accuracy: float
    claims: list[ClaimFaithfulness]
    tool_calls: list[str]


class PhaseGateReport(BaseModel):
    """Aggregate evaluation report for the phase gate."""

    avg_faithfulness: float
    avg_trajectory_precision: float
    avg_domain_accuracy: float
    passed: bool
    wiki_coverage: dict[str, int] = Field(default_factory=dict)
    question_results: list[EvalQuestionResult] = Field(default_factory=list)


# --- Scoring Functions ---


def compute_faithfulness(claims: list[ClaimFaithfulness]) -> float:
    """Compute faithfulness score from claim judgments.

    SUPPORTED / (SUPPORTED + NOT_SUPPORTED). UNCLEAR claims excluded
    from the denominator. Returns 1.0 if no scoreable claims.
    """
    supported = sum(1 for c in claims if c.judgment == FaithfulnessJudgment.SUPPORTED)
    not_supported = sum(1 for c in claims if c.judgment == FaithfulnessJudgment.NOT_SUPPORTED)

    denominator = supported + not_supported
    if denominator == 0:
        return 1.0

    return supported / denominator


def compute_trajectory_precision(
    actual_calls: list[str],
    expected_tools: list[str],
) -> float:
    """Compute trajectory precision.

    Fraction of actual calls that match an expected tool. Repeated
    identical consecutive calls after the first count as wasted
    (precision penalty). Returns 1.0 if no tool calls were made.
    """
    if not actual_calls:
        return 1.0

    expected_set = set(expected_tools)
    useful = 0
    total = len(actual_calls)

    for i, call in enumerate(actual_calls):
        # Consecutive duplicate = wasted
        if i > 0 and call == actual_calls[i - 1]:
            continue  # wasted — not counted as useful
        if call in expected_set:
            useful += 1

    return useful / total


def compute_phase_gate(
    results: list[EvalQuestionResult],
) -> PhaseGateReport:
    """Aggregate per-question scores into a PhaseGateReport.

    Averages each metric across all questions. Sets passed=True only
    if all three averages meet or exceed their thresholds.
    """
    if not results:
        return PhaseGateReport(
            avg_faithfulness=0.0,
            avg_trajectory_precision=0.0,
            avg_domain_accuracy=0.0,
            passed=False,
            question_results=results,
        )

    n = len(results)
    avg_faith = sum(r.faithfulness for r in results) / n
    avg_traj = sum(r.trajectory_precision for r in results) / n
    avg_acc = sum(r.domain_accuracy for r in results) / n

    passed = (
        avg_faith >= FAITHFULNESS_THRESHOLD
        and avg_traj >= TRAJECTORY_THRESHOLD
        and avg_acc >= ACCURACY_THRESHOLD
    )

    return PhaseGateReport(
        avg_faithfulness=avg_faith,
        avg_trajectory_precision=avg_traj,
        avg_domain_accuracy=avg_acc,
        passed=passed,
        question_results=results,
    )
```

**Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/unit/test_eval_metrics.py -v
```

Expected: All ~15 tests pass.

**Step 5: Commit**

```bash
git add code/shukketsu/evals/metrics.py tests/unit/test_eval_metrics.py
git commit -m "feat(evals): add pure scoring functions for phase gate metrics"
```

---

## Task 4: Eval Judge — LLM-as-Judge Module

**Files:**
- Create: `code/shukketsu/evals/judge.py`
- Test: `tests/unit/test_eval_judge.py` (new)

**Step 1: Write failing tests**

Create `tests/unit/test_eval_judge.py`:

```python
"""Tests for eval judge module — all LLM calls mocked."""

from unittest.mock import AsyncMock, patch

import pytest

from code.shukketsu.evals.judge import (
    _AccuracyScore,
    _ClaimExtraction,
    _FaithfulnessVerdict,
    _SingleVerdict,
    extract_claims,
    judge_domain_accuracy,
    judge_faithfulness,
)
from code.shukketsu.evals.metrics import ClaimFaithfulness, FaithfulnessJudgment


class TestExtractClaims:
    @patch("code.shukketsu.evals.judge.get_structured_output")
    async def test_returns_claim_list(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _ClaimExtraction(claims=["Claim A", "Claim B"])
        result = await extract_claims("Some answer text.")
        assert result == ["Claim A", "Claim B"]

    @patch("code.shukketsu.evals.judge.get_structured_output")
    async def test_error_returns_empty_list(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = Exception("LLM timeout")
        result = await extract_claims("Some answer.")
        assert result == []

    @patch("code.shukketsu.evals.judge.get_structured_output")
    async def test_calls_with_reasoning_backend(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _ClaimExtraction(claims=["X"])
        await extract_claims("answer")
        call_kwargs = mock_llm.call_args.kwargs
        from code.shukketsu.llm.structured import ModelBackend
        assert call_kwargs["backend"] == ModelBackend.REASONING


class TestJudgeFaithfulness:
    @patch("code.shukketsu.evals.judge.get_structured_output")
    async def test_returns_claim_faithfulness_list(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _FaithfulnessVerdict(
            judgments=[
                _SingleVerdict(
                    claim="Claim A",
                    judgment=FaithfulnessJudgment.SUPPORTED,
                    evidence_snippet="evidence here",
                ),
            ]
        )
        result = await judge_faithfulness(["Claim A"], ["evidence here"])
        assert len(result) == 1
        assert isinstance(result[0], ClaimFaithfulness)
        assert result[0].judgment == FaithfulnessJudgment.SUPPORTED

    @patch("code.shukketsu.evals.judge.get_structured_output")
    async def test_error_returns_all_unclear(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = Exception("Timeout")
        result = await judge_faithfulness(["Claim A", "Claim B"], ["evidence"])
        assert len(result) == 2
        assert all(c.judgment == FaithfulnessJudgment.UNCLEAR for c in result)


class TestJudgeDomainAccuracy:
    @patch("code.shukketsu.evals.judge.get_structured_output")
    async def test_returns_score(self, mock_llm: AsyncMock) -> None:
        mock_llm.return_value = _AccuracyScore(score=0.85, reasoning="Good")
        result = await judge_domain_accuracy("answer", "truth", ["fact1"])
        assert result == pytest.approx(0.85)

    @patch("code.shukketsu.evals.judge.get_structured_output")
    async def test_error_returns_zero(self, mock_llm: AsyncMock) -> None:
        mock_llm.side_effect = Exception("Crash")
        result = await judge_domain_accuracy("answer", "truth", ["fact1"])
        assert result == 0.0
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/unit/test_eval_judge.py -v
```

Expected: FAIL — module not found.

**Step 3: Implement judge.py**

Create `code/shukketsu/evals/judge.py`:

```python
"""LLM-as-judge calls for phase gate evaluation.

Uses Llama 70B via get_structured_output for claim extraction,
faithfulness judgment, and domain accuracy scoring.
"""

import logging

from pydantic import BaseModel, Field

from code.shukketsu.evals.metrics import ClaimFaithfulness, FaithfulnessJudgment
from code.shukketsu.llm.structured import ModelBackend, get_structured_output

logger = logging.getLogger(__name__)

# --- Internal Schemas ---


class _ClaimExtraction(BaseModel):
    claims: list[str]


class _SingleVerdict(BaseModel):
    claim: str
    judgment: FaithfulnessJudgment
    evidence_snippet: str | None = None


class _FaithfulnessVerdict(BaseModel):
    judgments: list[_SingleVerdict]


class _AccuracyScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


# --- System Prompts ---

_CLAIM_EXTRACTION_PROMPT = """\
Extract only verifiable factual statements from the answer. Skip opinions, \
hedging language ('might', 'could'), structural text ('In this section...'), and \
meta-commentary. Each claim should be a standalone statement that can be checked \
against evidence."""

_FAITHFULNESS_PROMPT = """\
For each claim, determine if the provided evidence supports it.

SUPPORTED: evidence directly states or strongly implies the claim.
NOT_SUPPORTED: evidence contradicts or does not mention the claim.
UNCLEAR: evidence is ambiguous or tangentially related.

Return a judgment for every claim provided."""

_ACCURACY_PROMPT = """\
You are a WoW TBC Rogue expert. Score how well the answer matches the ground truth. \
Check each key fact: is it present and correct? Are there contradictions? A score of \
1.0 means all key facts present and no errors. A score of 0.0 means completely wrong \
or missing all key facts."""


# --- Public Functions ---


async def extract_claims(answer: str) -> list[str]:
    """Extract verifiable factual claims from an answer.

    Returns empty list on failure (conservative — faithfulness = 1.0).
    """
    try:
        result: _ClaimExtraction = await get_structured_output(
            response_model=_ClaimExtraction,
            messages=[
                {"role": "system", "content": _CLAIM_EXTRACTION_PROMPT},
                {"role": "user", "content": answer},
            ],
            backend=ModelBackend.REASONING,
        )
        return result.claims
    except Exception as exc:
        logger.warning("Claim extraction failed: %s", exc)
        return []


async def judge_faithfulness(
    claims: list[str],
    evidence: list[str],
) -> list[ClaimFaithfulness]:
    """Judge whether each claim is supported by the evidence.

    Returns all UNCLEAR on failure (conservative — faithfulness = 1.0).
    """
    evidence_text = "\n".join(f"- {e}" for e in evidence)
    claims_text = "\n".join(f"- {c}" for c in claims)

    try:
        result: _FaithfulnessVerdict = await get_structured_output(
            response_model=_FaithfulnessVerdict,
            messages=[
                {"role": "system", "content": _FAITHFULNESS_PROMPT},
                {
                    "role": "user",
                    "content": f"Evidence:\n{evidence_text}\n\nClaims to judge:\n{claims_text}",
                },
            ],
            backend=ModelBackend.REASONING,
        )
        return [
            ClaimFaithfulness(
                claim=j.claim,
                judgment=j.judgment,
                evidence_snippet=j.evidence_snippet,
            )
            for j in result.judgments
        ]
    except Exception as exc:
        logger.warning("Faithfulness judgment failed: %s", exc)
        return [
            ClaimFaithfulness(claim=c, judgment=FaithfulnessJudgment.UNCLEAR)
            for c in claims
        ]


async def judge_domain_accuracy(
    answer: str,
    ground_truth: str,
    key_facts: list[str],
) -> float:
    """Score 0.0-1.0: does the answer contain the key facts?

    Returns 0.0 on failure (conservative — assumes wrong).
    """
    facts_text = "\n".join(f"- {f}" for f in key_facts)

    try:
        result: _AccuracyScore = await get_structured_output(
            response_model=_AccuracyScore,
            messages=[
                {"role": "system", "content": _ACCURACY_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Ground truth:\n{ground_truth}\n\n"
                        f"Key facts to check:\n{facts_text}\n\n"
                        f"Answer to evaluate:\n{answer}"
                    ),
                },
            ],
            backend=ModelBackend.REASONING,
        )
        return result.score
    except Exception as exc:
        logger.warning("Domain accuracy judgment failed: %s", exc)
        return 0.0
```

**Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/unit/test_eval_judge.py -v
```

Expected: All ~8 tests pass.

**Step 5: Commit**

```bash
git add code/shukketsu/evals/judge.py tests/unit/test_eval_judge.py
git commit -m "feat(evals): add LLM-as-judge module for claim extraction, faithfulness, and accuracy"
```

---

## Task 5: Eval Dataset — 30 Curated Questions

**Files:**
- Create: `code/shukketsu/evals/datasets/phase2_questions.json`

**Step 1: Create the datasets directory**

```bash
mkdir -p code/shukketsu/evals/datasets
```

**Step 2: Write the eval dataset**

Create `code/shukketsu/evals/datasets/phase2_questions.json` containing all 30 questions from the design doc. Each question is a JSON object with fields: `id`, `question`, `complexity`, `category`, `ground_truth`, `expected_tools` (list of strings), `key_facts` (list of strings), `spec` (string).

Copy the full content from the design doc (Section "Component 5: Eval Dataset", lines 374-432). Here is the structure — use the exact questions, ground truths, expected tools, and key facts from the design doc tables:

```json
[
  {
    "id": "q01",
    "question": "How much energy does Sinister Strike cost?",
    "complexity": "TRIVIAL",
    "category": "RETRIEVAL",
    "ground_truth": "Sinister Strike costs 45 energy at max rank. With 2/2 Improved Sinister Strike it costs 40 energy, but most PvE builds skip this talent.",
    "expected_tools": [],
    "key_facts": ["45 energy", "40 with Improved Sinister Strike"],
    "spec": "combat"
  },
  ...all 30 questions...
]
```

Fill in all 30 questions from the design doc. For MODERATE questions, `expected_tools` is `["rag_search"]`. For COMPLEX questions, `expected_tools` comes from the design doc table (e.g., `["rag_search", "graph_search"]` or just `["rag_search"]`).

**Step 3: Validate JSON is well-formed**

```bash
python3 -c "import json; data = json.load(open('code/shukketsu/evals/datasets/phase2_questions.json')); print(f'{len(data)} questions loaded')"
```

Expected: `30 questions loaded`

**Step 4: Commit**

```bash
git add code/shukketsu/evals/datasets/phase2_questions.json
git commit -m "feat(evals): add 30-question eval dataset for phase gate"
```

---

## Task 6: Seed Content for Integration Tests

**Files:**
- Create: `code/shukketsu/evals/datasets/seed_content/combat-basics.md`
- Create: `code/shukketsu/evals/datasets/seed_content/assassination-basics.md`
- Create: `code/shukketsu/evals/datasets/seed_content/subtlety-basics.md`

**Step 1: Create seed content directory**

```bash
mkdir -p code/shukketsu/evals/datasets/seed_content
```

**Step 2: Write combat-basics.md (~400 words)**

Create `code/shukketsu/evals/datasets/seed_content/combat-basics.md` covering the topics from the design doc (Section "Component 6: Seed Content", line 449-456):

- Combat Swords overview, stat priority (expertise > hit > agi > haste > AP > crit)
- Hit cap mechanics: 9% / 142 rating total, 4% / 64 rating with Precision talent
- DW penalty only on white hits
- Rotation: SnD uptime 100% > Rupture at 5 CP > SS filler
- Cooldown stacking: BF + AR + Haste Potion
- Key talents: Combat Potency (20% chance on OH hit for 15 energy), Surprise Attacks, Blade Flurry (2 min CD, 15 sec duration, 25 energy), Adrenaline Rush
- Phase 1 BiS trinkets: Dragonspine Trophy from Gruul (+325 haste proc, ~1 PPM), Bloodlust Brooch from badges
- Weapon speed importance: fast OH for Combat Potency (1.4s > 1.8s)
- Mongoose enchant: +120 agility and +30 haste rating for 15 seconds, ~1 PPM, dual stacks
- Consumables: Haste Potion, Flask of Relentless Assault, Warp Burger, Adamantite Weightstone
- Expertise soft cap: 26 expertise skill (~103 rating), Weapon Expertise talent gives 10 free
- Slice and Dice: increases attack speed by 30%
- Base energy regen: 20 per 2 seconds (10 per second)
- Key raid buffs: Windfury Totem (20% extra attack), Blessing of Might (+220 AP), Leader of the Pack (+5% crit)
- Expose Armor vs Sunder Armor: Improved EA 3075 vs Sunder 5 stacks 2600, mutually exclusive

Ensure every key fact from the TRIVIAL and MODERATE questions that reference combat or general topics is present in this file.

**Step 3: Write assassination-basics.md (~400 words)**

Create `code/shukketsu/evals/datasets/seed_content/assassination-basics.md` covering (line 459-465):

- Mutilate spec overview, weapon requirement (daggers both hands)
- Energy cost (60), combo points (2 base, up to 4 with double crit + Seal Fate)
- Poison setup: Instant Poison MH / Deadly Poison OH; or DP OH with Windfury Totem on MH
- Deadly Poison mechanics: 30% proc rate, 5 stacks, 180 Nature over 12s
- Envenom: consumes DP stacks (1 per CP), Nature damage, AP scaling, ignores armor
- Key talents: Find Weakness, Seal Fate, Cold Blood (3 min CD, guarantees crit), Vile Poisons (+20%)
- Competitive timing: Phase 3+ with T6 gear, consistently trails combat but gap narrows
- Stat priority differences from combat (hit less valuable beyond cap, no Combat Potency)

**Step 4: Write subtlety-basics.md (~300 words)**

Create `code/shukketsu/evals/datasets/seed_content/subtlety-basics.md` covering (line 467-473):

- Subtlety PvE niche overview
- Hemorrhage: 35 energy, 110% weapon damage, +42 physical debuff x10 charges
- Why lower DPS: no AR/BF, Hemo < SS damage per energy
- Hemo vs Sinister Strike comparison: Hemo 35 energy/110% weapon damage vs SS 45 energy/weapon+98
- Shadowstep: 25 yd teleport, +20% damage buff
- Preparation: resets CDs
- Key talents: Serrated Blades, Sinister Calling, Deadliness
- Rotation: SnD > Rupture > Hemo > Evis
- When viable: T4/T5 where DPS checks are lenient

**Step 5: Commit**

```bash
git add code/shukketsu/evals/datasets/seed_content/
git commit -m "feat(evals): add seed content files for integration test fixtures"
```

---

## Task 7: Eval Runner — Phase Gate Runner

**Files:**
- Create: `code/shukketsu/evals/phase2_gate.py`

This is the runner module that loads the dataset, runs questions through the system, and produces a PhaseGateReport. It calls the judge module and metrics module. This runs against real Ollama and a real database, so it is NOT unit tested — it's invoked by the tier 3 integration test. We create the file here; it's tested in Task 9.

**Step 1: Create phase2_gate.py**

```python
"""Phase 2 gate evaluation runner.

Loads the eval dataset, runs each question through the live system,
judges results via LLM, and produces a PhaseGateReport.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from code.shukketsu.agents.factory import AgentFactory
from code.shukketsu.agents.tasks import AgentRole, AgentTask, ResearchTask
from code.shukketsu.db.connection import get_connection, init_db
from code.shukketsu.evals.judge import extract_claims, judge_domain_accuracy, judge_faithfulness
from code.shukketsu.evals.metrics import (
    ClaimFaithfulness,
    EvalQuestionResult,
    PhaseGateReport,
    compute_faithfulness,
    compute_phase_gate,
    compute_trajectory_precision,
)
from code.shukketsu.ingest.embedder import get_embedder
from code.shukketsu.routing.router import classify_query
from code.shukketsu.tools.knowledge.search import RagSearchTool
from code.shukketsu.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_DATASET_PATH = Path(__file__).parent / "datasets" / "phase2_questions.json"


def _load_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the eval question dataset."""
    p = path or _DATASET_PATH
    with p.open() as f:
        return json.load(f)


async def run_phase_gate(
    db_path: Path | None = None,
    dataset_path: Path | None = None,
) -> PhaseGateReport:
    """Run the full phase gate evaluation.

    Loads the dataset, creates agents, runs each question through
    the live system, judges results, and produces a PhaseGateReport.
    """
    from code.shukketsu import config
    from code.shukketsu.knowledge.manager import KnowledgeManager
    from code.shukketsu.tools.knowledge.graph_search import GraphSearchTool

    dataset = _load_dataset(dataset_path)

    conn = get_connection(db_path) if db_path else get_connection()
    init_db(conn)
    embedder = get_embedder()

    registry = ToolRegistry()
    registry.register(RagSearchTool(conn=conn, embed_fn=embedder.embed_query))
    registry.register(GraphSearchTool(conn=conn))

    factory = AgentFactory()
    km = KnowledgeManager(conn, config.WIKI_PATH)

    researcher = factory.create(AgentRole.RESEARCHER, tool_registry=registry)
    orchestrator = factory.create(
        AgentRole.ORCHESTRATOR,
        tool_registry=registry,
        factory=factory,
        knowledge_manager=km,
    )

    results: list[EvalQuestionResult] = []

    for q in dataset:
        try:
            result = await _eval_question(q, researcher, orchestrator)
            results.append(result)
            logger.info(
                "  %s  faith=%.2f  traj=%.2f  acc=%.2f  [%s]",
                q["id"],
                result.faithfulness,
                result.trajectory_precision,
                result.domain_accuracy,
                q["complexity"],
            )
        except Exception as exc:
            logger.warning("Question %s failed: %s", q["id"], exc)
            results.append(
                EvalQuestionResult(
                    question_id=q["id"],
                    faithfulness=0.0,
                    trajectory_precision=0.0,
                    domain_accuracy=0.0,
                    claims=[],
                    tool_calls=[],
                )
            )

    report = compute_phase_gate(results)

    # Soft wiki coverage check
    try:
        articles = km.list_articles()
        coverage: dict[str, int] = {}
        for a in articles:
            spec = a.spec
            coverage[spec] = coverage.get(spec, 0) + 1
        report = report.model_copy(update={"wiki_coverage": coverage})
    except Exception:
        pass

    return report


async def _eval_question(
    q: dict[str, Any],
    researcher: Any,
    orchestrator: Any,
) -> EvalQuestionResult:
    """Evaluate a single question."""
    complexity = q["complexity"]
    question = q["question"]

    # Route and execute
    if complexity == "TRIVIAL":
        decision = await classify_query(question)
        if decision.direct_answer:
            answer = decision.direct_answer
            tool_calls: list[str] = []
            evidence: list[str] = []
        else:
            result = await researcher.execute(ResearchTask(query=question))
            answer = result.output
            tool_calls = [t.tool_name for t in result.trajectory]
            evidence = result.evidence
    elif complexity == "MODERATE":
        result = await researcher.execute(ResearchTask(query=question))
        answer = result.output
        tool_calls = [t.tool_name for t in result.trajectory]
        evidence = result.evidence
    else:
        result = await orchestrator.execute(AgentTask(query=question))
        answer = result.output
        tool_calls = [t.tool_name for t in result.trajectory]
        evidence = result.evidence

    # Judge
    claims_text = await extract_claims(answer)
    claim_judgments = await judge_faithfulness(claims_text, evidence) if claims_text else []
    accuracy = await judge_domain_accuracy(answer, q["ground_truth"], q["key_facts"])

    # Score
    faithfulness = compute_faithfulness(claim_judgments)
    traj_precision = compute_trajectory_precision(tool_calls, q["expected_tools"])

    return EvalQuestionResult(
        question_id=q["id"],
        faithfulness=faithfulness,
        trajectory_precision=traj_precision,
        domain_accuracy=accuracy,
        claims=claim_judgments,
        tool_calls=tool_calls,
    )


def _print_report(report: PhaseGateReport) -> None:
    """Print a human-readable report to stdout."""
    print("\nPhase 2 Gate Evaluation")
    print("=" * 50)
    print(f"Questions evaluated: {len(report.question_results)}")
    print()

    def _status(val: float, threshold: float) -> str:
        return "PASS" if val >= threshold else "FAIL"

    from code.shukketsu.evals.metrics import (
        ACCURACY_THRESHOLD,
        FAITHFULNESS_THRESHOLD,
        TRAJECTORY_THRESHOLD,
    )

    print(f"RAG Faithfulness:        {report.avg_faithfulness:.2f}  (threshold: {FAITHFULNESS_THRESHOLD})  {_status(report.avg_faithfulness, FAITHFULNESS_THRESHOLD)}")
    print(f"Trajectory Precision:    {report.avg_trajectory_precision:.2f}  (threshold: {TRAJECTORY_THRESHOLD})  {_status(report.avg_trajectory_precision, TRAJECTORY_THRESHOLD)}")
    print(f"Domain Accuracy:         {report.avg_domain_accuracy:.2f}  (threshold: {ACCURACY_THRESHOLD})  {_status(report.avg_domain_accuracy, ACCURACY_THRESHOLD)}")
    print()

    if report.wiki_coverage:
        print("Wiki Coverage (informational):")
        for spec, count in sorted(report.wiki_coverage.items()):
            print(f"  {spec:15s} {count} article(s)")
        print()

    print("Per-question breakdown:")
    for qr in report.question_results:
        print(f"  {qr.question_id}  faith={qr.faithfulness:.2f}  traj={qr.trajectory_precision:.2f}  acc={qr.domain_accuracy:.2f}")

    print()
    status = "PASSED" if report.passed else "FAILED"
    print(f"Result: {status}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = asyncio.run(run_phase_gate())
    _print_report(report)
    sys.exit(0 if report.passed else 1)
```

**Step 2: Verify module imports cleanly**

```bash
python3 -c "from code.shukketsu.evals.phase2_gate import run_phase_gate; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add code/shukketsu/evals/phase2_gate.py
git commit -m "feat(evals): add phase gate evaluation runner"
```

---

## Task 8: Wiring Smoke Tests (Tier 1 — CI-Safe)

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/conftest.py`
- Create: `tests/integration/test_wiring.py`

These tests use mocked LLM calls and verify the composition of the multi-agent system. No `@pytest.mark.integration` needed — they run with the unit suite.

**Step 1: Create integration test infrastructure**

Create `tests/integration/__init__.py` (empty file).

Create `tests/integration/conftest.py`:

```python
"""Integration test fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def seed_content_dir() -> Path:
    """Path to the eval seed content directory."""
    return Path(__file__).parent.parent.parent / "code" / "shukketsu" / "evals" / "datasets" / "seed_content"
```

**Step 2: Write wiring smoke tests**

Create `tests/integration/test_wiring.py`:

```python
"""Tier 1: Wiring smoke tests — all LLM calls mocked, CI-safe."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code.shukketsu.agents.factory import AgentFactory
from code.shukketsu.agents.tasks import (
    AgentResult,
    AgentRole,
    AgentTask,
    OrchestratorPlan,
    ResearchResult,
    ResearchTask,
    SubTask,
    TaskStatus,
)
from code.shukketsu.llm.schemas import ActionType, AgentStep, ToolCall
from code.shukketsu.tools.registry import ToolRegistry
from code.shukketsu.tools.schemas import Tool


class _DummyTool(Tool):
    name = "rag_search"
    description = "Search knowledge base."
    parameters_schema = {"query": {"type": "string", "description": "Query"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        return "Found result about rogues."


class _GraphTool(Tool):
    name = "graph_search"
    description = "Search knowledge graph."
    parameters_schema = {"entity": {"type": "string", "description": "Entity"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        return "Entity: combat swords -> has_stat -> expertise"


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_DummyTool())
    reg.register(_GraphTool())
    return reg


def _final(answer: str) -> AgentStep:
    return AgentStep(reasoning="Done.", action=ActionType.FINAL_ANSWER, answer=answer)


def _call(name: str, inp: dict[str, Any]) -> AgentStep:
    return AgentStep(
        reasoning="Need info.",
        action=ActionType.TOOL_CALL,
        tool_call=ToolCall(thought="go", tool_name=name, tool_input=inp),
    )


class TestFactoryCreatesAllRoles:
    def test_researcher(self) -> None:
        from code.shukketsu.agents.researcher import Researcher

        agent = AgentFactory().create(AgentRole.RESEARCHER)
        assert isinstance(agent, Researcher)

    def test_writer(self) -> None:
        from code.shukketsu.agents.writer import Writer

        agent = AgentFactory().create(AgentRole.WRITER, knowledge_manager=MagicMock())
        assert isinstance(agent, Writer)

    def test_editor(self) -> None:
        from code.shukketsu.agents.editor import Editor

        agent = AgentFactory().create(AgentRole.EDITOR, knowledge_manager=MagicMock())
        assert isinstance(agent, Editor)

    def test_orchestrator(self) -> None:
        from code.shukketsu.agents.orchestrator import Orchestrator

        factory = AgentFactory()
        agent = factory.create(AgentRole.ORCHESTRATOR, factory=factory)
        assert isinstance(agent, Orchestrator)


class TestFactoryToolConfigs:
    def test_researcher_accepts_tool_registry(self) -> None:
        reg = _registry()
        agent = AgentFactory().create(AgentRole.RESEARCHER, tool_registry=reg)
        assert agent.tool_registry is reg
        assert "rag_search" in [t.name for t in agent.tool_registry.list_tools()]

    def test_editor_accepts_tool_registry(self) -> None:
        reg = _registry()
        agent = AgentFactory().create(
            AgentRole.EDITOR, tool_registry=reg, knowledge_manager=MagicMock()
        )
        assert agent.tool_registry is reg

    def test_writer_receives_knowledge_manager(self) -> None:
        km = MagicMock()
        agent = AgentFactory().create(AgentRole.WRITER, knowledge_manager=km)
        assert agent._km is km

    def test_orchestrator_receives_factory_and_km(self) -> None:
        factory = AgentFactory()
        km = MagicMock()
        agent = factory.create(
            AgentRole.ORCHESTRATOR,
            factory=factory,
            knowledge_manager=km,
        )
        assert agent._factory is factory
        assert agent._km is km


class TestChatHandlerWiring:
    @patch("code.shukketsu.web.routers.chat.classify_query")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    async def test_moderate_routes_to_researcher(
        self, mock_agents: AsyncMock, mock_classify: AsyncMock
    ) -> None:
        from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity
        from code.shukketsu.web.routers.chat import _agent_response

        mock_classify.return_value = RoutingDecision(
            complexity=TaskComplexity.MODERATE,
            category=TaskCategory.RETRIEVAL,
            needs_tools=True,
            suggested_agent="researcher",
        )

        mock_researcher = AsyncMock()
        mock_researcher.execute.return_value = ResearchResult(
            task_id="t1",
            agent_role=AgentRole.RESEARCHER,
            status=TaskStatus.SUCCESS,
            output="The hit cap is 142.",
        )
        mock_agents.return_value = (mock_researcher, AsyncMock())

        ws = AsyncMock()
        from code.shukketsu.web.routers.chat import ChatSession

        session = ChatSession()
        await _agent_response(ws, session, "What is the hit cap?")

        mock_researcher.execute.assert_called_once()
        # Verify the done message contains the answer
        done_calls = [c for c in ws.send_json.call_args_list if c.args[0].get("type") == "done"]
        assert len(done_calls) == 1
        assert "142" in done_calls[0].args[0]["content"]

    @patch("code.shukketsu.web.routers.chat.classify_query")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    async def test_complex_routes_to_orchestrator(
        self, mock_agents: AsyncMock, mock_classify: AsyncMock
    ) -> None:
        from code.shukketsu.agents.tasks import OrchestratorResult
        from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity

        mock_classify.return_value = RoutingDecision(
            complexity=TaskComplexity.COMPLEX,
            category=TaskCategory.ANALYSIS,
            needs_tools=True,
            suggested_agent="orchestrator",
        )

        mock_orch = AsyncMock()
        mock_orch.execute.return_value = OrchestratorResult(
            task_id="t2",
            agent_role=AgentRole.ORCHESTRATOR,
            status=TaskStatus.SUCCESS,
            output="Comparison complete.",
            plan=OrchestratorPlan(reasoning="plan", subtasks=[]),
        )
        mock_agents.return_value = (AsyncMock(), mock_orch)

        ws = AsyncMock()
        from code.shukketsu.web.routers.chat import ChatSession

        session = ChatSession()
        await _agent_response(ws, session, "Compare combat vs mutilate")

        mock_orch.execute.assert_called_once()


class TestResearcherResultTrajectory:
    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_researcher_result_includes_trajectory(
        self, mock_loop: AsyncMock, mock_struct: AsyncMock
    ) -> None:
        from code.shukketsu.agents.researcher import Researcher, StructuredFindings
        from code.shukketsu.agents.tasks import Finding

        mock_loop.side_effect = [
            _call("rag_search", {"query": "hit cap"}),
            _final("142 rating"),
        ]
        mock_struct.return_value = StructuredFindings(
            findings=[Finding(claim="Hit cap is 142", confidence=0.9)],
            sufficient=True,
        )

        researcher = Researcher(tool_registry=_registry(), role=AgentRole.RESEARCHER)
        result = await researcher.execute(ResearchTask(query="hit cap"))

        assert len(result.trajectory) == 1
        assert result.trajectory[0].tool_name == "rag_search"


class TestErrorPropagation:
    @patch("code.shukketsu.web.routers.chat.classify_query")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    async def test_shukketsu_error_surfaces_as_ws_error(
        self, mock_agents: AsyncMock, mock_classify: AsyncMock
    ) -> None:
        from code.shukketsu.resilience.errors import LLMUnavailableError
        from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity
        from code.shukketsu.web.routers.chat import ChatSession, _agent_response

        mock_classify.return_value = RoutingDecision(
            complexity=TaskComplexity.MODERATE,
            category=TaskCategory.RETRIEVAL,
            needs_tools=True,
            suggested_agent="researcher",
        )
        mock_researcher = AsyncMock()
        mock_researcher.execute.side_effect = LLMUnavailableError("Server down")
        mock_agents.return_value = (mock_researcher, AsyncMock())

        ws = AsyncMock()
        session = ChatSession()
        await _agent_response(ws, session, "test")

        error_calls = [c for c in ws.send_json.call_args_list if c.args[0].get("type") == "error"]
        assert len(error_calls) == 1
        assert "Server down" in error_calls[0].args[0]["content"]

    @patch("code.shukketsu.web.routers.chat.classify_query")
    @patch("code.shukketsu.web.routers.chat._get_agents")
    async def test_unexpected_error_surfaces_generic_message(
        self, mock_agents: AsyncMock, mock_classify: AsyncMock
    ) -> None:
        from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity
        from code.shukketsu.web.routers.chat import ChatSession, _agent_response

        mock_classify.return_value = RoutingDecision(
            complexity=TaskComplexity.MODERATE,
            category=TaskCategory.RETRIEVAL,
            needs_tools=True,
            suggested_agent="researcher",
        )
        mock_researcher = AsyncMock()
        mock_researcher.execute.side_effect = RuntimeError("Unexpected")
        mock_agents.return_value = (mock_researcher, AsyncMock())

        ws = AsyncMock()
        session = ChatSession()
        await _agent_response(ws, session, "test")

        error_calls = [c for c in ws.send_json.call_args_list if c.args[0].get("type") == "error"]
        assert len(error_calls) == 1
        assert "unexpected error" in error_calls[0].args[0]["content"].lower()
```

**Step 3: Run tests to verify they pass**

```bash
python3 -m pytest tests/integration/test_wiring.py -v
```

Expected: All ~12 tests pass.

**Step 4: Commit**

```bash
git add tests/integration/__init__.py tests/integration/conftest.py tests/integration/test_wiring.py
git commit -m "test(integration): add tier 1 wiring smoke tests for multi-agent composition"
```

---

## Task 9: Live Agent Tests (Tier 2) + Phase Gate Tests (Tier 3)

**Files:**
- Create: `tests/integration/test_live_agents.py`
- Create: `tests/integration/test_phase2_gate.py`

These are `@pytest.mark.e2e` tests that require Ollama running. They do NOT run in the default `python3 -m pytest tests/unit/` invocation. They're run separately with:

```bash
python3 -m pytest tests/integration/ -m e2e -v
```

**Step 1: Update conftest.py with seeded_db fixture**

Modify `tests/integration/conftest.py` to add the `seeded_db` fixture:

```python
"""Integration test fixtures."""

import sqlite3
from pathlib import Path

import pytest

from code.shukketsu.db.connection import get_connection, init_db


@pytest.fixture
def seed_content_dir() -> Path:
    """Path to the eval seed content directory."""
    return Path(__file__).parent.parent.parent / "code" / "shukketsu" / "evals" / "datasets" / "seed_content"


@pytest.fixture
def integration_db(tmp_path: Path) -> sqlite3.Connection:
    """Fresh database for integration tests."""
    conn = get_connection(tmp_path / "integration.db")
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
async def seeded_db(integration_db: sqlite3.Connection, seed_content_dir: Path) -> sqlite3.Connection:
    """Ingest seed content files into a test database.

    Requires Ollama running (for nomic-embed-text embeddings).
    """
    from code.shukketsu.ingest.embedder import get_embedder
    from code.shukketsu.ingest.pipeline import IngestPipeline

    embedder = get_embedder()
    pipeline = IngestPipeline(conn=integration_db, embedder=embedder)

    for md_file in sorted(seed_content_dir.glob("*.md")):
        content = md_file.read_text()
        await pipeline.ingest(content, f"file://{md_file.name}", md_file.stem)

    return integration_db
```

**Step 2: Create live agent tests (Tier 2)**

Create `tests/integration/test_live_agents.py`:

```python
"""Tier 2: Live agent tests — real Ollama, real DB with seed data.

Run with: python3 -m pytest tests/integration/test_live_agents.py -m e2e -v
"""

import pytest

from code.shukketsu.agents.factory import AgentFactory
from code.shukketsu.agents.tasks import AgentRole, AgentTask, ResearchTask, TaskStatus
from code.shukketsu.ingest.embedder import get_embedder
from code.shukketsu.routing.router import classify_query
from code.shukketsu.tools.knowledge.graph_search import GraphSearchTool
from code.shukketsu.tools.knowledge.search import RagSearchTool
from code.shukketsu.tools.registry import ToolRegistry

pytestmark = pytest.mark.e2e


def _make_registry(conn):
    embedder = get_embedder()
    reg = ToolRegistry()
    reg.register(RagSearchTool(conn=conn, embed_fn=embedder.embed_query))
    reg.register(GraphSearchTool(conn=conn))
    return reg


class TestResearcherLive:
    async def test_answers_from_seeded_content(self, seeded_db) -> None:
        reg = _make_registry(seeded_db)
        factory = AgentFactory()
        researcher = factory.create(AgentRole.RESEARCHER, tool_registry=reg)

        result = await researcher.execute(
            ResearchTask(query="What is the hit cap for a combat rogue?")
        )

        assert result.status in (TaskStatus.SUCCESS, TaskStatus.PARTIAL)
        assert len(result.output) > 0


class TestRouterLive:
    async def test_classifies_trivial_question(self) -> None:
        decision = await classify_query("How much energy does Sinister Strike cost?")
        assert decision.complexity.value == "trivial"

    async def test_classifies_complex_question(self) -> None:
        decision = await classify_query(
            "Compare combat swords vs mutilate for Phase 1 raiding, "
            "including stat priorities, rotation differences, and gear requirements"
        )
        assert decision.complexity.value == "complex"


class TestOrchestratorLive:
    async def test_decomposes_and_dispatches(self, seeded_db) -> None:
        from code.shukketsu.agents.tasks import OrchestratorResult
        from code.shukketsu.knowledge.manager import KnowledgeManager

        from code.shukketsu import config

        reg = _make_registry(seeded_db)
        factory = AgentFactory()
        km = KnowledgeManager(seeded_db, config.WIKI_PATH)

        orch = factory.create(
            AgentRole.ORCHESTRATOR,
            tool_registry=reg,
            factory=factory,
            knowledge_manager=km,
        )
        result = await orch.execute(
            AgentTask(query="Compare combat swords vs mutilate for Phase 1 raiding")
        )

        assert isinstance(result, OrchestratorResult)
        assert len(result.output) > 0
```

**Step 3: Create phase gate tests (Tier 3)**

Create `tests/integration/test_phase2_gate.py`:

```python
"""Tier 3: Phase gate threshold tests — formal evaluation.

Run with: python3 -m pytest tests/integration/test_phase2_gate.py -m e2e -v
"""

import pytest

from code.shukketsu.evals.metrics import (
    ACCURACY_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    TRAJECTORY_THRESHOLD,
)
from code.shukketsu.evals.phase2_gate import run_phase_gate

pytestmark = pytest.mark.e2e


@pytest.fixture
async def phase_gate_report(seeded_db, tmp_path):
    """Run the full phase gate evaluation once for all tests."""
    report = await run_phase_gate(db_path=tmp_path / "integration.db")
    return report


class TestPhaseGate:
    async def test_faithfulness_above_threshold(self, phase_gate_report) -> None:
        assert phase_gate_report.avg_faithfulness >= FAITHFULNESS_THRESHOLD

    async def test_trajectory_precision_above_threshold(self, phase_gate_report) -> None:
        assert phase_gate_report.avg_trajectory_precision >= TRAJECTORY_THRESHOLD

    async def test_domain_accuracy_above_threshold(self, phase_gate_report) -> None:
        assert phase_gate_report.avg_domain_accuracy >= ACCURACY_THRESHOLD
```

**Step 4: Commit**

```bash
git add tests/integration/conftest.py tests/integration/test_live_agents.py tests/integration/test_phase2_gate.py
git commit -m "test(integration): add tier 2 live agent and tier 3 phase gate tests"
```

---

## Task 10: Lint, Type-Check, Full Test Suite

**Step 1: Run ruff check and format**

```bash
ruff check code/shukketsu/evals/ tests/unit/test_trajectory.py tests/unit/test_eval_metrics.py tests/unit/test_eval_judge.py tests/integration/ --fix
ruff format code/shukketsu/evals/ tests/unit/test_trajectory.py tests/unit/test_eval_metrics.py tests/unit/test_eval_judge.py tests/integration/
```

**Step 2: Run mypy**

```bash
python3 -m mypy code/shukketsu/evals/ code/shukketsu/agents/tasks.py code/shukketsu/agents/base.py code/shukketsu/agents/researcher.py code/shukketsu/agents/orchestrator.py
```

Fix any type errors.

**Step 3: Run full unit test suite**

```bash
python3 -m pytest tests/unit/ -v
```

Expected: ~720+ tests pass (690 existing + ~31 new).

**Step 4: Run tier 1 integration tests**

```bash
python3 -m pytest tests/integration/test_wiring.py -v
```

Expected: ~12 tests pass.

**Step 5: Fix any failures, re-run**

Iterate until all unit + tier 1 tests pass.

**Step 6: Commit any fixes**

```bash
git add -u
git commit -m "fix: resolve lint, type, and test issues for Step 10"
```

**Step 7: Run full pre-commit check**

```bash
ruff check . --fix && ruff format . && python3 -m mypy code/shukketsu/ && python3 -m pytest tests/unit/ tests/integration/test_wiring.py -v
```

Expected: All clean.

---

## Summary

| Task | New Files | Tests Added | Description |
|------|-----------|-------------|-------------|
| 1 | — | ~4 | ToolCallRecord model + AgentResult trajectory field |
| 2 | — | ~6 | Populate trajectory in BaseAgent, Researcher, Orchestrator |
| 3 | `evals/metrics.py` | ~15 | Pure scoring functions (faithfulness, trajectory, phase gate) |
| 4 | `evals/judge.py` | ~8 | LLM-as-judge (claim extraction, faithfulness, accuracy) |
| 5 | `evals/datasets/phase2_questions.json` | 0 | 30-question eval dataset |
| 6 | 3 seed content `.md` files | 0 | Seed content for integration test fixtures |
| 7 | `evals/phase2_gate.py` | 0 | Eval runner (tested by tier 3 integration tests) |
| 8 | `tests/integration/test_wiring.py` | ~12 | Tier 1 wiring smoke tests (CI-safe) |
| 9 | 2 integration test files | ~8 (e2e) | Tier 2 live + Tier 3 phase gate tests |
| 10 | — | 0 | Lint, type-check, full suite verification |

**Starting count: 690 tests**
**Expected after: ~733+ tests** (unit + tier 1 integration)
**e2e tests: ~8** (separate invocation with `-m e2e`)
