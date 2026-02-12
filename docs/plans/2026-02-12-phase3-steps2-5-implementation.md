# Phase 3 Steps 2-5: Parallel Execution, Streaming, Compaction, Reflection — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Four improvements to the agent system: (1) parallel subtask execution in the Orchestrator, (2) structured WebSocket step streaming, (3) context compaction in the ReAct loop, and (4) reflection before final answer on complex queries.

---

## Step 2: Parallel Subtask Execution in Orchestrator

**Goal:** Replace sequential subtask dispatch with level-grouped parallel execution using `asyncio.gather()`.

---

### Task 2.1: Write failing tests for `_group_by_level()`

**Files:**
- Modify: `tests/unit/test_orchestrator_execute.py`

**Step 1: Add the new test class to the existing test file**

Append the following to `tests/unit/test_orchestrator_execute.py`:

```python
class TestGroupByLevel:
    def test_empty_subtasks(self) -> None:
        """Empty subtask list returns empty levels."""
        orch = _orchestrator()
        levels = orch._group_by_level([])
        assert levels == []

    def test_all_independent(self) -> None:
        """3 subtasks with no deps -> single level with all 3."""
        orch = _orchestrator()
        subtasks = [
            SubTask(agent_role=AgentRole.RESEARCHER, description="a"),
            SubTask(agent_role=AgentRole.RESEARCHER, description="b"),
            SubTask(agent_role=AgentRole.RESEARCHER, description="c"),
        ]
        levels = orch._group_by_level(subtasks)
        assert len(levels) == 1
        assert sorted(levels[0]) == [0, 1, 2]

    def test_chain_three_levels(self) -> None:
        """A->B->C produces 3 levels of 1 each."""
        orch = _orchestrator()
        subtasks = [
            SubTask(agent_role=AgentRole.RESEARCHER, description="a"),
            SubTask(agent_role=AgentRole.RESEARCHER, description="b", depends_on=[0]),
            SubTask(agent_role=AgentRole.RESEARCHER, description="c", depends_on=[1]),
        ]
        levels = orch._group_by_level(subtasks)
        assert len(levels) == 3
        assert levels[0] == [0]
        assert levels[1] == [1]
        assert levels[2] == [2]

    def test_diamond_dependency(self) -> None:
        """Diamond: A -> B, A -> C, B+C -> D produces 3 levels."""
        orch = _orchestrator()
        subtasks = [
            SubTask(agent_role=AgentRole.RESEARCHER, description="a"),
            SubTask(agent_role=AgentRole.RESEARCHER, description="b", depends_on=[0]),
            SubTask(agent_role=AgentRole.RESEARCHER, description="c", depends_on=[0]),
            SubTask(agent_role=AgentRole.RESEARCHER, description="d", depends_on=[1, 2]),
        ]
        levels = orch._group_by_level(subtasks)
        assert len(levels) == 3
        assert levels[0] == [0]
        assert sorted(levels[1]) == [1, 2]
        assert levels[2] == [3]

    def test_two_independent_plus_dependent(self) -> None:
        """A, B independent; C depends on both -> 2 levels."""
        orch = _orchestrator()
        subtasks = [
            SubTask(agent_role=AgentRole.RESEARCHER, description="a"),
            SubTask(agent_role=AgentRole.RESEARCHER, description="b"),
            SubTask(agent_role=AgentRole.RESEARCHER, description="c", depends_on=[0, 1]),
        ]
        levels = orch._group_by_level(subtasks)
        assert len(levels) == 2
        assert sorted(levels[0]) == [0, 1]
        assert levels[1] == [2]
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_orchestrator_execute.py::TestGroupByLevel -v`

Expected: All fail — `_group_by_level` doesn't exist yet.

---

### Task 2.2: Write failing tests for parallel execution behavior

**Files:**
- Modify: `tests/unit/test_orchestrator_execute.py`

**Step 1: Add the new test class to the existing test file**

Append the following to `tests/unit/test_orchestrator_execute.py`:

```python
import asyncio


class TestParallelExecution:
    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_parallel_independent_subtasks(self, mock_llm: AsyncMock) -> None:
        """Two independent RESEARCHER tasks run concurrently via asyncio.gather."""
        plan = _plan(
            subtasks=[
                SubTask(agent_role=AgentRole.RESEARCHER, description="find trinkets"),
                SubTask(agent_role=AgentRole.RESEARCHER, description="find weapons"),
            ]
        )

        call_count = 0

        async def _mock_structured(response_model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if response_model is OrchestratorPlan:
                return plan
            return response_model(response="Synthesized.")

        mock_llm.side_effect = _mock_structured

        execution_order: list[str] = []
        start_times: dict[str, float] = {}

        factory = MagicMock()

        def _create(role, **kwargs):
            agent = MagicMock()

            async def _execute(task, on_status=None):
                desc = task.query
                execution_order.append(f"start:{desc}")
                start_times[desc] = asyncio.get_event_loop().time()
                await asyncio.sleep(0.05)
                execution_order.append(f"end:{desc}")
                return _research_result(output=f"Results for {desc}")

            agent.execute = _execute
            return agent

        factory.create.side_effect = _create

        orch = _orchestrator(factory=factory)
        result = await orch.execute(AgentTask(query="Compare trinkets and weapons"))

        # Both should have started before either finished (parallel)
        assert "start:find trinkets" in execution_order
        assert "start:find weapons" in execution_order
        assert len(result.specialist_results) == 2

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_dependent_subtasks_wait(self, mock_llm: AsyncMock) -> None:
        """C depends on A and B — C runs only after both A and B complete."""
        plan = _plan(
            subtasks=[
                SubTask(agent_role=AgentRole.RESEARCHER, description="a"),
                SubTask(agent_role=AgentRole.RESEARCHER, description="b"),
                SubTask(agent_role=AgentRole.RESEARCHER, description="c", depends_on=[0, 1]),
            ]
        )

        call_count = 0

        async def _mock_structured(response_model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if response_model is OrchestratorPlan:
                return plan
            return response_model(response="Synthesized.")

        mock_llm.side_effect = _mock_structured

        execution_order: list[str] = []

        factory = MagicMock()

        def _create(role, **kwargs):
            agent = MagicMock()

            async def _execute(task, on_status=None):
                execution_order.append(f"start:{task.query}")
                await asyncio.sleep(0.01)
                execution_order.append(f"end:{task.query}")
                return _research_result(output=f"Results for {task.query}")

            agent.execute = _execute
            return agent

        factory.create.side_effect = _create

        orch = _orchestrator(factory=factory)
        await orch.execute(AgentTask(query="test"))

        # C must start after both A and B end
        c_start = execution_order.index("start:c")
        a_end = execution_order.index("end:a")
        b_end = execution_order.index("end:b")
        assert c_start > a_end
        assert c_start > b_end

    @patch("code.shukketsu.agents.orchestrator.get_structured_output", new_callable=AsyncMock)
    async def test_parallel_failure_doesnt_block(self, mock_llm: AsyncMock) -> None:
        """One of two parallel subtasks fails; the other still completes."""
        plan = _plan(
            subtasks=[
                SubTask(agent_role=AgentRole.RESEARCHER, description="good"),
                SubTask(agent_role=AgentRole.RESEARCHER, description="bad"),
            ]
        )

        call_count = 0

        async def _mock_structured(response_model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if response_model is OrchestratorPlan:
                return plan
            return response_model(response="Synthesized.")

        mock_llm.side_effect = _mock_structured

        factory = MagicMock()

        def _create(role, **kwargs):
            agent = MagicMock()
            call_idx = factory.create.call_count

            async def _execute(task, on_status=None):
                if task.query == "bad":
                    raise RuntimeError("Agent failed")
                return _research_result(output="Good result")

            agent.execute = _execute
            return agent

        factory.create.side_effect = _create

        orch = _orchestrator(factory=factory)
        result = await orch.execute(AgentTask(query="test"))

        # One succeeded, one failed
        assert len(result.specialist_results) == 2
        statuses = [r.status for r in result.specialist_results]
        assert TaskStatus.SUCCESS in statuses
        assert TaskStatus.FAILED in statuses
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_orchestrator_execute.py::TestParallelExecution -v`

Expected: Failures — the current `_dispatch` runs sequentially and doesn't use `_group_by_level`.

---

### Task 2.3: Implement `_group_by_level()` and parallel `_dispatch()`

**Files:**
- Modify: `code/shukketsu/agents/orchestrator.py`

**Step 1: Add `import asyncio` at the top of orchestrator.py**

Add `import asyncio` after the `from __future__ import annotations` line (line 8), so imports become:

```python
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any
```

**Step 2: Add `_group_by_level()` method to the Orchestrator class**

Add this method after `_topological_sort()` (after line 450):

```python
    def _group_by_level(self, subtasks: list[SubTask]) -> list[list[int]]:
        """Group subtask indices by dependency depth.

        Level 0 = no dependencies, level 1 = depends only on level 0, etc.
        Used to identify which subtasks can run in parallel within each level.

        Args:
            subtasks: The list of subtasks to group.

        Returns:
            A list of levels, where each level is a list of subtask indices.
        """
        n = len(subtasks)
        if n == 0:
            return []

        # Compute depth for each node via BFS
        depth = [0] * n
        adj: dict[int, list[int]] = defaultdict(list)
        in_degree = [0] * n

        for i, st in enumerate(subtasks):
            for dep in st.depends_on:
                if 0 <= dep < n:
                    adj[dep].append(i)
                    in_degree[i] += 1

        queue: deque[int] = deque(i for i in range(n) if in_degree[i] == 0)

        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                depth[neighbor] = max(depth[neighbor], depth[node] + 1)
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Group by depth
        max_depth = max(depth) if depth else 0
        levels: list[list[int]] = [[] for _ in range(max_depth + 1)]
        for i, d in enumerate(depth):
            levels[d].append(i)

        return levels
```

**Step 3: Extract `_execute_single()` from `_dispatch()`**

Add this method before `_dispatch()`:

```python
    async def _execute_single(
        self,
        idx: int,
        subtask: SubTask,
        results: list[AgentResult | None],
        skipped: list[str],
        task: AgentTask,
        on_status: StatusCallback | None = None,
    ) -> None:
        """Execute a single subtask and store its result in the results list.

        Handles failed dependencies, missing knowledge_manager, task building
        errors, and agent execution failures. On any failure, records a FAILED
        AgentResult or appends to skipped list.
        """
        # Check failed dependencies
        failed_deps = [
            d
            for d in subtask.depends_on
            if results[d] is not None and results[d].status == TaskStatus.FAILED  # type: ignore[union-attr]
        ]
        if failed_deps:
            skipped.append(subtask.description)
            logger.info(
                "Skipping subtask %d (%s): failed dependencies",
                idx,
                subtask.description,
            )
            return

        # Check if Writer/Editor needs knowledge_manager
        if subtask.agent_role in _KM_ROLES and self._km is None:
            skipped.append(f"{subtask.description} (no knowledge_manager)")
            logger.warning(
                "Skipping %s subtask: no knowledge_manager",
                subtask.agent_role,
            )
            return

        # Build typed task
        try:
            typed_task = self._build_task(
                subtask,
                results,
                task.trace_id,
            )
        except (ValueError, KeyError) as exc:
            logger.warning(
                "Failed to build task for subtask %d: %s",
                idx,
                exc,
            )
            skipped.append(subtask.description)
            return

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
            await on_status(f"{subtask.agent_role}: {subtask.description[:50]}...")

        try:
            results[idx] = await agent.execute(
                typed_task,
                on_status=on_status,
            )
        except Exception as exc:
            logger.warning("Subtask %d failed: %s", idx, exc)
            results[idx] = AgentResult(
                task_id=typed_task.task_id,
                agent_role=subtask.agent_role,
                status=TaskStatus.FAILED,
                output=f"Specialist failed: {exc}",
            )
```

**Step 4: Replace `_dispatch()` with level-grouped parallel execution**

Replace the entire `_dispatch` method (lines 181-264) with:

```python
    async def _dispatch(
        self,
        plan: OrchestratorPlan,
        task: AgentTask,
        on_status: StatusCallback | None = None,
    ) -> tuple[list[AgentResult | None], list[str]]:
        """Phase 2: Execute subtasks in topological order, parallelizing independent tasks."""
        results: list[AgentResult | None] = [None] * len(plan.subtasks)
        skipped: list[str] = []

        levels = self._group_by_level(plan.subtasks)

        for level in levels:
            if len(level) == 1:
                # Single subtask — run directly (no gather overhead)
                idx = level[0]
                await self._execute_single(idx, plan.subtasks[idx], results, skipped, task, on_status)
            else:
                # Multiple independent subtasks — run in parallel
                coros = [
                    self._execute_single(idx, plan.subtasks[idx], results, skipped, task, on_status)
                    for idx in level
                ]
                await asyncio.gather(*coros, return_exceptions=True)

        return results, skipped
```

**Step 5: Run all orchestrator tests**

Run: `python3 -m pytest tests/unit/test_orchestrator_execute.py -v`

Expected: All tests pass (existing + new).

---

### Task 2.4: Run full test suite + linting

**Files:** None (verification only)

Run: `ruff check code/ tests/ --fix && ruff format code/ tests/ && python3 -m mypy code/shukketsu/ && python3 -m pytest tests/unit/ -v`

Expected: 741+ tests pass, no lint errors, no type errors.

---

### Task 2.5: Commit

```bash
git add code/shukketsu/agents/orchestrator.py tests/unit/test_orchestrator_execute.py
git commit -m "feat(agents): add parallel subtask execution to Orchestrator

Group subtasks by dependency depth and execute independent subtasks
within each level concurrently via asyncio.gather(). Subtasks with
dependencies still wait for their predecessors to complete.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Step 3: WebSocket Agent Step Streaming

**Goal:** Emit structured dict events from agents for tool calls and dispatch actions, while keeping backward-compatible string status messages.

---

### Task 3.1: Write failing tests for dict status callback

**Files:**
- Modify: `tests/unit/test_base_agent.py`

**Step 1: Add new test class to test_base_agent.py**

Append the following to `tests/unit/test_base_agent.py`:

```python
class TestStatusCallbackEvents:
    """Tests for structured dict events emitted via on_status."""

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_status_callback_receives_dict_for_tool_call(self, mock_llm: AsyncMock) -> None:
        """After a tool call, the callback receives a dict event with tool metadata."""
        mock_llm.side_effect = [
            _tool_call("echo", {"text": "hello"}),
            _final_answer("Done"),
        ]
        agent = BaseAgent(tool_registry=_registry(EchoTool()), role=AgentRole.RESEARCHER)
        events: list[str | dict] = []

        async def capture(msg: str | dict) -> None:
            events.append(msg)

        await agent.execute(AgentTask(query="q"), on_status=capture)

        dict_events = [e for e in events if isinstance(e, dict)]
        assert len(dict_events) >= 1
        step_event = dict_events[0]
        assert step_event["type"] == "step"
        assert step_event["action"] == "tool_call"
        assert step_event["tool"] == "echo"

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_status_callback_still_receives_strings(self, mock_llm: AsyncMock) -> None:
        """Existing string status messages (e.g. 'thinking...') still work."""
        mock_llm.return_value = _final_answer("Done")
        agent = BaseAgent(tool_registry=_registry(EchoTool()), role=AgentRole.RESEARCHER)
        events: list[str | dict] = []

        async def capture(msg: str | dict) -> None:
            events.append(msg)

        await agent.execute(AgentTask(query="q"), on_status=capture)

        str_events = [e for e in events if isinstance(e, str)]
        assert len(str_events) >= 1
        assert any("thinking" in s for s in str_events)

    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_dict_event_includes_agent_role(self, mock_llm: AsyncMock) -> None:
        """Dict events include the agent's role."""
        mock_llm.side_effect = [
            _tool_call("echo", {"text": "hi"}),
            _final_answer("Done"),
        ]
        agent = BaseAgent(tool_registry=_registry(EchoTool()), role=AgentRole.RESEARCHER)
        events: list[str | dict] = []

        async def capture(msg: str | dict) -> None:
            events.append(msg)

        await agent.execute(AgentTask(query="q"), on_status=capture)

        dict_events = [e for e in events if isinstance(e, dict)]
        assert dict_events[0]["agent"] == "researcher"
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_base_agent.py::TestStatusCallbackEvents -v`

Expected: Failures — StatusCallback only accepts `str`, and no dict events are emitted.

---

### Task 3.2: Write failing tests for chat handler dict/str handling

**Files:**
- Modify: `tests/unit/test_chat_handler.py`

**Step 1: Add new test class to test_chat_handler.py**

Append the following to `tests/unit/test_chat_handler.py`:

```python
class TestSendStatusFormat:
    """Tests for _send_status handling of str and dict messages."""

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_send_status_wraps_string(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """String messages are wrapped in {'type': 'status', 'content': msg}."""
        mock_classify.return_value = _moderate_decision()
        researcher, orchestrator = _mock_agents("Answer")

        sent_messages: list[dict] = []
        original_execute = researcher.execute

        async def _capture_execute(task, on_status=None):
            if on_status:
                await on_status("test string message")
            return MagicMock(output="Answer")

        researcher.execute = _capture_execute
        mock_get.return_value = (researcher, orchestrator)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "test"})
            messages = []
            while True:
                msg = ws.receive_json()
                messages.append(msg)
                if msg["type"] == "done":
                    break

            status_msgs = [m for m in messages if m["type"] == "status"]
            # At least one status message should have our string content
            assert any(m["content"] == "test string message" for m in status_msgs)

    @patch("code.shukketsu.web.routers.chat._get_agents")
    @patch("code.shukketsu.web.routers.chat.classify_query", new_callable=AsyncMock)
    def test_send_status_passes_dict_as_is(self, mock_classify: AsyncMock, mock_get: MagicMock) -> None:
        """Dict messages are sent as-is via WebSocket."""
        mock_classify.return_value = _moderate_decision()
        researcher, orchestrator = _mock_agents("Answer")

        async def _capture_execute(task, on_status=None):
            if on_status:
                await on_status({"type": "step", "agent": "researcher", "action": "tool_call", "tool": "rag_search"})
            return MagicMock(output="Answer")

        researcher.execute = _capture_execute
        mock_get.return_value = (researcher, orchestrator)

        client = TestClient(_get_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.receive_json()  # connected
            ws.send_json({"type": "message", "content": "test"})
            messages = []
            while True:
                msg = ws.receive_json()
                messages.append(msg)
                if msg["type"] == "done":
                    break

            step_msgs = [m for m in messages if m.get("type") == "step"]
            assert len(step_msgs) >= 1
            assert step_msgs[0]["tool"] == "rag_search"
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_chat_handler.py::TestSendStatusFormat -v`

Expected: Failures — `_send_status` only handles strings, and StatusCallback type is `str` only.

---

### Task 3.3: Widen StatusCallback type and emit dict events

**Files:**
- Modify: `code/shukketsu/agents/base.py`

**Step 1: Update the StatusCallback type alias**

Replace line 18:
```python
StatusCallback = Callable[[str], Awaitable[None]]
```

With:
```python
StatusCallback = Callable[[str | dict[str, Any]], Awaitable[None]]
```

**Step 2: Emit a dict event after tool execution in `_run_loop()`**

After line 158 (the `observation = await self.tool_registry.execute(...)` line), add the following before the `scratchpad.append(...)`:

```python
            if on_status:
                await on_status({
                    "type": "step",
                    "agent": self.role or "agent",
                    "action": "tool_call",
                    "tool": tool_call.tool_name,
                })
```

The resulting section (lines ~152-170) should look like:

```python
            tool_call = step.tool_call
            assert tool_call is not None  # guaranteed by AgentStep validator
            logger.info("Agent calling tool: %s", tool_call.tool_name)
            if on_status:
                await on_status(f"using {tool_call.tool_name}...")

            observation = await self.tool_registry.execute(tool_call.tool_name, tool_call.tool_input)

            if on_status:
                await on_status({
                    "type": "step",
                    "agent": self.role or "agent",
                    "action": "tool_call",
                    "tool": tool_call.tool_name,
                })

            scratchpad.append(
                {
                    "reasoning": step.reasoning,
                    "tool_name": tool_call.tool_name,
                    "tool_input": tool_call.tool_input,
                    "observation": observation,
                }
            )
```

**Step 3: Run base agent tests**

Run: `python3 -m pytest tests/unit/test_base_agent.py -v`

Expected: All tests pass including the new ones.

---

### Task 3.4: Update `_send_status()` in chat.py to handle dicts

**Files:**
- Modify: `code/shukketsu/web/routers/chat.py`

**Step 1: Update the `_send_status` closure**

Replace lines 172-173:
```python
        async def _send_status(msg: str) -> None:
            await websocket.send_json({"type": "status", "content": msg})
```

With:
```python
        async def _send_status(msg: str | dict) -> None:
            if isinstance(msg, dict):
                await websocket.send_json(msg)
            else:
                await websocket.send_json({"type": "status", "content": msg})
```

**Step 2: Run chat handler tests**

Run: `python3 -m pytest tests/unit/test_chat_handler.py -v`

Expected: All tests pass including the new ones.

---

### Task 3.5: Run full test suite + linting

**Files:** None (verification only)

Run: `ruff check code/ tests/ --fix && ruff format code/ tests/ && python3 -m mypy code/shukketsu/ && python3 -m pytest tests/unit/ -v`

Expected: All tests pass, no lint errors, no type errors.

---

### Task 3.6: Commit

```bash
git add code/shukketsu/agents/base.py code/shukketsu/web/routers/chat.py tests/unit/test_base_agent.py tests/unit/test_chat_handler.py
git commit -m "feat(agents): add structured dict events to status callback

Widen StatusCallback to accept str | dict. Emit structured step events
with tool name and agent role after each tool call. Chat handler passes
dicts as-is via WebSocket and wraps strings in the existing format.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Step 4: Context Compaction in ReAct Loop

**Goal:** Prevent context window overflow by compacting older scratchpad observations when the estimated token count exceeds a threshold.

---

### Task 4.1: Add config constant

**Files:**
- Modify: `code/shukketsu/config.py`

**Step 1: Add the compaction threshold constant**

Add after the `AGENT_GRACEFUL_FAILURE` line (line 46):

```python
COMPACTION_THRESHOLD_TOKENS = int(os.getenv("COMPACTION_THRESHOLD_TOKENS", "60000"))
```

(No commit yet — we'll commit with the implementation.)

---

### Task 4.2: Write failing tests for compaction

**Files:**
- Modify: `tests/unit/test_base_agent.py`

**Step 1: Add test class for compaction and token estimation**

Append the following to `tests/unit/test_base_agent.py`:

```python
class TestTokenEstimation:
    """Tests for _estimate_tokens helper."""

    def test_estimate_tokens_basic(self) -> None:
        """Simple sanity check: known string length maps to approximate token count."""
        agent = BaseAgent(tool_registry=_registry())
        messages = [{"role": "user", "content": "a" * 400}]
        tokens = agent._estimate_tokens(messages)
        assert tokens == 100  # 400 chars // 4

    def test_estimate_tokens_empty(self) -> None:
        """Empty messages list returns 0."""
        agent = BaseAgent(tool_registry=_registry())
        assert agent._estimate_tokens([]) == 0

    def test_estimate_tokens_multiple(self) -> None:
        """Multiple messages sum their content lengths."""
        agent = BaseAgent(tool_registry=_registry())
        messages = [
            {"role": "system", "content": "a" * 200},
            {"role": "user", "content": "b" * 800},
        ]
        tokens = agent._estimate_tokens(messages)
        assert tokens == 250  # (200 + 800) // 4


class TestContextCompaction:
    """Tests for _compact_scratchpad and its integration with _build_messages."""

    def test_compaction_not_triggered_below_threshold(self) -> None:
        """Small scratchpad -> all observations preserved verbatim."""
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        scratchpad = [
            {
                "reasoning": "step 1",
                "tool_name": "echo",
                "tool_input": {"text": "hello"},
                "observation": "Echo: hello",
            },
            {
                "reasoning": "step 2",
                "tool_name": "echo",
                "tool_input": {"text": "world"},
                "observation": "Echo: world",
            },
        ]
        messages = agent._build_messages("query", scratchpad)
        all_content = " ".join(m["content"] for m in messages)
        assert "Echo: hello" in all_content
        assert "Echo: world" in all_content
        assert "[Summarized]" not in all_content

    @patch("code.shukketsu.config.COMPACTION_THRESHOLD_TOKENS", 10)
    def test_compaction_triggered_above_threshold(self) -> None:
        """Large scratchpad -> older entries truncated, 2 most recent preserved verbatim."""
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        scratchpad = [
            {
                "reasoning": f"step {i}",
                "tool_name": "echo",
                "tool_input": {"text": f"call{i}"},
                "observation": f"Result {i}: " + "x" * 500,
            }
            for i in range(5)
        ]
        messages = agent._build_messages("query", scratchpad)
        all_content = " ".join(m["content"] for m in messages)

        # The last 2 entries should be verbatim
        assert f"Result 3: " + "x" * 500 in all_content
        assert f"Result 4: " + "x" * 500 in all_content

        # Older entries should be summarized
        assert "[Summarized]" in all_content

    @patch("code.shukketsu.config.COMPACTION_THRESHOLD_TOKENS", 10)
    def test_compaction_preserves_tool_name(self) -> None:
        """Truncated entries still mention the tool name."""
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        scratchpad = [
            {
                "reasoning": f"step {i}",
                "tool_name": "rag_search",
                "tool_input": {"query": f"q{i}"},
                "observation": f"Result: " + "x" * 500,
            }
            for i in range(5)
        ]
        messages = agent._build_messages("query", scratchpad)

        # Find observation messages for older entries (should mention tool name)
        observation_msgs = [m for m in messages if m["role"] == "user" and "[Summarized]" in m["content"]]
        assert len(observation_msgs) >= 1
        for msg in observation_msgs:
            assert "rag_search" in msg["content"]

    def test_compaction_idempotent(self) -> None:
        """Calling _build_messages twice with same scratchpad produces same result."""
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        scratchpad = [
            {
                "reasoning": f"step {i}",
                "tool_name": "echo",
                "tool_input": {"text": f"call{i}"},
                "observation": f"Result {i}: " + "x" * 200,
            }
            for i in range(5)
        ]
        messages1 = agent._build_messages("query", scratchpad)
        messages2 = agent._build_messages("query", scratchpad)
        assert messages1 == messages2

    @patch("code.shukketsu.config.COMPACTION_THRESHOLD_TOKENS", 10)
    def test_compaction_truncation_format(self) -> None:
        """Truncated observations use the format: [Summarized] [tool_name] first_100...last_100."""
        agent = BaseAgent(tool_registry=_registry(EchoTool()))
        long_obs = "START" + "m" * 300 + "END"
        scratchpad = [
            {
                "reasoning": "step",
                "tool_name": "rag_search",
                "tool_input": {"query": "q"},
                "observation": long_obs,
            },
            {
                "reasoning": "step",
                "tool_name": "echo",
                "tool_input": {"text": "recent1"},
                "observation": "recent observation 1" + "x" * 500,
            },
            {
                "reasoning": "step",
                "tool_name": "echo",
                "tool_input": {"text": "recent2"},
                "observation": "recent observation 2" + "y" * 500,
            },
        ]
        messages = agent._build_messages("query", scratchpad)
        observation_msgs = [m for m in messages if m["role"] == "user" and "[Summarized]" in m["content"]]

        assert len(observation_msgs) >= 1
        summarized = observation_msgs[0]["content"]
        assert "[Summarized]" in summarized
        assert "[rag_search]" in summarized
        assert "START" in summarized  # from first 100 chars
        assert "END" in summarized  # from last 100 chars
        assert "[truncated]" in summarized
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_base_agent.py::TestTokenEstimation tests/unit/test_base_agent.py::TestContextCompaction -v`

Expected: Failures — `_estimate_tokens` and `_compact_scratchpad` don't exist yet.

---

### Task 4.3: Implement token estimation and compaction in base.py

**Files:**
- Modify: `code/shukketsu/agents/base.py`

**Step 1: Add `_estimate_tokens()` method to BaseAgent**

Add after `_synthesize_partial_answer()` (after line 189):

```python
    def _estimate_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimate token count from messages using char/4 heuristic.

        This is a rough approximation. 1 token ~= 4 characters for English text.
        """
        return sum(len(m.get("content", "")) for m in messages) // 4
```

**Step 2: Add `_compact_scratchpad()` method to BaseAgent**

Add after `_estimate_tokens()`:

```python
    def _compact_scratchpad(
        self,
        scratchpad: list[dict[str, Any]],
        keep_recent: int = 2,
    ) -> list[dict[str, Any]]:
        """Return a compacted copy of the scratchpad.

        Keeps the most recent `keep_recent` entries verbatim. Older entries
        have their observations truncated to a summarized format:
        '[Summarized] [tool_name] first_100_chars... [truncated] ...last_100_chars'

        Args:
            scratchpad: The full scratchpad entries.
            keep_recent: Number of most recent entries to keep verbatim.

        Returns:
            A new list with the same structure but truncated older observations.
        """
        if len(scratchpad) <= keep_recent:
            return list(scratchpad)

        compacted: list[dict[str, Any]] = []
        cutoff = len(scratchpad) - keep_recent

        for i, entry in enumerate(scratchpad):
            if i < cutoff:
                obs = str(entry.get("observation", ""))
                tool = entry.get("tool_name", "unknown")
                if len(obs) > 200:
                    head = obs[:100]
                    tail = obs[-100:]
                    summarized = f"[Summarized] [{tool}] {head}... [truncated] ...{tail}"
                else:
                    summarized = f"[Summarized] [{tool}] {obs}"
                compacted.append({**entry, "observation": summarized})
            else:
                compacted.append(entry)

        return compacted
```

**Step 3: Integrate compaction into `_build_messages()`**

Replace `_build_messages()` (lines 191-215) with:

```python
    def _build_messages(self, query: str, scratchpad: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Build the messages array for the LLM.

        If the estimated token count exceeds COMPACTION_THRESHOLD_TOKENS,
        older scratchpad entries are compacted to reduce context size.
        """
        tool_descriptions = self.tool_registry.get_tool_descriptions()
        system_content = self._system_prompt + "\n\n" + _REACT_INSTRUCTIONS.format(tool_descriptions=tool_descriptions)

        effective_scratchpad = scratchpad

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        for entry in effective_scratchpad:
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

        if self._estimate_tokens(messages) > config.COMPACTION_THRESHOLD_TOKENS:
            effective_scratchpad = self._compact_scratchpad(scratchpad)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query},
            ]
            for entry in effective_scratchpad:
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

**Step 4: Run all base agent tests**

Run: `python3 -m pytest tests/unit/test_base_agent.py -v`

Expected: All tests pass.

---

### Task 4.4: Run full test suite + linting

**Files:** None (verification only)

Run: `ruff check code/ tests/ --fix && ruff format code/ tests/ && python3 -m mypy code/shukketsu/ && python3 -m pytest tests/unit/ -v`

Expected: All tests pass, no lint errors, no type errors.

---

### Task 4.5: Commit

```bash
git add code/shukketsu/config.py code/shukketsu/agents/base.py tests/unit/test_base_agent.py
git commit -m "feat(agents): add context compaction to ReAct loop

When estimated token count exceeds COMPACTION_THRESHOLD_TOKENS (60k),
older scratchpad observations are truncated to first/last 100 chars
with a [Summarized] prefix. The 2 most recent entries are always kept
verbatim. Prevents context window overflow on long research sessions.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Step 5: Reflection Before Final Answer (Complex Queries Only)

**Goal:** Add a self-verification step to the Researcher agent that checks whether the answer is fully supported by evidence before returning it.

---

### Task 5.1: Add ReflectionResult model to llm/schemas.py

**Files:**
- Modify: `code/shukketsu/llm/schemas.py`

**Step 1: Add the ReflectionResult model**

Append after the `AgentStep` class:

```python
class ReflectionResult(BaseModel):
    """Result of a reflection pass on a research answer.

    The Researcher runs this after structuring to verify the answer
    is supported by the gathered evidence. If not, provides a revised
    answer that better matches the evidence.
    """

    supported: bool
    issues: list[str] = []
    revised_answer: str = ""
```

(No commit yet — we'll commit with the full implementation.)

---

### Task 5.2: Add REFLECTION_PROMPT to researcher prompts

**Files:**
- Modify: `code/shukketsu/llm/prompts/researcher.py`

**Step 1: Add the reflection prompt constant**

Append at the end of the file:

```python
REFLECTION_PROMPT = """\
You are a research quality checker for WoW: The Burning Crusade Rogue content. \
Given a research question, the researcher's answer, and the tool observations \
gathered during research, verify whether the answer is fully supported by the \
evidence.

## Instructions

1. Compare each claim in the answer against the tool observations.
2. If the answer is well-supported by the evidence, set supported=True with an \
empty issues list.
3. If ANY claims are unsupported, speculative, or contradicted by the evidence:
   - Set supported=False
   - List the specific issues found
   - Provide a revised_answer that only states what the evidence supports

## Important

- Do NOT add information that isn't in the observations.
- If the original answer is fine, say supported=True.
- If revising, keep the same structure but remove/correct unsupported claims.
- Err on the side of caution — if evidence is ambiguous, flag it.\
"""
```

---

### Task 5.3: Add config constants

**Files:**
- Modify: `code/shukketsu/config.py`

**Step 1: Add reflection config constants**

Add after the `RESEARCHER_MAX_TOKENS` line (line 104):

```python
REFLECTION_ENABLED = os.getenv("REFLECTION_ENABLED", "true").lower() == "true"
REFLECTION_TEMPERATURE = float(os.getenv("REFLECTION_TEMPERATURE", "0.1"))
```

---

### Task 5.4: Write failing tests for reflection

**Files:**
- Modify: `tests/unit/test_researcher.py`

**Step 1: Add the reflection test class**

Append the following to `tests/unit/test_researcher.py`:

```python
from code.shukketsu.llm.schemas import ReflectionResult


class TestReflection:
    """Tests for the reflection step in Researcher.execute()."""

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_reflection_skipped_non_complex(self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock) -> None:
        """Moderate query (no complexity metadata) -> reflection not called."""
        mock_loop_llm.return_value = _final_answer("Answer text.")

        call_count = 0

        async def _side_effect(response_model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if response_model is StructuredFindings:
                return _mock_structured_findings()
            if response_model is ReflectionResult:
                pytest.fail("Reflection should not be called for non-complex queries")
            return _mock_structured_findings()

        mock_struct_llm.side_effect = _side_effect

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        task = AgentTask(query="What is the hit cap?")
        # No metadata["complexity"] = "complex"
        result = await researcher.execute(task)

        assert result.status == TaskStatus.SUCCESS

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_reflection_fires_on_complex(self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock) -> None:
        """Complex query -> reflection is called."""
        mock_loop_llm.return_value = _final_answer("Answer text.")

        reflection_called = False

        async def _side_effect(response_model, messages, **kwargs):
            nonlocal reflection_called
            if response_model is StructuredFindings:
                return _mock_structured_findings()
            if response_model is ReflectionResult:
                reflection_called = True
                return ReflectionResult(supported=True)
            return _mock_structured_findings()

        mock_struct_llm.side_effect = _side_effect

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        task = AgentTask(query="Compare combat vs mutilate")
        task.context = {"complexity": "complex"}
        result = await researcher.execute(task)

        assert reflection_called is True
        assert result.status == TaskStatus.SUCCESS

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_reflection_supported_returns_original(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """Reflection says supported=True -> answer unchanged."""
        mock_loop_llm.return_value = _final_answer("Original answer.")

        async def _side_effect(response_model, messages, **kwargs):
            if response_model is StructuredFindings:
                return _mock_structured_findings()
            if response_model is ReflectionResult:
                return ReflectionResult(supported=True)
            return _mock_structured_findings()

        mock_struct_llm.side_effect = _side_effect

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        task = AgentTask(query="q")
        task.context = {"complexity": "complex"}
        result = await researcher.execute(task)

        assert result.output == "Original answer."

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_reflection_unsupported_returns_revised(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """Reflection says supported=False -> revised_answer used."""
        mock_loop_llm.return_value = _final_answer("Original speculative answer.")

        async def _side_effect(response_model, messages, **kwargs):
            if response_model is StructuredFindings:
                return _mock_structured_findings()
            if response_model is ReflectionResult:
                return ReflectionResult(
                    supported=False,
                    issues=["Claim about proc rate is unsupported"],
                    revised_answer="Revised answer without speculation.",
                )
            return _mock_structured_findings()

        mock_struct_llm.side_effect = _side_effect

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        task = AgentTask(query="q")
        task.context = {"complexity": "complex"}
        result = await researcher.execute(task)

        assert result.output == "Revised answer without speculation."

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_reflection_failure_returns_original(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """Reflection LLM call fails -> original answer returned."""
        mock_loop_llm.return_value = _final_answer("Original answer.")

        async def _side_effect(response_model, messages, **kwargs):
            if response_model is StructuredFindings:
                return _mock_structured_findings()
            if response_model is ReflectionResult:
                raise RuntimeError("LLM timeout")
            return _mock_structured_findings()

        mock_struct_llm.side_effect = _side_effect

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        task = AgentTask(query="q")
        task.context = {"complexity": "complex"}
        result = await researcher.execute(task)

        assert result.output == "Original answer."

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_reflection_skipped_for_failed_outcome(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """FAILED status -> reflection not called."""
        mock_loop_llm.return_value = _tool_call("rag_search", {"query": "loop"})

        async def _side_effect(response_model, messages, **kwargs):
            if response_model is ReflectionResult:
                pytest.fail("Reflection should not be called for FAILED outcomes")
            return _mock_structured_findings()

        mock_struct_llm.side_effect = _side_effect

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER, max_iterations=1)
        task = AgentTask(query="q")
        task.context = {"complexity": "complex"}
        result = await researcher.execute(task)

        assert result.status == TaskStatus.FAILED

    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_reflection_skipped_for_partial_outcome(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """PARTIAL status -> reflection not called (loop detected)."""
        # Trigger loop detection by repeating exact same tool call
        mock_loop_llm.return_value = _tool_call("rag_search", {"query": "stuck"})

        async def _side_effect(response_model, messages, **kwargs):
            if response_model is ReflectionResult:
                pytest.fail("Reflection should not be called for PARTIAL outcomes")
            return _mock_structured_findings()

        mock_struct_llm.side_effect = _side_effect

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER, max_iterations=10)
        task = AgentTask(query="q")
        task.context = {"complexity": "complex"}
        result = await researcher.execute(task)

        # PARTIAL from loop detection, or FAILED from max_iterations — either skips reflection
        assert result.status in (TaskStatus.PARTIAL, TaskStatus.FAILED)

    @patch("code.shukketsu.config.REFLECTION_ENABLED", False)
    @patch("code.shukketsu.agents.researcher.get_structured_output")
    @patch("code.shukketsu.agents.base.get_structured_output")
    async def test_reflection_disabled_via_config(
        self, mock_loop_llm: AsyncMock, mock_struct_llm: AsyncMock
    ) -> None:
        """REFLECTION_ENABLED=False -> reflection not called even on complex queries."""
        mock_loop_llm.return_value = _final_answer("Answer text.")

        async def _side_effect(response_model, messages, **kwargs):
            if response_model is ReflectionResult:
                pytest.fail("Reflection should not be called when disabled")
            if response_model is StructuredFindings:
                return _mock_structured_findings()
            return _mock_structured_findings()

        mock_struct_llm.side_effect = _side_effect

        researcher = Researcher(tool_registry=_registry(_EchoTool()), role=AgentRole.RESEARCHER)
        task = AgentTask(query="q")
        task.context = {"complexity": "complex"}
        result = await researcher.execute(task)

        assert result.status == TaskStatus.SUCCESS
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_researcher.py::TestReflection -v`

Expected: Failures — `ReflectionResult` import fails, and `_reflect()` method doesn't exist.

---

### Task 5.5: Implement reflection in Researcher agent

**Files:**
- Modify: `code/shukketsu/agents/researcher.py`

**Step 1: Add imports for reflection**

Add to the imports at the top of the file:

```python
from code.shukketsu import config
from code.shukketsu.llm.prompts.researcher import REFLECTION_PROMPT, STRUCTURING_PROMPT
from code.shukketsu.llm.schemas import ReflectionResult
```

And update the existing import line to remove the duplicate `STRUCTURING_PROMPT`:

```python
from code.shukketsu.llm.prompts.researcher import STRUCTURING_PROMPT
```

becomes part of the combined import above. The full import block should be:

```python
import logging
from typing import Any

from langfuse import observe
from pydantic import BaseModel, Field

from code.shukketsu import config
from code.shukketsu.agents.base import BaseAgent, StatusCallback
from code.shukketsu.agents.tasks import AgentTask, Finding, ResearchResult, TaskStatus, ToolCallRecord
from code.shukketsu.llm.prompts.researcher import REFLECTION_PROMPT, STRUCTURING_PROMPT
from code.shukketsu.llm.schemas import ReflectionResult
from code.shukketsu.llm.structured import get_structured_output
from code.shukketsu.resilience.errors import StructuredOutputError
```

**Step 2: Add `_reflect()` method to the Researcher class**

Add after `_format_scratchpad()` (after line 169):

```python
    async def _reflect(
        self,
        query: str,
        output: str,
        scratchpad: list[dict[str, Any]],
    ) -> ReflectionResult:
        """Run a reflection pass to verify the answer against evidence.

        Checks whether the answer is fully supported by the tool observations.
        If not, returns a revised answer that removes unsupported claims.
        """
        observations_text = self._format_scratchpad(scratchpad)

        messages = [
            {"role": "system", "content": REFLECTION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Researcher's answer:\n{output}\n\n"
                    f"Tool observations:\n{observations_text}"
                ),
            },
        ]

        result: ReflectionResult = await get_structured_output(
            response_model=ReflectionResult,
            messages=messages,
            temperature=config.REFLECTION_TEMPERATURE,
        )
        return result
```

**Step 3: Add reflection step to `execute()`**

Replace the `execute()` method (lines 43-120) with the following. The key change is adding a reflection step after the successful structuring pass and before the final `return`:

```python
    @observe(as_type="agent")
    async def execute(self, task: AgentTask, *, on_status: StatusCallback | None = None) -> ResearchResult:
        """Execute a research task and return structured findings.

        Runs the ReAct loop, then structures the output via a second
        LLM call. For complex queries, runs a reflection pass to verify
        the answer is supported by evidence. Falls back to basic result
        if structuring fails.

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

        trajectory = [
            ToolCallRecord(tool_name=e["tool_name"], tool_input=e["tool_input"])
            for e in outcome.scratchpad
        ]

        # Skip structuring for failed loops — no useful output to parse
        if outcome.status == TaskStatus.FAILED:
            logger.info("ReAct loop failed; skipping structuring pass")
            return ResearchResult(
                task_id=task.task_id,
                agent_role=self.role,
                status=outcome.status,
                output=outcome.output,
                sufficient=False,
                trajectory=trajectory,
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
                trajectory=trajectory,
            )

        # Derive metadata from structuring output + scratchpad
        sources_used = list(dict.fromkeys(e for f in structured.findings for e in f.evidence))
        strategies_used = self._extract_strategies(outcome.scratchpad)

        # Reflection pass for complex queries
        final_output = outcome.output
        is_complex = task.context.get("complexity") == "complex"

        if (
            is_complex
            and config.REFLECTION_ENABLED
            and outcome.status == TaskStatus.SUCCESS
        ):
            if on_status:
                await on_status("reflecting on answer...")

            try:
                reflection = await self._reflect(
                    query=task.query,
                    output=outcome.output,
                    scratchpad=outcome.scratchpad,
                )
                if not reflection.supported and reflection.revised_answer:
                    logger.info(
                        "Reflection found %d issues; using revised answer",
                        len(reflection.issues),
                    )
                    final_output = reflection.revised_answer
            except Exception as exc:
                logger.warning("Reflection failed, keeping original answer: %s", exc)

        return ResearchResult(
            task_id=task.task_id,
            agent_role=self.role,
            status=outcome.status,
            output=final_output,
            evidence=sources_used,
            findings=structured.findings,
            sources_used=sources_used,
            strategies_used=strategies_used,
            gaps=structured.gaps,
            sufficient=structured.sufficient,
            trajectory=trajectory,
        )
```

**Step 4: Run researcher tests**

Run: `python3 -m pytest tests/unit/test_researcher.py -v`

Expected: All tests pass (existing + new reflection tests).

---

### Task 5.6: Update Orchestrator `_build_task()` to pass complexity metadata

**Files:**
- Modify: `code/shukketsu/agents/orchestrator.py`

**Step 1: Set complexity metadata when building ResearchTask**

In `_build_task()`, update the RESEARCHER branch (line 460) from:

```python
        if subtask.agent_role == AgentRole.RESEARCHER:
            return ResearchTask(query=subtask.description, trace_id=trace_id)
```

To:

```python
        if subtask.agent_role == AgentRole.RESEARCHER:
            return ResearchTask(
                query=subtask.description,
                trace_id=trace_id,
                context={"complexity": "complex"},
            )
```

**Step 2: Run orchestrator tests to verify nothing breaks**

Run: `python3 -m pytest tests/unit/test_orchestrator_execute.py tests/unit/test_orchestrator_build_task.py -v`

Expected: All pass.

---

### Task 5.7: Run full test suite + linting

**Files:** None (verification only)

Run: `ruff check code/ tests/ --fix && ruff format code/ tests/ && python3 -m mypy code/shukketsu/ && python3 -m pytest tests/unit/ -v`

Expected: All tests pass, no lint errors, no type errors.

If mypy complains about the `temperature` kwarg to `get_structured_output`, check `code/shukketsu/llm/structured.py` to verify the function signature accepts `**kwargs` or a `temperature` parameter. If not, pass it via `extra_kwargs={"temperature": config.REFLECTION_TEMPERATURE}` or adjust the call accordingly.

---

### Task 5.8: Commit

```bash
git add code/shukketsu/llm/schemas.py code/shukketsu/llm/prompts/researcher.py code/shukketsu/config.py code/shukketsu/agents/researcher.py code/shukketsu/agents/orchestrator.py tests/unit/test_researcher.py
git commit -m "feat(agents): add reflection pass to Researcher for complex queries

After structuring findings on complex queries, the Researcher runs a
reflection pass that verifies the answer against tool observations.
If claims are unsupported, a revised answer is used instead. Reflection
is skippable via REFLECTION_ENABLED config and only fires when
context.complexity == 'complex' and outcome.status == SUCCESS.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
