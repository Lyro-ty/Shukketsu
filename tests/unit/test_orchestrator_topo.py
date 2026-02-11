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
            _sub(),  # 0: no deps
            _sub(depends_on=[0]),  # 1: depends on 0
            _sub(depends_on=[1]),  # 2: depends on 1
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
            _sub(),  # 0: A
            _sub(),  # 1: B
            _sub(depends_on=[0, 1]),  # 2: C
            _sub(depends_on=[2]),  # 3: D
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
