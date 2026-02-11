"""Tests for OrchestratorPlan validation."""

from code.shukketsu.agents.tasks import AgentRole, OrchestratorPlan, SubTask


def _plan(*subtasks: SubTask) -> OrchestratorPlan:
    """Build an OrchestratorPlan from subtasks."""
    return OrchestratorPlan(reasoning="test", subtasks=list(subtasks))


def _sub(
    role: AgentRole = AgentRole.RESEARCHER,
    desc: str = "test",
    depends_on: list[int] | None = None,
    **kw: object,
) -> SubTask:
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
