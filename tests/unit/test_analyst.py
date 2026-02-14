"""Tests for the Analyst agent and its task protocol models."""

from code.shukketsu.agents.analyst import Analyst
from code.shukketsu.agents.tasks import AgentRole, AnalysisResult, AnalysisTask, TaskStatus
from code.shukketsu.tools.registry import ToolRegistry


class TestAnalystTaskProtocol:
    def test_agent_role_analyst_exists(self) -> None:
        assert AgentRole.ANALYST == "analyst"

    def test_agent_role_analyst_in_enum(self) -> None:
        assert "analyst" in [r.value for r in AgentRole]

    def test_analysis_task_validates(self) -> None:
        task = AnalysisTask(query="What's my DPS?")
        assert task.query == "What's my DPS?"
        assert task.import_string == ""

    def test_analysis_task_with_import_string(self) -> None:
        task = AnalysisTask(query="Sim my character", import_string='rogue="Test"')
        assert task.import_string == 'rogue="Test"'

    def test_analysis_task_with_comparison_mode(self) -> None:
        task = AnalysisTask(query="Compare trinkets", comparison_mode=True)
        assert task.comparison_mode is True

    def test_analysis_task_with_optimization_slot(self) -> None:
        task = AnalysisTask(query="Best trinket?", optimization_slot="trinket_1")
        assert task.optimization_slot == "trinket_1"

    def test_analysis_task_inherits_agent_task(self) -> None:
        task = AnalysisTask(query="test")
        assert task.task_id  # UUID is generated
        assert task.trace_id  # UUID is generated
        assert task.context == {}

    def test_analysis_result_validates(self) -> None:
        result = AnalysisResult(
            task_id="test",
            agent_role=AgentRole.ANALYST,
            status=TaskStatus.SUCCESS,
            output="DPS: 1500",
            dps_mean=1500.0,
        )
        assert result.dps_mean == 1500.0

    def test_analysis_result_optional_fields(self) -> None:
        result = AnalysisResult(
            task_id="t",
            agent_role="analyst",
            status="success",
            output="test",
        )
        assert result.dps_mean is None
        assert result.stat_weights == {}
        assert result.recommendations == []
        assert result.comparison == {}

    def test_analysis_result_with_stat_weights(self) -> None:
        result = AnalysisResult(
            task_id="t",
            agent_role="analyst",
            status="success",
            output="test",
            stat_weights={"hit_rating": 2.15, "agility": 1.8},
        )
        assert result.stat_weights["hit_rating"] == 2.15


class TestAnalystAgent:
    def test_analyst_initializes(self) -> None:
        agent = Analyst(
            tool_registry=ToolRegistry(),
            role=AgentRole.ANALYST,
            system_prompt="test prompt",
        )
        assert agent.role == AgentRole.ANALYST

    def test_analyst_has_system_prompt(self) -> None:
        agent = Analyst(
            tool_registry=ToolRegistry(),
            role=AgentRole.ANALYST,
            system_prompt="You are a DPS analyst.",
        )
        assert "DPS" in agent._system_prompt

    def test_extract_dps_mean_pattern(self) -> None:
        assert Analyst._extract_dps("Mean DPS: 1523.4") == 1523.4

    def test_extract_dps_trailing_pattern(self) -> None:
        assert Analyst._extract_dps("Your character does 2000 DPS") == 2000.0

    def test_extract_dps_integer(self) -> None:
        assert Analyst._extract_dps("DPS: 1500") == 1500.0

    def test_extract_dps_colon_with_space(self) -> None:
        assert Analyst._extract_dps("DPS 1234.5") == 1234.5

    def test_extract_dps_no_match(self) -> None:
        assert Analyst._extract_dps("No numbers here") is None

    def test_extract_dps_empty_string(self) -> None:
        assert Analyst._extract_dps("") is None


class TestAnalystFactory:
    def test_factory_creates_analyst(self) -> None:
        from code.shukketsu.agents.factory import AgentFactory

        factory = AgentFactory()
        agent = factory.create(AgentRole.ANALYST)
        assert isinstance(agent, Analyst)

    def test_factory_analyst_has_correct_class(self) -> None:
        from code.shukketsu.agents.factory import _ROLE_CLASSES

        assert _ROLE_CLASSES[AgentRole.ANALYST] is Analyst

    def test_analyst_system_prompt_non_empty(self) -> None:
        from code.shukketsu.llm.prompts.analyst import ANALYST_SYSTEM_PROMPT

        assert ANALYST_SYSTEM_PROMPT
        assert "DPS" in ANALYST_SYSTEM_PROMPT

    def test_analyst_system_prompt_mentions_sim(self) -> None:
        from code.shukketsu.llm.prompts.analyst import ANALYST_SYSTEM_PROMPT

        assert "sim" in ANALYST_SYSTEM_PROMPT.lower()

    def test_analyst_max_iterations_in_config(self) -> None:
        from code.shukketsu import config

        assert config.ANALYST_MAX_ITERATIONS >= 1

    def test_analyst_gets_custom_iteration_limit(self) -> None:
        from code.shukketsu.agents.factory import AgentFactory

        factory = AgentFactory()
        agent = factory.create(AgentRole.ANALYST)
        from code.shukketsu import config

        assert agent.max_iterations == config.ANALYST_MAX_ITERATIONS
