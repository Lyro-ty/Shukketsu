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
