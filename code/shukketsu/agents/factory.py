"""Factory for creating configured agent instances.

Each agent role gets a different system prompt, iteration limit, and
potentially a different class. Tool registries are injected by the caller
(or default to empty).
"""

import logging
from typing import Any

from code.shukketsu import config
from code.shukketsu.agents.analyst import Analyst
from code.shukketsu.agents.base import BaseAgent
from code.shukketsu.agents.editor import Editor
from code.shukketsu.agents.orchestrator import Orchestrator
from code.shukketsu.agents.researcher import Researcher
from code.shukketsu.agents.tasks import AgentRole
from code.shukketsu.agents.writer import Writer
from code.shukketsu.llm.prompts.analyst import ANALYST_SYSTEM_PROMPT
from code.shukketsu.llm.prompts.editor import EDITOR_SYSTEM_PROMPT
from code.shukketsu.llm.prompts.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT
from code.shukketsu.llm.prompts.researcher import RESEARCHER_SYSTEM_PROMPT
from code.shukketsu.llm.prompts.writer import WRITER_SYSTEM_PROMPT
from code.shukketsu.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_ROLE_PROMPTS: dict[AgentRole, str] = {
    AgentRole.RESEARCHER: RESEARCHER_SYSTEM_PROMPT,
    AgentRole.WRITER: WRITER_SYSTEM_PROMPT,
    AgentRole.EDITOR: EDITOR_SYSTEM_PROMPT,
    AgentRole.ORCHESTRATOR: ORCHESTRATOR_SYSTEM_PROMPT,
    AgentRole.ANALYST: ANALYST_SYSTEM_PROMPT,
}

_ROLE_MAX_ITERATIONS: dict[AgentRole, int] = {
    AgentRole.RESEARCHER: config.RESEARCHER_MAX_ITERATIONS,
    AgentRole.ANALYST: config.ANALYST_MAX_ITERATIONS,
}

_ROLE_CLASSES: dict[AgentRole, type[BaseAgent]] = {
    AgentRole.RESEARCHER: Researcher,
    AgentRole.WRITER: Writer,
    AgentRole.EDITOR: Editor,
    AgentRole.ORCHESTRATOR: Orchestrator,
    AgentRole.ANALYST: Analyst,
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
        **kwargs: Any,
    ) -> BaseAgent:
        """Create an agent configured for the given role.

        Args:
            role: The specialist role to configure.
            tool_registry: Optional pre-configured tool registry.
                If None, a new empty registry is created.
            **kwargs: Additional keyword arguments forwarded to the agent
                constructor (e.g. ``knowledge_manager`` for Writer).

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
            **kwargs,
        )

        logger.info("Created %s agent (%s, max_iter=%d)", role.value, cls.__name__, max_iter)
        return agent
