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
