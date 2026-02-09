"""Tool base class for the agent tool system."""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Abstract base class for agent tools.

    Subclasses must define name, description, parameters_schema,
    and implement the execute method.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]

    @abstractmethod
    async def execute(self, tool_input: dict[str, Any]) -> str:
        """Execute the tool with the given input.

        Args:
            tool_input: Dictionary of parameters matching parameters_schema.

        Returns:
            A string observation describing the tool's output.
        """
