"""Tool registration and dispatch for the agent system."""

import logging
from typing import Any

from langfuse import get_client, observe

from code.shukketsu.resilience.errors import ToolNotFoundError
from code.shukketsu.tools.schemas import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for agent tools. Stores tools by name and dispatches execution."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Overwrites any existing tool with the same name."""
        logger.info("Registered tool: %s", tool.name)
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the tool with the given name.

        Raises:
            ToolNotFoundError: If no tool with that name is registered.
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool not found: {name}")
        return self._tools[name]

    def get_tool_descriptions(self) -> str:
        """Format all tool descriptions for inclusion in the agent system prompt."""
        if not self._tools:
            return "No tools available."

        parts: list[str] = []
        for tool in self._tools.values():
            params = ""
            for param_name, param_info in tool.parameters_schema.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                optional = " (optional)" if param_info.get("optional") else ""
                params += f"\n  - {param_name} ({param_type}{optional}): {param_desc}"

            parts.append(f"Tool: {tool.name}\nDescription: {tool.description}\nParameters:{params}")

        return "\n\n".join(parts)

    @observe(as_type="tool")
    async def execute(self, name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool by name, returning the observation string.

        Returns an error observation (not an exception) for unknown tools
        or execution failures, so the agent can see what went wrong.
        """
        langfuse = get_client()
        langfuse.update_current_span(metadata={"tool_name": name, "tool_input": tool_input})

        if name not in self._tools:
            logger.warning("Tool not found: %s", name)
            return f"Error: Tool '{name}' not found. Available tools: {', '.join(self._tools.keys())}"

        try:
            result = await self._tools[name].execute(tool_input)
            logger.debug("Tool %s returned: %s", name, result[:200])
            return result
        except Exception as exc:
            logger.exception("Tool %s failed: %s", name, exc)
            return f"Error executing tool '{name}': {exc}"

    def __len__(self) -> int:
        return len(self._tools)

    def has(self, name: str) -> bool:
        """Check if a tool is registered by name."""
        return name in self._tools

    def __contains__(self, name: str) -> bool:
        return name in self._tools
