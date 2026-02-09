"""Tests for the ToolRegistry."""

from typing import Any

import pytest

from code.shukketsu.resilience.errors import ToolNotFoundError
from code.shukketsu.tools.registry import ToolRegistry
from code.shukketsu.tools.schemas import Tool


class FakeTool(Tool):
    name = "fake"
    description = "A fake tool."
    parameters_schema = {"input": {"type": "string", "description": "Input text"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        return f"fake result: {tool_input.get('input', '')}"


class AnotherFakeTool(Tool):
    name = "another"
    description = "Another fake tool."
    parameters_schema = {"value": {"type": "integer", "description": "A number"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        return f"another result: {tool_input.get('value', 0)}"


class FailingTool(Tool):
    name = "failing"
    description = "A tool that fails."
    parameters_schema = {}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        raise RuntimeError("Tool exploded")


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_adds_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeTool())
        assert "fake" in registry

    def test_register_overwrites_duplicate(self) -> None:
        tool1 = FakeTool()
        tool2 = FakeTool()
        registry = ToolRegistry()
        registry.register(tool1)
        registry.register(tool2)
        assert registry.get("fake") is tool2

    def test_get_returns_registered_tool(self) -> None:
        tool = FakeTool()
        registry = ToolRegistry()
        registry.register(tool)
        assert registry.get("fake") is tool

    def test_get_raises_for_unknown(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError, match="nonexistent"):
            registry.get("nonexistent")

    def test_len(self) -> None:
        registry = ToolRegistry()
        assert len(registry) == 0
        registry.register(FakeTool())
        assert len(registry) == 1
        registry.register(AnotherFakeTool())
        assert len(registry) == 2

    def test_contains(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeTool())
        assert "fake" in registry
        assert "missing" not in registry

    async def test_execute_dispatches_correctly(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeTool())
        registry.register(AnotherFakeTool())
        result = await registry.execute("fake", {"input": "hello"})
        assert "fake result: hello" in result

    async def test_execute_unknown_returns_error(self) -> None:
        registry = ToolRegistry()
        result = await registry.execute("nonexistent", {})
        assert "error" in result.lower()
        assert "nonexistent" in result

    async def test_execute_catches_tool_failure(self) -> None:
        registry = ToolRegistry()
        registry.register(FailingTool())
        result = await registry.execute("failing", {})
        assert "error" in result.lower()

    def test_descriptions_includes_all_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(FakeTool())
        registry.register(AnotherFakeTool())
        desc = registry.get_tool_descriptions()
        assert "fake" in desc
        assert "another" in desc
        assert "A fake tool" in desc

    def test_descriptions_empty_registry(self) -> None:
        registry = ToolRegistry()
        desc = registry.get_tool_descriptions()
        assert "no tools" in desc.lower() or desc == ""
