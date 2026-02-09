"""Tests for the Tool abstract base class."""

from typing import Any

import pytest

from code.shukketsu.tools.schemas import Tool


class DummyTool(Tool):
    """Concrete tool for testing."""

    name = "dummy"
    description = "A dummy tool for testing."
    parameters_schema = {"query": {"type": "string", "description": "The search query"}}

    async def execute(self, tool_input: dict[str, Any]) -> str:
        return f"dummy result for {tool_input.get('query', 'unknown')}"


class TestToolBaseClass:
    """Tests for the abstract Tool base class."""

    def test_cannot_instantiate_abstract_tool(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            Tool()  # type: ignore[abstract]

    def test_concrete_tool_can_be_instantiated(self) -> None:
        tool = DummyTool()
        assert tool is not None

    async def test_execute_returns_string(self) -> None:
        tool = DummyTool()
        result = await tool.execute({"query": "test"})
        assert isinstance(result, str)
        assert "dummy result" in result

    def test_has_name(self) -> None:
        assert DummyTool().name == "dummy"

    def test_has_description(self) -> None:
        assert DummyTool().description == "A dummy tool for testing."

    def test_has_parameters_schema(self) -> None:
        schema = DummyTool().parameters_schema
        assert isinstance(schema, dict)
        assert "query" in schema
