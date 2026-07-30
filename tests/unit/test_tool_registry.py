"""Tests for ToolRegistry: schema shape, dispatch, and error handling."""

from src.tools.base import Tool
from src.tools.registry import ToolRegistry


def _boom(_arguments: dict) -> str:
    raise RuntimeError("kaboom")


def test_to_ollama_schema_shape() -> None:
    registry = ToolRegistry()
    schema = registry.to_ollama_schema()

    names = {entry["function"]["name"] for entry in schema}
    assert names == {"calculator", "get_current_datetime"}
    assert all(entry["type"] == "function" for entry in schema)
    assert all("parameters" in entry["function"] for entry in schema)


def test_execute_dispatches_to_the_right_tool() -> None:
    registry = ToolRegistry()
    assert registry.execute("calculator", {"expression": "2 + 2"}) == "4"


def test_execute_unknown_tool_returns_error_string() -> None:
    registry = ToolRegistry()
    result = registry.execute("does_not_exist", {})
    assert result == "Error: unknown tool 'does_not_exist'"


def test_execute_swallows_tool_exceptions() -> None:
    broken_tool = Tool(name="broken", description="always fails", parameters={}, execute=_boom)
    registry = ToolRegistry(tools=[broken_tool])

    result = registry.execute("broken", {})

    assert result.startswith("Error: tool 'broken' failed")
