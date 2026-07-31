"""Registry of tools the model can call, and the glue to Ollama's schema."""

import logging

from src.tools.base import Tool
from src.tools.calculator import calculator_tool
from src.tools.datetime_tool import current_datetime_tool

logger = logging.getLogger(__name__)

DEFAULT_TOOLS = [calculator_tool, current_datetime_tool]


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools = {t.name: t for t in (tools if tools is not None else DEFAULT_TOOLS)}

    def to_ollama_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, arguments: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            return await tool.execute(arguments)
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return f"Error: tool '{name}' failed ({exc})"
