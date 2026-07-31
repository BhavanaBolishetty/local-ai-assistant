"""Runs the bounded, multi-step tool-calling loop and exposes each step.

Phase 4 had a fixed 3-iteration loop hidden inside `ChatService`; this
generalizes it into a first-class agent with a higher step budget so the
API/UI can show a real reasoning trace instead of a fleeting indicator.
"""

from collections.abc import AsyncIterator

from src.ai.ollama_client import OllamaClient
from src.models.agent import AgentStep
from src.models.chat import ChatStreamEvent, ChatTurnResult
from src.models.message import Message, MessageRole, ToolCall
from src.tools.registry import ToolRegistry

_MAX_STEPS = 8


class AgentRunner:
    def __init__(self, ollama_client: OllamaClient, tool_registry: ToolRegistry) -> None:
        self._ollama_client = ollama_client
        self._tool_registry = tool_registry

    async def run(self, history: list[Message], model: str | None = None) -> ChatTurnResult:
        """Run to completion (non-streaming) and return the final reply
        plus the full step trace."""
        steps: list[AgentStep] = []
        total_latency_ms = 0.0
        result: ChatTurnResult | None = None

        for iteration in range(_MAX_STEPS):
            offer_tools = self._tools_schema() if iteration < _MAX_STEPS - 1 else None
            result = await self._ollama_client.chat(history, model=model, tools=offer_tools)
            total_latency_ms += result.latency_ms
            if not result.reply.tool_calls:
                break
            steps.extend(await self._apply_tool_calls(history, result.reply.tool_calls))

        return ChatTurnResult(
            reply=result.reply, model=result.model, latency_ms=total_latency_ms, steps=steps
        )

    async def run_streaming(
        self, history: list[Message], model: str | None = None
    ) -> AsyncIterator[ChatStreamEvent]:
        """Streaming variant: yields `content` events live as real tokens
        arrive, and a `step` event once each round's tool calls resolve."""
        for iteration in range(_MAX_STEPS):
            offer_tools = self._tools_schema() if iteration < _MAX_STEPS - 1 else None
            round_tool_calls: list[ToolCall] = []

            stream = self._ollama_client.stream_chat(history, model=model, tools=offer_tools)
            async for chunk in stream:
                if chunk.tool_calls:
                    round_tool_calls.extend(chunk.tool_calls)
                if chunk.content:
                    yield ChatStreamEvent(type="content", text=chunk.content)

            if not round_tool_calls:
                return

            for step in await self._apply_tool_calls(history, round_tool_calls):
                yield ChatStreamEvent(type="step", step=step)

    def _tools_schema(self) -> list[dict]:
        return self._tool_registry.to_ollama_schema()

    async def _apply_tool_calls(
        self, history: list[Message], tool_calls: list[ToolCall]
    ) -> list[AgentStep]:
        """Append the assistant's tool-call request and each tool's result
        to the (scratch, per-turn) working history so the next round can
        see them, and return one `AgentStep` per call for the trace."""
        history.append(Message(role=MessageRole.ASSISTANT, content="", tool_calls=tool_calls))
        steps: list[AgentStep] = []
        for call in tool_calls:
            result_text = await self._tool_registry.execute(call.name, call.arguments)
            history.append(
                Message(role=MessageRole.TOOL, content=result_text, tool_call_id=call.id)
            )
            steps.append(
                AgentStep(tool_name=call.name, arguments=call.arguments, result=result_text)
            )
        return steps
