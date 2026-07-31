"""Tests for AgentRunner's bounded multi-step tool-calling loop.

Uses small hand-written fake collaborators (no mocking framework) —
consistent with this project's existing test style.
"""

from collections.abc import AsyncIterator

from src.agents.runner import _MAX_STEPS, AgentRunner
from src.models.chat import ChatTurnResult, StreamChunk
from src.models.message import Message, MessageRole, ToolCall


class _FakeOllamaClient:
    """Returns one canned `ChatTurnResult` per call to `chat`, in order."""

    def __init__(self, results: list[ChatTurnResult]) -> None:
        self._results = list(results)
        self.calls: list[dict] = []

    async def chat(self, history, model=None, tools=None):
        self.calls.append({"tools": tools})
        return self._results.pop(0)


class _FakeStreamingOllamaClient:
    """Yields one canned list of `StreamChunk`s per call to `stream_chat`, in order."""

    def __init__(self, rounds: list[list[StreamChunk]]) -> None:
        self._rounds = list(rounds)

    async def stream_chat(self, history, model=None, tools=None) -> AsyncIterator[StreamChunk]:
        for chunk in self._rounds.pop(0):
            yield chunk


class _FakeToolRegistry:
    def __init__(self) -> None:
        self.executed: list[tuple[str, dict]] = []

    def to_ollama_schema(self) -> list[dict]:
        return [{"type": "function", "function": {"name": "fake_tool"}}]

    async def execute(self, name: str, arguments: dict) -> str:
        self.executed.append((name, arguments))
        return f"result-for-{name}"


def _assistant_reply(content: str, tool_calls: list[ToolCall] | None = None) -> Message:
    return Message(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)


async def test_run_returns_immediately_when_no_tool_call_requested() -> None:
    ollama_client = _FakeOllamaClient(
        [ChatTurnResult(reply=_assistant_reply("Hello there"), model="m", latency_ms=10.0)]
    )
    runner = AgentRunner(ollama_client, _FakeToolRegistry())

    result = await runner.run([Message(role=MessageRole.USER, content="hi")])

    assert result.reply.content == "Hello there"
    assert result.steps == []
    assert result.latency_ms == 10.0


async def test_run_executes_a_tool_call_and_loops_for_the_final_answer() -> None:
    tool_call = ToolCall(id="call_1", name="fake_tool", arguments={"x": 1})
    ollama_client = _FakeOllamaClient(
        [
            ChatTurnResult(
                reply=_assistant_reply("", tool_calls=[tool_call]), model="m", latency_ms=5.0
            ),
            ChatTurnResult(reply=_assistant_reply("Final answer"), model="m", latency_ms=7.0),
        ]
    )
    tool_registry = _FakeToolRegistry()
    runner = AgentRunner(ollama_client, tool_registry)

    result = await runner.run([Message(role=MessageRole.USER, content="do the thing")])

    assert result.reply.content == "Final answer"
    assert result.latency_ms == 12.0
    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "fake_tool"
    assert result.steps[0].result == "result-for-fake_tool"
    assert tool_registry.executed == [("fake_tool", {"x": 1})]


async def test_run_forces_a_final_answer_once_the_step_budget_is_exhausted() -> None:
    tool_call = ToolCall(id="call_1", name="fake_tool", arguments={})
    # The model keeps wanting to call a tool every round...
    looping_result = ChatTurnResult(
        reply=_assistant_reply("", tool_calls=[tool_call]), model="m", latency_ms=1.0
    )
    # ...until the final round, where tools are withheld and it must answer.
    results = [looping_result] * (_MAX_STEPS - 1) + [
        ChatTurnResult(reply=_assistant_reply("Giving up gracefully"), model="m", latency_ms=1.0)
    ]
    ollama_client = _FakeOllamaClient(results)
    runner = AgentRunner(ollama_client, _FakeToolRegistry())

    result = await runner.run([Message(role=MessageRole.USER, content="loop forever")])

    assert result.reply.content == "Giving up gracefully"
    assert len(ollama_client.calls) == _MAX_STEPS
    assert ollama_client.calls[-1]["tools"] is None


async def test_run_streaming_yields_content_then_step_then_final_content() -> None:
    tool_call = ToolCall(id="call_1", name="fake_tool", arguments={"q": "x"})
    ollama_client = _FakeStreamingOllamaClient(
        [
            [StreamChunk(tool_calls=[tool_call])],
            [StreamChunk(content="Final "), StreamChunk(content="answer")],
        ]
    )
    runner = AgentRunner(ollama_client, _FakeToolRegistry())

    events = [
        event
        async for event in runner.run_streaming([Message(role=MessageRole.USER, content="hi")])
    ]

    assert [e.type for e in events] == ["step", "content", "content"]
    assert events[0].step is not None
    assert events[0].step.tool_name == "fake_tool"
    assert events[1].text == "Final "
    assert events[2].text == "answer"
