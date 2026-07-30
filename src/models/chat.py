"""Domain result of a single chat turn.

Deliberately distinct from `api.schemas.chat.ChatResponse`: this is what the
service/AI layers return internally, while the schema is the HTTP wire
format. Renaming an API response field should never force a change here.
"""

from dataclasses import dataclass
from typing import Literal

from src.models.message import Message, ToolCall


@dataclass(frozen=True, slots=True)
class ChatTurnResult:
    reply: Message
    model: str
    latency_ms: float


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """`OllamaClient.stream_chat`'s internal parsed-chunk shape."""

    content: str = ""
    tool_calls: list[ToolCall] | None = None


@dataclass(frozen=True, slots=True)
class ChatStreamEvent:
    """What `ChatService.stream_message` yields to the API route.

    Carries only the tool's *name* (for the UI's "Using X..." indicator),
    never its arguments — those stay internal to the tool-calling loop.
    """

    type: Literal["content", "tool_call"]
    text: str = ""
    tool_name: str = ""
