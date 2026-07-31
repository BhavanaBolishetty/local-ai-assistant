"""Domain result of a single chat turn.

Deliberately distinct from `api.schemas.chat.ChatResponse`: this is what the
service/AI layers return internally, while the schema is the HTTP wire
format. Renaming an API response field should never force a change here.
"""

from dataclasses import dataclass, field
from typing import Literal

from src.models.agent import AgentStep
from src.models.message import Message, ToolCall


@dataclass(frozen=True, slots=True)
class ChatTurnResult:
    reply: Message
    model: str
    latency_ms: float
    steps: list[AgentStep] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """`OllamaClient.stream_chat`'s internal parsed-chunk shape."""

    content: str = ""
    tool_calls: list[ToolCall] | None = None


@dataclass(frozen=True, slots=True)
class ChatStreamEvent:
    """What `AgentRunner`/`ChatService` yield to the API route."""

    type: Literal["content", "step"]
    text: str = ""
    step: AgentStep | None = None
