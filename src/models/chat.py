"""Domain result of a single chat turn.

Deliberately distinct from `api.schemas.chat.ChatResponse`: this is what the
service/AI layers return internally, while the schema is the HTTP wire
format. Renaming an API response field should never force a change here.
"""

from dataclasses import dataclass

from src.models.message import Message


@dataclass(frozen=True, slots=True)
class ChatTurnResult:
    reply: Message
    model: str
    latency_ms: float
