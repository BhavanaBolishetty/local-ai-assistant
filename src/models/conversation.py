"""Domain model for a conversation: an ordered collection of messages.

Mutable (unlike `Message`) because messages are appended to it over the
conversation's lifetime. In Phase 2, `repositories/conversation_repository.py`
will persist this object as-is, unchanged by this model's shape.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from src.models.message import Message


@dataclass(slots=True)
class Conversation:
    id: str = field(default_factory=lambda: str(uuid4()))
    title: str = "New Chat"
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
