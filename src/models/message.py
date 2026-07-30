"""Domain model for a single chat message.

A plain `dataclass`, not Pydantic: this object is only ever constructed
internally from already-valid data (never parsed from untrusted input), so
we don't need Pydantic's validation machinery here. Validation of raw
request bodies happens once, at the API boundary, in `api/schemas`.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
