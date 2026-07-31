"""Request/response contracts for the /chat endpoints.

Pydantic here (not dataclasses) because this is the HTTP boundary: these
models parse and validate untrusted client input before anything else
touches it.
"""

from pydantic import BaseModel, Field


class ChatMessageIn(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessageIn] = Field(min_length=1)
    model: str | None = None
    conversation_id: str | None = None


class AgentStepOut(BaseModel):
    tool_name: str
    arguments: dict
    result: str


class ChatResponse(BaseModel):
    reply: str
    model: str
    latency_ms: float
    conversation_id: str
    steps: list[AgentStepOut] = []
