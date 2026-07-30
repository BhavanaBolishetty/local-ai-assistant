"""Request/response contracts for the /conversations endpoints."""

from datetime import datetime

from pydantic import BaseModel

from src.api.schemas.chat import ChatMessageIn


class ConversationSummaryOut(BaseModel):
    id: str
    title: str
    created_at: datetime


class ConversationOut(ConversationSummaryOut):
    messages: list[ChatMessageIn]
