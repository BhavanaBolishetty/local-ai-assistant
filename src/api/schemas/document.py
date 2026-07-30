"""Request/response contracts for the /documents endpoints."""

from datetime import datetime

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: str
    filename: str
    uploaded_at: datetime
    chunk_count: int
