"""Request/response contracts for the /voice endpoints."""

from pydantic import BaseModel, Field


class TranscriptionOut(BaseModel):
    text: str


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1)
