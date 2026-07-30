"""Chat endpoints: the HTTP boundary for talking to the assistant."""

import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from src.ai.ollama_client import OllamaClient
from src.api.schemas.chat import ChatRequest, ChatResponse
from src.core.config import Settings, get_settings
from src.models.message import Message, MessageRole
from src.services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def get_chat_service(
    request: Request, settings: Settings = Depends(get_settings)
) -> ChatService:
    """Build a ChatService using the shared, connection-pooled http client
    created once at app startup (see api/main.py's lifespan)."""
    ollama_client = OllamaClient(
        http_client=request.app.state.http_client, default_model=settings.ollama_model
    )
    return ChatService(ollama_client)


def _to_domain_messages(request: ChatRequest) -> list[Message]:
    return [Message(role=MessageRole(m.role), content=m.content) for m in request.messages]


@router.post("", response_model=ChatResponse)
async def send_chat_message(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    result = await chat_service.send_message(_to_domain_messages(request), model=request.model)
    return ChatResponse(
        reply=result.reply.content, model=result.model, latency_ms=result.latency_ms
    )


@router.post("/stream")
async def stream_chat_message(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    history = _to_domain_messages(request)
    return StreamingResponse(
        chat_service.stream_message(history, model=request.model),
        media_type="text/plain",
    )
