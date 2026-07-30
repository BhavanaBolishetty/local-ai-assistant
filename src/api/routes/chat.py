"""Chat endpoints: the HTTP boundary for talking to the assistant."""

import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.ai.ollama_client import OllamaClient
from src.api.schemas.chat import ChatRequest, ChatResponse
from src.core.config import Settings, get_settings
from src.db.session import get_db_session
from src.models.message import Message, MessageRole
from src.rag.retriever import RagRetriever
from src.rag.vector_store import VectorStore
from src.repositories.conversation_repository import ConversationRepository
from src.services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def get_chat_service(
    request: Request,
    settings: Settings = Depends(get_settings),
    db_session: AsyncSession = Depends(get_db_session),
) -> ChatService:
    """Build a ChatService using the shared, connection-pooled http client
    and vector store created once at app startup (see api/main.py's lifespan)."""
    ollama_client = OllamaClient(
        http_client=request.app.state.http_client,
        default_model=settings.ollama_model,
        embedding_model=settings.ollama_embedding_model,
    )
    repository = ConversationRepository(db_session)
    vector_store: VectorStore = request.app.state.vector_store
    rag_retriever = RagRetriever(
        ollama_client,
        vector_store,
        top_k=settings.rag_top_k,
        max_distance=settings.rag_max_distance,
    )
    return ChatService(ollama_client, repository, rag_retriever)


def _to_domain_messages(request: ChatRequest) -> list[Message]:
    return [Message(role=MessageRole(m.role), content=m.content) for m in request.messages]


@router.post("", response_model=ChatResponse)
async def send_chat_message(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid4())
    result = await chat_service.send_message(
        _to_domain_messages(request), conversation_id, model=request.model
    )
    return ChatResponse(
        reply=result.reply.content,
        model=result.model,
        latency_ms=result.latency_ms,
        conversation_id=conversation_id,
    )


@router.post("/stream")
async def stream_chat_message(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    conversation_id = request.conversation_id or str(uuid4())
    history = _to_domain_messages(request)
    return StreamingResponse(
        chat_service.stream_message(history, conversation_id, model=request.model),
        media_type="text/plain",
        headers={"X-Conversation-Id": conversation_id},
    )
