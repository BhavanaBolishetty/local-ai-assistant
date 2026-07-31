"""Chat endpoints: the HTTP boundary for talking to the assistant."""

import json
import logging
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.runner import AgentRunner
from src.ai.ollama_client import OllamaClient
from src.api.schemas.chat import AgentStepOut, ChatRequest, ChatResponse
from src.core.config import Settings, get_settings
from src.db.session import get_db_session
from src.models.chat import ChatStreamEvent
from src.models.message import Message, MessageRole
from src.rag.retriever import RagRetriever
from src.rag.vector_store import VectorStore
from src.repositories.conversation_repository import ConversationRepository
from src.services.chat_service import ChatService
from src.tools.calculator import calculator_tool
from src.tools.date_duration import date_duration_tool
from src.tools.datetime_tool import current_datetime_tool
from src.tools.document_search import build_search_documents_tool
from src.tools.registry import ToolRegistry

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
    tool_registry = ToolRegistry(
        tools=[
            calculator_tool,
            current_datetime_tool,
            date_duration_tool,
            build_search_documents_tool(rag_retriever),
        ]
    )
    agent_runner = AgentRunner(ollama_client, tool_registry)
    return ChatService(ollama_client, repository, rag_retriever, agent_runner)


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
        steps=[
            AgentStepOut(tool_name=s.tool_name, arguments=s.arguments, result=s.result)
            for s in result.steps
        ],
    )


async def _to_ndjson(events: AsyncIterator[ChatStreamEvent]) -> AsyncIterator[str]:
    """Serialize each event to one NDJSON line.

    Content and step events are distinguished here (rather than inline in
    the reply text) so the client can show a step trace without it
    leaking into the persisted/displayed message.
    """
    async for event in events:
        if event.type == "step" and event.step is not None:
            payload = {
                "type": "step",
                "tool_name": event.step.tool_name,
                "arguments": event.step.arguments,
                "result": event.step.result,
            }
        else:
            payload = {"type": "content", "text": event.text}
        yield json.dumps(payload) + "\n"


@router.post("/stream")
async def stream_chat_message(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    conversation_id = request.conversation_id or str(uuid4())
    history = _to_domain_messages(request)
    events = chat_service.stream_message(history, conversation_id, model=request.model)
    return StreamingResponse(
        _to_ndjson(events),
        media_type="application/x-ndjson",
        headers={"X-Conversation-Id": conversation_id},
    )
