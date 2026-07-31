"""Vision endpoint: ask a question about an attached image."""

import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, Form, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.ai.ollama_client import OllamaClient
from src.core.config import Settings, get_settings
from src.db.session import get_db_session
from src.repositories.conversation_repository import ConversationRepository
from src.services.vision_service import VisionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vision", tags=["vision"])


def get_vision_service(
    request: Request,
    settings: Settings = Depends(get_settings),
    db_session: AsyncSession = Depends(get_db_session),
) -> VisionService:
    ollama_client = OllamaClient(
        http_client=request.app.state.http_client,
        default_model=settings.ollama_model,
        embedding_model=settings.ollama_embedding_model,
    )
    repository = ConversationRepository(db_session)
    return VisionService(ollama_client, repository, settings.ollama_vision_model)


@router.post("/ask")
async def ask_about_image(
    image: UploadFile,
    text: str = Form(...),
    conversation_id: str | None = Form(None),
    vision_service: VisionService = Depends(get_vision_service),
) -> StreamingResponse:
    resolved_conversation_id = conversation_id or str(uuid4())
    image_bytes = await image.read()
    return StreamingResponse(
        vision_service.ask_streaming(resolved_conversation_id, text, image_bytes),
        media_type="text/plain",
        headers={"X-Conversation-Id": resolved_conversation_id},
    )
