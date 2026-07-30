"""Conversation history endpoints: list/load/delete past conversations."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.chat import ChatMessageIn
from src.api.schemas.conversation import ConversationOut, ConversationSummaryOut
from src.db.session import get_db_session
from src.repositories.conversation_repository import ConversationRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


def get_conversation_repository(
    db_session: AsyncSession = Depends(get_db_session),
) -> ConversationRepository:
    return ConversationRepository(db_session)


@router.get("", response_model=list[ConversationSummaryOut])
async def list_conversations(
    repository: ConversationRepository = Depends(get_conversation_repository),
) -> list[ConversationSummaryOut]:
    summaries = await repository.list_summaries()
    return [
        ConversationSummaryOut(id=c.id, title=c.title, created_at=c.created_at) for c in summaries
    ]


@router.get("/{conversation_id}", response_model=ConversationOut)
async def get_conversation(
    conversation_id: str,
    repository: ConversationRepository = Depends(get_conversation_repository),
) -> ConversationOut:
    conversation = await repository.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationOut(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        messages=[
            ChatMessageIn(role=m.role.value, content=m.content) for m in conversation.messages
        ],
    )


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    repository: ConversationRepository = Depends(get_conversation_repository),
) -> None:
    await repository.delete(conversation_id)
