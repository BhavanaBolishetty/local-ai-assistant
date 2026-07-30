"""Persistence for conversations, abstracting SQLite away from services.

Translates between `ConversationORM`/`MessageORM` (the on-disk shape) and
the framework-agnostic domain dataclasses in `src/models/` (what services
and API routes actually work with) — mirrors how `OllamaClient` is the only
module that knows Ollama's wire format.
"""

from sqlalchemy import delete as sql_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import noload, selectinload

from src.db.models import ConversationORM, MessageORM
from src.models.conversation import Conversation
from src.models.message import Message, MessageRole


def _to_domain_message(row: MessageORM) -> Message:
    return Message(role=MessageRole(row.role), content=row.content, created_at=row.created_at)


def _to_domain_conversation(row: ConversationORM, *, with_messages: bool) -> Conversation:
    return Conversation(
        id=row.id,
        title=row.title,
        created_at=row.created_at,
        messages=[_to_domain_message(m) for m in row.messages] if with_messages else [],
    )


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _fetch(self, conversation_id: str, *, with_messages: bool) -> ConversationORM | None:
        """Load a conversation row, always eagerly (never lazily).

        `session.get()` can return a stale, already-cached instance whose
        `messages` was never populated, and touching it afterward would hit
        the `lazy="raise"` guard. An explicit `select()` with `selectinload`
        forces the relationship to be (re)loaded as part of this query.
        """
        if with_messages:
            loader_option = selectinload(ConversationORM.messages)
        else:
            loader_option = noload(ConversationORM.messages)
        stmt = (
            select(ConversationORM)
            .where(ConversationORM.id == conversation_id)
            .options(loader_option)
        )
        return await self._session.scalar(stmt)

    async def get_or_create(self, conversation_id: str) -> Conversation:
        row = await self._fetch(conversation_id, with_messages=True)
        if row is None:
            new_row = ConversationORM(id=conversation_id)
            self._session.add(new_row)
            await self._session.commit()
            return _to_domain_conversation(new_row, with_messages=False)
        return _to_domain_conversation(row, with_messages=True)

    async def add_message(self, conversation_id: str, message: Message) -> None:
        self._session.add(
            MessageORM(
                conversation_id=conversation_id,
                role=message.role.value,
                content=message.content,
                created_at=message.created_at,
            )
        )
        await self._session.commit()

    async def get(self, conversation_id: str) -> Conversation | None:
        row = await self._fetch(conversation_id, with_messages=True)
        return _to_domain_conversation(row, with_messages=True) if row is not None else None

    async def list_summaries(self) -> list[Conversation]:
        stmt = (
            select(ConversationORM)
            .options(noload(ConversationORM.messages))
            .order_by(ConversationORM.created_at.desc())
        )
        result = await self._session.scalars(stmt)
        return [_to_domain_conversation(row, with_messages=False) for row in result]

    async def set_title(self, conversation_id: str, title: str) -> None:
        row = await self._session.get(ConversationORM, conversation_id)
        if row is not None:
            row.title = title
            await self._session.commit()

    async def delete(self, conversation_id: str) -> None:
        """Bulk-delete messages then the conversation row directly, rather
        than loading the `messages` relationship to let ORM cascade handle
        it (which `lazy="raise"` would reject)."""
        await self._session.execute(
            sql_delete(MessageORM).where(MessageORM.conversation_id == conversation_id)
        )
        await self._session.execute(
            sql_delete(ConversationORM).where(ConversationORM.id == conversation_id)
        )
        await self._session.commit()
