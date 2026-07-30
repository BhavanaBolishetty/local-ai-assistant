"""SQLAlchemy ORM models — the on-disk row shapes.

These are DB-only: nothing outside `src/db` and `src/repositories` should
import them. Services and API routes only ever see the framework-agnostic
domain dataclasses in `src/models/`, which `ConversationRepository` builds
from these rows.
"""

from datetime import UTC, datetime

from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ConversationORM(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(default="New Chat")
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

    # lazy="raise": under the async engine, an implicit (lazy) load can't
    # actually perform I/O and blows up with a confusing MissingGreenlet
    # error. Forcing "raise" means any spot that forgot to eager-load this
    # relationship fails loudly and immediately instead of depending on
    # identity-map cache state. See ConversationRepository, which always
    # loads this explicitly via `selectinload`/`noload`.
    messages: Mapped[list["MessageORM"]] = relationship(
        back_populates="conversation",
        order_by="MessageORM.id",
        lazy="raise",
    )


class MessageORM(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    role: Mapped[str]
    content: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

    conversation: Mapped[ConversationORM] = relationship(back_populates="messages")
