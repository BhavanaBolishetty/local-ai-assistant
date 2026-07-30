"""Async engine/session setup for the SQLite-backed persistence layer."""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import get_settings
from src.db.models import Base

settings = get_settings()
_engine = create_async_engine(settings.database_url)
_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def init_db() -> None:
    """Create tables that don't exist yet. Call once, at process startup."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding one `AsyncSession` per request."""
    async with _session_factory() as session:
        yield session
