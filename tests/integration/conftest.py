"""Builds the real FastAPI app for integration tests, with every route's
model/persistence dependency overridden so tests run fast, offline, and
never touch the real data/ directory.

Deliberately does *not* run the app's real `lifespan` (see
`src/api/main.py`) — that would trigger a real init_db, a real Chroma
connection at the dev `chroma_path`, and real Whisper/Piper model
downloads. Every route dependency is overridden instead, so nothing ever
reads the (never-populated) `app.state` those would normally set up.
"""

from collections.abc import AsyncIterator

import pytest
from fakes import FakeOllamaClient, FakeSynthesisService, FakeTranscriptionService
from fastapi import Depends
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.main import app
from src.api.routes.chat import get_chat_service
from src.api.routes.documents import get_document_service
from src.api.routes.vision import get_vision_service
from src.api.routes.voice import get_synthesis_service, get_transcription_service
from src.db.models import Base
from src.db.session import get_db_session
from src.rag.vector_store import VectorStore
from src.repositories.conversation_repository import ConversationRepository
from src.services.chat_service import ChatService
from src.services.document_service import DocumentService
from src.services.vision_service import VisionService


@pytest.fixture
async def client(tmp_path) -> AsyncIterator[AsyncClient]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fake_ollama = FakeOllamaClient()
    vector_store = VectorStore(str(tmp_path / "chroma"))

    def override_get_chat_service(
        db_session: AsyncSession = Depends(get_db_session),
    ) -> ChatService:
        return ChatService(fake_ollama, ConversationRepository(db_session))

    def override_get_document_service() -> DocumentService:
        return DocumentService(fake_ollama, vector_store)

    def override_get_vision_service(
        db_session: AsyncSession = Depends(get_db_session),
    ) -> VisionService:
        return VisionService(fake_ollama, ConversationRepository(db_session), "fake-vision-model")

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_chat_service] = override_get_chat_service
    app.dependency_overrides[get_document_service] = override_get_document_service
    app.dependency_overrides[get_vision_service] = override_get_vision_service
    app.dependency_overrides[get_transcription_service] = lambda: FakeTranscriptionService()
    app.dependency_overrides[get_synthesis_service] = lambda: FakeSynthesisService()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client

    app.dependency_overrides.clear()
    await engine.dispose()
