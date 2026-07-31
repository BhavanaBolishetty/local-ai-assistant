"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from src.api.routes.chat import router as chat_router
from src.api.routes.conversations import router as conversations_router
from src.api.routes.documents import router as documents_router
from src.api.routes.voice import router as voice_router
from src.core.config import get_settings
from src.core.logging import configure_logging
from src.db.session import init_db
from src.rag.vector_store import VectorStore
from src.voice.synthesis import SynthesisService
from src.voice.transcription import TranscriptionService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage resources that must live for the whole process lifetime.

    A single connection-pooled httpx.AsyncClient is created here instead of
    per-request, then closed cleanly on shutdown. `VectorStore` similarly
    holds a persistent Chroma connection, and the voice services hold
    loaded models, so all are built once here rather than per-request.
    """
    configure_logging()
    settings = get_settings()
    await init_db()
    app.state.http_client = httpx.AsyncClient(
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_request_timeout_seconds,
    )
    app.state.vector_store = VectorStore(settings.chroma_path)
    app.state.transcription_service = await TranscriptionService.create(
        settings.whisper_model_size, settings.whisper_download_root
    )
    app.state.synthesis_service = await SynthesisService.create(
        settings.piper_voice, settings.piper_voices_dir
    )
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="Local AI Assistant API", version="0.1.0", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(documents_router)
app.include_router(voice_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
