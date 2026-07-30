"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from src.api.routes.chat import router as chat_router
from src.core.config import get_settings
from src.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage resources that must live for the whole process lifetime.

    A single connection-pooled httpx.AsyncClient is created here instead of
    per-request, then closed cleanly on shutdown.
    """
    configure_logging()
    settings = get_settings()
    app.state.http_client = httpx.AsyncClient(
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_request_timeout_seconds,
    )
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="Local AI Assistant API", version="0.1.0", lifespan=lifespan)
app.include_router(chat_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
