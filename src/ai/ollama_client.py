"""Thin async client around Ollama's REST API.

This is the only module in the codebase that knows Ollama's request/response
shape. Everything above it (services, API routes) speaks in domain models
(`Message`, `ChatTurnResult`), so swapping the backend later touches only
this file.
"""

import json
import logging
import time
from collections.abc import AsyncIterator

import httpx

from src.models.chat import ChatTurnResult
from src.models.message import Message, MessageRole

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        default_model: str,
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        self._http_client = http_client
        self._default_model = default_model
        self._embedding_model = embedding_model

    async def chat(self, messages: list[Message], model: str | None = None) -> ChatTurnResult:
        """Send a full conversation and get back one complete reply."""
        resolved_model = model or self._default_model
        payload = self._build_payload(messages, resolved_model, stream=False)

        started = time.perf_counter()
        response = await self._http_client.post("/api/chat", json=payload)
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - started) * 1000

        content = response.json()["message"]["content"]
        logger.info("Ollama chat completed in %.0fms (model=%s)", elapsed_ms, resolved_model)

        reply = Message(role=MessageRole.ASSISTANT, content=content)
        return ChatTurnResult(reply=reply, model=resolved_model, latency_ms=elapsed_ms)

    async def stream_chat(
        self, messages: list[Message], model: str | None = None
    ) -> AsyncIterator[str]:
        """Send a full conversation and yield reply content as it's generated.

        Ollama's streaming format is newline-delimited JSON objects (NDJSON),
        one per token/chunk, with a final object carrying `"done": true`.
        """
        resolved_model = model or self._default_model
        payload = self._build_payload(messages, resolved_model, stream=True)

        async with self._http_client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
                if chunk.get("done"):
                    break

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, in the same order they were given."""
        response = await self._http_client.post(
            "/api/embed", json={"model": self._embedding_model, "input": texts}
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    @staticmethod
    def _build_payload(messages: list[Message], model: str, *, stream: bool) -> dict:
        return {
            "model": model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "stream": stream,
        }
