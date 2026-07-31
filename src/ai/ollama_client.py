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

from src.models.chat import ChatTurnResult, StreamChunk
from src.models.message import Message, MessageRole, ToolCall

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

    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
    ) -> ChatTurnResult:
        """Send a full conversation and get back one complete reply."""
        resolved_model = model or self._default_model
        payload = self._build_payload(
            messages, resolved_model, stream=False, tools=tools, images=images
        )

        started = time.perf_counter()
        response = await self._http_client.post("/api/chat", json=payload)
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - started) * 1000

        message = response.json()["message"]
        logger.info("Ollama chat completed in %.0fms (model=%s)", elapsed_ms, resolved_model)

        reply = Message(
            role=MessageRole.ASSISTANT,
            content=message.get("content", ""),
            tool_calls=_parse_tool_calls(message.get("tool_calls")),
        )
        return ChatTurnResult(reply=reply, model=resolved_model, latency_ms=elapsed_ms)

    async def stream_chat(
        self,
        messages: list[Message],
        model: str | None = None,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Send a full conversation and yield chunks as they're generated.

        Ollama's streaming format is newline-delimited JSON objects (NDJSON),
        one per token/chunk, with a final object carrying `"done": true`. A
        chunk requesting a tool call carries `tool_calls` with empty content
        (verified against qwen2.5:3b-instruct) — normal replies stream real
        content with `tool_calls` always empty, so this doesn't affect the
        common case's streaming behavior at all.
        """
        resolved_model = model or self._default_model
        payload = self._build_payload(
            messages, resolved_model, stream=True, tools=tools, images=images
        )

        async with self._http_client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                message = chunk.get("message", {})
                content = message.get("content", "")
                tool_calls = _parse_tool_calls(message.get("tool_calls"))
                if content or tool_calls:
                    yield StreamChunk(content=content, tool_calls=tool_calls)
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
    def _build_payload(
        messages: list[Message],
        model: str,
        *,
        stream: bool,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
    ) -> dict:
        wire_messages = [OllamaClient._serialize_message(m) for m in messages]
        if images and wire_messages:
            wire_messages[-1]["images"] = images
        payload: dict = {"model": model, "messages": wire_messages, "stream": stream}
        if tools:
            payload["tools"] = tools
        return payload

    @staticmethod
    def _serialize_message(message: Message) -> dict:
        wire: dict = {"role": message.role.value, "content": message.content}
        if message.tool_calls:
            wire["tool_calls"] = [
                {"function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in message.tool_calls
            ]
        return wire


def _parse_tool_calls(raw: list[dict] | None) -> list[ToolCall] | None:
    if not raw:
        return None
    return [
        ToolCall(
            id=tc.get("id", ""),
            name=tc["function"]["name"],
            arguments=tc["function"]["arguments"],
        )
        for tc in raw
    ]
