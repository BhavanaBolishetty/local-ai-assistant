"""Fake collaborators for integration tests — stand in for Ollama and the
local voice models so route tests are fast, deterministic, and don't
need Ollama/Whisper/Piper actually running."""

from collections.abc import AsyncIterator

from src.models.chat import ChatTurnResult, StreamChunk
from src.models.message import Message, MessageRole


class FakeOllamaClient:
    """Duck-types `OllamaClient`'s `chat`/`stream_chat`/`embed` methods
    with a canned reply — enough for every service that depends on
    `OllamaClient` (`ChatService`, `DocumentService`, `VisionService`,
    `RagRetriever`, `AgentRunner`), since none of them care that the
    reply isn't a real model's output."""

    def __init__(self, reply_text: str = "This is a fake reply.") -> None:
        self.reply_text = reply_text

    async def chat(self, messages, model=None, tools=None, images=None) -> ChatTurnResult:
        reply = Message(role=MessageRole.ASSISTANT, content=self.reply_text)
        return ChatTurnResult(reply=reply, model=model or "fake-model", latency_ms=1.0)

    async def stream_chat(
        self, messages, model=None, tools=None, images=None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(content=self.reply_text)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeTranscriptionService:
    async def transcribe(self, audio) -> str:
        return "fake transcribed text"


class FakeSynthesisService:
    async def synthesize(self, text: str) -> bytes:
        return b"RIFF\x00\x00\x00\x00WAVEfake-audio-bytes"
