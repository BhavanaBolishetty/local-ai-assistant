"""Business logic for a chat turn.

Currently a thin orchestrator over `OllamaClient`, but this is the seam
where Phase 2 (persistence), Phase 3 (RAG context injection), and Phase 5
(agent tool-use decisions) will plug in — without the API layer changing.
"""

import logging
from collections.abc import AsyncIterator

from src.ai.ollama_client import OllamaClient
from src.models.chat import ChatTurnResult
from src.models.message import Message, MessageRole
from src.utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, ollama_client: OllamaClient) -> None:
        self._ollama_client = ollama_client

    async def send_message(
        self, conversation_history: list[Message], model: str | None = None
    ) -> ChatTurnResult:
        history = self._with_system_prompt(conversation_history)
        return await self._ollama_client.chat(history, model=model)

    async def stream_message(
        self, conversation_history: list[Message], model: str | None = None
    ) -> AsyncIterator[str]:
        history = self._with_system_prompt(conversation_history)
        async for chunk in self._ollama_client.stream_chat(history, model=model):
            yield chunk

    @staticmethod
    def _with_system_prompt(history: list[Message]) -> list[Message]:
        if not history:
            raise ValueError("conversation_history must contain at least one message")
        if history[0].role == MessageRole.SYSTEM:
            return history
        system_message = Message(role=MessageRole.SYSTEM, content=load_prompt("system_prompt"))
        return [system_message, *history]
