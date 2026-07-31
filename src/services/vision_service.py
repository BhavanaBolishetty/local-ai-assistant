"""Business logic for a one-shot, image-attached chat turn.

Distinct from `ChatService` because the vision model (moondream) has no
tool-calling support — a vision turn is a single call to a different
model, not routed through `AgentRunner`/`RagRetriever`. The image itself
is never persisted or resent on later turns (one-shot, by design): only
the text Q&A becomes part of the conversation history.
"""

import base64
import logging
from collections.abc import AsyncIterator

from src.ai.ollama_client import OllamaClient
from src.models.message import Message, MessageRole
from src.repositories.conversation_repository import ConversationRepository
from src.utils.text import derive_title

logger = logging.getLogger(__name__)


class VisionService:
    def __init__(
        self,
        ollama_client: OllamaClient,
        repository: ConversationRepository,
        vision_model: str,
    ) -> None:
        self._ollama_client = ollama_client
        self._repository = repository
        self._vision_model = vision_model

    async def ask_streaming(
        self, conversation_id: str, text: str, image_bytes: bytes
    ) -> AsyncIterator[str]:
        stored_text = f"[Image attached] {text}"
        conversation = await self._repository.get_or_create(conversation_id)
        if not conversation.messages:
            await self._repository.set_title(conversation_id, derive_title(stored_text))
        user_message = Message(role=MessageRole.USER, content=stored_text)
        await self._repository.add_message(conversation_id, user_message)

        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        full_text = ""
        # No system prompt: moondream is a narrow image-QA model, not an
        # instruct chat model — this project's assistant-persona prompt
        # isn't written for it.
        async for chunk in self._ollama_client.stream_chat(
            [user_message], model=self._vision_model, images=[image_b64]
        ):
            if chunk.content:
                full_text += chunk.content
                yield chunk.content

        reply = Message(role=MessageRole.ASSISTANT, content=full_text)
        await self._repository.add_message(conversation_id, reply)
