"""Business logic for a chat turn.

Thin orchestrator over `OllamaClient`, `ConversationRepository` (Phase 2
persistence), `RagRetriever` (Phase 3 RAG context injection), and now
`AgentRunner` (Phase 5's multi-step tool-calling loop) — the seam the
earlier phases' docstrings called out, wired up without any change to
the API layer's shape.
"""

import logging
from collections.abc import AsyncIterator

from src.agents.runner import AgentRunner
from src.ai.ollama_client import OllamaClient
from src.models.chat import ChatStreamEvent, ChatTurnResult
from src.models.message import Message, MessageRole
from src.rag.retriever import RagRetriever
from src.repositories.conversation_repository import ConversationRepository
from src.utils.prompt_loader import load_prompt
from src.utils.text import derive_title

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        ollama_client: OllamaClient,
        repository: ConversationRepository,
        rag_retriever: RagRetriever | None = None,
        agent_runner: AgentRunner | None = None,
    ) -> None:
        self._ollama_client = ollama_client
        self._repository = repository
        self._rag_retriever = rag_retriever
        self._agent_runner = agent_runner

    async def send_message(
        self,
        conversation_history: list[Message],
        conversation_id: str,
        model: str | None = None,
    ) -> ChatTurnResult:
        new_message = await self._persist_new_user_message(conversation_history, conversation_id)
        context = await self._build_rag_context(new_message)
        history = self._with_system_prompt(conversation_history, context)

        if self._agent_runner is not None:
            result = await self._agent_runner.run(history, model=model)
        else:
            result = await self._ollama_client.chat(history, model=model)

        await self._repository.add_message(conversation_id, result.reply)
        return result

    async def stream_message(
        self,
        conversation_history: list[Message],
        conversation_id: str,
        model: str | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        new_message = await self._persist_new_user_message(conversation_history, conversation_id)
        context = await self._build_rag_context(new_message)
        history = self._with_system_prompt(conversation_history, context)

        full_text = ""
        if self._agent_runner is not None:
            async for event in self._agent_runner.run_streaming(history, model=model):
                if event.type == "content":
                    full_text += event.text
                yield event
        else:
            async for chunk in self._ollama_client.stream_chat(history, model=model):
                if chunk.content:
                    full_text += chunk.content
                    yield ChatStreamEvent(type="content", text=chunk.content)

        reply = Message(role=MessageRole.ASSISTANT, content=full_text)
        await self._repository.add_message(conversation_id, reply)

    async def _persist_new_user_message(
        self, conversation_history: list[Message], conversation_id: str
    ) -> Message:
        """Get-or-create the conversation row and append the latest user
        message (the last entry in the client-supplied history) to it.

        The client resends the full history each turn, so only the newest
        message is new to the database.
        """
        conversation = await self._repository.get_or_create(conversation_id)
        new_message = conversation_history[-1]
        if not conversation.messages:
            await self._repository.set_title(conversation_id, derive_title(new_message.content))
        await self._repository.add_message(conversation_id, new_message)
        return new_message

    async def _build_rag_context(self, new_message: Message) -> str | None:
        if self._rag_retriever is None:
            return None
        return await self._rag_retriever.build_context(new_message.content)

    @staticmethod
    def _with_system_prompt(
        history: list[Message], rag_context: str | None = None
    ) -> list[Message]:
        if not history:
            raise ValueError("conversation_history must contain at least one message")
        if history[0].role == MessageRole.SYSTEM:
            return history

        prompt = load_prompt("system_prompt")
        if rag_context:
            prompt = f"{prompt}\n\nRelevant context from the user's documents:\n\n{rag_context}"
        system_message = Message(role=MessageRole.SYSTEM, content=prompt)
        return [system_message, *history]
