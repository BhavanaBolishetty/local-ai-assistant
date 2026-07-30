"""Business logic for a chat turn.

Thin orchestrator over `OllamaClient`, `ConversationRepository` (Phase 2
persistence), `RagRetriever` (Phase 3 RAG context injection), and now
`ToolRegistry` (Phase 4 tool calling) — the seam the earlier phases'
docstrings called out, wired up without any change to the API layer's
shape. Phase 5 (agent planning) will plug in the same way.
"""

import logging
from collections.abc import AsyncIterator

from src.ai.ollama_client import OllamaClient
from src.models.chat import ChatStreamEvent, ChatTurnResult
from src.models.message import Message, MessageRole, ToolCall
from src.rag.retriever import RagRetriever
from src.repositories.conversation_repository import ConversationRepository
from src.tools.registry import ToolRegistry
from src.utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_TITLE_MAX_LENGTH = 50
_MAX_TOOL_ITERATIONS = 3


class ChatService:
    def __init__(
        self,
        ollama_client: OllamaClient,
        repository: ConversationRepository,
        rag_retriever: RagRetriever | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._ollama_client = ollama_client
        self._repository = repository
        self._rag_retriever = rag_retriever
        self._tool_registry = tool_registry

    async def send_message(
        self,
        conversation_history: list[Message],
        conversation_id: str,
        model: str | None = None,
    ) -> ChatTurnResult:
        new_message = await self._persist_new_user_message(conversation_history, conversation_id)
        context = await self._build_rag_context(new_message)
        history = self._with_system_prompt(conversation_history, context)
        tools_schema = self._tools_schema()

        result: ChatTurnResult | None = None
        for iteration in range(_MAX_TOOL_ITERATIONS):
            offer_tools = tools_schema if iteration < _MAX_TOOL_ITERATIONS - 1 else None
            result = await self._ollama_client.chat(history, model=model, tools=offer_tools)
            if not result.reply.tool_calls:
                break
            self._apply_tool_calls(history, result.reply.tool_calls)

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
        tools_schema = self._tools_schema()

        full_text = ""
        for iteration in range(_MAX_TOOL_ITERATIONS):
            offer_tools = tools_schema if iteration < _MAX_TOOL_ITERATIONS - 1 else None
            round_tool_calls: list[ToolCall] = []
            full_text = ""
            stream = self._ollama_client.stream_chat(history, model=model, tools=offer_tools)
            async for chunk in stream:
                if chunk.tool_calls:
                    round_tool_calls.extend(chunk.tool_calls)
                if chunk.content:
                    full_text += chunk.content
                    yield ChatStreamEvent(type="content", text=chunk.content)

            if not round_tool_calls:
                break

            for call in round_tool_calls:
                yield ChatStreamEvent(type="tool_call", tool_name=call.name)
            self._apply_tool_calls(history, round_tool_calls)

        reply = Message(role=MessageRole.ASSISTANT, content=full_text)
        await self._repository.add_message(conversation_id, reply)

    def _tools_schema(self) -> list[dict] | None:
        return self._tool_registry.to_ollama_schema() if self._tool_registry else None

    def _apply_tool_calls(self, history: list[Message], tool_calls: list[ToolCall]) -> None:
        """Append the assistant's tool-call request and each tool's result
        to the (scratch, per-turn) working history so the next round can
        see them. Never persisted — only the original user message and
        the final assistant reply get saved via `ConversationRepository`.
        """
        history.append(Message(role=MessageRole.ASSISTANT, content="", tool_calls=tool_calls))
        for call in tool_calls:
            result_text = self._tool_registry.execute(call.name, call.arguments)
            history.append(
                Message(role=MessageRole.TOOL, content=result_text, tool_call_id=call.id)
            )

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
            title = self._derive_title(new_message.content)
            await self._repository.set_title(conversation_id, title)
        await self._repository.add_message(conversation_id, new_message)
        return new_message

    async def _build_rag_context(self, new_message: Message) -> str | None:
        if self._rag_retriever is None:
            return None
        return await self._rag_retriever.build_context(new_message.content)

    @staticmethod
    def _derive_title(content: str) -> str:
        stripped = content.strip()
        if len(stripped) <= _TITLE_MAX_LENGTH:
            return stripped
        return stripped[: _TITLE_MAX_LENGTH - 3] + "..."

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
