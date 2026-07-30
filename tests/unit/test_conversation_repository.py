"""Tests for ConversationRepository against an in-memory SQLite database."""

from src.models.message import Message, MessageRole
from src.repositories.conversation_repository import ConversationRepository


async def test_get_or_create_creates_new_conversation(repository: ConversationRepository) -> None:
    conversation = await repository.get_or_create("conv-1")

    assert conversation.id == "conv-1"
    assert conversation.title == "New Chat"
    assert conversation.messages == []


async def test_get_or_create_is_idempotent(repository: ConversationRepository) -> None:
    first = await repository.get_or_create("conv-1")
    await repository.set_title("conv-1", "Renamed")

    second = await repository.get_or_create("conv-1")

    assert first.id == second.id
    assert second.title == "Renamed"


async def test_add_message_and_get_preserves_order(repository: ConversationRepository) -> None:
    await repository.get_or_create("conv-1")
    await repository.add_message("conv-1", Message(role=MessageRole.USER, content="hi"))
    await repository.add_message("conv-1", Message(role=MessageRole.ASSISTANT, content="hello"))

    conversation = await repository.get("conv-1")

    assert conversation is not None
    assert [m.content for m in conversation.messages] == ["hi", "hello"]
    assert conversation.messages[0].role == MessageRole.USER
    assert conversation.messages[1].role == MessageRole.ASSISTANT


async def test_get_returns_none_for_missing_conversation(
    repository: ConversationRepository,
) -> None:
    assert await repository.get("missing") is None


async def test_list_summaries_orders_newest_first_and_omits_messages(
    repository: ConversationRepository,
) -> None:
    await repository.get_or_create("conv-1")
    await repository.add_message("conv-1", Message(role=MessageRole.USER, content="hi"))
    await repository.get_or_create("conv-2")

    summaries = await repository.list_summaries()

    assert [s.id for s in summaries] == ["conv-2", "conv-1"]
    assert all(s.messages == [] for s in summaries)


async def test_delete_removes_conversation_and_messages(
    repository: ConversationRepository,
) -> None:
    await repository.get_or_create("conv-1")
    await repository.add_message("conv-1", Message(role=MessageRole.USER, content="hi"))

    await repository.delete("conv-1")

    assert await repository.get("conv-1") is None
