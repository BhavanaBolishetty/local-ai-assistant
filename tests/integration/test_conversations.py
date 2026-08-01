"""Integration tests for /conversations not already covered by test_chat.py."""


async def test_get_nonexistent_conversation_returns_404(client) -> None:
    response = await client.get("/conversations/does-not-exist")

    assert response.status_code == 404


async def test_list_conversations_empty_by_default(client) -> None:
    response = await client.get("/conversations")

    assert response.status_code == 200
    assert response.json() == []


async def test_delete_conversation(client) -> None:
    sent = await client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    conversation_id = sent.json()["conversation_id"]

    delete = await client.delete(f"/conversations/{conversation_id}")
    assert delete.status_code == 204

    after = await client.get(f"/conversations/{conversation_id}")
    assert after.status_code == 404
