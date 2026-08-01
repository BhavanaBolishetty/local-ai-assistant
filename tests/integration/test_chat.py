"""Integration tests for /chat: routing, validation, and persistence,
against a fake OllamaClient rather than a real model."""


async def test_send_chat_message_returns_fake_reply_and_persists(client) -> None:
    response = await client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "This is a fake reply."
    assert body["steps"] == []
    conversation_id = body["conversation_id"]
    assert conversation_id

    history = await client.get(f"/conversations/{conversation_id}")
    assert history.status_code == 200
    messages = history.json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "Hello"
    assert messages[1]["content"] == "This is a fake reply."


async def test_send_chat_message_reuses_conversation_id(client) -> None:
    first = await client.post(
        "/chat",
        json={
            "conversation_id": "fixed-id",
            "messages": [{"role": "user", "content": "First"}],
        },
    )
    assert first.json()["conversation_id"] == "fixed-id"

    second = await client.post(
        "/chat",
        json={
            "conversation_id": "fixed-id",
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "This is a fake reply."},
                {"role": "user", "content": "Second"},
            ],
        },
    )
    assert second.json()["conversation_id"] == "fixed-id"

    history = await client.get("/conversations/fixed-id")
    contents = [m["content"] for m in history.json()["messages"]]
    assert contents == ["First", "This is a fake reply.", "Second", "This is a fake reply."]


async def test_empty_messages_is_rejected(client) -> None:
    response = await client.post("/chat", json={"messages": []})

    assert response.status_code == 422


async def test_invalid_role_is_rejected(client) -> None:
    response = await client.post(
        "/chat",
        json={"messages": [{"role": "system-admin", "content": "hi"}]},
    )

    assert response.status_code == 422
