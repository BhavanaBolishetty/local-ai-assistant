"""Integration test for /vision/ask against a fake OllamaClient."""


async def test_ask_about_image_streams_reply_and_persists(client) -> None:
    response = await client.post(
        "/vision/ask",
        files={"image": ("photo.png", b"not-really-a-png", "image/png")},
        data={"text": "What is this?", "conversation_id": "vision-convo"},
    )

    assert response.status_code == 200
    assert response.headers["x-conversation-id"] == "vision-convo"
    assert response.text == "This is a fake reply."

    history = await client.get("/conversations/vision-convo")
    messages = history.json()["messages"]
    assert messages[0]["content"] == "[Image attached] What is this?"
    assert messages[1]["content"] == "This is a fake reply."
