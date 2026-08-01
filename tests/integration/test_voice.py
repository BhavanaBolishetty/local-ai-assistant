"""Integration tests for /voice/transcribe and /voice/speak against fake
Whisper/Piper stand-ins (see tests/integration/fakes.py) — no real model
loading needed to verify routing and response shape."""


async def test_transcribe_returns_fake_text(client) -> None:
    response = await client.post(
        "/voice/transcribe",
        files={"file": ("recording.wav", b"not-really-audio", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "fake transcribed text"}


async def test_speak_returns_wav_bytes(client) -> None:
    response = await client.post("/voice/speak", json={"text": "Hello there"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")


async def test_speak_rejects_empty_text(client) -> None:
    response = await client.post("/voice/speak", json={"text": ""})

    assert response.status_code == 422
