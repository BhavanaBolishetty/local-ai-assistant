"""Text-to-speech via Piper.

Voice download/load and `.synthesize_wav()` are both blocking, synchronous
calls — wrapped in `anyio.to_thread.run_sync`, same treatment
`TranscriptionService`/`VectorStore` give their respective sync libraries.
"""

import wave
from io import BytesIO
from pathlib import Path

import anyio
from piper import PiperVoice
from piper.download_voices import download_voice


class SynthesisService:
    def __init__(self, voice: PiperVoice) -> None:
        self._voice = voice

    @classmethod
    async def create(cls, voice_name: str, voices_dir: str) -> "SynthesisService":
        """Download the voice (idempotent — a cheap local no-op once the
        files already exist) and load it, off the event loop thread."""

        def _load() -> PiperVoice:
            download_dir = Path(voices_dir)
            download_dir.mkdir(parents=True, exist_ok=True)
            download_voice(voice_name, download_dir)
            return PiperVoice.load(download_dir / f"{voice_name}.onnx")

        voice = await anyio.to_thread.run_sync(_load)
        return cls(voice)

    async def synthesize(self, text: str) -> bytes:
        return await anyio.to_thread.run_sync(self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)
        return buffer.getvalue()
