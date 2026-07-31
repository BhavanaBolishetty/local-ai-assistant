"""Speech-to-text via faster-whisper.

Model construction and `.transcribe()` are both blocking, synchronous
calls (faster-whisper has no asyncio support) — wrapped in
`anyio.to_thread.run_sync`, the same treatment `VectorStore` gives
ChromaDB's sync client.
"""

from typing import BinaryIO

import anyio
from faster_whisper import WhisperModel


class TranscriptionService:
    def __init__(self, model: WhisperModel) -> None:
        self._model = model

    @classmethod
    async def create(cls, model_size: str, download_root: str) -> "TranscriptionService":
        """Load (and, on first use, download) the Whisper model off the
        event loop thread — this can take a while the first time."""
        model = await anyio.to_thread.run_sync(
            lambda: WhisperModel(model_size, device="cpu", download_root=download_root)
        )
        return cls(model)

    async def transcribe(self, audio: BinaryIO) -> str:
        return await anyio.to_thread.run_sync(self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: BinaryIO) -> str:
        segments, _info = self._model.transcribe(audio)
        return "".join(segment.text for segment in segments).strip()
