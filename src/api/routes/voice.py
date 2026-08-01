"""Voice endpoints: speech-to-text and text-to-speech."""

import logging
from io import BytesIO

from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.responses import Response

from src.api.schemas.voice import SpeakRequest, TranscriptionOut
from src.voice.synthesis import SynthesisService
from src.voice.transcription import TranscriptionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])


def get_transcription_service(request: Request) -> TranscriptionService:
    """Built once at app startup (see api/main.py's lifespan) since
    loading the Whisper model is expensive."""
    return request.app.state.transcription_service


def get_synthesis_service(request: Request) -> SynthesisService:
    """Built once at app startup — loading the Piper voice is expensive."""
    return request.app.state.synthesis_service


@router.post("/transcribe", response_model=TranscriptionOut)
async def transcribe_audio(
    file: UploadFile,
    transcription_service: TranscriptionService = Depends(get_transcription_service),
) -> TranscriptionOut:
    raw = await file.read()
    text = await transcription_service.transcribe(BytesIO(raw))
    return TranscriptionOut(text=text)


@router.post("/speak")
async def speak_text(
    speak_request: SpeakRequest,
    synthesis_service: SynthesisService = Depends(get_synthesis_service),
) -> Response:
    wav_bytes = await synthesis_service.synthesize(speak_request.text)
    return Response(content=wav_bytes, media_type="audio/wav")
