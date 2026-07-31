"""Voice endpoints: speech-to-text and text-to-speech."""

import logging
from io import BytesIO

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import Response

from src.api.schemas.voice import SpeakRequest, TranscriptionOut
from src.voice.synthesis import SynthesisService
from src.voice.transcription import TranscriptionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/transcribe", response_model=TranscriptionOut)
async def transcribe_audio(request: Request, file: UploadFile) -> TranscriptionOut:
    transcription_service: TranscriptionService = request.app.state.transcription_service
    raw = await file.read()
    text = await transcription_service.transcribe(BytesIO(raw))
    return TranscriptionOut(text=text)


@router.post("/speak")
async def speak_text(request: Request, speak_request: SpeakRequest) -> Response:
    synthesis_service: SynthesisService = request.app.state.synthesis_service
    wav_bytes = await synthesis_service.synthesize(speak_request.text)
    return Response(content=wav_bytes, media_type="audio/wav")
