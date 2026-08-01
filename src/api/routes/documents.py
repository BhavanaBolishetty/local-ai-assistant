"""Document upload/list/delete endpoints for the RAG knowledge base."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from src.ai.ollama_client import OllamaClient
from src.api.schemas.document import DocumentOut
from src.core.config import Settings, get_settings
from src.rag.text_extraction import SUPPORTED_EXTENSIONS, extract_text
from src.rag.vector_store import VectorStore
from src.services.document_service import DocumentService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


def get_document_service(
    request: Request, settings: Settings = Depends(get_settings)
) -> DocumentService:
    ollama_client = OllamaClient(
        http_client=request.app.state.http_client,
        default_model=settings.ollama_model,
        embedding_model=settings.ollama_embedding_model,
    )
    vector_store: VectorStore = request.app.state.vector_store
    return DocumentService(
        ollama_client,
        vector_store,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )


def _to_document_out(summary) -> DocumentOut:
    return DocumentOut(
        id=summary.id,
        filename=summary.filename,
        uploaded_at=summary.uploaded_at,
        chunk_count=summary.chunk_count,
    )


@router.post(
    "",
    response_model=DocumentOut,
    status_code=201,
    summary="Upload a document (.txt/.md/.pdf/.docx) to the RAG knowledge base",
)
async def upload_document(
    file: UploadFile,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentOut:
    filename = file.filename or "untitled"
    if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
        allowed = ", ".join(SUPPORTED_EXTENSIONS)
        raise HTTPException(status_code=400, detail=f"Only {allowed} files are supported")

    raw = await file.read()
    try:
        text = extract_text(filename, raw)
        summary = await document_service.ingest(filename, text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _to_document_out(summary)


@router.get("", response_model=list[DocumentOut], summary="List uploaded documents")
async def list_documents(
    document_service: DocumentService = Depends(get_document_service),
) -> list[DocumentOut]:
    summaries = await document_service.list_documents()
    return [_to_document_out(s) for s in summaries]


@router.delete("/{document_id}", status_code=204, summary="Delete a document and its chunks")
async def delete_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
) -> None:
    await document_service.delete(document_id)
