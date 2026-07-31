"""Business logic for ingesting uploaded documents into the RAG store.

Mirrors `ChatService`: a thin orchestrator over the lower layers
(`chunker`, `OllamaClient.embed`, `VectorStore`), so the API route never
has to know how ingestion actually works.
"""

import logging
from datetime import UTC, datetime
from uuid import uuid4

from src.ai.ollama_client import OllamaClient
from src.models.document import DocumentSummary
from src.rag.chunker import chunk_text
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        ollama_client: OllamaClient,
        vector_store: VectorStore,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        self._ollama_client = ollama_client
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def ingest(self, filename: str, text: str) -> DocumentSummary:
        chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
        if not chunks:
            raise ValueError("Document has no extractable text")

        # Re-uploading a file with the same name replaces the old entry
        # instead of leaving a confusing stale duplicate that RAG search
        # would otherwise keep surfacing alongside the new version.
        existing = [d for d in await self._vector_store.list_documents() if d.filename == filename]
        for document in existing:
            await self._vector_store.delete_document(document.id)

        embeddings = await self._ollama_client.embed(chunks)
        document_id = str(uuid4())
        await self._vector_store.add_document(document_id, filename, chunks, embeddings)
        logger.info(
            "Ingested %s into %d chunks (document_id=%s)", filename, len(chunks), document_id
        )

        return DocumentSummary(
            id=document_id,
            filename=filename,
            uploaded_at=datetime.now(UTC),
            chunk_count=len(chunks),
        )

    async def list_documents(self) -> list[DocumentSummary]:
        return await self._vector_store.list_documents()

    async def delete(self, document_id: str) -> None:
        await self._vector_store.delete_document(document_id)
