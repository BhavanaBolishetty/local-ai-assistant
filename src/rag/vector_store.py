"""Chroma-backed vector store for RAG chunks.

The only module in the codebase that imports `chromadb` — mirrors how
`OllamaClient`/`ConversationRepository` are the sole modules that know
their respective backend's shape. Chroma's client is sync-only, so every
public method offloads its call via `anyio.to_thread.run_sync` to keep an
async interface consistent with the rest of the app.
"""

from datetime import UTC, datetime

import anyio
import chromadb

from src.models.document import DocumentSummary, RetrievedChunk

_COLLECTION_NAME = "documents"


class VectorStore:
    def __init__(self, chroma_path: str) -> None:
        client = chromadb.PersistentClient(path=chroma_path)
        self._collection = client.get_or_create_collection(
            name=_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    async def has_documents(self) -> bool:
        count = await anyio.to_thread.run_sync(self._collection.count)
        return count > 0

    async def add_document(
        self,
        document_id: str,
        filename: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Store pre-computed embeddings. Never lets Chroma fall back to
        its own default embedding function (which would silently try to
        download a separate ONNX model)."""
        uploaded_at = datetime.now(UTC).isoformat()
        ids = [f"{document_id}:{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
                "uploaded_at": uploaded_at,
            }
            for i in range(len(chunks))
        ]
        await anyio.to_thread.run_sync(
            lambda: self._collection.add(
                ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
            )
        )

    async def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        result = await anyio.to_thread.run_sync(
            lambda: self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        )
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []
        return [
            RetrievedChunk(content=doc, filename=meta["filename"], distance=dist)
            for doc, meta, dist in zip(documents, metadatas, distances, strict=True)
        ]

    async def list_documents(self) -> list[DocumentSummary]:
        result = await anyio.to_thread.run_sync(lambda: self._collection.get(include=["metadatas"]))

        by_document: dict[str, dict] = {}
        for metadata in result["metadatas"]:
            document_id = metadata["document_id"]
            entry = by_document.setdefault(
                document_id,
                {
                    "filename": metadata["filename"],
                    "uploaded_at": metadata["uploaded_at"],
                    "chunk_count": 0,
                },
            )
            entry["chunk_count"] += 1
            entry["uploaded_at"] = min(entry["uploaded_at"], metadata["uploaded_at"])

        summaries = [
            DocumentSummary(
                id=document_id,
                filename=entry["filename"],
                uploaded_at=datetime.fromisoformat(entry["uploaded_at"]),
                chunk_count=entry["chunk_count"],
            )
            for document_id, entry in by_document.items()
        ]
        return sorted(summaries, key=lambda d: d.uploaded_at, reverse=True)

    async def delete_document(self, document_id: str) -> None:
        await anyio.to_thread.run_sync(
            lambda: self._collection.delete(where={"document_id": document_id})
        )
