"""Turns a user's message into a formatted context block, if relevant.

The seam between the vector store and `ChatService` — everything above
this only ever sees a plain string (or `None`), never chunks/distances.
"""

from src.ai.ollama_client import OllamaClient
from src.rag.vector_store import VectorStore


class RagRetriever:
    def __init__(
        self,
        ollama_client: OllamaClient,
        vector_store: VectorStore,
        top_k: int = 4,
        max_distance: float = 0.5,
    ) -> None:
        self._ollama_client = ollama_client
        self._vector_store = vector_store
        self._top_k = top_k
        self._max_distance = max_distance

    async def build_context(self, query: str) -> str | None:
        if not await self._vector_store.has_documents():
            return None

        [embedding] = await self._ollama_client.embed([query])
        chunks = await self._vector_store.query(embedding, self._top_k)
        relevant = [c for c in chunks if c.distance <= self._max_distance]
        if not relevant:
            return None

        return "\n\n".join(f"[From {c.filename}]\n{c.content}" for c in relevant)
