"""Domain models for uploaded documents and RAG retrieval results.

Plain dataclasses, like `Message`/`Conversation` — these flow between
`src/rag`, `src/services`, and the API layer without any of them knowing
Chroma's or Ollama's shapes.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    id: str
    filename: str
    uploaded_at: datetime
    chunk_count: int


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    content: str
    filename: str
    distance: float
