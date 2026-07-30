"""Tests for VectorStore against a real (temp-dir) Chroma instance.

Uses hand-written embedding vectors so no live Ollama call is needed —
embedding quality itself is verified manually against the running app.
"""

import pytest

from src.rag.vector_store import VectorStore


@pytest.fixture
def vector_store(tmp_path) -> VectorStore:
    return VectorStore(str(tmp_path / "chroma"))


async def test_has_documents_false_when_empty(vector_store: VectorStore) -> None:
    assert await vector_store.has_documents() is False


async def test_add_and_query_returns_relevant_chunks(vector_store: VectorStore) -> None:
    await vector_store.add_document(
        "doc-1", "notes.txt", ["chunk one", "chunk two"], [[1.0, 0.0], [0.0, 1.0]]
    )
    assert await vector_store.has_documents() is True

    results = await vector_store.query([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].content == "chunk one"
    assert results[0].filename == "notes.txt"
    assert results[0].distance == pytest.approx(0.0)


async def test_list_documents_groups_by_document_and_counts_chunks(
    vector_store: VectorStore,
) -> None:
    await vector_store.add_document("doc-1", "a.txt", ["a1", "a2"], [[1.0, 0.0], [0.9, 0.1]])
    await vector_store.add_document("doc-2", "b.md", ["b1"], [[0.0, 1.0]])

    summaries = await vector_store.list_documents()

    by_id = {s.id: s for s in summaries}
    assert by_id["doc-1"].filename == "a.txt"
    assert by_id["doc-1"].chunk_count == 2
    assert by_id["doc-2"].filename == "b.md"
    assert by_id["doc-2"].chunk_count == 1


async def test_delete_document_removes_its_chunks(vector_store: VectorStore) -> None:
    await vector_store.add_document("doc-1", "a.txt", ["a1"], [[1.0, 0.0]])

    await vector_store.delete_document("doc-1")

    assert await vector_store.has_documents() is False
    assert await vector_store.list_documents() == []
