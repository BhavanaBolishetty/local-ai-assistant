"""Tests for the paragraph-aware chunker."""

from src.rag.chunker import chunk_text


def test_empty_text_produces_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_paragraphs_are_packed_into_one_chunk() -> None:
    text = "Paragraph A.\n\nParagraph B."
    assert chunk_text(text, chunk_size=100, overlap=10) == ["Paragraph A.\n\nParagraph B."]


def test_paragraphs_split_across_chunks_with_overlap() -> None:
    paragraphs = ["0123456789", "abcdefghij", "ABCDEFGHIJ"]
    chunks = chunk_text("\n\n".join(paragraphs), chunk_size=15, overlap=5)

    assert chunks == [
        "0123456789",
        "56789\n\nabcdefghij",
        "fghij\n\nABCDEFGHIJ",
    ]


def test_oversized_paragraph_is_split_on_whitespace() -> None:
    words = [f"word{i}" for i in range(50)]
    paragraph = " ".join(words)

    chunks = chunk_text(paragraph, chunk_size=30, overlap=0)

    assert len(chunks) > 1
    assert all(len(chunk) <= 30 for chunk in chunks)
    combined = " ".join(chunks)
    assert all(word in combined for word in words)
