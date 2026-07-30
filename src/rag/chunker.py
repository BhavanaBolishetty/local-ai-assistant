"""Paragraph-aware text chunking for RAG ingestion.

Hand-rolled rather than pulling in a framework (e.g. langchain) — this
project prefers writing the small stuff itself. Splits on blank lines
first, greedily packs paragraphs up to `chunk_size` characters, and falls
back to splitting an oversized paragraph on whitespace. Consecutive
chunks repeat the last `overlap` characters of their predecessor so
context isn't lost at a chunk boundary.
"""

import re


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current.strip())
            current = current[-overlap:] if overlap else ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            pieces = _split_oversized(paragraph, chunk_size)
        else:
            pieces = [paragraph]
        for piece in pieces:
            if current and len(current) + 2 + len(piece) > chunk_size:
                flush()
            current = f"{current}\n\n{piece}" if current else piece

    flush()
    return chunks


def _split_oversized(paragraph: str, chunk_size: int) -> list[str]:
    """Greedily pack a single (too-long) paragraph's words into pieces."""
    words = paragraph.split()
    pieces: list[str] = []
    current_words: list[str] = []
    current_len = 0

    for word in words:
        added_len = len(word) + (1 if current_words else 0)
        if current_words and current_len + added_len > chunk_size:
            pieces.append(" ".join(current_words))
            current_words = []
            added_len = len(word)
            current_len = 0
        current_words.append(word)
        current_len += added_len

    if current_words:
        pieces.append(" ".join(current_words))
    return pieces
