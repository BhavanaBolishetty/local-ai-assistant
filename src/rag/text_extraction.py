"""Turns raw uploaded file bytes into plain text, per file type.

The only module that knows how to read a PDF or DOCX's internal format —
everything downstream (`chunker`, `DocumentService`) only ever sees a
plain string, exactly as it did when only `.txt`/`.md` were supported.
"""

from io import BytesIO

import pymupdf
from docx import Document

SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf", ".docx")


def extract_text(filename: str, raw: bytes) -> str:
    lowered = filename.lower()
    if lowered.endswith((".txt", ".md")):
        text = _extract_plain_text(raw)
    elif lowered.endswith(".pdf"):
        text = _extract_pdf(raw)
    elif lowered.endswith(".docx"):
        text = _extract_docx(raw)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    return _normalize(text)


def _normalize(text: str) -> str:
    """Some PDF fonts encode dashes/special punctuation with glyph IDs
    that don't map to a real Unicode codepoint, which surfaces as U+FFFD
    (the replacement character) in extracted text — e.g. "Jan 2023 <?>
    Jul 2024" instead of "Jan 2023 - Jul 2024". A plain hyphen is a safe,
    readable stand-in, since this almost always occurs where some kind of
    dash/separator was intended."""
    return text.replace("�", "-")


def _extract_plain_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("File must be UTF-8 text") from exc


def _extract_pdf(raw: bytes) -> str:
    """Parse the PDF with PyMuPDF (MuPDF bindings), not pypdf.

    Real-world PDFs are often missing a clean trailer/xref/startxref
    (certain export tools, minor truncation in transit) even though
    ordinary PDF viewers open them without complaint. pypdf's stricter
    parser rejected such files outright (`Stream has ended unexpectedly`,
    `startxref not found`, ...) one error at a time; PyMuPDF recovers
    all of those same cases directly, with no special-casing needed.
    """
    try:
        with pymupdf.open(stream=raw, filetype="pdf") as document:
            # sort=True orders extracted text by visual position (top-to-
            # bottom, left-to-right) instead of the PDF's internal content
            # stream order — many real-world PDFs (templated resumes,
            # multi-column layouts, tables) write content in an order that
            # doesn't match how it's meant to be read, which otherwise
            # scrambles section order (e.g. "Work Experience" and
            # "Education" content interleaving) badly enough to confuse
            # what's actually being read.
            pages = (page.get_text(sort=True) for page in document)
            return "\n\n".join(p.strip() for p in pages if p.strip())
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {exc}") from exc


def _extract_docx(raw: bytes) -> str:
    try:
        document = Document(BytesIO(raw))
        paragraphs = (p.text for p in document.paragraphs)
        return "\n\n".join(p for p in paragraphs if p.strip())
    except Exception as exc:
        raise ValueError(f"Could not read DOCX: {exc}") from exc
