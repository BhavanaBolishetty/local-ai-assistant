"""Turns raw uploaded file bytes into plain text, per file type.

The only module that knows how to read a PDF or DOCX's internal format —
everything downstream (`chunker`, `DocumentService`) only ever sees a
plain string, exactly as it did when only `.txt`/`.md` were supported.
"""

from io import BytesIO

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf", ".docx")


def extract_text(filename: str, raw: bytes) -> str:
    lowered = filename.lower()
    if lowered.endswith((".txt", ".md")):
        return _extract_plain_text(raw)
    if lowered.endswith(".pdf"):
        return _extract_pdf(raw)
    if lowered.endswith(".docx"):
        return _extract_docx(raw)
    raise ValueError(f"Unsupported file type: {filename}")


def _extract_plain_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("File must be UTF-8 text") from exc


def _extract_pdf(raw: bytes) -> str:
    reader = PdfReader(BytesIO(raw))
    pages = (page.extract_text() or "" for page in reader.pages)
    return "\n\n".join(p for p in pages if p.strip())


def _extract_docx(raw: bytes) -> str:
    document = Document(BytesIO(raw))
    paragraphs = (p.text for p in document.paragraphs)
    return "\n\n".join(p for p in paragraphs if p.strip())
