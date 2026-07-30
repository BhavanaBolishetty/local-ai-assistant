"""Tests for per-file-type text extraction."""

from io import BytesIO

import pytest
from docx import Document
from pypdf import PdfWriter

from src.rag.text_extraction import extract_text


def test_extracts_plain_text_for_txt_and_md() -> None:
    assert extract_text("notes.txt", b"hello world") == "hello world"
    assert extract_text("notes.md", b"# Title") == "# Title"


def test_invalid_utf8_raises_value_error() -> None:
    with pytest.raises(ValueError):
        extract_text("notes.txt", b"\xff\xfe\x00\x01")


def test_unsupported_extension_raises_value_error() -> None:
    with pytest.raises(ValueError):
        extract_text("archive.zip", b"whatever")


def test_extracts_text_from_docx() -> None:
    document = Document()
    document.add_paragraph("First paragraph.")
    document.add_paragraph("Second paragraph.")
    buffer = BytesIO()
    document.save(buffer)

    text = extract_text("notes.docx", buffer.getvalue())

    assert "First paragraph." in text
    assert "Second paragraph." in text


def test_pdf_extraction_does_not_crash_on_valid_pdf() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)

    text = extract_text("notes.pdf", buffer.getvalue())

    assert text == ""
