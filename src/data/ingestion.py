"""ingestion.py: Load and normalize uploaded documents into pages."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, cast

from pypdf import PdfReader

from src.core.config import settings
from src.core.models import Document, Page, SourceType
from src.core.utils import clean_ocr_text



def _read_size(uploaded_file: BinaryIO) -> int:
    """Return file size in bytes without changing the caller's read position."""

    current_position = uploaded_file.tell()  # preserve caller's read position
    uploaded_file.seek(0, 2)
    size = uploaded_file.tell()
    uploaded_file.seek(current_position)
    return size



def validate_uploaded_file(uploaded_file: BinaryIO) -> str:
    """Validate file extension and size, then return the normalized source type."""

    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if suffix not in settings.allowed_file_types:
        raise ValueError(f"Unsupported file type: .{suffix}")

    size_mb = _read_size(uploaded_file) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise ValueError(f"File exceeds {settings.max_file_size_mb}MB limit")
    return suffix



def extract_pdf(uploaded_file: BinaryIO) -> list[Page]:
    """Read a PDF and return cleaned page text as Page objects."""

    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    pages: list[Page] = []
    for index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        pages.append(Page(page_number=index, text=clean_ocr_text(raw_text)))
    return pages



def extract_text_file(uploaded_file: BinaryIO) -> list[Page]:
    """Read a text/markdown file and return one cleaned Page object."""

    uploaded_file.seek(0)
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="ignore")  # ignore odd encodings instead of failing upload
    else:
        text = str(raw)
    return [Page(page_number=1, text=clean_ocr_text(text))]



def build_document(uploaded_file: BinaryIO, session_id: str) -> Document:
    """Build a Document model from one uploaded file."""

    source_type = cast(SourceType, validate_uploaded_file(uploaded_file))
    if source_type == "pdf":
        pages = extract_pdf(uploaded_file)
    else:
        pages = extract_text_file(uploaded_file)

    return Document(
        filename=Path(uploaded_file.name).name,
        session_id=session_id,
        source_type=source_type,
        pages=pages,
    )
