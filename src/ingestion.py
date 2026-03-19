from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader

from src.config import settings
from src.models import Document, Page
from src.utils import clean_text


def _read_size(uploaded_file: BinaryIO) -> int:
    current_position = uploaded_file.tell()
    uploaded_file.seek(0, 2)
    size = uploaded_file.tell()
    uploaded_file.seek(current_position)
    return size


def validate_uploaded_file(uploaded_file: BinaryIO) -> str:
    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if suffix not in settings.allowed_file_types:
        raise ValueError(f"Unsupported file type: .{suffix}")

    size_mb = _read_size(uploaded_file) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise ValueError(f"File exceeds {settings.max_file_size_mb}MB limit")
    return suffix


def extract_pdf(uploaded_file: BinaryIO) -> list[Page]:
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    pages: list[Page] = []
    for index, page in enumerate(reader.pages, start=1):
        pages.append(Page(page_number=index, text=clean_text(page.extract_text() or "")))
    return pages


def extract_text_file(uploaded_file: BinaryIO) -> list[Page]:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = str(raw)
    return [Page(page_number=1, text=clean_text(text))]


def build_document(uploaded_file: BinaryIO, session_id: str) -> Document:
    source_type = validate_uploaded_file(uploaded_file)
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
