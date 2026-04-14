"""ingestion.py: Load and normalize uploaded documents into pages."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader

from src.core.config import settings
from src.core.models import Document, Page
from src.core.utils import clean_text



def _read_size(uploaded_file: BinaryIO) -> int:
    """Read file size without changing the current file pointer.
    
    Args:
        uploaded_file (BinaryIO): Input parameter.
    
    Returns:
        int: Computed integer result.
    """

    current_position = uploaded_file.tell()  # preserve caller's read position
    uploaded_file.seek(0, 2)
    size = uploaded_file.tell()
    uploaded_file.seek(current_position)
    return size



def validate_uploaded_file(uploaded_file: BinaryIO) -> str:
    """Validate file type and size before ingestion.
    
    Args:
        uploaded_file (BinaryIO): Input parameter.
    
    Returns:
        str: Formatted text result.
    """

    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if suffix not in settings.allowed_file_types:
        raise ValueError(f"Unsupported file type: .{suffix}")

    size_mb = _read_size(uploaded_file) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise ValueError(f"File exceeds {settings.max_file_size_mb}MB limit")
    return suffix



def extract_pdf(uploaded_file: BinaryIO) -> list[Page]:
    """Extract text from a PDF into page objects.
    
    Args:
        uploaded_file (BinaryIO): Input parameter.
    
    Returns:
        list[Page]: List of results.
    """

    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    pages: list[Page] = []
    for index, page in enumerate(reader.pages, start=1):
        pages.append(Page(page_number=index, text=clean_text(page.extract_text() or "")))
    return pages



def extract_text_file(uploaded_file: BinaryIO) -> list[Page]:
    """Extract text from a plain text or markdown file.
    
    Args:
        uploaded_file (BinaryIO): Input parameter.
    
    Returns:
        list[Page]: List of results.
    """

    uploaded_file.seek(0)
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="ignore")  # ignore odd encodings instead of failing upload
    else:
        text = str(raw)
    return [Page(page_number=1, text=clean_text(text))]



def build_document(uploaded_file: BinaryIO, session_id: str) -> Document:
    """Build a Document model from an uploaded file.
    
    Args:
        uploaded_file (BinaryIO): Input parameter.
        session_id (str): Session identifier for the current chat.
    
    Returns:
        Document: Result value.
    """

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
