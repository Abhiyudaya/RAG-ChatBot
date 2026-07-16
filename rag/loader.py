"""
STAGE 1 of the RAG pipeline: LOAD
--------------------------------
Read raw source documents off disk and turn them into plain text.

This is the "load your source documents" step. Real systems load from PDFs,
web pages, wikis, databases, transcripts, etc. Here we support the three most
common local formats: .txt, .md and .pdf.

Each document becomes a (source, text) pair. `source` is kept so that later,
when we cite an answer, we can tell the user *which file* a fact came from.
"""

from __future__ import annotations

import os
from typing import List, Tuple

# Only import pypdf lazily so that people without PDFs don't need it installed.
SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf_file(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "Reading PDFs requires 'pypdf'. Install it with: pip install pypdf"
        ) from e

    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        # extract_text() can return None for image-only pages.
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_document(path: str) -> str:
    """Load a single document and return its text."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf_file(path)
    return _read_text_file(path)


def load_folder(folder: str) -> List[Tuple[str, str]]:
    """
    Walk a folder recursively and load every supported document.

    Returns a list of (source_path, text) tuples.
    """
    documents: List[Tuple[str, str]] = []
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            path = os.path.join(root, name)
            text = load_document(path)
            if text.strip():
                documents.append((path, text))
    if not documents:
        raise FileNotFoundError(
            f"No supported documents ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) "
            f"found in '{folder}'."
        )
    return documents
