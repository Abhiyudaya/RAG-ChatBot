"""
STAGE 2 of the RAG pipeline: CHUNK
----------------------------------
Split each document into smaller passages ("chunks").

Why chunk at all? You want to retrieve *just the relevant paragraph*, not a
whole 50-page PDF. Chunk size is a genuine trade-off:
  - Too big  -> you retrieve lots of irrelevant text and waste the context window.
  - Too small -> you lose the surrounding context that makes a passage meaningful.

We use a sliding window measured in WORDS, with OVERLAP so that a sentence
split across a boundary still appears (whole) in at least one chunk.

Rough intuition: 1 token ~= 0.75 words, so 200 words ~= 260 tokens per chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Chunk:
    text: str
    source: str          # which file this came from (for citations)
    chunk_index: int     # position of this chunk within its document
    metadata: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 200,
    overlap: int = 40,
) -> List[Chunk]:
    """
    Split one document's text into overlapping word-windows.

    chunk_size / overlap are counts of WORDS.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: List[Chunk] = []
    index = 0
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        if not window:
            break
        chunks.append(
            Chunk(
                text=" ".join(window),
                source=source,
                chunk_index=index,
            )
        )
        index += 1
        # Stop once this window reached the end of the document.
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(
    documents: List[Tuple[str, str]],
    chunk_size: int = 200,
    overlap: int = 40,
) -> List[Chunk]:
    """Chunk a list of (source, text) documents into a flat list of Chunks."""
    all_chunks: List[Chunk] = []
    for source, text in documents:
        all_chunks.extend(chunk_text(text, source, chunk_size, overlap))
    return all_chunks
