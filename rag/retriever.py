"""
STAGE 5a of the RAG pipeline: RETRIEVE
--------------------------------------
At QUERY time: embed the user's question with the SAME embedder used at
ingest, then search the store for the most relevant chunks.

This is a thin orchestration layer over the embedder and the vector store.
It exposes the retrieval "mode" so you can feel the difference between
semantic-only, keyword-only, and hybrid retrieval.
"""

from __future__ import annotations

from typing import List, Tuple

from .chunker import Chunk
from .embedder import Embedder
from .store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid",
        alpha: float = 0.7,
    ) -> List[Tuple[Chunk, float]]:
        if mode == "keyword":
            return self.store.keyword_search(query, k=k)

        query_vec = self.embedder.embed_one(query)
        if mode == "semantic":
            return self.store.semantic_search(query_vec, k=k)
        if mode == "hybrid":
            return self.store.hybrid_search(query, query_vec, k=k, alpha=alpha)
        raise ValueError(f"Unknown retrieval mode: {mode!r}")
