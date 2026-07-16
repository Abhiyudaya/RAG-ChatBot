"""
STAGE 4 + part of STAGE 5: STORE and SEARCH
-------------------------------------------
A minimal vector database, built from scratch with numpy so you can see
*exactly* what a production vector DB (FAISS, Chroma, pgvector, Pinecone,
Weaviate, Qdrant, Milvus...) does under the hood.

It stores:
  - an (N, dim) matrix of chunk embeddings
  - the parallel list of Chunk objects (text + source + metadata)

It supports three kinds of search:
  1. semantic_search  -> pure vector similarity (meaning-based)
  2. keyword_search   -> simple term-overlap score (exact-word based)
  3. hybrid_search    -> weighted blend of the two (usually the strongest)

Production stores use Approximate Nearest Neighbor (ANN) indexes so this
search stays fast over millions of vectors. Here we do an exact brute-force
dot product, which is perfectly fine for thousands of chunks and much easier
to understand.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import asdict
from typing import List, Tuple

import numpy as np

from .chunker import Chunk

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Scale scores into [0, 1] so semantic and keyword scores are comparable."""
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class VectorStore:
    def __init__(self):
        self.embeddings: np.ndarray | None = None   # shape (N, dim), unit vectors
        self.chunks: List[Chunk] = []

    # ------------------------------------------------------------------ build
    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Number of chunks and embeddings must match.")
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.chunks.extend(chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    # ----------------------------------------------------------------- search
    def semantic_search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[Chunk, float]]:
        """Nearest chunks by cosine similarity (dot product on unit vectors)."""
        if self.embeddings is None:
            return []
        sims = self.embeddings @ query_vec        # (N,) cosine similarities
        k = min(k, len(self.chunks))
        # argpartition gets the top-k fast, then we sort just those k.
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [(self.chunks[i], float(sims[i])) for i in top_idx]

    def keyword_search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        A lightweight keyword score: how many query terms appear in the chunk,
        weighted by how often. This is a tiny stand-in for BM25 and is great at
        catching exact terms (names, codes, error numbers) that embeddings blur.
        """
        q_terms = set(_tokenize(query))
        if not q_terms:
            return []
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for i, chunk in enumerate(self.chunks):
            counts = Counter(_tokenize(chunk.text))
            total = sum(counts.values()) or 1
            score = sum(counts[t] for t in q_terms) / total
            scores[i] = score
        k = min(k, len(self.chunks))
        top_idx = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def hybrid_search(
        self,
        query: str,
        query_vec: np.ndarray,
        k: int = 5,
        alpha: float = 0.7,
    ) -> List[Tuple[Chunk, float]]:
        """
        Blend semantic and keyword scores.
        alpha = weight on semantic (0.7 -> 70% semantic, 30% keyword).
        """
        if self.embeddings is None:
            return []
        sims = _minmax(self.embeddings @ query_vec)

        q_terms = set(_tokenize(query))
        kw = np.zeros(len(self.chunks), dtype=np.float32)
        if q_terms:
            for i, chunk in enumerate(self.chunks):
                counts = Counter(_tokenize(chunk.text))
                total = sum(counts.values()) or 1
                kw[i] = sum(counts[t] for t in q_terms) / total
        kw = _minmax(kw)

        combined = alpha * sims + (1 - alpha) * kw
        k = min(k, len(self.chunks))
        top_idx = np.argsort(-combined)[:k]
        return [(self.chunks[i], float(combined[i])) for i in top_idx]

    # ------------------------------------------------------------ persistence
    def save(self, directory: str) -> None:
        """
        Persist the index to disk. This is the whole point of the OFFLINE
        indexing phase: build once, then load instantly at query time.
        """
        os.makedirs(directory, exist_ok=True)
        if self.embeddings is not None:
            np.save(os.path.join(directory, "embeddings.npy"), self.embeddings)
        with open(os.path.join(directory, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        store = cls()
        emb_path = os.path.join(directory, "embeddings.npy")
        chunks_path = os.path.join(directory, "chunks.json")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(
                f"No index found in '{directory}'. Run ingest.py first."
            )
        store.embeddings = np.load(emb_path) if os.path.exists(emb_path) else None
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        store.chunks = [Chunk(**c) for c in data]
        return store
