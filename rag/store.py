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
  2. keyword_search   -> BM25 lexical ranking (exact-word based)
  3. hybrid_search    -> weighted blend of the two (usually the strongest)

Production stores use Approximate Nearest Neighbor (ANN) indexes so this
search stays fast over millions of vectors. Here we do an exact brute-force
dot product, which is perfectly fine for thousands of chunks and much easier
to understand.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np

from .chunker import Chunk

# Standard BM25 parameters (Robertson/Zaragoza defaults).
BM25_K1 = 1.5
BM25_B = 0.75

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
        self._bm25: dict | None = None               # lazily built BM25 statistics

    # ------------------------------------------------------------------ build
    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Number of chunks and embeddings must match.")
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.chunks.extend(chunks)
        self._bm25 = None   # corpus changed -> BM25 statistics are stale

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

    def _build_bm25(self) -> dict:
        """Precompute per-chunk term counts, lengths and document frequencies."""
        tfs: List[Dict[str, int]] = []
        lengths = np.zeros(len(self.chunks), dtype=np.float32)
        df: Counter = Counter()
        for i, chunk in enumerate(self.chunks):
            tf = Counter(_tokenize(chunk.text))
            tfs.append(tf)
            lengths[i] = sum(tf.values())
            df.update(tf.keys())
        avgdl = float(lengths.mean()) if len(lengths) else 0.0
        return {"tfs": tfs, "lengths": lengths, "df": df, "avgdl": avgdl}

    def bm25_scores(self, query: str) -> np.ndarray:
        """
        Okapi BM25 score of every chunk for `query`:

            score = sum over query terms t of
                    idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avglen))

        Unlike a raw term-overlap count, BM25 down-weights common words (idf),
        saturates repeated terms (k1) and normalises for chunk length (b).
        """
        n = len(self.chunks)
        scores = np.zeros(n, dtype=np.float32)
        q_terms = set(_tokenize(query))
        if n == 0 or not q_terms:
            return scores
        if self._bm25 is None:
            self._bm25 = self._build_bm25()
        tfs, lengths = self._bm25["tfs"], self._bm25["lengths"]
        df, avgdl = self._bm25["df"], self._bm25["avgdl"] or 1.0

        for t in q_terms:
            n_t = df.get(t, 0)
            if n_t == 0:
                continue
            idf = math.log(1.0 + (n - n_t + 0.5) / (n_t + 0.5))
            for i in range(n):
                tf = tfs[i].get(t, 0)
                if tf:
                    denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * lengths[i] / avgdl)
                    scores[i] += idf * tf * (BM25_K1 + 1) / denom
        return scores

    def keyword_search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        """BM25 keyword ranking -- great at exact terms (names, codes, numbers)."""
        scores = self.bm25_scores(query)
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
        Blend dense (semantic) and BM25 (keyword) scores.
        alpha = weight on semantic (0.7 -> 70% semantic, 30% keyword).
        Both score sets are min-max normalised to [0, 1] first.
        """
        if self.embeddings is None:
            return []
        sims = _minmax(self.embeddings @ query_vec)
        kw = _minmax(self.bm25_scores(query))

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
