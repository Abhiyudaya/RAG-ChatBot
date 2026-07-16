"""
STAGE 3 of the RAG pipeline: EMBED
----------------------------------
Turn each chunk of text into a VECTOR (a list of numbers) that captures its
*meaning*. Chunks with similar meaning end up close together in vector space,
even when they use completely different words. This is what lets retrieval
match "how do I reset my password?" to a passage about "recovering account
access".

We use a small, fast, FREE local model from sentence-transformers
(all-MiniLM-L6-v2, 384 dimensions). No API key, no per-token cost, and the
whole embedding step is visible to you.

IMPORTANT: the *same* model must embed both the documents (at ingest time)
and the query (at question time), or the vectors won't be comparable.

We ask the model for NORMALIZED (unit-length) vectors. With unit vectors,
cosine similarity is just a dot product, which keeps the vector-store math
in store.py simple and fast.
"""

from __future__ import annotations

from typing import List

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        # Imported lazily so importing this module is cheap and error messages
        # are clear if the dependency is missing.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Embeddings require 'sentence-transformers'. Install it with:\n"
                "    pip install sentence-transformers"
            ) from e
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts. Returns an (N, dim) float32 numpy array of
        unit-normalized vectors.
        """
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,   # unit vectors -> cosine == dot product
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 64,
        )
        return vectors.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string; returns a 1-D vector of shape (dim,)."""
        return self.embed([text])[0]
