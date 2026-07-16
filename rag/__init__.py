"""A minimal, from-scratch Retrieval-Augmented Generation (RAG) pipeline.

Each module maps to one stage of the pipeline:
    loader    -> LOAD documents
    chunker   -> CHUNK them into passages
    embedder  -> EMBED passages into vectors
    store     -> STORE vectors + SEARCH them
    retriever -> RETRIEVE relevant chunks for a query
    generator -> AUGMENT the prompt + GENERATE a grounded answer
"""

from .chunker import Chunk, chunk_documents, chunk_text
from .embedder import Embedder
from .generator import Generator
from .loader import load_folder
from .retriever import Retriever
from .store import VectorStore

__all__ = [
    "Chunk",
    "chunk_documents",
    "chunk_text",
    "Embedder",
    "Generator",
    "load_folder",
    "Retriever",
    "VectorStore",
]
