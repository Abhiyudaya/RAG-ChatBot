"""
api.py -- serve the RAG pipeline over HTTP with FastAPI.

    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health   liveness + index size
    POST /ask      {"question": "...", "k": 4, "mode": "hybrid"} -> answer + citations
    POST /search   same body, retrieval only (no LLM call, no API key needed)

Config via environment variables:
    INDEX_DIR   folder built by ingest.py   (default: index)
    LLM_API_KEY / LLM_BASE_URL / LLM_MODEL  see rag/generator.py
"""

import os
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import Embedder, Generator, Retriever, VectorStore

INDEX_DIR = os.environ.get("INDEX_DIR", "index")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the index and embedding model once at startup, not per request.
    store = VectorStore.load(INDEX_DIR)
    state["store"] = store
    state["retriever"] = Retriever(Embedder(), store)
    state["generator"] = None   # created lazily so /search works without an LLM key
    yield
    state.clear()


app = FastAPI(title="RAG Document Q&A", version="1.0.0", lifespan=lifespan)


class Query(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(4, ge=1, le=20)
    mode: Literal["semantic", "keyword", "hybrid"] = "hybrid"
    model: Optional[str] = None


class Source(BaseModel):
    id: int
    source: str
    score: float
    text: str


class SearchResponse(BaseModel):
    sources: List[Source]


class AskResponse(SearchResponse):
    answer: str


def _retrieve(q: Query):
    results = state["retriever"].retrieve(q.question, k=q.k, mode=q.mode)
    sources = [
        Source(id=i, source=os.path.basename(c.source), score=s, text=c.text)
        for i, (c, s) in enumerate(results, start=1)
    ]
    return results, sources


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["store"])}


@app.post("/search", response_model=SearchResponse)
def search(q: Query):
    _, sources = _retrieve(q)
    return SearchResponse(sources=sources)


@app.post("/ask", response_model=AskResponse)
def ask(q: Query):
    results, sources = _retrieve(q)
    try:
        if state["generator"] is None or (q.model and q.model != state["generator"].model):
            state["generator"] = Generator(model=q.model) if q.model else Generator()
        answer = state["generator"].answer(q.question, results)
    except EnvironmentError as e:   # missing LLM_API_KEY
        raise HTTPException(status_code=503, detail=str(e))
    return AskResponse(answer=answer, sources=sources)
