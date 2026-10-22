# RAG Chatbot — a from-scratch Retrieval-Augmented Generation pipeline

A complete, end-to-end RAG system with **no LangChain / LlamaIndex** — every
stage of the pipeline is written out so you can see exactly what those
frameworks hide. Point it at a folder of your own documents (`.txt`, `.md`,
`.pdf`), build an index, and chat with them. Answers are **grounded** in your
documents and **cite their sources**.

## How the code maps to the RAG concepts

| Stage | File | What it does |
|------|------|--------------|
| 1. Load | `rag/loader.py` | Read `.txt` / `.md` / `.pdf` into plain text |
| 2. Chunk | `rag/chunker.py` | Split docs into overlapping word-windows |
| 3. Embed | `rag/embedder.py` | Turn each chunk into a meaning-vector (local, free) |
| 4. Store + Search | `rag/store.py` | numpy vector DB: semantic (cosine), keyword (BM25) & hybrid search + save/load |
| 5a. Retrieve | `rag/retriever.py` | Embed the query, fetch the top-k chunks |
| 5b. Augment + Generate | `rag/generator.py` | Build the prompt, call the LLM, return a cited answer |

Entry points:

- `ingest.py` — the **offline indexing phase** (load → chunk → embed → store)
- `chat.py` — the **online query phase** (retrieve → augment → generate) as a CLI
- `api.py` — the same query phase served over HTTP with **FastAPI**
- `eval.py` — measures retrieval quality (hit@k, MRR) for each retrieval mode

## Setup

```bash
# 1. Install dependencies (sentence-transformers pulls in torch; first run is slow)
pip install -r requirements.txt

# 2. Set your LLM key. Works with ANY OpenAI-compatible provider.
export LLM_API_KEY=...          # your key (any provider)
export LLM_MODEL=gpt-4o-mini    # a model your provider offers
# export LLM_BASE_URL=...       # set this for non-OpenAI providers (see table below)
```

### Which provider? Set `LLM_BASE_URL` + `LLM_MODEL` to match your key

| Provider | `LLM_BASE_URL` | example `LLM_MODEL` |
|----------|----------------|---------------------|
| OpenAI | *(leave unset)* | `gpt-4o-mini` |
| OpenRouter | `https://openrouter.ai/api/v1` | `openai/gpt-4o-mini` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Together | `https://api.together.xyz/v1` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat` |
| Fireworks | `https://api.fireworks.ai/inference/v1` | `accounts/fireworks/models/llama-v3p3-70b-instruct` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.0-flash` |
| Ollama (local, free) | `http://localhost:11434/v1` | `llama3.1` (set `LLM_API_KEY=ollama`) |
| LM Studio (local, free) | `http://localhost:1234/v1` | *(whatever you loaded)* |

Use the model names your provider actually lists — the ones above are just examples.
Only `rag/generator.py` talks to the LLM; everything else is provider-independent.

## Run it

```bash
# Build the index from the included sample doc (or drop your own files in docs/)
python ingest.py --docs docs/ --index index/

# Chat with your documents
python chat.py --index index/
```

Example session (against the included sample FAQ):

```
You: how long is a password reset link valid?
  Retrieved:
    [1] sample.md (score 0.71): To reset your password, go to the Nimbus login page ...
Assistant: A password reset link is valid for 30 minutes; if it expires you
must request a new one. [1]

You: what is the refund policy?
Assistant: I don't know based on the provided documents.
```

That second answer is the whole point: when the context doesn't contain the
answer, the model says so instead of hallucinating.

## Serve it over HTTP (FastAPI)

```bash
python ingest.py --docs docs/ --index index/
uvicorn api:app --port 8000
```

| Endpoint | What it does |
|----------|--------------|
| `GET /health` | Liveness check + number of indexed chunks |
| `POST /search` | Retrieval only — no LLM call, no API key needed |
| `POST /ask` | Retrieve + generate a cited answer |

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "how long is a password reset link valid?", "k": 4, "mode": "hybrid"}'
```

The response contains the `answer` plus the numbered `sources` (file, score,
text) that its `[1]`, `[2]` citations refer to. Interactive docs are at
`http://localhost:8000/docs`.

## Run it in Docker

```bash
docker compose up --build
```

The image installs CPU-only PyTorch, bakes in the `all-MiniLM-L6-v2` embedding
model and builds the index from `docs/` at build time, so retrieval runs fully
offline inside the container. Pass `LLM_API_KEY` / `LLM_BASE_URL` /
`LLM_MODEL` through your shell or a `.env` file to enable `/ask`.

## Evaluate retrieval

```bash
python eval.py --k 3
```

`eval_set.json` holds labelled questions about the sample FAQ — a mix of
exact-term questions ("0.08 USD per compute-hour") and paraphrased ones ("how
much cheaper are yearly subscriptions?"). For each retrieval mode, `eval.py`
reports **hit@k** (did a top-k chunk contain the answer?) and **MRR** (how high
was it ranked?). No LLM key is needed. Add questions for your own documents to
get numbers that mean something for your corpus.

## Things to try (to *feel* how each part matters)

- **Change retrieval mode:** `python chat.py --mode semantic` vs `--mode keyword`
  vs `--mode hybrid`. Ask about an exact term (like a specific number) and watch
  keyword/hybrid win; ask a paraphrased question and watch semantic win.
- **See what's retrieved:** add `--show-chunks` to print the full retrieved text.
- **Break the chunking:** re-ingest with `--chunk-size 20` (too small) or
  `--chunk-size 1000 --overlap 0` (too big) and watch answer quality change.
- **Swap the model:** `python chat.py --model <a-model-your-provider-offers>`
  (or set `LLM_MODEL` once in your environment).
- **Add your own docs:** drop PDFs/markdown into `docs/`, re-run `ingest.py`.

## How this maps to production

This is deliberately minimal and honest about it. To productionize:

- **Vector store:** swap `store.py`'s brute-force search for FAISS, Chroma,
  pgvector, Pinecone, Weaviate, Qdrant, or Milvus (they use ANN indexes for
  scale). The `add` / `search` / `save` / `load` interface stays the same.
- **Hybrid search:** BM25 is implemented from scratch in `store.py`; at scale,
  use your DB's built-in full-text search, and consider reciprocal rank fusion
  instead of a weighted score blend.
- **Re-ranking:** retrieve a larger set (say top 20), then re-order with a
  cross-encoder re-ranker and keep the best 4. This is one of the biggest
  quality wins.
- **Evaluation:** `eval.py` measures retrieval; also measure generation
  (is the answer faithful to the chunks?), e.g. with an LLM judge.
- **Serving:** `api.py` is a starting point; add streaming responses,
  logging/tracing, caching, auth, and guardrails.

## Requirements

Python 3.9+. See `requirements.txt`.
```
```
