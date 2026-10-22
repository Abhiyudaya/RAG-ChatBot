FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image far smaller than the default CUDA build.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY rag/ rag/
COPY api.py chat.py ingest.py eval.py eval_set.json ./
COPY docs/ docs/

# Bake the embedding model into the image so the container runs fully offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Build the index at image build time (offline indexing phase).
RUN python ingest.py --docs docs --index index

ENV INDEX_DIR=index
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
