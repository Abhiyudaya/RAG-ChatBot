"""
ingest.py -- the OFFLINE INDEXING phase.

Run this once (and again whenever your documents change):

    python ingest.py --docs docs/ --index index/

It loads every document, chunks them, embeds the chunks, and saves the
resulting vector index to disk so chat.py can load it instantly.
"""

import argparse
import time

from rag import Embedder, VectorStore, chunk_documents, load_folder


def main():
    parser = argparse.ArgumentParser(description="Build a RAG index from a folder of documents.")
    parser.add_argument("--docs", default="docs", help="Folder containing your documents.")
    parser.add_argument("--index", default="index", help="Folder to write the index to.")
    parser.add_argument("--chunk-size", type=int, default=200, help="Chunk size in words.")
    parser.add_argument("--overlap", type=int, default=40, help="Overlap between chunks in words.")
    args = parser.parse_args()

    t0 = time.time()

    print(f"[1/4] Loading documents from '{args.docs}' ...")
    documents = load_folder(args.docs)
    print(f"      loaded {len(documents)} document(s)")

    print(f"[2/4] Chunking (size={args.chunk_size} words, overlap={args.overlap}) ...")
    chunks = chunk_documents(documents, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"      produced {len(chunks)} chunk(s)")

    print("[3/4] Embedding chunks (loads the embedding model on first run) ...")
    embedder = Embedder()
    embeddings = embedder.embed([c.text for c in chunks])
    print(f"      embedded to {embeddings.shape[1]}-dimensional vectors")

    print(f"[4/4] Saving index to '{args.index}' ...")
    store = VectorStore()
    store.add(embeddings, chunks)
    store.save(args.index)

    print(f"\nDone in {time.time() - t0:.1f}s. Now run:  python chat.py --index {args.index}")


if __name__ == "__main__":
    main()
