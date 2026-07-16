"""
chat.py -- the ONLINE QUERY phase.

Loads the index built by ingest.py and answers your questions in a loop:

    python chat.py --index index/

For each question it:
    1. retrieves the most relevant chunks,
    2. shows you which sources were retrieved (so retrieval is never a black box),
    3. asks Claude to answer grounded in those chunks, with citations.

Requires an LLM key (any OpenAI-compatible provider):  export LLM_API_KEY=...
See rag/generator.py for LLM_BASE_URL / LLM_MODEL settings per provider.
"""

import argparse
import os

from rag import Embedder, Generator, Retriever, VectorStore


def main():
    parser = argparse.ArgumentParser(description="Chat with your documents via RAG.")
    parser.add_argument("--index", default="index", help="Folder containing the built index.")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["semantic", "keyword", "hybrid"],
        help="Retrieval strategy. Try switching this to feel the difference.",
    )
    parser.add_argument("--model", default=None, help="Override the Claude model.")
    parser.add_argument("--show-chunks", action="store_true", help="Print retrieved chunk text.")
    args = parser.parse_args()

    print("Loading index and embedding model ...")
    store = VectorStore.load(args.index)
    embedder = Embedder()
    retriever = Retriever(embedder, store)
    generator = Generator(model=args.model) if args.model else Generator()
    print(f"Ready. {len(store)} chunks indexed. Retrieval mode: {args.mode}. "
          f"Type a question (or 'exit').\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        results = retriever.retrieve(query, k=args.k, mode=args.mode)

        # Transparency: always show what was retrieved.
        print("\n  Retrieved:")
        for i, (chunk, score) in enumerate(results, start=1):
            preview = chunk.text[:90].replace("\n", " ")
            print(f"    [{i}] {os.path.basename(chunk.source)} "
                  f"(score {score:.3f}): {preview}...")
            if args.show_chunks:
                print(f"        {chunk.text}\n")

        answer = generator.answer(query, results)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
