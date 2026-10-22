"""
eval.py -- measure retrieval quality for each mode on a small labelled set.

    python eval.py                         # uses docs/ and eval_set.json
    python eval.py --chunk-size 40 --k 3

For every question in eval_set.json we check whether a retrieved chunk contains
the expected answer string, and report per mode:
    hit@k  fraction of questions whose answer appears in the top-k chunks
    MRR    mean reciprocal rank of the first chunk containing the answer

No LLM or API key needed -- this evaluates RETRIEVAL only. Whatever numbers it
prints are the numbers to quote; they depend on your docs and eval set.
"""

import argparse
import json

from rag import Embedder, Retriever, VectorStore, chunk_documents, load_folder

MODES = ["semantic", "keyword", "hybrid"]


def evaluate(retriever: Retriever, items: list, mode: str, k: int) -> dict:
    hits, rr = 0, 0.0
    for item in items:
        results = retriever.retrieve(item["question"], k=k, mode=mode)
        needle = item["answer"].lower()
        for rank, (chunk, _score) in enumerate(results, start=1):
            if needle in chunk.text.lower():
                hits += 1
                rr += 1.0 / rank
                break
    n = len(items)
    return {"hit@k": hits / n, "mrr": rr / n}


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval modes.")
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--eval-set", default="eval_set.json")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--overlap", type=int, default=10)
    args = parser.parse_args()

    with open(args.eval_set, encoding="utf-8") as f:
        items = json.load(f)

    chunks = chunk_documents(load_folder(args.docs), chunk_size=args.chunk_size, overlap=args.overlap)
    embedder = Embedder()
    store = VectorStore()
    store.add(embedder.embed([c.text for c in chunks]), chunks)
    retriever = Retriever(embedder, store)

    print(f"{len(items)} questions, {len(chunks)} chunks "
          f"(size={args.chunk_size}, overlap={args.overlap}), k={args.k}\n")
    print(f"{'mode':<10} {'hit@' + str(args.k):>8} {'MRR':>8}")
    for mode in MODES:
        m = evaluate(retriever, items, mode, args.k)
        print(f"{mode:<10} {m['hit@k']:>8.2f} {m['mrr']:>8.2f}")


if __name__ == "__main__":
    main()
