"""
STAGE 5b of the RAG pipeline: AUGMENT + GENERATE
------------------------------------------------
Take the retrieved chunks, build an AUGMENTED prompt ("here is some context,
answer the question using it"), and ask an LLM to generate a grounded answer.

PROVIDER-AGNOSTIC: this uses the OpenAI-compatible Chat Completions interface,
which is supported by almost every LLM provider. You bring whatever LLM key
you have and point it at the matching base URL + model:

    Provider        LLM_BASE_URL                                  example LLM_MODEL
    -----------     ------------------------------------------    ---------------------------
    OpenAI          (leave unset)                                 gpt-4o-mini
    OpenRouter      https://openrouter.ai/api/v1                  openai/gpt-4o-mini
    Groq            https://api.groq.com/openai/v1                llama-3.3-70b-versatile
    Together        https://api.together.xyz/v1                   meta-llama/Llama-3.3-70B-Instruct-Turbo
    DeepSeek        https://api.deepseek.com                       deepseek-chat
    Fireworks       https://api.fireworks.ai/inference/v1         accounts/fireworks/models/llama-v3p3-70b-instruct
    Google Gemini   https://generativelanguage.googleapis.com/v1beta/openai   gemini-2.0-flash
    Ollama (local)  http://localhost:11434/v1                     llama3.1        (LLM_API_KEY can be "ollama")
    LM Studio       http://localhost:1234/v1                      (whatever you loaded)

Configure via environment variables:
    LLM_API_KEY   your key            (falls back to OPENAI_API_KEY)
    LLM_BASE_URL  provider base URL   (optional; unset = OpenAI)
    LLM_MODEL     model name          (optional; overridden by chat.py --model)

Two details do most of the work of making RAG reliable:
  1. GROUNDING INSTRUCTION: answer ONLY from the provided context; say
     "I don't know" when it isn't there. This suppresses hallucination.
  2. CITATIONS: context chunks are numbered [1], [2], ... and we ask the model
     to cite the numbers it used, giving traceability back to the source doc.
"""

from __future__ import annotations

import os
from typing import List, Tuple

from .chunker import Chunk

# Used only if neither chat.py --model nor the LLM_MODEL env var is set.
FALLBACK_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context passages. Follow these rules strictly:\n"
    "1. Base your answer only on the context below. Do not use outside knowledge.\n"
    "2. If the context does not contain the answer, say: "
    "\"I don't know based on the provided documents.\" Do not guess.\n"
    "3. Cite the passages you used with their bracketed numbers, e.g. [1], [2].\n"
    "4. Be concise and accurate."
)


def build_context_block(chunks: List[Tuple[Chunk, float]]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    lines = []
    for i, (chunk, _score) in enumerate(chunks, start=1):
        src = os.path.basename(chunk.source)
        lines.append(f"[{i}] (source: {src})\n{chunk.text}")
    return "\n\n".join(lines)


def build_user_prompt(query: str, chunks: List[Tuple[Chunk, float]]) -> str:
    context = build_context_block(chunks)
    return (
        f"Context passages:\n\n{context}\n\n"
        f"----\n"
        f"Question: {query}\n\n"
        f"Answer using only the context above, and cite the passage numbers you use."
    )


class Generator:
    def __init__(self, model: str | None = None, max_tokens: int = 1024):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Generation requires the OpenAI SDK (it speaks to any "
                "OpenAI-compatible provider). Install it with:\n"
                "    pip install openai"
            ) from e

        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Set your LLM API key first:  export LLM_API_KEY=...\n"
                "(any OpenAI-compatible provider works -- see rag/generator.py)"
            )
        base_url = os.environ.get("LLM_BASE_URL")  # None = OpenAI's default endpoint

        # base_url=None is fine; the SDK uses the default OpenAI endpoint then.
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or os.environ.get("LLM_MODEL") or FALLBACK_MODEL
        self.max_tokens = max_tokens

    def answer(self, query: str, chunks: List[Tuple[Chunk, float]]) -> str:
        if not chunks:
            return "I don't know based on the provided documents. (Nothing was retrieved.)"

        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,   # if a newer OpenAI model rejects this,
                                          # rename to max_completion_tokens
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(query, chunks)},
            ],
        )
        return resp.choices[0].message.content or ""
