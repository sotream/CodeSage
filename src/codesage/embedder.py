"""Turns code chunks into vectors.

Uses a local sentence-transformers model rather than a hosted embeddings API:
it avoids per-chunk network calls and a second API key, at the cost of a
larger initial model download and being limited to that model's quality
ceiling. For a codebase-QA tool where re-indexing runs often (every commit,
potentially), avoiding network round-trips per chunk was judged the better
trade-off for v0.1.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from codesage.models import CodeChunk, EmbeddedChunk


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> SentenceTransformer:
    """Loads (and caches) the embedding model.

    Cached at module level because loading a sentence-transformers model is
    expensive (seconds, plus a one-time download); repeated calls within a
    process must reuse the same instance rather than reloading per batch.
    """
    return SentenceTransformer(model_name)


def _embed_sync(model_name: str, chunks: list[CodeChunk]) -> list[EmbeddedChunk]:
    model = _load_model(model_name)
    vectors = model.encode([chunk.content for chunk in chunks], convert_to_numpy=True)
    return [
        EmbeddedChunk(chunk=chunk, vector=vector.tolist())
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


async def embed_chunks(chunks: list[CodeChunk], *, model_name: str) -> list[EmbeddedChunk]:
    """Embeds a batch of chunks off the event loop.

    Model inference is CPU-bound, so `asyncio.to_thread` is used purely to
    keep the CLI's event loop responsive while a large batch encodes — this
    is not a concurrency win the way it is for I/O in `chunker.py`.
    """
    if not chunks:
        return []
    return await asyncio.to_thread(_embed_sync, model_name, chunks)


async def embed_query(query: str, *, model_name: str) -> list[float]:
    """Embeds a single query string using the same model as the corpus."""
    model = _load_model(model_name)
    vector = await asyncio.to_thread(model.encode, query)
    return list(vector)
