"""Deterministic fake embedder for tests: no model download, content-based vectors.

A bag-of-words hash vector keeps similarity meaningful (shared words score higher), so eval
tests can have real hits and misses, while staying exactly reproducible.
"""

from __future__ import annotations

import hashlib
import re

from codesage.models import CodeChunk, EmbeddedChunk

_DIM = 64


def embed_text(text: str) -> list[float]:
    vector = [0.0] * _DIM
    for token in re.findall(r"[a-z_]+", text.lower()):
        bucket = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % _DIM
        vector[bucket] += 1.0
    return vector


class CountingEmbedder:
    """Counts how often, and for which chunks, the embedder is asked to work."""

    def __init__(self) -> None:
        self.calls = 0
        self.embedded: list[str] = []

    async def __call__(self, chunks: list[CodeChunk]) -> list[EmbeddedChunk]:
        self.calls += 1
        self.embedded.extend(chunk.chunk_id for chunk in chunks)
        return [EmbeddedChunk(chunk=c, vector=embed_text(c.content)) for c in chunks]

    async def embed_query(self, query: str) -> list[float]:
        return embed_text(query)
