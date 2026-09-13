"""A minimal vector store: cosine similarity over an in-memory numpy matrix.

Trade-off: no chromadb/FAISS/pgvector dependency. For a single-developer or
single-repo tool, a full vector database is solving a scale problem this tool
doesn't have yet (YAGNI) — an in-memory matrix with JSON persistence is
enough for a codebase up to roughly tens of thousands of chunks, and it keeps
`index`/`query` at O(n) with no server process to run. If CodeSage needs to
serve many repositories concurrently, this module is the one to replace —
its interface (`add`, `search`, `save`, `load`) is small on purpose so that
swap doesn't ripple into the rest of the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from codesage.models import CodeChunk, EmbeddedChunk, RetrievedChunk


class VectorStore:
    """In-memory cosine-similarity index over `EmbeddedChunk`s."""

    def __init__(self) -> None:
        self._chunks: list[CodeChunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Appends embedded chunks to the index."""
        if not embedded_chunks:
            return
        new_vectors = np.array([ec.vector for ec in embedded_chunks], dtype=np.float32)
        new_vectors = _normalize_rows(new_vectors)
        self._matrix = (
            new_vectors if self._matrix is None else np.vstack([self._matrix, new_vectors])
        )
        self._chunks.extend(ec.chunk for ec in embedded_chunks)

    def search(self, query_vector: list[float], *, top_k: int) -> list[RetrievedChunk]:
        """Returns the `top_k` chunks most similar to `query_vector`."""
        if self._matrix is None or not self._chunks:
            return []
        query = _normalize_rows(np.array([query_vector], dtype=np.float32))[0]
        scores = self._matrix @ query
        top_indices = np.argsort(-scores)[:top_k]
        return [
            RetrievedChunk(chunk=self._chunks[i], score=float(scores[i])) for i in top_indices
        ]

    def save(self, path: Path) -> None:
        """Persists the index as JSON. Simple, human-inspectable, fine at this scale."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [chunk.model_dump(mode="json") for chunk in self._chunks],
            "vectors": self._matrix.tolist() if self._matrix is not None else [],
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> VectorStore:
        """Loads a previously saved index. Returns an empty store if none exists yet."""
        store = cls()
        if not path.exists():
            return store
        payload = json.loads(path.read_text())
        store._chunks = [CodeChunk.model_validate(c) for c in payload["chunks"]]
        vectors = payload["vectors"]
        store._matrix = np.array(vectors, dtype=np.float32) if vectors else None
        return store


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalizes each row so a dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized: np.ndarray = matrix / norms
    return normalized
