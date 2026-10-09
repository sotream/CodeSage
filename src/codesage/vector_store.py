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
import os
from pathlib import Path

import numpy as np

from codesage.models import (
    INDEX_FORMAT_VERSION,
    CodeChunk,
    EmbeddedChunk,
    IndexMeta,
    RetrievedChunk,
)


class IncompatibleIndexError(Exception):
    """The index file cannot be used as is (old format, corrupt): rebuild it."""


class VectorStore:
    """In-memory cosine-similarity index over `EmbeddedChunk`s."""

    def __init__(self) -> None:
        self._chunks: list[CodeChunk] = []
        self._matrix: np.ndarray | None = None
        self.meta = IndexMeta()

    @property
    def chunks(self) -> list[CodeChunk]:
        """A copy, so callers (eval, indexer) cannot desync the chunk list from the matrix."""
        return list(self._chunks)

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

    def remove_paths(self, paths: set[str]) -> None:
        """Drops every chunk (and its vector) that belongs to one of `paths`."""
        if not paths or self._matrix is None:
            return
        keep = [i for i, chunk in enumerate(self._chunks) if chunk.path not in paths]
        self._chunks = [self._chunks[i] for i in keep]
        self._matrix = self._matrix[keep] if keep else None

    def search(self, query_vector: list[float], *, top_k: int) -> list[RetrievedChunk]:
        """Returns the `top_k` chunks most similar to `query_vector`."""
        if self._matrix is None or not self._chunks:
            return []
        query = _normalize_rows(np.array([query_vector], dtype=np.float32))[0]
        scores = self._matrix @ query
        # Ties are broken by chunk id so ranking does not depend on insertion order: an index
        # updated incrementally must rank exactly like one rebuilt from scratch.
        top_indices = sorted(
            range(len(self._chunks)), key=lambda i: (-float(scores[i]), self._chunks[i].chunk_id)
        )[:top_k]
        return [
            RetrievedChunk(chunk=self._chunks[i], score=float(scores[i])) for i in top_indices
        ]

    def save(self, path: Path) -> None:
        """Persists the index as JSON, atomically: a crash mid-write cannot corrupt the old file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **self.meta.model_dump(mode="json"),
            "chunks": [chunk.model_dump(mode="json") for chunk in self._chunks],
            "vectors": self._matrix.tolist() if self._matrix is not None else [],
        }
        tmp = path.with_name(path.name + ".tmp")
        try:
            with tmp.open("w") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> VectorStore:
        """Loads a saved index. A missing file gives an empty store; an unusable one raises
        `IncompatibleIndexError` so callers can rebuild instead of crashing."""
        store = cls()
        if not path.exists():
            return store
        try:
            payload = json.loads(path.read_text())
            version = payload.get("format_version")
            if version != INDEX_FORMAT_VERSION:
                raise IncompatibleIndexError(
                    f"index at {path} has format version {version}, expected {INDEX_FORMAT_VERSION}"
                )
            store.meta = IndexMeta.model_validate({k: payload[k] for k in IndexMeta.model_fields})
            store._chunks = [CodeChunk.model_validate(c) for c in payload["chunks"]]
            vectors = payload["vectors"]
        except (ValueError, KeyError, AttributeError) as err:
            raise IncompatibleIndexError(f"index at {path} is unreadable: {err}") from err
        store._matrix = np.array(vectors, dtype=np.float32) if vectors else None
        return store


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalizes each row so a dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized: np.ndarray = matrix / norms
    return normalized
