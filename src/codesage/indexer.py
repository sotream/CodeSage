"""Incremental indexing keyed by per-file content hash.

A file is the unit of staleness: if its SHA-256 is unchanged, its chunks and vectors are kept
untouched; otherwise the whole file is re-chunked and re-embedded. Chunk-level diffing is not
worth it at this scale (see ADR 0006). The embed step is injected so tests can count calls
without a model.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import BaseModel

from codesage.chunker import CHUNKER_VERSION, chunk_source, discover_files
from codesage.models import CodeChunk, EmbeddedChunk, IndexMeta
from codesage.vector_store import IncompatibleIndexError, VectorStore

EmbedFn = Callable[[list[CodeChunk]], Awaitable[list[EmbeddedChunk]]]


class IndexReport(BaseModel):
    files_added: int
    files_changed: int
    files_removed: int
    files_unchanged: int
    chunks_added: int
    chunks_removed: int
    chunks_unchanged: int
    full_rebuild_reason: str | None = None
    seconds: float = 0.0

    def summary(self) -> str:
        lines = []
        if self.full_rebuild_reason:
            lines.append(f"Full rebuild: {self.full_rebuild_reason}")
        lines.append(
            f"Files: {self.files_added} added, {self.files_changed} changed, "
            f"{self.files_removed} removed, {self.files_unchanged} unchanged"
        )
        lines.append(
            f"Chunks: {self.chunks_added} added, {self.chunks_removed} removed, "
            f"{self.chunks_unchanged} unchanged"
        )
        lines.append(f"Done in {self.seconds:.1f}s")
        return "\n".join(lines)


def load_store(path: Path) -> tuple[VectorStore, str | None]:
    """Loads the existing index, or an empty store plus the reason it had to be discarded."""
    try:
        return VectorStore.load(path), None
    except IncompatibleIndexError as err:
        return VectorStore(), str(err)


def _read_and_hash(root: Path, path: Path) -> tuple[str, str, str] | None:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        # Deleted between discovery and read: treating it as absent makes it a removal, which
        # is what it is, instead of aborting the whole run.
        return None
    # The bytes are read once: hashing them and chunking their decoded text agree on what the
    # file was, even if it changes on disk while the index runs.
    return (
        str(path.relative_to(root)),
        hashlib.sha256(data).hexdigest(),
        data.decode("utf-8", errors="ignore"),
    )


def _rebuild_reason(meta: IndexMeta, model_name: str) -> str | None:
    if not meta.embedding_model:
        return "no existing index"
    if meta.embedding_model != model_name:
        return f"embedding model changed ({meta.embedding_model} -> {model_name})"
    if meta.chunker_version != CHUNKER_VERSION:
        return f"chunker version changed ({meta.chunker_version} -> {CHUNKER_VERSION})"
    return None


async def build_index(
    root: Path,
    store: VectorStore,
    embed: EmbedFn,
    *,
    model_name: str,
    force_reason: str | None = None,
) -> tuple[VectorStore, IndexReport]:
    """Brings `store` in line with `root`, embedding only new or changed files.

    Returns the store to save: a fresh one on a full rebuild, otherwise `store` mutated in place.
    Nothing is mutated until embedding has succeeded, so a failure leaves the caller's store
    (and, since saving is atomic, the file on disk) exactly as it was.
    """
    started = time.perf_counter()
    entries = await asyncio.gather(
        *(asyncio.to_thread(_read_and_hash, root, p) for p in discover_files(root))
    )
    current = {e[0]: (e[1], e[2]) for e in entries if e is not None}

    reason = force_reason or _rebuild_reason(store.meta, model_name)
    previous = store
    if reason:
        store = VectorStore()
    known = store.meta.files

    added = [p for p in current if p not in known]
    changed = [p for p in current if p in known and known[p] != current[p][0]]
    removed = [p for p in known if p not in current]

    new_chunks = [c for p in added + changed for c in chunk_source(p, current[p][1])]
    embedded = await embed(new_chunks) if new_chunks else []

    stale = set(changed) | set(removed)
    chunks_removed = (
        len(previous.chunks) if reason else sum(1 for c in store.chunks if c.path in stale)
    )
    store.remove_paths(stale)
    store.add(embedded)
    store.meta = IndexMeta(
        embedding_model=model_name,
        chunker_version=CHUNKER_VERSION,
        files={p: digest for p, (digest, _) in current.items()},
    )

    return store, IndexReport(
        files_added=len(added),
        files_changed=len(changed),
        files_removed=len(removed),
        files_unchanged=len(current) - len(added) - len(changed),
        chunks_added=len(embedded),
        chunks_removed=chunks_removed,
        chunks_unchanged=len(store.chunks) - len(embedded),
        full_rebuild_reason=reason,
        seconds=time.perf_counter() - started,
    )
