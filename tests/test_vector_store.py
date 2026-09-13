from __future__ import annotations

from pathlib import Path

from codesage.models import ChunkKind, CodeChunk, EmbeddedChunk
from codesage.vector_store import VectorStore


def _chunk(name: str) -> CodeChunk:
    return CodeChunk(
        path="sample.py",
        kind=ChunkKind.FUNCTION,
        name=name,
        start_line=1,
        end_line=2,
        content=f"def {name}(): pass",
    )


def test_search_ranks_closest_vector_first() -> None:
    store = VectorStore()
    store.add(
        [
            EmbeddedChunk(chunk=_chunk("near"), vector=[1.0, 0.0]),
            EmbeddedChunk(chunk=_chunk("far"), vector=[0.0, 1.0]),
        ]
    )

    results = store.search([1.0, 0.01], top_k=2)

    assert [r.chunk.name for r in results] == ["near", "far"]
    assert results[0].score > results[1].score


def test_search_on_empty_store_returns_empty_list() -> None:
    store = VectorStore()
    assert store.search([1.0, 0.0], top_k=5) == []


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    store = VectorStore()
    store.add([EmbeddedChunk(chunk=_chunk("alpha"), vector=[0.5, 0.5])])
    index_path = tmp_path / "index.json"

    store.save(index_path)
    loaded = VectorStore.load(index_path)

    results = loaded.search([0.5, 0.5], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.name == "alpha"


def test_load_missing_file_returns_empty_store(tmp_path: Path) -> None:
    store = VectorStore.load(tmp_path / "missing.json")
    assert store.search([1.0, 0.0], top_k=1) == []
