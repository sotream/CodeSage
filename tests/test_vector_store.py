from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from codesage.models import ChunkKind, CodeChunk, EmbeddedChunk, IndexMeta
from codesage.vector_store import IncompatibleIndexError, VectorStore


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


def _at(path: str, name: str) -> CodeChunk:
    return _chunk(name).model_copy(update={"path": path})


def test_remove_paths_drops_chunks_and_vectors() -> None:
    store = VectorStore()
    store.add(
        [
            EmbeddedChunk(chunk=_chunk("keep"), vector=[1.0, 0.0]),
            EmbeddedChunk(chunk=_at("x.py", "drop"), vector=[0.0, 1.0]),
        ]
    )
    store.remove_paths({"x.py"})
    assert [c.name for c in store.chunks] == ["keep"]
    assert [r.chunk.name for r in store.search([0.0, 1.0], top_k=5)] == ["keep"]


def test_remove_paths_of_everything_leaves_a_searchable_empty_store() -> None:
    store = VectorStore()
    store.add([EmbeddedChunk(chunk=_chunk("a"), vector=[1.0, 0.0])])
    store.remove_paths({"sample.py"})
    assert store.chunks == []
    assert store.search([1.0, 0.0], top_k=3) == []


def test_search_breaks_ties_by_chunk_id_not_insertion_order() -> None:
    first = EmbeddedChunk(chunk=_at("b.py", "b"), vector=[1.0, 0.0])
    second = EmbeddedChunk(chunk=_at("a.py", "a"), vector=[1.0, 0.0])
    forward, backward = VectorStore(), VectorStore()
    forward.add([first, second])
    backward.add([second, first])

    def ids(store: VectorStore) -> list[str]:
        return [r.chunk.chunk_id for r in store.search([1.0, 0.0], top_k=2)]

    assert ids(forward) == ids(backward)


def test_meta_roundtrips(tmp_path: Path) -> None:
    store = VectorStore()
    store.meta = IndexMeta(embedding_model="m", chunker_version=3, files={"a.py": "abc"})
    store.add([EmbeddedChunk(chunk=_chunk("a"), vector=[1.0, 0.0])])
    store.save(tmp_path / "i.json")
    loaded = VectorStore.load(tmp_path / "i.json")
    assert loaded.meta == store.meta


def test_load_rejects_an_unversioned_v1_index(tmp_path: Path) -> None:
    path = tmp_path / "i.json"
    path.write_text(json.dumps({"chunks": [], "vectors": []}))
    with pytest.raises(IncompatibleIndexError, match="format"):
        VectorStore.load(path)


def test_load_rejects_a_corrupt_index(tmp_path: Path) -> None:
    path = tmp_path / "i.json"
    path.write_text("{not json")
    with pytest.raises(IncompatibleIndexError):
        VectorStore.load(path)


def test_failed_save_keeps_the_old_index_and_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "i.json"
    old = VectorStore()
    old.add([EmbeddedChunk(chunk=_chunk("old"), vector=[1.0, 0.0])])
    old.save(path)
    before = path.read_bytes()

    new = VectorStore()
    new.add([EmbeddedChunk(chunk=_chunk("new"), vector=[0.0, 1.0])])

    def boom(*_: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        new.save(path)

    assert path.read_bytes() == before
    assert [c.name for c in VectorStore.load(path).chunks] == ["old"]
    assert list(tmp_path.glob("*.tmp")) == []


def test_load_rejects_an_index_with_fewer_vectors_than_chunks(tmp_path: Path) -> None:
    path = tmp_path / "i.json"
    store = VectorStore()
    store.add(
        [
            EmbeddedChunk(chunk=_chunk("a"), vector=[1.0, 0.0]),
            EmbeddedChunk(chunk=_chunk("b"), vector=[0.0, 1.0]),
        ]
    )
    store.save(path)
    payload = json.loads(path.read_text())
    payload["vectors"] = payload["vectors"][:1]
    path.write_text(json.dumps(payload))

    with pytest.raises(IncompatibleIndexError, match="2 chunks but 1 vectors"):
        VectorStore.load(path)


def test_save_syncs_the_directory_so_the_rename_survives_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced_dirs: list[bool] = []
    real_fsync = os.fsync

    def spy(fd: int) -> None:
        synced_dirs.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)
    store = VectorStore()
    store.add([EmbeddedChunk(chunk=_chunk("a"), vector=[1.0, 0.0])])
    store.save(tmp_path / "i.json")

    # Once for the data file, then once for the directory that holds the new name.
    assert synced_dirs == [False, True]
