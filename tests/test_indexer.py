from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fakes import CountingEmbedder

from codesage import indexer
from codesage.evaluation import EvalCase, evaluate
from codesage.indexer import IndexReport, build_index, load_store
from codesage.models import CodeChunk, EmbeddedChunk
from codesage.vector_store import VectorStore

MODEL = "fake-model"

FILES = {
    "cfg.py": "def parse_config():\n    return 'read yaml settings file'\n",
    "mail.py": "def send_mail():\n    return 'smtp server message'\n",
    "pkg/util.py": "def clamp(value):\n    return 'limit value floor ceiling'\n",
}


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    _write(root, FILES)
    return root


def run(
    root: Path,
    index_path: Path,
    embedder: CountingEmbedder,
    *,
    model: str = MODEL,
    force: str | None = None,
) -> IndexReport:
    """One CLI-like cycle: load, build, save."""
    store, problem = load_store(index_path)
    store, report = asyncio.run(
        build_index(root, store, embedder, model_name=model, force_reason=force or problem)
    )
    store.save(index_path)
    return report


def ids(store: VectorStore) -> set[str]:
    return {c.chunk_id for c in store.chunks}


def test_first_run_is_a_full_build(repo: Path, tmp_path: Path) -> None:
    embedder = CountingEmbedder()
    report = run(repo, tmp_path / "i.json", embedder)
    assert report.full_rebuild_reason == "no existing index"
    assert (report.files_added, report.files_unchanged) == (3, 0)
    assert report.chunks_added == 3
    assert embedder.calls == 1


def test_unchanged_repo_makes_zero_embedder_calls(repo: Path, tmp_path: Path) -> None:
    index = tmp_path / "i.json"
    run(repo, index, CountingEmbedder())
    before = ids(VectorStore.load(index))

    embedder = CountingEmbedder()
    report = run(repo, index, embedder)

    assert embedder.calls == 0
    assert report.full_rebuild_reason is None
    assert (report.files_added, report.files_changed, report.files_removed) == (0, 0, 0)
    assert report.files_unchanged == 3
    assert report.chunks_unchanged == 3
    assert ids(VectorStore.load(index)) == before


def test_edited_file_is_the_only_one_re_embedded(repo: Path, tmp_path: Path) -> None:
    index = tmp_path / "i.json"
    run(repo, index, CountingEmbedder())

    (repo / "mail.py").write_text(
        "def send_mail():\n    return 'smtp relay'\n\n\ndef extra():\n    pass\n"
    )
    embedder = CountingEmbedder()
    report = run(repo, index, embedder)

    assert {i.split(":")[0] for i in embedder.embedded} == {"mail.py"}
    assert len(embedder.embedded) == 2
    assert (report.files_changed, report.files_unchanged) == (1, 2)
    assert (report.chunks_added, report.chunks_removed, report.chunks_unchanged) == (2, 1, 2)
    mail_ids = {i for i in ids(VectorStore.load(index)) if i.startswith("mail.py")}
    assert mail_ids == {"mail.py:1-2", "mail.py:5-6"}


def test_deleted_file_loses_its_chunks(repo: Path, tmp_path: Path) -> None:
    index = tmp_path / "i.json"
    run(repo, index, CountingEmbedder())
    (repo / "mail.py").unlink()

    embedder = CountingEmbedder()
    report = run(repo, index, embedder)

    assert embedder.calls == 0
    assert report.files_removed == 1
    assert all(not i.startswith("mail.py") for i in ids(VectorStore.load(index)))


def test_rename_removes_old_path_and_adds_new(repo: Path, tmp_path: Path) -> None:
    index = tmp_path / "i.json"
    run(repo, index, CountingEmbedder())
    (repo / "mail.py").rename(repo / "email.py")

    embedder = CountingEmbedder()
    report = run(repo, index, embedder)

    assert (report.files_added, report.files_removed) == (1, 1)
    paths = {c.path for c in VectorStore.load(index).chunks}
    assert "email.py" in paths and "mail.py" not in paths
    assert {i.split(":")[0] for i in embedder.embedded} == {"email.py"}


def test_interrupted_embedding_leaves_the_old_index_valid(repo: Path, tmp_path: Path) -> None:
    index = tmp_path / "i.json"
    run(repo, index, CountingEmbedder())
    before = index.read_bytes()
    (repo / "mail.py").write_text("def send_mail():\n    return 'changed'\n")

    async def exploding(chunks: list[CodeChunk]) -> list[EmbeddedChunk]:
        raise KeyboardInterrupt

    store, _ = load_store(index)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(build_index(repo, store, exploding, model_name=MODEL))

    assert index.read_bytes() == before
    assert len(VectorStore.load(index).chunks) == 3


def test_model_change_forces_full_rebuild(repo: Path, tmp_path: Path) -> None:
    index = tmp_path / "i.json"
    run(repo, index, CountingEmbedder())

    embedder = CountingEmbedder()
    report = run(repo, index, embedder, model="other-model")

    assert report.full_rebuild_reason is not None
    assert "embedding model changed" in report.full_rebuild_reason
    assert len(embedder.embedded) == 3
    assert VectorStore.load(index).meta.embedding_model == "other-model"


def test_chunker_version_change_forces_full_rebuild(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = tmp_path / "i.json"
    run(repo, index, CountingEmbedder())
    monkeypatch.setattr(indexer, "CHUNKER_VERSION", 99)

    embedder = CountingEmbedder()
    report = run(repo, index, embedder)

    assert report.full_rebuild_reason is not None
    assert "chunker version" in report.full_rebuild_reason
    assert len(embedder.embedded) == 3


def test_old_format_index_triggers_rebuild_not_a_crash(repo: Path, tmp_path: Path) -> None:
    index = tmp_path / "i.json"
    index.write_text('{"chunks": [], "vectors": []}')

    embedder = CountingEmbedder()
    report = run(repo, index, embedder)

    assert report.full_rebuild_reason is not None
    assert "format version" in report.full_rebuild_reason
    assert len(embedder.embedded) == 3


def test_full_flag_re_embeds_everything(repo: Path, tmp_path: Path) -> None:
    index = tmp_path / "i.json"
    run(repo, index, CountingEmbedder())

    embedder = CountingEmbedder()
    report = run(repo, index, embedder, force="--full requested")

    assert report.full_rebuild_reason == "--full requested"
    assert len(embedder.embedded) == 3
    assert report.chunks_removed == 3


def test_syntax_error_and_non_utf8_files_index_and_stay_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "broken.py").write_text("def oops(:\n")
    (root / "binary.py").write_bytes(b"\xff\xfe\x00 not utf8")
    index = tmp_path / "i.json"

    first = run(root, index, CountingEmbedder())
    embedder = CountingEmbedder()
    second = run(root, index, embedder)

    assert first.files_added == 2
    assert embedder.calls == 0
    assert second.files_unchanged == 2


def test_empty_repo_and_deleting_every_file_save_a_loadable_index(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    index = tmp_path / "i.json"
    report = run(root, index, CountingEmbedder())
    assert report.files_added == 0
    assert VectorStore.load(index).chunks == []

    (root / "a.py").write_text("def f():\n    pass\n")
    run(root, index, CountingEmbedder())
    (root / "a.py").unlink()
    run(root, index, CountingEmbedder())
    assert VectorStore.load(index).chunks == []


def test_summary_mentions_counts_reason_and_time(repo: Path, tmp_path: Path) -> None:
    report = run(repo, tmp_path / "i.json", CountingEmbedder())
    text = report.summary()
    assert "Full rebuild" in text and "no existing index" in text
    assert "3 added" in text and "unchanged" in text
    assert text.splitlines()[-1].startswith("Done in ")


CASES = [
    EvalCase(question="read the yaml settings file", path="cfg.py", name="parse_config"),
    EvalCase(question="smtp server message", path="mail.py", name="send_mail"),
    EvalCase(question="limit value floor ceiling", path="pkg/helpers.py", name="clamp"),
    EvalCase(question="brand new feature flags", path="added.py", name="flags"),
]


def test_incremental_run_scores_identically_to_a_full_rebuild(repo: Path, tmp_path: Path) -> None:
    incremental_index = tmp_path / "inc.json"
    run(repo, incremental_index, CountingEmbedder())

    # edit one file, delete one, add one, rename one
    (repo / "mail.py").write_text("def send_mail():\n    return 'smtp server message relay'\n")
    (repo / "cfg.py").unlink()
    (repo / "added.py").write_text("def flags():\n    return 'brand new feature flags'\n")
    (repo / "pkg" / "util.py").rename(repo / "pkg" / "helpers.py")
    run(repo, incremental_index, CountingEmbedder())

    full_index = tmp_path / "full.json"
    run(repo, full_index, CountingEmbedder())

    incremental, full = VectorStore.load(incremental_index), VectorStore.load(full_index)
    assert ids(incremental) == ids(full)

    present = {(c.path, c.name) for c in full.chunks}
    cases = [c for c in CASES if (c.path, c.name) in present]
    assert len(cases) == 3
    embedder = CountingEmbedder()
    report_inc = asyncio.run(evaluate(cases, incremental, embedder.embed_query))
    report_full = asyncio.run(evaluate(cases, full, embedder.embed_query))
    assert report_inc == report_full
    assert report_full.overall.hit_at[1] > 0


def test_file_deleted_between_discovery_and_read_is_treated_as_removed(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = tmp_path / "i.json"
    run(repo, index, CountingEmbedder())
    real_discover = indexer.discover_files

    def racy_discover(root: Path) -> list[Path]:
        found = real_discover(root)
        (root / "mail.py").unlink()  # vanishes after discovery, before the read
        return found

    monkeypatch.setattr(indexer, "discover_files", racy_discover)
    report = run(repo, index, CountingEmbedder())

    assert report.files_removed == 1
    assert all(not i.startswith("mail.py") for i in ids(VectorStore.load(index)))
