from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes import CountingEmbedder
from typer.testing import CliRunner

from codesage import cli
from codesage.models import CodeChunk, EmbeddedChunk

runner = CliRunner()


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> CountingEmbedder:
    embedder = CountingEmbedder()

    async def _embed_chunks(chunks: list[CodeChunk], *, model_name: str) -> list[EmbeddedChunk]:
        return await embedder(chunks)

    async def _embed_query(query: str, *, model_name: str) -> list[float]:
        return await embedder.embed_query(query)

    monkeypatch.setattr(cli, "embed_chunks", _embed_chunks)
    monkeypatch.setattr(cli, "embed_query", _embed_query)
    return embedder


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "cfg.py").write_text("def parse_config():\n    return 'read yaml settings file'\n")
    (root / "mail.py").write_text("def send_mail():\n    return 'smtp server message'\n")
    return root


def test_index_then_eval_end_to_end(tmp_path: Path, fake_embedder: CountingEmbedder) -> None:
    root = _repo(tmp_path)
    index_path = tmp_path / "idx.json"
    cases = tmp_path / "e.jsonl"
    cases.write_text(
        '{"question": "read yaml settings file", "path": "cfg.py", "name": "parse_config"}\n'
        '{"question": "smtp server message", "path": "mail.py", "name": "send_mail"}\n'
    )

    out = runner.invoke(cli.app, ["index", str(root), "--index-path", str(index_path)])
    assert out.exit_code == 0, out.output
    assert index_path.exists()

    result = runner.invoke(cli.app, ["eval", str(cases), "--index-path", str(index_path)])
    assert result.exit_code == 0, result.output
    assert "hit@1=1.000" in result.output


def test_eval_on_missing_index_is_a_clear_error(tmp_path: Path) -> None:
    cases = tmp_path / "e.jsonl"
    cases.write_text('{"question": "q", "path": "a.py", "name": "f"}\n')
    result = runner.invoke(cli.app, ["eval", str(cases), "--index-path", str(tmp_path / "no.json")])
    assert result.exit_code == 1
    assert "missing or empty" in result.output
    assert "Traceback" not in result.output


def test_eval_rejects_small_depth(tmp_path: Path, fake_embedder: CountingEmbedder) -> None:
    root = _repo(tmp_path)
    index_path = tmp_path / "idx.json"
    runner.invoke(cli.app, ["index", str(root), "--index-path", str(index_path)])
    cases = tmp_path / "e.jsonl"
    cases.write_text('{"question": "q", "path": "cfg.py", "name": "parse_config"}\n')
    result = runner.invoke(
        cli.app, ["eval", str(cases), "--index-path", str(index_path), "--depth", "2"]
    )
    assert result.exit_code == 1
    assert "depth" in result.output


def test_second_index_run_embeds_nothing_and_says_so(
    tmp_path: Path, fake_embedder: CountingEmbedder
) -> None:
    root = _repo(tmp_path)
    index_path = tmp_path / "idx.json"
    runner.invoke(cli.app, ["index", str(root), "--index-path", str(index_path)])
    calls_after_first = fake_embedder.calls

    result = runner.invoke(cli.app, ["index", str(root), "--index-path", str(index_path)])

    assert result.exit_code == 0, result.output
    assert fake_embedder.calls == calls_after_first
    assert "0 added, 0 changed, 0 removed, 2 unchanged" in result.output


def test_full_flag_forces_a_rebuild(tmp_path: Path, fake_embedder: CountingEmbedder) -> None:
    root = _repo(tmp_path)
    index_path = tmp_path / "idx.json"
    runner.invoke(cli.app, ["index", str(root), "--index-path", str(index_path)])
    calls_after_first = fake_embedder.calls

    result = runner.invoke(cli.app, ["index", str(root), "--full", "--index-path", str(index_path)])

    assert fake_embedder.calls == calls_after_first + 1
    assert "Full rebuild: --full requested" in result.output


def test_old_index_prints_a_clear_message_and_rebuilds(
    tmp_path: Path, fake_embedder: CountingEmbedder
) -> None:
    root = _repo(tmp_path)
    index_path = tmp_path / "idx.json"
    index_path.write_text(json.dumps({"chunks": [], "vectors": []}))

    result = runner.invoke(cli.app, ["index", str(root), "--index-path", str(index_path)])

    assert result.exit_code == 0, result.output
    assert "format version" in result.output
    assert "Traceback" not in result.output


def test_eval_refuses_an_index_built_with_another_model(
    tmp_path: Path, fake_embedder: CountingEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path)
    index_path = tmp_path / "idx.json"
    monkeypatch.setenv("CODESAGE_EMBEDDING_MODEL", "model-a")
    runner.invoke(cli.app, ["index", str(root), "--index-path", str(index_path)])
    cases = tmp_path / "e.jsonl"
    cases.write_text('{"question": "q", "path": "cfg.py", "name": "parse_config"}\n')

    monkeypatch.setenv("CODESAGE_EMBEDDING_MODEL", "model-b")
    result = runner.invoke(cli.app, ["eval", str(cases), "--index-path", str(index_path)])

    assert result.exit_code == 1
    assert "model-a" in result.output and "model-b" in result.output


def test_eval_on_an_old_format_index_is_a_clear_error(tmp_path: Path) -> None:
    index_path = tmp_path / "old.json"
    index_path.write_text(json.dumps({"chunks": [], "vectors": []}))
    cases = tmp_path / "e.jsonl"
    cases.write_text('{"question": "q", "path": "a.py", "name": "f"}\n')

    result = runner.invoke(cli.app, ["eval", str(cases), "--index-path", str(index_path)])

    assert result.exit_code == 1
    assert "format version" in result.output
    assert "Traceback" not in result.output


def test_eval_with_a_missing_eval_file_is_a_clear_error(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["eval", str(tmp_path / "nope.jsonl")])
    assert result.exit_code == 1
    assert "nope.jsonl" in result.output
    assert "Traceback" not in result.output


def test_index_of_a_mistyped_root_fails_and_keeps_the_existing_index(
    tmp_path: Path, fake_embedder: CountingEmbedder
) -> None:
    root = _repo(tmp_path)
    index_path = tmp_path / "idx.json"
    runner.invoke(cli.app, ["index", str(root), "--index-path", str(index_path)])
    before = index_path.read_bytes()

    result = runner.invoke(
        cli.app, ["index", str(tmp_path / "no-such-dir"), "--index-path", str(index_path)]
    )

    assert result.exit_code == 1
    assert "not a directory" in result.output
    assert index_path.read_bytes() == before
