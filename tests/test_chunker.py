from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from codesage.chunker import _chunks_from_source, chunk_repository
from codesage.models import ChunkKind

SAMPLE_SOURCE = textwrap.dedent(
    '''
    class Greeter:
        def greet(self, name: str) -> str:
            return f"Hello, {name}"

    def standalone() -> int:
        return 42
    '''
).strip()


def test_chunks_from_source_splits_class_and_function() -> None:
    chunks = _chunks_from_source("sample.py", SAMPLE_SOURCE)

    kinds = {chunk.name: chunk.kind for chunk in chunks}
    assert kinds["Greeter"] == ChunkKind.CLASS
    assert kinds["standalone"] == ChunkKind.FUNCTION
    # nested method is also indexed independently, since it's an equally
    # valid retrieval target on its own
    assert kinds["greet"] == ChunkKind.FUNCTION


def test_chunks_from_source_falls_back_on_syntax_error() -> None:
    broken_source = "def broken(:\n    pass"
    chunks = _chunks_from_source("broken.py", broken_source)

    assert len(chunks) == 1
    assert chunks[0].kind == ChunkKind.MODULE
    assert chunks[0].content == broken_source


def test_chunks_from_source_falls_back_on_empty_file() -> None:
    chunks = _chunks_from_source("empty.py", "")

    assert len(chunks) == 1
    assert chunks[0].kind == ChunkKind.MODULE


@pytest.mark.asyncio
async def test_chunk_repository_skips_excluded_dirs(tmp_path: Path) -> None:
    (tmp_path / "real.py").write_text("def a():\n    pass\n")
    excluded = tmp_path / "__pycache__"
    excluded.mkdir()
    (excluded / "ignored.py").write_text("def b():\n    pass\n")

    chunks = await chunk_repository(tmp_path)

    paths = {chunk.path for chunk in chunks}
    assert "real.py" in paths
    assert not any("__pycache__" in p for p in paths)
