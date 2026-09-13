"""Splits a Python codebase into retrievable chunks.

Trade-off: chunking is scoped to Python's `ast` module only, not a general
multi-language parser (e.g. tree-sitter). That keeps the dependency footprint
at zero for this stage, at the cost of only supporting Python codebases in
v0.1. Adding another language means adding another `_chunks_from_*` function,
not touching this module's public API.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

from codesage.models import ChunkKind, CodeChunk

_TOP_LEVEL_NODE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _kind_for_node(node: ast.AST) -> ChunkKind:
    """Maps an AST node type to a `ChunkKind`."""
    if isinstance(node, ast.ClassDef):
        return ChunkKind.CLASS
    return ChunkKind.FUNCTION


def _chunks_from_source(path: str, source: str) -> list[CodeChunk]:
    """Parses one file's source into function/class-level chunks.

    Falls back to a single module-level chunk on a `SyntaxError` — better to
    index an unparsed file whole than to silently drop it from the corpus.
    """
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return [
            CodeChunk(
                path=path,
                kind=ChunkKind.MODULE,
                name=Path(path).stem,
                start_line=1,
                end_line=len(source.splitlines()) or 1,
                content=source,
            )
        ]

    lines = source.splitlines()
    chunks: list[CodeChunk] = []
    for node in ast.walk(tree):
        if not isinstance(node, _TOP_LEVEL_NODE_TYPES):
            continue
        end_line = getattr(node, "end_lineno", node.lineno)
        chunks.append(
            CodeChunk(
                path=path,
                kind=_kind_for_node(node),
                name=node.name,
                start_line=node.lineno,
                end_line=end_line,
                content="\n".join(lines[node.lineno - 1 : end_line]),
            )
        )

    if not chunks:
        chunks.append(
            CodeChunk(
                path=path,
                kind=ChunkKind.MODULE,
                name=Path(path).stem,
                start_line=1,
                end_line=len(lines) or 1,
                content=source,
            )
        )
    return chunks


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


_DEFAULT_EXCLUDE_DIRS = frozenset({".git", ".venv", "__pycache__", "node_modules"})


async def chunk_repository(
    root: Path, *, exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS
) -> list[CodeChunk]:
    """Walks `root` and chunks every `.py` file concurrently.

    File reads are blocking I/O; `asyncio.to_thread` moves each read off the
    event loop so a large repository's files are read concurrently instead of
    one at a time, without needing a thread pool the caller has to manage.
    """
    py_files = [
        p for p in root.rglob("*.py") if not any(part in exclude_dirs for part in p.parts)
    ]

    async def _process(path: Path) -> list[CodeChunk]:
        source = await asyncio.to_thread(_read_file, path)
        return _chunks_from_source(str(path.relative_to(root)), source)

    results = await asyncio.gather(*(_process(p) for p in py_files))
    return [chunk for file_chunks in results for chunk in file_chunks]
