"""Command-line entry point.

Kept thin on purpose: every command below only wires together functions from
`indexer`, `embedder`, `vector_store`, `evaluation`, and `qa` — none of the actual logic
lives here, so the pipeline stays testable without invoking the CLI.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from anthropic import AsyncAnthropic

from codesage.config import get_settings
from codesage.embedder import embed_chunks, embed_query
from codesage.evaluation import DEFAULT_DEPTH, evaluate, format_report, load_cases
from codesage.indexer import build_index, load_store
from codesage.models import CodeChunk, EmbeddedChunk
from codesage.qa import generate_answer, retrieve
from codesage.vector_store import IncompatibleIndexError, VectorStore

app = typer.Typer(help="CodeSage: ask questions about a codebase.")

_PATH_ARG = typer.Argument(help="Root directory of the codebase to index.")
_QUESTION_ARG = typer.Argument(help="Question to ask about the indexed codebase.")
_EVAL_ARG = typer.Argument(help="JSONL file of (question, expected path and name) cases.")
_INDEX_PATH_OPT = typer.Option(
    "--index-path", help="Index file to write or read. Defaults to CODESAGE_INDEX_PATH."
)
_FULL_OPT = typer.Option("--full", help="Ignore the existing index and rebuild everything.")
_DEPTH_OPT = typer.Option(help="Retrieval depth; MRR is reported at this cutoff.")
_MISSES_OPT = typer.Option(help="How many worst misses to list.")


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(1)


@app.command()
def index(
    path: Path = _PATH_ARG,
    full: Annotated[bool, _FULL_OPT] = False,
    index_path: Annotated[Path | None, _INDEX_PATH_OPT] = None,
) -> None:
    """Indexes the codebase, re-embedding only files that changed since the last run."""
    settings = get_settings()
    target = index_path or Path(settings.index_path)
    if not path.is_dir():
        # An unreadable root looks like "every file was deleted" and would silently empty the index.
        raise _fail(f"{path} is not a directory")

    async def _embed(chunks: list[CodeChunk]) -> list[EmbeddedChunk]:
        return await embed_chunks(chunks, model_name=settings.embedding_model)

    async def _run() -> None:
        store, problem = load_store(target)
        force = "--full requested" if full else problem
        store, report = await build_index(
            path, store, _embed, model_name=settings.embedding_model, force_reason=force
        )
        store.save(target)
        typer.echo(report.summary())
        typer.echo(f"Index saved to {target}")

    asyncio.run(_run())


@app.command(name="eval")
def eval_command(
    eval_file: Path = _EVAL_ARG,
    index_path: Annotated[Path | None, _INDEX_PATH_OPT] = None,
    depth: Annotated[int, _DEPTH_OPT] = DEFAULT_DEPTH,
    show_misses: Annotated[int, _MISSES_OPT] = 5,
) -> None:
    """Measures retrieval (hit@k, MRR) against an eval file. No LLM call, no API key."""
    settings = get_settings()
    target = index_path or Path(settings.index_path)

    async def _run() -> None:
        try:
            cases = load_cases(eval_file)
            store = VectorStore.load(target)
            if not store.chunks:
                raise _fail(
                    f"index at {target} is missing or empty; "
                    f"run `codesage index <repo> --index-path {target}` first"
                )
            if store.meta.embedding_model != settings.embedding_model:
                raise _fail(
                    f"index was built with embedding model '{store.meta.embedding_model}' but "
                    f"CODESAGE_EMBEDDING_MODEL is '{settings.embedding_model}'; "
                    "queries and chunks must use the same model"
                )

            async def _embed(query: str) -> list[float]:
                return await embed_query(query, model_name=settings.embedding_model)

            report = await evaluate(cases, store, _embed, depth=depth)
        except IncompatibleIndexError as err:
            raise _fail(f"{err}; run `codesage index` to rebuild it") from err
        except (ValueError, OSError) as err:
            raise _fail(str(err)) from err
        typer.echo(format_report(report, show_misses=show_misses))

    asyncio.run(_run())


@app.command()
def ask(question: str = _QUESTION_ARG) -> None:
    """Answers a question using the previously built index."""
    settings = get_settings()

    async def _run() -> None:
        store = VectorStore.load(Path(settings.index_path))
        query_vector = await embed_query(question, model_name=settings.embedding_model)
        retrieved = retrieve(store, query_vector, top_k=settings.top_k)
        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        answer = await generate_answer(
            question, retrieved, client=client, model=settings.anthropic_model
        )
        typer.echo(answer.text)
        if answer.sources:
            typer.echo("\nSources:")
            for source in answer.sources:
                location = f"{source.path}:{source.start_line}-{source.end_line}"
                typer.echo(f"  - {location} ({source.name})")

    asyncio.run(_run())


if __name__ == "__main__":
    app()
