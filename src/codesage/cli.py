"""Command-line entry point.

Kept thin on purpose: every command below only wires together functions from
`chunker`, `embedder`, `vector_store`, and `qa` — none of the actual logic
lives here, so the pipeline stays testable without invoking the CLI.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from anthropic import AsyncAnthropic

from codesage.chunker import chunk_repository
from codesage.config import get_settings
from codesage.embedder import embed_chunks, embed_query
from codesage.qa import generate_answer, retrieve
from codesage.vector_store import VectorStore

app = typer.Typer(help="CodeSage: ask questions about a codebase.")

_PATH_ARG = typer.Argument(help="Root directory of the codebase to index.")
_QUESTION_ARG = typer.Argument(help="Question to ask about the indexed codebase.")


@app.command()
def index(path: Path = _PATH_ARG) -> None:
    """Chunks, embeds, and persists an index for the given codebase."""
    settings = get_settings()

    async def _run() -> None:
        chunks = await chunk_repository(path)
        typer.echo(f"Chunked {len(chunks)} functions/classes from {path}")
        embedded = await embed_chunks(chunks, model_name=settings.embedding_model)
        store = VectorStore()
        store.add(embedded)
        store.save(Path(settings.index_path))
        typer.echo(f"Index saved to {settings.index_path}")

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
