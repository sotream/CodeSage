# CodeSage

RAG pipeline that answers questions about a Python codebase: chunk, embed, store, retrieve, answer. Not an
agent (see [ADR 0004](docs/adr/0004-rag-not-agent.md)). Favour small modules with one clear boundary each.

## Stack

Python 3.11+, uv, pydantic v2, sentence-transformers (local), numpy, Anthropic SDK, typer, pytest, ruff, mypy strict.

## Commands

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run mypy src
```

## Architecture in five lines

1. `chunker.py` splits files into function and class chunks with `ast`.
2. `embedder.py` turns chunks into vectors with a local model.
3. `vector_store.py` is an in-memory cosine index persisted as JSON. Its interface is `add`, `search`, `save`, `load`.
4. `qa.py` keeps `retrieve` and `generate_answer` separate so each is testable alone.
5. `cli.py` only wires the above together; no logic lives there.

## Hard rules

- Type hints everywhere; `mypy --strict` must pass. No bare `Any`.
- Before finishing any task run `pytest`, `ruff check .` and `mypy src`. Do not claim something works without running it.
- Never read or print `.env` files or API keys.
- Conventional Commits, small changes. Comments explain why, not what.
- A new dependency needs an ADR or a line in the README trade-offs.
- Decisions go in `docs/adr/`, using `template.md`.
