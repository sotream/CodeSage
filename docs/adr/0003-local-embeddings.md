# 0003. Local embedding model instead of a hosted API

- Status: accepted
- Date: 2026-09-13

## Context

Indexing embeds every chunk and may run on each commit. A hosted embeddings API adds a network call per
batch, a per-token cost and a second API key and vendor.

## Decision

Use `sentence-transformers` with `all-MiniLM-L6-v2` by default (`CODESAGE_EMBEDDING_MODEL`). The model is
loaded once and cached with `lru_cache`. Embedding is CPU-bound, so `asyncio.to_thread` keeps the CLI
responsive but does not add real concurrency.

## Consequences

- Indexing works offline and costs nothing per chunk; only the question and retrieved excerpts leave the
  machine, in the `ask` step.
- First run downloads the model (a few hundred MB incl. torch).
- A general-purpose text model is not trained on code. Retrieval quality is unmeasured until an evaluation
  set exists (see Roadmap in the README).
