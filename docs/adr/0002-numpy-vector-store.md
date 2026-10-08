# 0002. In-memory numpy index instead of a vector database

- Status: accepted
- Date: 2026-09-13

## Context

CodeSage indexes one repository at a time: thousands of chunks, not millions. A vector database (Chroma,
pgvector, FAISS service) adds a process or dependency to run, configure and secure.

## Decision

`VectorStore` keeps an L2-normalised float32 matrix in memory. Search is one matrix-vector product, so
dot product equals cosine similarity. The index persists as JSON. The public interface is `add`, `search`,
`save`, `load`.

## Consequences

- No server, no extra dependency, and the index file can be read by a human.
- Search is O(n) per query and the whole index must fit in memory; fine up to tens of thousands of chunks.
- JSON is slow and large for big indexes. Switch to `.npy` plus a JSON sidecar before JSON becomes the
  bottleneck.
- Replacing the store with pgvector or Chroma touches only `vector_store.py` because callers use that
  small interface.
