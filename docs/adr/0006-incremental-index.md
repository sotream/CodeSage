# 0006. Incremental index keyed by file content hash

- Status: accepted
- Date: 2026-10-09

## Context

`index` re-embedded the whole repository on every run, and embedding is the expensive step. Most runs
change a handful of files.

## Decision

Store a SHA-256 of each file's raw bytes in the index metadata, next to the embedding model name, the
chunker version and a format version (index format v2). `index` hashes the tree, then re-chunks and
re-embeds only new or changed files, drops the chunks of deleted files and leaves everything else
untouched. A missing index, `--full`, an old or corrupt format, a different embedding model or a
different chunker version forces a full rebuild, with the reason printed. The index is written to a
temporary file in the same directory, fsynced and renamed over the old one, and only after all
embedding succeeded, so an interrupted run leaves the previous index valid. The chunker version is
bumped by hand when chunk output changes for the same source.

Incremental and full results are compared by eval ranks, not by raw floats: with a deterministic
embedder they are exactly equal (tested), but with the real model batch composition can change float
low bits (about 1e-7), so vectors from an incremental and a full run may differ in the last digits
while ranking identically. Search breaks score ties by chunk id so ranking does not depend on the order
chunks were inserted.

## Consequences

- An unchanged repo costs zero embedder calls; one edited file costs only its chunks.
- Granularity is the file. A one-line edit re-embeds every chunk of that file. Chunk-level diffing would
  save little at this scale and adds matching logic that can go wrong.
- A chunk embeds only its own text, so no cross-file context exists to go stale: changing a callee does
  not invalidate its callers' vectors. If chunks ever embed cross-file context (imports, call graph),
  this decision must be revisited.
- A rename is a delete plus an add, so the renamed file is re-embedded even though its hash is known.
  Reusing vectors by matching hashes is possible and not done (YAGNI).
- Not handled: two `index` runs at once (last writer wins, no lock). A file edited while `index` runs
  is read once, so the index stays self-consistent and is at most one run behind.
- Changing `chunker.py` output without bumping `CHUNKER_VERSION` leaves stale chunks: a human step the
  tests cannot enforce.
