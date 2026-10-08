# 0001. Chunk by AST node, not by fixed window

- Status: accepted
- Date: 2026-09-13

## Context

Retrieval quality depends on what one chunk contains. A fixed-length window can cut a function in half, so
the vector represents an incomplete unit of logic and the answer prompt shows a broken excerpt.

## Decision

`chunker.py` parses each file with Python's `ast` and emits one chunk per function, async function or
class, with path, name and line range. A file with no such nodes becomes one module chunk. A file that fails
to parse (`SyntaxError`) is indexed whole instead of being dropped.

## Consequences

- Chunks are semantically whole and carry exact line numbers for source citations.
- Only Python is supported. A new language means a new chunker function, not a change to the pipeline.
- Large functions stay one chunk (`max_function_lines` documents this limit); very long chunks can dilute
  the embedding.
- `ast.walk` also visits methods inside classes, so a class chunk and its methods can overlap. Retrieval
  may return both. Acceptable now; revisit when evaluating retrieval quality.
