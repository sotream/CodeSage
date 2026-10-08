# 0004. A retrieval pipeline, not an agent

- Status: accepted
- Date: 2026-09-13

## Context

"Answer a question about this codebase" can be built as an agent loop (plan, call tools, re-plan) or as a
single retrieve-then-answer pipeline.

## Decision

Build a pipeline: chunk, embed, store, retrieve, answer. Retrieval and generation are separate functions in
`qa.py`. The system prompt tells the model to answer only from the excerpts and to say so when they do not
contain the answer.

## Consequences

- One LLM call per question: predictable latency, cost and failure surface, and retrieval can be tested
  without calling the model.
- Questions that need several hops (follow a call chain across files) or running code will answer poorly.
  If that matters, an agent with search tools earns its cost; this ADR would then be superseded.
