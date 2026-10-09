# 0005. Evaluate retrieval with hit-rate@k and MRR

- Status: accepted
- Date: 2026-10-09

## Context

Retrieval quality was unmeasured (ADR 0003 says so). Chunking, model and index changes need a number to
compare, not an impression. Answer quality depends on the LLM, costs money on every run and is not
deterministic.

## Decision

Measure `retrieve` only. Each case is a question plus the expected `(path, function/class/method name)`;
a chunk is a hit when both match. Report hit@1, hit@3, hit@5 and MRR@10 (reciprocal rank, 0 beyond
depth 10), overall and per tag (`direct`, `paraphrase`, `method`, `class`; tags overlap).
The eval needs no API key and is deterministic for a fixed model. Metrics code is pure and unit-tested
with a fake embedder; the run with the real model is a manual command, not part of CI (model download,
time).

The dataset is 33 questions over two fixed targets: CodeSage's own source at `d8633d5` and
`humanize` at `785e5dc`, both fetched at the pinned SHA by `scripts/fetch_eval_repos.sh`, never vendored.
I wrote the questions from reading the source before running anything, and kept the failures in.

## Consequences

- Retrieval changes become comparable, and the incremental index (ADR 0006) can be checked against a
  full rebuild by scores.
- Small dataset: one case is about 3 points of hit-rate, so differences of a few points are noise. No
  confidence intervals are claimed. Two repos only, both small and in Python.
- Author-written questions share the author's vocabulary, which favours the model; the paraphrase tag
  is a partial counterweight.
- Name-level matching ignores whether the retrieved chunk is enough to answer, and cannot tell
  same-named methods in one file apart (the chunker does not qualify methods with their class). The
  dataset avoids such cases.
- It says nothing about answer quality, hallucination or prompt quality; that needs a separate,
  paid, non-deterministic evaluation.
