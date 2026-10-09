"""Retrieval evaluation: hit-rate@k and MRR, computed over `retrieve` only.

No LLM call is involved, so a run is free, deterministic and needs no API key. The query
embedder is injected so unit tests can use a fake and never load a model.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from codesage.models import CodeChunk, RetrievedChunk
from codesage.qa import retrieve
from codesage.vector_store import VectorStore

HIT_KS = (1, 3, 5)
# MRR is truncated at this depth (reported as MRR@10): a rank beyond it counts as a miss.
DEFAULT_DEPTH = 10
_TOP_SHOWN = 3

QueryEmbedder = Callable[[str], Awaitable[list[float]]]


class EvalCase(BaseModel):
    """One question and the chunk that should answer it.

    `name` is the bare function/class/method name, matching `CodeChunk.name`: the chunker does
    not qualify methods with their class, so two same-named methods in one file are
    indistinguishable (see ADR 0005).
    """

    model_config = ConfigDict(frozen=True)

    question: str
    path: str
    name: str
    tags: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    case: EvalCase
    rank: int | None
    top: list[str]


class Metrics(BaseModel):
    n: int
    hit_at: dict[int, float]
    mrr: float


class EvalReport(BaseModel):
    depth: int
    overall: Metrics
    by_tag: dict[str, Metrics]
    results: list[CaseResult]


def load_cases(path: Path) -> list[EvalCase]:
    """Reads a JSONL eval file. Errors name the file and line so a dataset typo is easy to find."""
    cases: list[EvalCase] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(EvalCase.model_validate_json(line))
        except ValidationError as err:
            first = err.errors()[0]
            where = ".".join(str(part) for part in first["loc"]) or "line"
            raise ValueError(
                f"{path}:{number}: invalid eval case ({where}: {first['msg']})"
            ) from err
    if not cases:
        raise ValueError(f"{path}: no eval cases found")
    return cases


def rank_of(case: EvalCase, retrieved: list[RetrievedChunk]) -> int | None:
    """1-based position of the first chunk matching the case's path and name, else None."""
    for position, item in enumerate(retrieved, start=1):
        if item.chunk.path == case.path and item.chunk.name == case.name:
            return position
    return None


def metrics(results: list[CaseResult]) -> Metrics:
    n = len(results)
    if n == 0:
        return Metrics(n=0, hit_at={k: 0.0 for k in HIT_KS}, mrr=0.0)
    hit_at = {k: sum(1 for r in results if r.rank is not None and r.rank <= k) / n for k in HIT_KS}
    mrr = sum(1 / r.rank for r in results if r.rank is not None) / n
    return Metrics(n=n, hit_at=hit_at, mrr=mrr)


def summarize(results: list[CaseResult], *, depth: int) -> EvalReport:
    tags = sorted({tag for r in results for tag in r.case.tags})
    return EvalReport(
        depth=depth,
        overall=metrics(results),
        by_tag={tag: metrics([r for r in results if tag in r.case.tags]) for tag in tags},
        results=results,
    )


def missing_targets(cases: list[EvalCase], chunks: list[CodeChunk]) -> list[EvalCase]:
    """Cases whose expected (path, name) is not an indexed chunk: a dataset or index mismatch."""
    known = {(c.path, c.name) for c in chunks}
    return [case for case in cases if (case.path, case.name) not in known]


async def evaluate(
    cases: list[EvalCase],
    store: VectorStore,
    embed_query: QueryEmbedder,
    *,
    depth: int = DEFAULT_DEPTH,
) -> EvalReport:
    """Runs every case through `retrieve` and summarizes ranks."""
    if depth < max(HIT_KS):
        raise ValueError(f"depth must be at least {max(HIT_KS)} (largest k), got {depth}")
    missing = missing_targets(cases, store.chunks)
    if missing:
        # Without this a typo in the dataset would silently score as a retrieval miss.
        listed = "; ".join(f"{c.path}:{c.name}" for c in missing[:5])
        raise ValueError(f"{len(missing)} expected target(s) not in the index: {listed}")
    results: list[CaseResult] = []
    for case in cases:
        retrieved = retrieve(store, await embed_query(case.question), top_k=depth)
        results.append(
            CaseResult(
                case=case,
                rank=rank_of(case, retrieved),
                top=[f"{r.chunk.path}:{r.chunk.name}" for r in retrieved[:_TOP_SHOWN]],
            )
        )
    return summarize(results, depth=depth)


def _row(label: str, m: Metrics, depth: int) -> str:
    hits = "  ".join(f"hit@{k}={m.hit_at[k]:.3f}" for k in HIT_KS)
    return f"{label:<12} n={m.n:<3} {hits}  MRR@{depth}={m.mrr:.3f}"


def format_report(report: EvalReport, *, show_misses: int = 5) -> str:
    lines = [_row("overall", report.overall, report.depth)]
    lines += [_row(tag, m, report.depth) for tag, m in report.by_tag.items()]
    misses = sorted(
        (r for r in report.results if r.rank != 1),
        key=lambda r: (r.rank is None, r.rank or 0),
        reverse=True,
    )[:show_misses]
    if misses:
        lines.append("")
        lines.append(f"Worst {len(misses)} miss(es):")
        for r in misses:
            got = f"not in top {report.depth}" if r.rank is None else f"rank {r.rank}"
            lines.append(f"  [{got}] {r.case.question}")
            lines.append(f"    expected {r.case.path}:{r.case.name}; top: {', '.join(r.top)}")
    return "\n".join(lines)
