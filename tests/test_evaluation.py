from __future__ import annotations

from pathlib import Path

import pytest
from fakes import CountingEmbedder

from codesage.evaluation import (
    CaseResult,
    EvalCase,
    evaluate,
    format_report,
    load_cases,
    metrics,
    missing_targets,
    rank_of,
    summarize,
)
from codesage.models import ChunkKind, CodeChunk, RetrievedChunk
from codesage.vector_store import VectorStore


def _chunk(path: str, name: str, content: str = "") -> CodeChunk:
    return CodeChunk(
        path=path,
        kind=ChunkKind.FUNCTION,
        name=name,
        start_line=1,
        end_line=2,
        content=content or f"def {name}(): pass",
    )


def _case(path: str = "a.py", name: str = "f", tags: list[str] | None = None) -> EvalCase:
    return EvalCase(question="q", path=path, name=name, tags=tags or [])


def _retrieved(*pairs: tuple[str, str]) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk=_chunk(p, n), score=0.5) for p, n in pairs]


def test_rank_of_is_one_based_and_requires_path_and_name() -> None:
    retrieved = _retrieved(("b.py", "f"), ("a.py", "g"), ("a.py", "f"))
    assert rank_of(_case("a.py", "f"), retrieved) == 3


def test_rank_of_returns_none_when_absent() -> None:
    assert rank_of(_case(), _retrieved(("b.py", "f"))) is None


def test_metrics_hit_at_k_and_mrr() -> None:
    results = [
        CaseResult(case=_case(), rank=1, top=[]),
        CaseResult(case=_case(), rank=4, top=[]),
        CaseResult(case=_case(), rank=None, top=[]),
        CaseResult(case=_case(), rank=3, top=[]),
    ]
    m = metrics(results)
    assert m.n == 4
    assert m.hit_at == {1: 0.25, 3: 0.5, 5: 0.75}
    assert m.mrr == pytest.approx((1 + 0.25 + 0 + 1 / 3) / 4)


def test_metrics_on_no_results_is_all_zero() -> None:
    m = metrics([])
    assert m.n == 0
    assert m.mrr == 0.0
    assert m.hit_at == {1: 0.0, 3: 0.0, 5: 0.0}


def test_summarize_breaks_down_by_tag() -> None:
    results = [
        CaseResult(case=_case(tags=["direct"]), rank=1, top=[]),
        CaseResult(case=_case(tags=["paraphrase", "method"]), rank=None, top=[]),
    ]
    report = summarize(results, depth=10)
    assert set(report.by_tag) == {"direct", "paraphrase", "method"}
    assert report.by_tag["direct"].hit_at[1] == 1.0
    assert report.by_tag["method"].hit_at[1] == 0.0
    assert report.overall.n == 2


def test_load_cases_skips_blank_lines(tmp_path: Path) -> None:
    f = tmp_path / "e.jsonl"
    f.write_text(
        '{"question": "q1", "path": "a.py", "name": "f", "tags": ["direct"]}\n\n'
        '{"question": "q2", "path": "b.py", "name": "g"}\n'
    )
    cases = load_cases(f)
    assert [c.name for c in cases] == ["f", "g"]
    assert cases[1].tags == []


def test_load_cases_error_names_file_and_line(tmp_path: Path) -> None:
    f = tmp_path / "e.jsonl"
    f.write_text('{"question": "q1", "path": "a.py", "name": "f"}\nnot json\n')
    with pytest.raises(ValueError, match=r"e\.jsonl:2"):
        load_cases(f)


def test_load_cases_rejects_missing_field(tmp_path: Path) -> None:
    f = tmp_path / "e.jsonl"
    f.write_text('{"question": "q1", "path": "a.py"}\n')
    with pytest.raises(ValueError, match=r"e\.jsonl:1"):
        load_cases(f)


def test_load_cases_rejects_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "e.jsonl"
    f.write_text("\n")
    with pytest.raises(ValueError, match="no eval cases"):
        load_cases(f)


def test_missing_targets_lists_cases_not_in_index() -> None:
    chunks = [_chunk("a.py", "f")]
    missing = missing_targets([_case("a.py", "f"), _case("a.py", "typo")], chunks)
    assert [c.name for c in missing] == ["typo"]


async def _store_with(embedder: CountingEmbedder, chunks: list[CodeChunk]) -> VectorStore:
    store = VectorStore()
    store.add(await embedder(chunks))
    return store


async def test_evaluate_ranks_the_matching_chunk_first() -> None:
    embedder = CountingEmbedder()
    store = await _store_with(
        embedder,
        [
            _chunk("a.py", "parse_config", "def parse_config(): read yaml settings file"),
            _chunk("b.py", "send_mail", "def send_mail(): smtp server message"),
        ],
    )
    cases = [
        EvalCase(question="read the yaml settings file", path="a.py", name="parse_config"),
        EvalCase(question="smtp message", path="b.py", name="send_mail"),
    ]
    report = await evaluate(cases, store, embedder.embed_query)
    assert report.overall.hit_at[1] == 1.0
    assert report.overall.mrr == 1.0


async def test_evaluate_fails_loudly_on_target_missing_from_index() -> None:
    embedder = CountingEmbedder()
    store = await _store_with(embedder, [_chunk("a.py", "f")])
    with pytest.raises(ValueError, match="not in the index"):
        await evaluate([_case("a.py", "typo")], store, embedder.embed_query)


async def test_evaluate_rejects_depth_below_largest_k() -> None:
    embedder = CountingEmbedder()
    store = await _store_with(embedder, [_chunk("a.py", "f")])
    with pytest.raises(ValueError, match="depth"):
        await evaluate([_case()], store, embedder.embed_query, depth=3)


def test_format_report_lists_worst_misses_first() -> None:
    results = [
        CaseResult(case=EvalCase(question="found", path="a.py", name="f"), rank=1, top=[]),
        CaseResult(case=EvalCase(question="late", path="a.py", name="g"), rank=4, top=["x.py:y"]),
        CaseResult(case=EvalCase(question="gone", path="a.py", name="h"), rank=None, top=[]),
    ]
    text = format_report(summarize(results, depth=10), show_misses=2)
    assert "hit@1" in text and "MRR@10" in text
    assert text.index("gone") < text.index("late")
    assert "found" not in text
