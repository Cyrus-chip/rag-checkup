"""体检流水线:出题 -> 提问 -> 打分 -> 汇总 -> 建议."""
from __future__ import annotations

from typing import Callable

from .connectors.base import KBConnector
from .models import CheckupResult, Document, QAResult, Question
from .question_gen import QuestionGenerator
from .scoring import Judge, Scorer
from .suggestions import build_suggestions


def _validate_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")


def run_checkup(
    connector: KBConnector,
    n_per_doc: int = 4,
    judge: Judge | None = None,
    top_k: int = 5,
    progress: Callable[[str], None] | None = None,
    question_generator: QuestionGenerator | Callable[[list[Document], int], list[Question]] | None = None,
    questions: list[Question] | None = None,
) -> CheckupResult:
    _validate_positive_integer("n_per_doc", n_per_doc)
    _validate_positive_integer("top_k", top_k)
    if questions is not None and not questions:
        raise RuntimeError("question set is empty; cannot run checkup")

    def log(msg: str):
        if progress:
            progress(msg)

    log("拉取知识库文档…")
    docs = connector.list_documents()
    if not docs:
        raise RuntimeError("知识库文档为空,无法出题")
    log(f"拉到 {len(docs)} 篇文档,正在出题…")
    if questions is None:
        generator = question_generator or QuestionGenerator()
        if hasattr(generator, "generate"):
            questions = generator.generate(docs, n_per_doc=n_per_doc)
        else:
            questions = generator(docs, n_per_doc)
    if not questions:
        raise RuntimeError("question set is empty; cannot run checkup")
    log(f"生成 {len(questions)} 道测试题,逐题提问打分…")

    scorer = Scorer(judge)
    cases = []
    connector_errors = 0
    for i, q in enumerate(questions, 1):
        query_error_type = ""
        try:
            answer, chunks, citations = connector.query(q.text, top_k=top_k)
            qa = QAResult(question=q, answer=answer, chunks=chunks, citations=citations)
        except Exception as exc:
            # Exception text can contain request URLs, headers, or provider
            # details.  Keep the report useful without copying any of it into
            # the result; the type is enough to diagnose the connector class.
            connector_errors += 1
            query_error_type = type(exc).__name__
            error = f"connector query failed for question {q.text!r}: {query_error_type}"
            qa = QAResult(question=q, answer="", status="connector_error", error=error)
        case = scorer.score(qa)
        cases.append(case)
        if case.status == "connector_error":
            log(f"  [{i}/{len(questions)}] connector error ({query_error_type})")
        else:
            log(f"  [{i}/{len(questions)}] {q.text[:24]}…")

    dims = ["correctness", "citation", "refusal"]
    aggregates = {d: round(sum(c.scores[d] for c in cases) / len(cases), 1) for d in dims}
    aggregates["total"] = round(sum(aggregates[d] for d in dims) / 3, 1)
    aggregates["bad_rate"] = round(sum(1 for c in cases if c.is_bad) / len(cases) * 100, 1)
    aggregates["n_cases"] = len(cases)
    aggregates["n_bad"] = sum(1 for c in cases if c.is_bad)

    suggestions = build_suggestions(cases)
    meta = {
        "connector": connector.name,
        "n_docs": len(docs),
        "judge": type(scorer.judge).__name__,
        "judge_fallbacks": scorer.judge_fallbacks,
        "judge_errors": scorer.judge_errors,
        "connector_errors": connector_errors,
        "top_k": top_k,
    }
    log(f"体检完成:总分 {aggregates['total']},bad case {aggregates['n_bad']}/{aggregates['n_cases']}")
    return CheckupResult(cases=cases, aggregates=aggregates, suggestions=suggestions, meta=meta)
