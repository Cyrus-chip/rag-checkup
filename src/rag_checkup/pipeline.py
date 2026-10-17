"""体检流水线:出题 -> 提问 -> 打分 -> 汇总 -> 建议."""
from __future__ import annotations

from typing import Callable

from .connectors.base import KBConnector
from .models import CheckupResult, QAResult
from .question_gen import QuestionGenerator
from .scoring import Judge, Scorer
from .suggestions import build_suggestions


def run_checkup(
    connector: KBConnector,
    n_per_doc: int = 4,
    judge: Judge | None = None,
    top_k: int = 5,
    progress: Callable[[str], None] | None = None,
) -> CheckupResult:
    def log(msg: str):
        if progress:
            progress(msg)

    log("拉取知识库文档…")
    docs = connector.list_documents()
    if not docs:
        raise RuntimeError("知识库文档为空,无法出题")
    log(f"拉到 {len(docs)} 篇文档,正在出题…")
    questions = QuestionGenerator().generate(docs, n_per_doc=n_per_doc)
    log(f"生成 {len(questions)} 道测试题,逐题提问打分…")

    scorer = Scorer(judge)
    cases = []
    for i, q in enumerate(questions, 1):
        answer, chunks, citations = connector.query(q.text, top_k=top_k)
        cases.append(scorer.score(QAResult(question=q, answer=answer, chunks=chunks, citations=citations)))
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
        "top_k": top_k,
    }
    log(f"体检完成:总分 {aggregates['total']},bad case {aggregates['n_bad']}/{aggregates['n_cases']}")
    return CheckupResult(cases=cases, aggregates=aggregates, suggestions=suggestions, meta=meta)
