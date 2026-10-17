"""P1:A/B 对比——同一测试集跑两套配置,告诉你"改哪个参数涨了多少分"."""
from __future__ import annotations

from typing import Callable

from .connectors.base import KBConnector
from .models import CheckupResult
from .pipeline import run_checkup
from .scoring import Judge


def ab_compare(
    make_connector_a: Callable[[], KBConnector],
    make_connector_b: Callable[[], KBConnector],
    label_a: str = "A",
    label_b: str = "B",
    n_per_doc: int = 4,
    judge: Judge | None = None,
    top_k_a: int = 5,
    top_k_b: int = 5,
    progress: Callable[[str], None] | None = None,
) -> dict:
    ra: CheckupResult = run_checkup(make_connector_a(), n_per_doc=n_per_doc, judge=judge,
                                   top_k=top_k_a, progress=progress)
    rb: CheckupResult = run_checkup(make_connector_b(), n_per_doc=n_per_doc, judge=judge,
                                   top_k=top_k_b, progress=progress)
    dims = ["correctness", "citation", "refusal", "total"]
    rows = []
    for d in dims:
        delta = round(rb.aggregates[d] - ra.aggregates[d], 1)
        rows.append({"dim": d, "a": ra.aggregates[d], "b": rb.aggregates[d], "delta": delta})
    winner = label_b if rb.aggregates["total"] >= ra.aggregates["total"] else label_a
    return {"label_a": label_a, "label_b": label_b, "rows": rows,
            "winner": winner, "delta_total": round(rb.aggregates["total"] - ra.aggregates["total"], 1),
            "result_a": ra, "result_b": rb}
