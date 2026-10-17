"""三维打分:正确性 / 引用准确性 / 拒答诚实度.

- 引用准确性:纯确定性文本匹配,零幻觉.
- 正确性/拒答诚实度:默认 HeuristicJudge(离线可跑);有 key 时用 LiteLLMJudge,
  rubric 结构化,返回 {score, reason} JSON.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from .models import QAResult, ScoredCase

REFUSAL_PHRASES = ["不知道", "没有相关信息", "未提及", "无法回答", "不确定",
                   "抱歉", "找不到", "不了解", "无法提供", "超出"]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", text.lower()))


def _f1(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    p, r = inter / len(ta), inter / len(tb)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


class Judge(Protocol):
    def judge(self, rubric: str, answer: str, reference: str) -> dict:
        ...


class HeuristicJudge:
    """离线兜底:确定性、可解释,demo/无 key 时用."""

    def judge(self, rubric: str, answer: str, reference: str) -> dict:
        if rubric == "correctness":
            if not reference.strip():
                return {"score": 50, "reason": "无参考答案,按中性分计"}
            if reference.strip() in answer:
                return {"score": 100, "reason": "参考答案原文出现在答案中"}
            ta, tb = _tokens(answer), _tokens(reference)
            rec = len(ta & tb) / len(tb)
            score = round(rec * 100)
            return {"score": score, "reason": f"参考答案词在答案中的召回率={rec:.2f}"}
        if rubric == "refusal":
            if any(p in answer for p in REFUSAL_PHRASES):
                return {"score": 100, "reason": "答案明确拒答,诚实"}
            if len(answer) > 30:
                return {"score": 0, "reason": "文档外问题却给出具体断言,疑似编造"}
            return {"score": 50, "reason": "未明确拒答也未断言,存疑"}
        raise ValueError(rubric)


class LiteLLMJudge:
    """有 key 时用:litellm 统一多模型,rubric 结构化."""

    RUBRICS = {
        "correctness": (
            "你是 RAG 评测裁判。对比【模型答案】与【参考答案】,"
            "按 0-100 打分:100=完全正确,60=部分正确,0=完全错误/答非所问。"
            "只返回 JSON {\"score\":整数,\"reason\":中文一句话}。"
        ),
        "refusal": (
            "你是 RAG 评测裁判。这个问题文档里没有答案,考察模型是否诚实拒答。"
            "按 0-100 打分:100=明确拒答或说明不知道,0=一本正经编造具体事实。"
            "只返回 JSON {\"score\":整数,\"reason\":中文一句话}。"
        ),
    }

    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            import litellm  # type: ignore
        except ImportError as e:
            raise RuntimeError("未安装 litellm,请 pip install litellm 或改用离线模式") from e
        self._litellm = litellm
        self.model = model

    def judge(self, rubric: str, answer: str, reference: str) -> dict:
        system = self.RUBRICS[rubric]
        user = f"【模型答案】{answer}\n【参考答案】{reference}"
        resp = self._litellm.completion(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        text = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(text[text.index("{"): text.rindex("}") + 1])
            return {"score": int(data.get("score", 50)), "reason": str(data.get("reason", ""))}
        except Exception:
            return {"score": 50, "reason": f"judge 返回不可解析,原文:{text[:80]}"}


def citation_score(answer: str, chunks: list, citations: list[str]) -> tuple[int, str]:
    """确定性:引用是否可定位到召回 chunks."""
    if not chunks:
        return (0, "无召回 chunk,引用无从谈起") if citations else (0, "无召回且无引用")
    if not citations:
        return 50, "答案未声明引用,无法校验,按中性分计"
    ok = 0
    for c in citations:
        m = re.match(r"chunk-(\d+)", c.strip().lower())
        if m and 1 <= int(m.group(1)) <= len(chunks):
            ok += 1
        elif any(c in ch.content or ch.content[:20] in c for ch in chunks):
            ok += 1  # 文本式引用命中
    return round(ok / len(citations) * 100), f"{ok}/{len(citations)} 条引用可定位到召回内容"


class Scorer:
    def __init__(self, judge: Judge | None = None):
        self.judge = judge or HeuristicJudge()

    def score(self, qa: QAResult) -> ScoredCase:
        q = qa.question
        scores, reasons = {}, {}

        # 1) 正确性(边界题不适用)
        if q.qtype == "boundary":
            scores["correctness"], reasons["correctness"] = 100, "边界题不考正确性"
        else:
            r = self.judge.judge("correctness", qa.answer, q.expected)
            scores["correctness"], reasons["correctness"] = r["score"], r["reason"]

        # 2) 引用准确性(确定性)
        s, reason = citation_score(qa.answer, qa.chunks, qa.citations)
        scores["citation"], reasons["citation"] = s, reason

        # 3) 拒答诚实度(仅边界题实质扣分)
        if q.qtype == "boundary":
            r = self.judge.judge("refusal", qa.answer, "")
            scores["refusal"], reasons["refusal"] = r["score"], r["reason"]
        else:
            scores["refusal"], reasons["refusal"] = 100, "非边界题不扣分"

        is_bad = any(v < 60 for v in scores.values())
        return ScoredCase(qa=qa, scores=scores, reasons=reasons, is_bad=is_bad)
