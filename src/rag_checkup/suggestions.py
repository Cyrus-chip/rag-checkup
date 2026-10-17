"""调参建议规则库 v1:从 bad case 模式到"改哪个参数".这是 KnowEval 没有的闭环."""
from __future__ import annotations

from .models import ScoredCase


def build_suggestions(cases: list[ScoredCase]) -> list[dict]:
    bad = [c for c in cases if c.is_bad]
    if not bad:
        return [{"title": "知识库状态健康", "detail": "全部测试题通过,继续保持。建议每周跑一次体检做回归。", "level": "ok"}]
    n = len(cases)
    out: list[dict] = []

    def ratio(pred) -> float:
        return sum(1 for c in bad if pred(c)) / n

    # 规则1:检索没召回
    no_recall = [c for c in bad if not c.qa.chunks]
    if no_recall:
        out.append({"title": "检索召回不足", "level": "warn",
                    "detail": f"{len(no_recall)} 道题完全没有召回 chunk。"
                              "建议:调大 top-k(如 5→10)、检查切分粒度是否过粗、确认 embedding 模型与文档语言匹配。"})
    # 规则2:引用对不上
    if ratio(lambda c: c.scores["citation"] < 60) >= 0.15:
        out.append({"title": "引用不可靠", "level": "warn",
                    "detail": f"{ratio(lambda c: c.scores['citation'] < 60):.0%} 的 case 引用分数低于 60。"
                              "建议:检查答案生成 prompt 是否要求显式引用 chunk 编号;切分时保留足够上下文,避免引用指向碎片。"})
    # 规则3:边界题编造
    if ratio(lambda c: c.qa.question.qtype == "boundary" and c.scores["refusal"] < 60) >= 0.15:
        out.append({"title": "模型在编造答案", "level": "bad",
                    "detail": f"{ratio(lambda c: c.qa.question.qtype == 'boundary' and c.scores['refusal'] < 60):.0%} "
                              "的文档外问题得到了编造的回答。建议:在系统提示词中加入"
                              "\"不知道就说不知道\"的拒答指令,并给拒答加上固定模板。"})
    # 规则4:改写题明显弱于事实题
    f = [c.scores["correctness"] for c in cases if c.qa.question.qtype == "factual"]
    p = [c.scores["correctness"] for c in cases if c.qa.question.qtype == "paraphrase"]
    if f and p and (sum(f) / len(f) - sum(p) / len(p)) >= 20:
        out.append({"title": "鲁棒性不足", "level": "warn",
                    "detail": "改写题得分比事实题低 20 分以上,说明检索过度依赖字面匹配。"
                              "建议:换用语义更强的 embedding、打开混合检索(关键词+向量)或加 query 改写。"})
    # 规则5:正确性整体偏低但引用尚可 -> 生成环节问题
    if sum(c.scores["correctness"] for c in cases) / n < 60 and \
       sum(c.scores["citation"] for c in cases) / n >= 60:
        out.append({"title": "生成环节拖后腿", "level": "warn",
                    "detail": "引用没问题但答案正确性低,问题在生成环节:可能 prompt 没约束好格式,"
                              "或模型能力不足。建议:收紧答案 prompt(要求只基于引用作答)、必要时升级模型。"})
    if not out:
        out.append({"title": "个别 bad case", "level": "info",
                    "detail": "bad case 分散,无明显系统性模式。建议逐条回放 bad case,人工确认是否为真问题。"})
    return out
