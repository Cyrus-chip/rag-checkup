"""报告中英双语(P1)."""
from __future__ import annotations

STRINGS = {
    "zh": {
        "title": "RAG 健康体检报告",
        "total": "综合评分",
        "dims": {"correctness": "答案正确性", "citation": "引用准确性", "refusal": "拒答诚实度"},
        "bad_cases": "Bad Case 回放",
        "suggestions": "调参建议",
        "question": "问题", "answer": "模型答案", "chunks": "召回内容",
        "citations": "声明引用", "reason": "扣分原因", "copy": "复制",
        "meta": "体检信息", "connector": "知识库", "n_cases": "测试题数",
        "bad_rate": "Bad Case 占比", "judge": "裁判", "all_ok": "全部通过",
        "expand": "展开", "type_names": {"factual": "事实问答", "boundary": "边界拒答", "paraphrase": "改写鲁棒"},
    },
    "en": {
        "title": "RAG Health Checkup Report",
        "total": "Overall Score",
        "dims": {"correctness": "Correctness", "citation": "Citation Accuracy", "refusal": "Refusal Honesty"},
        "bad_cases": "Bad Case Replay",
        "suggestions": "Tuning Suggestions",
        "question": "Question", "answer": "Answer", "chunks": "Retrieved Chunks",
        "citations": "Citations", "reason": "Why Deducted", "copy": "Copy",
        "meta": "Checkup Info", "connector": "Knowledge Base", "n_cases": "# Questions",
        "bad_rate": "Bad Case Rate", "judge": "Judge", "all_ok": "All Passed",
        "expand": "Expand", "type_names": {"factual": "Factual", "boundary": "Boundary", "paraphrase": "Paraphrase"},
    },
}


def t(lang: str) -> dict:
    return STRINGS.get(lang, STRINGS["zh"])
