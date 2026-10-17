"""三类测试题自动生成:事实问答 / 边界拒答 / 改写鲁棒.

默认离线模板(零 key 可用);传入 llm 可调用(可调用对象 prompt->str)
则用 LLM 出题,失败自动回退到模板.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from .models import Document, Question

BOUNDARY_TEMPLATES = [
    "星云笔记本支持火星快递吗？",
    "星云笔记本创始人昨天晚饭吃了什么？",
    "可以用星云笔记本挖比特币吗？",
    "星云笔记本支持以旧换新吗？",
]

PARAPHRASE_MAP = [
    ("为多少？", "是多少？"),
    ("是什么？", "指的是什么？"),
    ("有哪些？", "包括哪些？"),
]


def make_litellm_callback(
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    litellm_module: Any | None = None,
) -> Callable[[str], str]:
    """Build the prompt -> text callback used for LLM question generation.

    ``api_key`` is passed explicitly to LiteLLM instead of relying on a
    provider-specific environment variable.  ``litellm_module`` is an
    optional injection point for offline tests; normal callers leave it unset.
    """
    if litellm_module is None:
        try:
            import litellm  # type: ignore
        except ImportError as e:
            raise RuntimeError("未安装 litellm,请 pip install 'rag-checkup[llm]' 或改用离线模式") from e
        litellm_module = litellm

    def complete(prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        if api_key:
            kwargs["api_key"] = api_key
        response = litellm_module.completion(**kwargs)
        message = response.choices[0].message
        return message.content or ""

    return complete


def _sentences(text: str) -> list[str]:
    parts = re.split(r"[。；！？]", text)
    return [p.strip() for p in parts if len(p.strip()) >= 8]


def _factual_from_sentence(sentence: str, doc: Document) -> Question | None:
    s = sentence.strip()
    clause = re.split(r"[，,]", s)[0].strip()  # 只看第一分句,答案才够短
    patterns = [
        (r"(.+?)为(.+)", lambda a, b: (f"{a}为多少？", b)),
        (r"(.+?)是(.+)", lambda a, b: (f"{a}是什么？", b)),
        (r"(.+?)搭载(.+)", lambda a, b: (f"{a}搭载了什么？", b)),
        (r"(.+?)支持(.+)", lambda a, b: (f"{a}支持什么？", b)),
    ]
    for pat, make in patterns:
        m = re.match(pat, clause)
        if not m:
            continue
        a, b = m.group(1).strip(), m.group(2).strip()
        # "人为/认为"里的"为"不是系动词(a 会被切成"人"),整句跳过该模式
        if pat.startswith("(.+?)为") and (clause.startswith("人为") or clause.startswith("认为")):
            continue
        if 1 <= len(a) <= 25 and 2 <= len(b) <= 25:
            text, expected = make(a, b)
            return Question(text=text, qtype="factual",
                            source_doc_id=doc.id, expected=expected, note=f"原文:{s}")
    return Question(text=f"请简述：{clause}", qtype="factual",
                    source_doc_id=doc.id, expected=clause, note="兜底题")


def _paraphrase(q: Question) -> Question | None:
    for src, dst in PARAPHRASE_MAP:
        if q.text.endswith(src):
            return Question(text=q.text[: -len(src)] + dst, qtype="paraphrase",
                            source_doc_id=q.source_doc_id, expected=q.expected,
                            note=f"改写自:{q.text}")
    return None


class QuestionGenerator:
    def __init__(self, llm: Callable[[str], str] | None = None):
        self.llm = llm

    def generate(self, docs: list[Document], n_per_doc: int = 4) -> list[Question]:
        if isinstance(n_per_doc, bool) or not isinstance(n_per_doc, int) or n_per_doc <= 0:
            raise ValueError("n_per_doc must be a positive integer")
        questions: list[Question] = []
        for doc in docs:
            if self.llm:
                try:
                    questions.extend(self._llm_generate(doc, n_per_doc))
                    continue
                except Exception:
                    pass  # 回退模板
            questions.extend(self._template_generate(doc, n_per_doc))
        # 边界题:全部加入,覆盖"编造"与"检索缺失"两种 bad case 形态
        for t in BOUNDARY_TEMPLATES:
            questions.append(Question(text=t, qtype="boundary", expected="", note="文档外问题,应拒答"))
        # 去重
        seen, uniq = set(), []
        for q in questions:
            if q.text not in seen:
                seen.add(q.text)
                uniq.append(q)
        return uniq

    def _template_generate(self, doc: Document, n_per_doc: int) -> list[Question]:
        factual: list[Question] = []
        for s in _sentences(doc.content):
            q = _factual_from_sentence(s, doc)
            if q and q.text not in {x.text for x in factual}:
                factual.append(q)
            if len(factual) >= n_per_doc:
                break
        out = list(factual)
        for q in factual[:2]:
            p = _paraphrase(q)
            if p:
                out.append(p)
        return out

    def _llm_generate(self, doc: Document, n: int) -> list[Question]:
        prompt = (
            "你是 RAG 评测出题人。根据下面这篇文档出 {n} 道测试题,"
            "题型 distribution:事实问答/边界拒答/改写鲁棒大致均分。"
            "只返回 JSON 数组,每项 {{\"text\":题面,\"qtype\":factual|boundary|paraphrase,"
            "\"expected\":参考答案(边界题为空字符串)}}。\n文档标题:{t}\n文档内容:{c}"
        ).format(n=n, t=doc.title, c=doc.content[:2000])
        raw = self.llm(prompt)
        items = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
        return [Question(text=i["text"], qtype=i.get("qtype", "factual"),
                         source_doc_id=doc.id, expected=i.get("expected", "")) for i in items]

    def to_csv(self, questions: list[Question], path: str) -> None:
        import csv
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["text", "qtype", "source_doc_id", "expected", "note"])
            for q in questions:
                w.writerow([q.text, q.qtype, q.source_doc_id, q.expected, q.note])
