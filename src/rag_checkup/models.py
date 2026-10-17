"""核心数据模型."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """知识库中的一篇文档(或一个 chunk)."""

    id: str
    title: str
    content: str


@dataclass
class Question:
    """一道测试题."""

    text: str
    qtype: str  # factual | boundary | paraphrase
    source_doc_id: str = ""
    expected: str = ""  # 事实题的参考答案;边界题为空
    note: str = ""


@dataclass
class RetrievedChunk:
    doc_id: str
    title: str
    content: str
    score: float = 0.0


@dataclass
class QAResult:
    question: Question
    answer: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)  # 答案中声明引用的 chunk 标识


@dataclass
class ScoredCase:
    qa: QAResult
    scores: dict = field(default_factory=dict)  # correctness / citation / refusal -> 0-100
    reasons: dict = field(default_factory=dict)
    is_bad: bool = False  # 任一维度 < 60 即为 bad case


@dataclass
class CheckupResult:
    cases: list[ScoredCase] = field(default_factory=list)
    aggregates: dict = field(default_factory=dict)  # 各维度均分 + 总分
    suggestions: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
