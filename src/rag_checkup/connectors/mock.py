"""Mock 知识库连接器:零 key 跑通全流程,演示各种 bad case.

内置虚构知识库"星云笔记本 NebulaBook",行为全部确定、可复现:
- 事实/改写题 -> 句子级检索 + 抽取式正确答案(好 case)
- 边界题(含特定关键词) -> 一本正经编造答案(拒答诚实度 bad case)
- "电池续航"问题 -> 答案正确但引用了不存在的 chunk(引用准确性 bad case)
- "以旧换新"问题 -> 召回为空(检索缺失 bad case,答案诚实拒答)
"""
from __future__ import annotations

import re

from .base import KBConnector
from ..models import Document, RetrievedChunk

DOCS: list[Document] = [
    Document(
        id="doc-1",
        title="星云笔记本 Pro 配置说明",
        content=(
            "星云笔记本 Pro 搭载 16GB 内存与 1TB 固态硬盘,重量仅 1.2kg。"
            "电池续航为 18 小时,支持 100W 快充,充电 30 分钟可用 8 小时。"
            "屏幕为 14 英寸 2.8K OLED,刷新率 120Hz。"
        ),
    ),
    Document(
        id="doc-2",
        title="星云笔记本售后政策",
        content=(
            "星云笔记本支持 7 天无理由退货,15 天只换不修。"
            "整机保修 1 年,电池与充电器保修 2 年。"
            "人为损坏不在保修范围内,维修需提供购买凭证。"
        ),
    ),
    Document(
        id="doc-3",
        title="星云笔记本快捷键手册",
        content=(
            "Ctrl+Shift+S 为区域截图,Fn+F5 切换静音。"
            "长按电源键 10 秒可强制关机。"
            "Fn+空格 切换键盘背光三档亮度。"
        ),
    ),
]

BOUNDARY_KEYS = ["火星", "创始人", "比特币", "昨天晚饭", "彩票", "外星人"]

FABRICATED = "支持的,星云笔记本全系支持火星快递,下单后 3 个工作日内由轨道货运送达,运费 2999 元。"
NO_RECALL = "抱歉,没有找到相关内容。"


def _shingles(text: str, n: int = 2) -> list[str]:
    t = re.sub(r"\s+", "", text)
    return [t[i:i + n] for i in range(len(t) - n + 1)]


class MockConnector(KBConnector):
    name = "mock"

    def __init__(self):
        self.docs = DOCS
        self._sentences: list[tuple[Document, str]] = []
        for d in DOCS:
            for s in re.split(r"[。；！？，,]", d.content):
                s = s.strip()
                if len(s) >= 6:
                    self._sentences.append((d, s))

    def list_documents(self, limit: int = 20) -> list[Document]:
        return self.docs[:limit]

    def query(self, question: str, top_k: int = 5) -> tuple[str, list[RetrievedChunk], list[str]]:
        # 1) 边界题:一本正经编造 -> 拒答诚实度 bad case
        if any(k in question for k in BOUNDARY_KEYS):
            d, s = self._sentences[0]
            ch = RetrievedChunk(doc_id=d.id, title=d.title, content=s, score=0.2)
            return FABRICATED, [ch], ["chunk-1"]
        # 2) 检索缺失:召回为空,答案诚实拒答 -> 引用维度 bad case
        if "以旧换新" in question:
            return NO_RECALL, [], []
        # 3) 句子级检索:问题 shingle 在句子中的命中数打分
        qsh = set(_shingles(question))
        scored = sorted(
            ((sum(1 for sh in qsh if sh in s), d, s) for d, s in self._sentences),
            key=lambda x: -x[0],
        )
        top = [(d, s, h) for h, d, s in scored[:top_k] if h > 0]
        if not top:
            return NO_RECALL, [], []
        chunks = [RetrievedChunk(doc_id=d.id, title=d.title, content=s, score=float(h))
                  for d, s, h in top]
        # 4) 电池续航:答案正确但引用造假 -> 引用准确性 bad case
        if "电池续航" in question:
            return f"根据文档:{chunks[0].content}。", chunks, ["chunk-9"]
        # 5) 默认:抽取式好答案
        return f"根据文档:{chunks[0].content}。", chunks, ["chunk-1"]
