"""连接器抽象：知识库怎么连，体检就怎么测."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Document, RetrievedChunk


class KBConnector(ABC):
    """所有知识库连接器都要实现这两个动作."""

    name: str = "base"

    @abstractmethod
    def list_documents(self, limit: int = 20) -> list[Document]:
        """拉取文档(供出题用)."""

    @abstractmethod
    def query(self, question: str, top_k: int = 5) -> tuple[str, list[RetrievedChunk], list[str]]:
        """问一个问题,返回 (答案, 召回 chunks, 答案声明的引用).

        引用是答案里声称依据的 chunk 标识(如下标或文本片段),
        用于"引用准确性"维度的确定性校验.
        """
