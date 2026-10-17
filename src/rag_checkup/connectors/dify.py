"""Dify 知识库连接器(REST 直调,不依赖 SDK).

两种模式:
- dataset 模式(默认):用 dataset-scoped API key,只读文档与检索.
  query() 走 /retrieve 接口,答案取 top-1 召回片段(抽取式),
  测的是"检索+引用"质量——这正是 RAG 最痛的一环.
- app 模式:额外传入 app_key,走 /chat-messages 做端到端问答.
"""
from __future__ import annotations

import httpx

from .base import KBConnector
from .http import request_with_retry
from ..models import Document, RetrievedChunk


class DifyConnector(KBConnector):
    name = "dify"

    def __init__(
        self,
        api_key: str,
        dataset_id: str,
        base_url: str = "https://api.dify.ai/v1",
        app_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_backoff: float = 0.1,
    ):
        if not api_key or not dataset_id:
            raise ValueError("Dify 需要 api_key 和 dataset_id(可用环境变量 RAGCHECKUP_DIFY_KEY 传入 key)")
        self.api_key = api_key
        self.dataset_id = dataset_id
        self.base_url = base_url.rstrip("/")
        self.app_key = app_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def _headers(self, key: str) -> dict:
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def list_documents(self, limit: int = 20) -> list[Document]:
        docs: list[Document] = []
        page, page_size = 1, min(limit, 100)
        with httpx.Client(timeout=self.timeout) as client:
            while len(docs) < limit:
                r = self._request(client, "GET",
                    f"{self.base_url}/datasets/{self.dataset_id}/documents",
                    headers=self._headers(self.api_key),
                    params={"page": page, "limit": page_size},
                )
                r.raise_for_status()
                data = r.json().get("data", [])
                if not data:
                    break
                for d in data:
                    segs = self._segments(client, d["id"])
                    content = "\n".join(s["content"] for s in segs if s.get("content"))
                    if content.strip():
                        docs.append(Document(id=d["id"], title=d.get("name", d["id"]), content=content))
                    if len(docs) >= limit:
                        break
                if not r.json().get("has_more"):
                    break
                page += 1
        return docs

    def _segments(self, client: httpx.Client, document_id: str) -> list[dict]:
        r = self._request(client, "GET",
            f"{self.base_url}/datasets/{self.dataset_id}/documents/{document_id}/segments",
            headers=self._headers(self.api_key),
        )
        r.raise_for_status()
        return r.json().get("data", [])

    def query(self, question: str, top_k: int = 5) -> tuple[str, list[RetrievedChunk], list[str]]:
        if self.app_key:
            return self._chat(question=question, top_k=top_k)

        with httpx.Client(timeout=self.timeout) as client:
            r = self._request(client, "POST",
                f"{self.base_url}/datasets/{self.dataset_id}/retrieve",
                headers=self._headers(self.api_key),
                json={
                    "query": question,
                    "retrieval_model": {
                        "search_method": "semantic_search",
                        "reranking_enable": False,
                        "top_k": top_k,
                        "score_threshold_enabled": False,
                    },
                },
            )
            r.raise_for_status()
            records = r.json().get("records", [])

        chunks = [
            RetrievedChunk(
                doc_id=rec.get("segment", {}).get("document_id", ""),
                title=rec.get("segment", {}).get("document", {}).get("name", ""),
                content=rec.get("segment", {}).get("content", ""),
                score=float(rec.get("score") or 0),
            )
            for rec in records
        ]

        # dataset 模式:抽取式答案,引用即 top-1
        answer = chunks[0].content if chunks else "未检索到相关内容。"
        citations = ["chunk-1"] if chunks else []
        return answer, chunks, citations

    def _request(self, client: httpx.Client, method: str, url: str, **kwargs) -> httpx.Response:
        return request_with_retry(
            client, method, url, max_retries=self.max_retries,
            retry_backoff=self.retry_backoff, **kwargs,
        )

    def _chat(self, question: str, top_k: int = 5) -> tuple[str, list[RetrievedChunk], list[str]]:
        with httpx.Client(timeout=self.timeout) as client:
            r = self._request(client, "POST",
                f"{self.base_url}/chat-messages",
                headers=self._headers(self.app_key or ""),
                json={"inputs": {}, "query": question, "response_mode": "blocking", "user": "rag-checkup"},
            )
            r.raise_for_status()
            payload = r.json()
        answer = payload.get("answer", "")
        chunks: list[RetrievedChunk] = []
        for res in payload.get("retriever_resources", []) or []:
            if not isinstance(res, dict):
                continue
            content = res.get("content") or res.get("text") or ""
            if not isinstance(content, str):
                content = str(content)
            try:
                score = float(res.get("score") or 0)
            except (TypeError, ValueError):
                score = 0.0
            chunks.append(RetrievedChunk(
                doc_id=str(res.get("document_id") or res.get("segment_id") or ""),
                title=str(res.get("document_name") or res.get("dataset_name") or ""),
                content=content,
                score=score,
            ))
        # The app's resources are the evidence for this answer.  Do not use a
        # second dataset retrieve response: its ordering can differ from the
        # app's citations and make an otherwise valid citation look wrong.
        return answer, chunks, [f"chunk-{i}" for i, chunk in enumerate(chunks, 1)
                                if chunk.content]
