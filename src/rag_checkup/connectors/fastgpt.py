"""FastGPT 知识库连接器(REST 直调,不依赖 SDK).

FastGPT 的知识库问答走 chat completions 接口(detail=true 会回传引用).
文档拉取走 dataset 数据接口;连不上时给出明确报错,不静默吞掉.
"""
from __future__ import annotations

import httpx

from .base import KBConnector
from ..models import Document, RetrievedChunk


class FastGPTConnector(KBConnector):
    name = "fastgpt"

    def __init__(
        self,
        api_key: str,
        dataset_ids: list[str] | None = None,
        base_url: str = "https://api.fastgpt.in/api",
        timeout: float = 60.0,
    ):
        if not api_key:
            raise ValueError("FastGPT 需要 api_key(可用环境变量 RAGCHECKUP_FASTGPT_KEY 传入)")
        self.api_key = api_key
        self.dataset_ids = dataset_ids or []
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def list_documents(self, limit: int = 20) -> list[Document]:
        # FastGPT 的 dataset 数据接口按 collection 组织;这里做尽力拉取,
        # 拉不到就抛出明确的 dataset/key 权限错误.
        docs: list[Document] = []
        with httpx.Client(timeout=self.timeout) as client:
            for ds_id in self.dataset_ids[:5]:
                try:
                    r = client.get(
                        f"{self.base_url}/core/dataset/collection/list",
                        headers=self._headers(),
                        params={"parentId": ds_id, "pageNum": 1, "pageSize": 50},
                    )
                    r.raise_for_status()
                    for col in r.json().get("data", {}).get("list", []):
                        text = self._collection_text(client, col.get("datasetId"), col.get("_id"))
                        if text.strip():
                            docs.append(Document(id=col.get("_id", ""), title=col.get("name", ""), content=text))
                        if len(docs) >= limit:
                            return docs
                except httpx.HTTPError as e:
                    raise RuntimeError(f"FastGPT 拉取文档失败(dataset={ds_id}):{e}") from e
        if not docs:
            raise RuntimeError("FastGPT 未拉到任何文档内容,请检查 datasetId / key 权限.")
        return docs

    def _collection_text(self, client: httpx.Client, dataset_id: str, collection_id: str) -> str:
        try:
            r = client.post(
                f"{self.base_url}/core/dataset/data/list",
                headers=self._headers(),
                json={"collectionId": collection_id, "pageNum": 1, "pageSize": 50, "searchText": ""},
            )
            r.raise_for_status()
            items = r.json().get("data", {}).get("list", [])
            return "\n".join(f"{it.get('q', '')}\n{it.get('a', '')}".strip() for it in items)
        except httpx.HTTPError:
            return ""

    def query(self, question: str, top_k: int = 5) -> tuple[str, list[RetrievedChunk], list[str]]:
        body: dict = {
            "chatId": "rag-checkup",
            "stream": False,
            "detail": True,
            "messages": [{"role": "user", "content": question}],
        }
        if self.dataset_ids:
            body["datasetIds"] = self.dataset_ids
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/v1/chat/completions", headers=self._headers(), json=body)
            r.raise_for_status()
            payload = r.json()

        choice = (payload.get("choices") or [{}])[0]
        answer = (choice.get("message") or {}).get("content", "")
        chunks, citations = [], []
        for i, q in enumerate(payload.get("responseData", []) or [], start=1):
            for item in q.get("quoteList", []) or []:
                chunks.append(RetrievedChunk(
                    doc_id=str(item.get("id", "")),
                    title=item.get("sourceName", ""),
                    content=item.get("text", "") or item.get("q", ""),
                    score=float(item.get("score") or 0),
                ))
                citations.append(f"chunk-{i}")
                i += 1
        return answer, chunks, citations
