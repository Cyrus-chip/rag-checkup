"""rag-checkup 测试:全部离线,不调任何真实 API/Key."""
import os
import sys
from types import SimpleNamespace

import pytest
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag_checkup.connectors.mock import MockConnector, DOCS
from rag_checkup.connectors.dify import DifyConnector
from rag_checkup.connectors.fastgpt import FastGPTConnector
from rag_checkup import cli
from rag_checkup.models import QAResult, Question, RetrievedChunk
from rag_checkup.pipeline import run_checkup
from rag_checkup.question_gen import QuestionGenerator, make_litellm_callback
from rag_checkup.report import render, write_report
from rag_checkup.scoring import LiteLLMJudge, Scorer, citation_score
from rag_checkup.suggestions import build_suggestions
from rag_checkup.ab_compare import ab_compare


def test_question_gen_three_types():
    gen = QuestionGenerator()
    qs = gen.generate(DOCS, n_per_doc=4)
    types = {q.qtype for q in qs}
    assert {"factual", "boundary", "paraphrase"} <= types
    assert len({q.text for q in qs}) == len(qs), "题目去重失败"
    factual = [q for q in qs if q.qtype == "factual"]
    assert all(q.expected for q in factual), "事实题必须有参考答案"


def test_question_gen_csv(tmp_path):
    gen = QuestionGenerator()
    qs = gen.generate(DOCS[:1], n_per_doc=2)
    p = str(tmp_path / "q.csv")
    gen.to_csv(qs, p)
    with open(p, encoding="utf-8-sig") as f:
        head = f.readline()
    assert "text" in head and "qtype" in head


def test_question_gen_fake_llm_callback_is_offline():
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return '[{"text":"电池续航是多少？","qtype":"factual","expected":"18 小时"}]'

    questions = QuestionGenerator(llm=fake_llm).generate(DOCS[:1], n_per_doc=2)

    assert len(prompts) == 1
    assert questions[0].text == "电池续航是多少？"
    assert questions[0].expected == "18 小时"


def test_litellm_components_receive_explicit_key_without_network():
    class FakeLiteLLM:
        def __init__(self, content):
            self.content = content
            self.calls = []

        def completion(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
            )

    generator_llm = FakeLiteLLM(
        '[{"text":"测试题？","qtype":"factual","expected":"答案"}]'
    )
    callback = make_litellm_callback(
        model="fake-model", api_key="test-key", litellm_module=generator_llm
    )
    assert callback("出一道题")
    assert generator_llm.calls[0]["api_key"] == "test-key"

    judge_llm = FakeLiteLLM('{"score": 77, "reason": "mock"}')
    judged = LiteLLMJudge(
        model="fake-model", api_key="test-key", litellm_module=judge_llm
    ).judge("correctness", "答案", "参考")
    assert judged == {"score": 77, "reason": "mock"}
    assert judge_llm.calls[0]["api_key"] == "test-key"


def test_cli_model_is_passed_to_question_generator_and_judge(monkeypatch):
    calls = {}

    def fake_callback(*, model, api_key):
        calls["generator_model"] = model
        calls["generator_key"] = api_key
        return lambda prompt: ""

    class FakeJudge:
        def __init__(self, *, model, api_key):
            calls["judge_model"] = model
            calls["judge_key"] = api_key

    monkeypatch.setenv("RAGCHECKUP_LLM_KEY", "test-key")
    monkeypatch.setattr(cli, "make_litellm_callback", fake_callback)
    monkeypatch.setattr(cli, "LiteLLMJudge", FakeJudge)

    generator, judge = cli._llm_components(True, model="provider/custom-model")

    assert generator.llm is not None
    assert isinstance(judge, FakeJudge)
    assert calls == {
        "generator_model": "provider/custom-model",
        "generator_key": "test-key",
        "judge_model": "provider/custom-model",
        "judge_key": "test-key",
    }


def test_litellm_judge_runtime_failure_uses_heuristic_fallback():
    class FailingLiteLLM:
        def completion(self, **kwargs):
            raise RuntimeError("offline")

    judge = LiteLLMJudge(litellm_module=FailingLiteLLM())
    judged = judge.judge(
        "correctness", "答案包含 18 小时", "18 小时"
    )

    assert judged["score"] == 100
    assert "参考答案" in judged["reason"]
    assert "fallback" in judged["reason"]
    assert judge.fallback_count == 1
    assert judge.error_count == 1


def test_litellm_judge_out_of_range_score_falls_back_and_stays_in_range():
    class OutOfRangeLiteLLM:
        def completion(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content='{"score": 1000, "reason": "bad"}')
                )]
            )

    judge = LiteLLMJudge(litellm_module=OutOfRangeLiteLLM())
    judged = judge.judge("correctness", "答案包含 18 小时", "18 小时")

    assert 0 <= judged["score"] <= 100
    assert judged["score"] == 100  # heuristic fallback, not the invalid 1000
    assert "error/fallback" in judged["reason"]
    assert judge.fallback_count == 1
    assert judge.error_count == 1


def test_pipeline_records_judge_fallbacks_and_report_exposes_them():
    class OutOfRangeLiteLLM:
        def completion(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content='{"score": 1000, "reason": "bad"}')
                )]
            )

    result = run_checkup(
        MockConnector(), n_per_doc=1,
        judge=LiteLLMJudge(litellm_module=OutOfRangeLiteLLM()),
    )

    assert result.meta["judge_fallbacks"] > 0
    assert result.meta["judge_errors"] == result.meta["judge_fallbacks"]
    assert all(0 <= score <= 100 for case in result.cases for score in case.scores.values())
    report = render(result)
    assert "裁判回退次数" in report
    assert str(result.meta["judge_fallbacks"]) in report


def test_pipeline_records_provider_failures_in_meta():
    class FailingLiteLLM:
        def completion(self, **kwargs):
            raise RuntimeError("offline provider")

    result = run_checkup(
        MockConnector(), n_per_doc=1,
        judge=LiteLLMJudge(litellm_module=FailingLiteLLM()),
    )

    assert result.meta["judge_fallbacks"] > 0
    assert result.meta["judge_errors"] > 0


def test_mock_behaviors():
    m = MockConnector()
    assert len(m.list_documents()) == 3
    # 边界题编造
    ans, chunks, cites = m.query("星云笔记本支持火星快递吗？")
    assert "火星" in ans and len(ans) > 30
    # 电池续航题引用造假
    ans, chunks, cites = m.query("电池续航为多少？")
    assert cites == ["chunk-9"] and chunks
    # 以旧换新:召回为空,答案诚实拒答
    ans, chunks, cites = m.query("星云笔记本支持以旧换新吗？")
    assert chunks == [] and cites == [] and "抱歉" in ans
    # 普通事实题:句子级检索命中正确文档
    ans, chunks, cites = m.query("整机保修为多少？")
    assert chunks and cites == ["chunk-1"]
    assert "保修" in chunks[0].content
    # 快捷键题命中 doc-3
    ans, chunks, cites = m.query("Ctrl+Shift+S为多少？")
    assert chunks and chunks[0].doc_id == "doc-3"


def test_citation_score_deterministic():
    chunks = [RetrievedChunk("d1", "t", "电池续航为 18 小时", 0.9)]
    s, _ = citation_score("答案", chunks, ["chunk-1"])
    assert s == 100
    s, _ = citation_score("答案", chunks, ["chunk-9"])
    assert s == 0
    s, _ = citation_score("答案", chunks, [])
    assert s == 50
    s, _ = citation_score("答案", [], ["chunk-1"])
    assert s == 0


def test_scorer_bad_cases():
    scorer = Scorer()
    # 拒答 bad case
    qa = QAResult(Question("火星快递?", "boundary"), "支持的,星云笔记本全系支持火星快递,3 个工作日送达,运费 2999 元。", [], ["chunk-1"])
    c = scorer.score(qa)
    assert c.scores["refusal"] == 0 and c.is_bad
    # 好 case
    qa2 = QAResult(
        Question("电池续航为多少?", "factual", expected="18 小时"),
        "电池续航为 18 小时",
        [RetrievedChunk("d1", "t", "电池续航为 18 小时", 0.9)],
        ["chunk-1"],
    )
    c2 = scorer.score(qa2)
    assert c2.scores["citation"] == 100 and not c2.is_bad


def test_pipeline_end_to_end():
    result = run_checkup(MockConnector(), n_per_doc=3)
    agg = result.aggregates
    assert agg["n_cases"] > 0
    assert 0 <= agg["total"] <= 100
    assert agg["n_bad"] >= 2, "mock 应至少检出边界编造与引用造假两个 bad case"
    assert result.suggestions, "应有调参建议"


@pytest.mark.parametrize("kwargs", [{"n_per_doc": 0}, {"n_per_doc": -1}, {"top_k": 0}, {"top_k": -1}])
def test_pipeline_rejects_non_positive_sampling_options(kwargs):
    with pytest.raises(RuntimeError, match="positive integer"):
        run_checkup(MockConnector(), **kwargs)


def test_pipeline_rejects_empty_question_set():
    with pytest.raises(RuntimeError, match="question set is empty"):
        run_checkup(MockConnector(), questions=[])


def test_suggestions_rules():
    result = run_checkup(MockConnector(), n_per_doc=3)
    titles = [s["title"] for s in result.suggestions]
    assert any("编造" in t for t in titles), "应触发'模型在编造答案'规则"


def test_report_renders(tmp_path):
    result = run_checkup(MockConnector(), n_per_doc=2)
    html = render(result, lang="zh")
    assert "RAG 健康体检报告" in html
    assert "<svg" in html, "应内联 SVG 雷达图"
    assert "Bad Case" in html
    p = str(tmp_path / "r.html")
    write_report(result, p, lang="en")
    with open(p, encoding="utf-8") as f:
        en = f.read()
    assert "RAG Health Checkup Report" in en


def test_write_report_creates_missing_parent_directories(tmp_path):
    result = run_checkup(MockConnector(), n_per_doc=1)
    p = tmp_path / "nested" / "reports" / "checkup.html"
    write_report(result, p)
    assert p.is_file()


def test_fastgpt_citation_ids_are_global_across_quote_groups(monkeypatch):
    payload = {
        "choices": [{"message": {"content": "答案"}}],
        "responseData": [
            {"quoteList": [{"id": "a", "text": "第一段", "score": 0.9},
                           {"id": "b", "text": "第二段", "score": 0.8}]},
            {"quoteList": [{"id": "c", "text": "第三段", "score": 0.7}]},
        ],
    }

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("rag_checkup.connectors.fastgpt.httpx.Client", FakeClient)
    connector = FastGPTConnector(api_key="test-key")
    _, chunks, citations = connector.query("问题")

    assert len(chunks) == 3
    assert citations == ["chunk-1", "chunk-2", "chunk-3"]
    assert len(citations) == len(set(citations))


def test_ab_compare():
    cmp = ab_compare(MockConnector, MockConnector, label_a="k=3", label_b="k=10",
                     n_per_doc=2, top_k_a=1, top_k_b=5)
    assert "rows" in cmp and cmp["winner"] in ("k=3", "k=10", "tie")
    dims = {r["dim"] for r in cmp["rows"]}
    assert {"correctness", "citation", "refusal", "total"} <= dims


def test_ab_compare_reuses_one_generated_question_set():
    asked_a, asked_b = [], []

    class RecordingConnector(MockConnector):
        def __init__(self, asked):
            super().__init__()
            self.asked = asked

        def query(self, question, top_k=5):
            self.asked.append(question)
            return super().query(question, top_k=top_k)

    class CountingGenerator:
        calls = 0

        def generate(self, docs, n_per_doc=4):
            self.calls += 1
            return [Question("整机保修为多少？", "factual", expected="1 年")]

    generator = CountingGenerator()
    cmp = ab_compare(
        lambda: RecordingConnector(asked_a),
        lambda: RecordingConnector(asked_b),
        n_per_doc=1,
        question_generator=generator,
    )

    assert generator.calls == 1
    assert asked_a == asked_b == ["整机保修为多少？"]
    assert cmp["result_a"].cases[0].qa.question is cmp["result_b"].cases[0].qa.question


def test_no_real_keys_required():
    # demo 链路全程不读 key
    for var in ("RAGCHECKUP_DIFY_KEY", "RAGCHECKUP_FASTGPT_KEY", "OPENAI_API_KEY"):
        os.environ.pop(var, None)
    result = run_checkup(MockConnector(), n_per_doc=2)
    assert result.aggregates["n_cases"] > 0


def test_connector_query_failure_is_one_bad_case_and_does_not_abort_report():
    class PartiallyFailingConnector(MockConnector):
        name = "partial"

        def query(self, question, top_k=5):
            if question == "会失败的问题":
                raise RuntimeError("Authorization: Bearer api-key-must-not-appear")
            return super().query(question, top_k=top_k)

    questions = [
        Question("会失败的问题", "factual", expected="答案"),
        Question("整机保修为多少？", "factual", expected="1 年"),
    ]
    result = run_checkup(PartiallyFailingConnector(), questions=questions)

    assert len(result.cases) == 2
    failed, passed = result.cases
    assert failed.status == failed.qa.status == "connector_error"
    assert failed.error == failed.qa.error
    assert "会失败的问题" in failed.error
    assert "RuntimeError" in failed.error
    assert "api-key-must-not-appear" not in failed.error
    assert passed.status == "ok"
    assert result.meta["connector_errors"] == 1
    assert any(s["title"] == "连接器请求失败" for s in result.suggestions)
    report_zh = render(result, lang="zh")
    report_en = render(result, lang="en")
    assert "连接器错误次数" in report_zh
    assert "Connector errors" in report_en
    assert "会失败的问题" in report_zh and "RuntimeError" in report_zh
    assert "api-key-must-not-appear" not in report_zh


class _ContractClient:
    def __init__(self, routes, calls):
        self.routes = routes
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def _response(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        result = self.routes(method, url, kwargs)
        if isinstance(result, list):
            result = result.pop(0)
        if isinstance(result, Exception):
            raise result
        status, payload = result
        return httpx.Response(
            status,
            json=payload,
            request=httpx.Request(method, url),
        )

    def get(self, url, **kwargs):
        return self._response("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._response("POST", url, **kwargs)


def test_dify_dataset_http_contract_maps_documents_segments_and_retrieve(monkeypatch):
    calls = []

    def routes(method, url, kwargs):
        if url.endswith("/documents"):
            assert method == "GET"
            assert kwargs["params"] == {"page": 1, "limit": 20}
            return 200, {"data": [{"id": "doc-1", "name": "产品文档"}], "has_more": False}
        if url.endswith("/documents/doc-1/segments"):
            return 200, {"data": [{"content": "答案段落"}]}
        if url.endswith("/retrieve"):
            assert kwargs["json"]["query"] == "产品问题"
            assert kwargs["json"]["retrieval_model"]["top_k"] == 3
            return 200, {"records": [{
                "score": 0.91,
                "segment": {"document_id": "doc-1", "content": "召回段落",
                             "document": {"name": "产品文档"}},
            }]}
        raise AssertionError(url)

    monkeypatch.setattr(
        "rag_checkup.connectors.dify.httpx.Client",
        lambda **kwargs: _ContractClient(routes, calls),
    )
    connector = DifyConnector(api_key="dify-key", dataset_id="ds-1", base_url="http://dify")

    docs = connector.list_documents()
    answer, chunks, citations = connector.query("产品问题", top_k=3)

    assert docs[0].id == "doc-1" and docs[0].content == "答案段落"
    assert answer == "召回段落"
    assert chunks[0].doc_id == "doc-1" and chunks[0].score == 0.91
    assert citations == ["chunk-1"]
    assert [method for method, _, _ in calls] == ["GET", "GET", "POST"]


def test_dify_app_uses_app_retriever_resources_as_citation_chunks(monkeypatch):
    calls = []

    def routes(method, url, kwargs):
        assert method == "POST" and url.endswith("/chat-messages")
        return 200, {
            "answer": "应用答案【1】",
            "retriever_resources": [{
                "document_id": "app-doc",
                "document_name": "应用证据",
                "segment_id": "seg-1",
                "content": "应用返回的证据",
                "score": 0.88,
            }],
        }

    monkeypatch.setattr(
        "rag_checkup.connectors.dify.httpx.Client",
        lambda **kwargs: _ContractClient(routes, calls),
    )
    connector = DifyConnector(
        api_key="dataset-key", dataset_id="ds-1", app_key="app-key", base_url="http://dify",
    )

    answer, chunks, citations = connector.query("问题", top_k=5)

    assert answer == "应用答案【1】"
    assert chunks == [RetrievedChunk("app-doc", "应用证据", "应用返回的证据", 0.88)]
    assert citations == ["chunk-1"]
    assert len(calls) == 1
    assert calls[0][1].endswith("/chat-messages")
    assert citation_score(answer, chunks, citations)[0] == 100


def test_dify_auth_failure_is_not_retried(monkeypatch):
    calls = []

    def routes(method, url, kwargs):
        return 401, {"message": "invalid key"}

    monkeypatch.setattr(
        "rag_checkup.connectors.dify.httpx.Client",
        lambda **kwargs: _ContractClient(routes, calls),
    )
    connector = DifyConnector(
        api_key="secret-key", dataset_id="ds-1", base_url="http://dify", retry_backoff=0,
    )

    with pytest.raises(httpx.HTTPStatusError):
        connector.query("问题")
    assert len(calls) == 1


def test_fastgpt_contract_failure_is_explicit_and_transient_5xx_retries(monkeypatch):
    calls = []
    responses = [
        (503, {"message": "temporarily unavailable"}),
        (200, {
            "choices": [{"message": {"content": "FastGPT 答案"}}],
            "responseData": [{"quoteList": [{
                "id": "quote-1", "sourceName": "来源", "text": "FastGPT 证据", "score": 0.7,
            }]}],
        }),
    ]

    def routes(method, url, kwargs):
        assert method == "POST" and url.endswith("/v1/chat/completions")
        return responses.pop(0)

    monkeypatch.setattr(
        "rag_checkup.connectors.fastgpt.httpx.Client",
        lambda **kwargs: _ContractClient(routes, calls),
    )
    connector = FastGPTConnector(
        api_key="fastgpt-key", base_url="http://fastgpt", retry_backoff=0,
    )

    answer, chunks, citations = connector.query("问题")

    assert answer == "FastGPT 答案"
    assert chunks[0].content == "FastGPT 证据"
    assert citations == ["chunk-1"]
    assert len(calls) == 2


def test_fastgpt_auth_failure_is_not_retried(monkeypatch):
    calls = []

    def routes(method, url, kwargs):
        return 403, {"message": "forbidden"}

    monkeypatch.setattr(
        "rag_checkup.connectors.fastgpt.httpx.Client",
        lambda **kwargs: _ContractClient(routes, calls),
    )
    connector = FastGPTConnector(
        api_key="secret-key", base_url="http://fastgpt", retry_backoff=0,
    )

    with pytest.raises(httpx.HTTPStatusError):
        connector.query("问题")
    assert len(calls) == 1
