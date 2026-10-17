"""rag-checkup 测试:全部离线,不调任何真实 API/Key."""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag_checkup.connectors.mock import MockConnector, DOCS
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


def test_litellm_judge_runtime_failure_uses_heuristic_fallback():
    class FailingLiteLLM:
        def completion(self, **kwargs):
            raise RuntimeError("offline")

    judged = LiteLLMJudge(litellm_module=FailingLiteLLM()).judge(
        "correctness", "答案包含 18 小时", "18 小时"
    )

    assert judged["score"] == 100
    assert "参考答案" in judged["reason"]


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
