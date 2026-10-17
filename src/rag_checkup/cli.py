"""rag-checkup 命令行:demo / run / ab."""
from __future__ import annotations

import os

import typer

from . import __version__
from .ab_compare import ab_compare
from .connectors.dify import DifyConnector
from .connectors.fastgpt import FastGPTConnector
from .connectors.mock import MockConnector
from .pipeline import run_checkup
from .question_gen import QuestionGenerator, make_litellm_callback
from .report import write_report
from .scoring import HeuristicJudge, LiteLLMJudge

app = typer.Typer(help="10 分钟给知识库问答做一次体检.")


def _llm_components(use_llm: bool, model: str = "gpt-4o-mini"):
    """Return the question generator and judge for one CLI run.

    Both components receive the same explicit key.  If the optional LLM
    dependency is unavailable, or no key is configured, the complete run stays
    usable through the deterministic generator/judge.
    """
    if not use_llm:
        return QuestionGenerator(), HeuristicJudge()

    key = os.environ.get("RAGCHECKUP_LLM_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        typer.echo("未找到 LLM key,用离线启发式出题和裁判(可用 RAGCHECKUP_LLM_KEY 启用 LLM).")
        return QuestionGenerator(), HeuristicJudge()

    try:
        generator = QuestionGenerator(llm=make_litellm_callback(model=model, api_key=key))
        judge = LiteLLMJudge(model=model, api_key=key)
        return generator, judge
    except RuntimeError as e:
        typer.echo(f"LLM 不可用({e}),用离线启发式出题和裁判.")
        return QuestionGenerator(), HeuristicJudge()


def _judge(use_llm: bool, model: str = "gpt-4o-mini"):
    """Compatibility helper for callers that only need the judge."""
    return _llm_components(use_llm, model=model)[1]


def _validate_positive_option(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise typer.BadParameter(f"{name} must be a positive integer", param_hint=f"--{name}")


@app.command()
def demo(
    questions: int = typer.Option(4, "--questions", "-n", help="每篇文档出几道事实题"),
    out: str = typer.Option("report.html", "--out", "-o", help="报告输出路径"),
    lang: str = typer.Option("zh", "--lang", help="报告语言 zh|en"),
    use_llm: bool = typer.Option(False, "--llm", help="用 LLM 出题+裁判(需 key)"),
    model: str = typer.Option("gpt-4o-mini", "--model", help="LLM 模型名(用于出题和裁判)"),
):
    """零 key 演示:内置 mock 知识库,10 分钟跑通全流程."""
    _validate_positive_option("questions", questions)
    typer.echo("== rag-checkup demo 模式:内置 mock 知识库,无需任何 key ==")
    generator, judge = _llm_components(use_llm, model=model)
    result = run_checkup(MockConnector(), n_per_doc=questions, judge=judge,
                         question_generator=generator,
                         progress=typer.echo)
    path = write_report(result, out, lang=lang, version=__version__)
    typer.echo(f"报告已生成:{path}  总分:{result.aggregates['total']}")


@app.command()
def run(
    platform: str = typer.Option(..., "--platform", "-p", help="dify | fastgpt"),
    api_key: str = typer.Option("", "--api-key", help="API key(默认读环境变量)"),
    dataset_id: str = typer.Option("", "--dataset-id", help="Dify dataset ID / FastGPT dataset IDs(逗号分隔)"),
    base_url: str = typer.Option("", "--base-url", help="自托管服务地址(可选)"),
    app_key: str = typer.Option("", "--app-key", help="Dify App key(端到端问答模式,可选)"),
    questions: int = typer.Option(4, "--questions", "-n"),
    top_k: int = typer.Option(5, "--top-k"),
    out: str = typer.Option("report.html", "--out", "-o"),
    lang: str = typer.Option("zh", "--lang", help="zh|en"),
    use_llm: bool = typer.Option(False, "--llm", help="用 LLM 出题+裁判(需 RAGCHECKUP_LLM_KEY)"),
    model: str = typer.Option("gpt-4o-mini", "--model", help="LLM 模型名(用于出题和裁判)"),
):
    """连真实知识库做体检."""
    _validate_positive_option("questions", questions)
    _validate_positive_option("top-k", top_k)
    key = api_key or os.environ.get("RAGCHECKUP_DIFY_KEY" if platform == "dify"
                                    else "RAGCHECKUP_FASTGPT_KEY", "")
    if platform == "dify":
        conn = DifyConnector(api_key=key, dataset_id=dataset_id,
                             base_url=base_url or "https://api.dify.ai/v1",
                             app_key=app_key or None)
    elif platform == "fastgpt":
        conn = FastGPTConnector(api_key=key,
                                dataset_ids=[d.strip() for d in dataset_id.split(",") if d.strip()],
                                base_url=base_url or "https://api.fastgpt.in/api")
    else:
        raise typer.BadParameter("platform 只能是 dify 或 fastgpt")
    generator, judge = _llm_components(use_llm, model=model)
    result = run_checkup(conn, n_per_doc=questions, judge=judge,
                         question_generator=generator,
                         top_k=top_k, progress=typer.echo)
    path = write_report(result, out, lang=lang, version=__version__)
    typer.echo(f"报告已生成:{path}  总分:{result.aggregates['total']}")


@app.command(name="ab")
def ab_cmd(
    platform: str = typer.Option(..., "--platform", "-p", help="dify | fastgpt | mock(演示)"),
    api_key: str = typer.Option("", "--api-key"),
    dataset_id: str = typer.Option("", "--dataset-id"),
    base_url: str = typer.Option("", "--base-url"),
    top_k_a: int = typer.Option(3, "--top-k-a"),
    top_k_b: int = typer.Option(10, "--top-k-b"),
    questions: int = typer.Option(4, "--questions", "-n"),
    use_llm: bool = typer.Option(False, "--llm"),
    model: str = typer.Option("gpt-4o-mini", "--model", help="LLM 模型名(用于出题和裁判)"),
):
    """A/B 对比:同一测试集跑两套 top-k,看改参数涨多少分."""
    _validate_positive_option("questions", questions)
    _validate_positive_option("top-k-a", top_k_a)
    _validate_positive_option("top-k-b", top_k_b)
    key = api_key or os.environ.get("RAGCHECKUP_DIFY_KEY" if platform == "dify"
                                    else "RAGCHECKUP_FASTGPT_KEY", "")

    def make_a():
        if platform == "mock":
            return MockConnector()
        if platform == "dify":
            return DifyConnector(api_key=key, dataset_id=dataset_id, base_url=base_url or "https://api.dify.ai/v1")
        return FastGPTConnector(api_key=key, dataset_ids=[dataset_id], base_url=base_url or "https://api.fastgpt.in/api")

    def make_b():
        return make_a()

    generator, judge = _llm_components(use_llm, model=model)
    cmp = ab_compare(make_a, make_b, label_a=f"top_k={top_k_a}", label_b=f"top_k={top_k_b}",
                     n_per_doc=questions, judge=judge, question_generator=generator,
                     top_k_a=top_k_a, top_k_b=top_k_b, progress=typer.echo)
    typer.echo(f"\nA/B 对比结果:胜者 {cmp['winner']},总分差 {cmp['delta_total']:+}")
    for r in cmp["rows"]:
        typer.echo(f"  {r['dim']:12s} A={r['a']:<6} B={r['b']:<6} Δ={r['delta']:+}")
    write_report(cmp["result_b"], "report_ab_b.html", version=__version__)
    typer.echo("B 组报告已生成:report_ab_b.html")


@app.command()
def version():
    typer.echo(f"rag-checkup {__version__}")


if __name__ == "__main__":
    app()
