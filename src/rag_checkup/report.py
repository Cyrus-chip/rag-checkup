"""单文件中文/英文 HTML 体检报告:Jinja2 + 内联 SVG,手机可看,可分享."""
from __future__ import annotations

import html
from datetime import datetime

from jinja2 import Template

from .i18n import t
from .models import CheckupResult

TEMPLATE = """<!DOCTYPE html>
<html lang="{{ lang }}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ S.title }}</title>
<style>
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;max-width:860px;margin:0 auto;padding:16px;color:#1f2937;background:#f9fafb}
.card{background:#fff;border-radius:12px;padding:20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
h1{font-size:22px;margin:0 0 4px}.sub{color:#6b7280;font-size:13px}
.total{font-size:56px;font-weight:800;color:{{ color }}}
.grid{display:flex;gap:12px;flex-wrap:wrap}.stat{flex:1;min-width:140px;background:#f3f4f6;border-radius:8px;padding:12px;text-align:center}
.stat b{font-size:24px;display:block}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;color:#fff;background:{{ color }}}
.bad{border-left:4px solid #ef4444}.ok{border-left:4px solid #10b981}
details{margin:10px 0;border:1px solid #e5e7eb;border-radius:8px;padding:10px}
summary{cursor:pointer;font-weight:600}
pre{background:#f3f4f6;padding:10px;border-radius:8px;white-space:pre-wrap;font-size:13px}
button{border:1px solid #d1d5db;background:#fff;border-radius:6px;padding:4px 12px;cursor:pointer;font-size:12px}
.dim{display:flex;align-items:center;gap:8px;margin:6px 0;font-size:14px}
.bar{height:10px;border-radius:5px;background:#e5e7eb;flex:1}.fill{height:10px;border-radius:5px}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{border:1px solid #e5e7eb;padding:6px 8px;text-align:left}
.sug{border-left:4px solid #f59e0b;background:#fffbeb;padding:10px;border-radius:0 8px 8px 0;margin:8px 0}
.sug.ok{border-color:#10b981;background:#ecfdf5}.sug.bad{border-color:#ef4444;background:#fef2f2}
.footer{color:#9ca3af;font-size:12px;text-align:center;margin-top:24px}
</style></head><body>
<div class="card">
<h1>{{ S.title }}</h1>
<div class="sub">{{ now }} · {{ S.connector }}: {{ meta.connector }} · {{ S.n_cases }}: {{ agg.n_cases }} · {{ S.judge }}: {{ meta.judge }} · {{ S.judge_fallbacks }}: {{ meta.get('judge_fallbacks', 0) }} · {{ S.judge_errors }}: {{ meta.get('judge_errors', 0) }}</div>
<div style="display:flex;gap:20px;align-items:center;margin-top:12px;flex-wrap:wrap">
<div><div class="total">{{ agg.total }}</div><div class="sub">{{ S.total }}</div></div>
<div>{{ radar|safe }}</div>
<div><span class="badge">{{ agg.n_bad }}/{{ agg.n_cases }} bad</span>
<div class="sub" style="margin-top:6px">{{ S.bad_rate }}: {{ agg.bad_rate }}%</div></div>
</div>
<div style="margin-top:12px">
{% for d in dims %}<div class="dim"><span style="width:90px">{{ S.dims[d] }}</span>
<div class="bar"><div class="fill" style="width:{{ agg[d] }}%;background:{{ dcolor[d] }}"></div></div>
<span style="width:44px;text-align:right">{{ agg[d] }}</span></div>{% endfor %}
</div></div>

<div class="card"><h2>{{ S.suggestions }}</h2>
{% for s in suggestions %}<div class="sug {{ s.level }}"><b>{{ s.title }}</b><div style="font-size:13px;margin-top:4px">{{ s.detail }}</div></div>{% endfor %}
</div>

<div class="card"><h2>{{ S.bad_cases }}</h2>
{% if bad_cases %}
{% for c in bad_cases %}<details class="{{ 'bad' }}">
<summary>[{{ c.qa.question.qtype }}] {{ c.qa.question.text|e }} — 最低分 {{ c.min_score }}</summary>
<p><b>{{ S.question }}:</b> {{ c.qa.question.text|e }}</p>
<p><b>{{ S.answer }}:</b></p><pre>{{ c.qa.answer|e }}</pre>
<p><b>{{ S.chunks }} ({{ c.qa.chunks|length }}):</b></p>
{% for ch in c.qa.chunks %}<pre>{{ ch.content|e }}</pre>{% endfor %}
{% if not c.qa.chunks %}<p style="color:#ef4444">—</p>{% endif %}
<p><b>{{ S.citations }}:</b> {{ c.qa.citations|join(', ')|e or '—' }}</p>
<p><b>{{ S.reason }}:</b></p>
<ul>{% for k,v in c.scores.items() %}<li>{{ S.dims[k] }}: {{ v }} — {{ c.reasons[k]|e }}</li>{% endfor %}</ul>
<button onclick="navigator.clipboard.writeText(this.parentElement.innerText)">{{ S.copy }}</button>
</details>{% endfor %}
{% else %}<p>🎉 {{ S.all_ok }}!</p>{% endif %}
</div>

<div class="card"><h2>{{ S.meta }}</h2>
<table>{% for k,v in meta.items() %}<tr><th>{{ k }}</th><td>{{ v }}</td></tr>{% endfor %}</table></div>
<div class="footer">rag-checkup v{{ version }} · 本地运行,Key 与数据不出内网</div>
</body></html>
"""


def _radar(agg: dict, lang: str) -> str:
    import math
    cx, cy, r = 90, 80, 60
    dims = ["correctness", "citation", "refusal"]
    labels = [t(lang)["dims"][d] for d in dims]
    pts = []
    for i, d in enumerate(dims):
        ang = -math.pi / 2 + i * 2 * math.pi / 3
        rr = r * agg[d] / 100
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    spokes = "".join(
        f'<line x1="{cx}" y1="{cy}" x2="{cx + r * math.cos(-math.pi/2 + i*2*math.pi/3):.1f}" '
        f'y2="{cy + r * math.sin(-math.pi/2 + i*2*math.pi/3):.1f}" stroke="#e5e7eb"/>'
        for i in range(3))
    texts = "".join(
        f'<text x="{cx + (r+16) * math.cos(-math.pi/2 + i*2*math.pi/3):.1f}" '
        f'y="{cy + (r+16) * math.sin(-math.pi/2 + i*2*math.pi/3):.1f}" font-size="11" '
        f'text-anchor="middle" fill="#6b7280">{html.escape(labels[i])}</text>'
        for i in range(3))
    return (f'<svg width="180" height="160"><polygon points="{poly}" fill="rgba(59,130,246,.25)" '
            f'stroke="#3b82f6" stroke-width="2"/>{spokes}{texts}</svg>')


def render(result: CheckupResult, lang: str = "zh", version: str = "0.1.0") -> str:
    S = t(lang)
    agg = result.aggregates
    total = agg["total"]
    color = "#10b981" if total >= 80 else "#f59e0b" if total >= 60 else "#ef4444"
    dcolor = {"correctness": "#3b82f6", "citation": "#8b5cf6", "refusal": "#ec4899"}
    bad_cases = []
    for c in sorted(result.cases, key=lambda x: min(x.scores.values())):
        if c.is_bad:
            c.min_score = min(c.scores.values())
            bad_cases.append(c)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return Template(TEMPLATE).render(
        S=S, lang=lang, agg=agg, dims=["correctness", "citation", "refusal"],
        radar=_radar(agg, lang), color=color, dcolor=dcolor,
        bad_cases=bad_cases, suggestions=result.suggestions,
        meta=result.meta, now=now, version=version,
    )


def write_report(result: CheckupResult, path: str, lang: str = "zh", version: str = "0.1.0") -> str:
    from pathlib import Path

    from . import __version__
    html_text = render(result, lang=lang, version=version or __version__)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html_text)
    return path
