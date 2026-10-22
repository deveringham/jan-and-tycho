"""Interactive explorer over a sweep's `results.parquet` (SPEC Phase 5, optional).

One self-contained HTML file (plotly.js inlined, data embedded as JSON): pick any column
for x, y and colour, filter by scenario, toggle log axes, and switch between points and
box plots. Needs the `rudolf[explorer]` extra (plotly); without it nothing is written.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

_DEFAULT_Y = ("holdout_rms_arcsec", "holdout_rms_au", "runtime_s")

_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__NAME__ explorer</title>
<style>
:root{--bg:#fff;--fg:#222;--muted:#666;--line:#ddd}
@media (prefers-color-scheme:dark){:root{--bg:#16181d;--fg:#e6e6e6;--muted:#9aa;--line:#333}}
body{margin:0;padding:12px 16px;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
h1{font-size:18px;margin:0 0 8px}.bar{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;
margin-bottom:8px}label{color:var(--muted);font-size:12px}select{max-width:260px}
#plot{width:100%;height:calc(100vh - 120px);min-height:420px}
</style>
<script>__PLOTLY__</script></head><body>
<h1>__NAME__ — results explorer</h1>
<div class="bar">
<label>x <select id="x"></select></label><label>y <select id="y"></select></label>
<label>colour <select id="c"></select></label><label>scenario <select id="s"></select></label>
<label><input type="checkbox" id="lx"> log x</label>
<label><input type="checkbox" id="ly"> log y</label>
<label><input type="checkbox" id="box"> box plot</label>
</div><div id="plot"></div>
<script>
const DATA = __DATA__, COLS = __COLS__, DEF = __DEFAULTS__;
const $ = id => document.getElementById(id);
function fill(sel, opts, val){ for (const o of opts){ const e=document.createElement('option');
  e.value=e.textContent=o; sel.appendChild(e);} sel.value=val; }
fill($('x'), COLS, DEF.x); fill($('y'), COLS, DEF.y); fill($('c'), COLS, DEF.c);
const scen = ['(all)', ...new Set(DATA.map(r => r.scenario))]; fill($('s'), scen, '(all)');
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
function draw(){
  const x=$('x').value, y=$('y').value, c=$('c').value, s=$('s').value, box=$('box').checked;
  const rows = DATA.filter(r => s==='(all)' || r.scenario===s);
  const groups = {};
  for (const r of rows){ const k=String(r[c]); (groups[k] = groups[k] || []).push(r); }
  const traces = Object.entries(groups).sort().map(([k, g]) => ({
    type: box ? 'box' : 'scatter', mode: 'markers', name: k,
    x: g.map(r => r[x]), y: g.map(r => r[y]), boxpoints: box ? 'all' : undefined,
    text: g.map(r => `${r.label} · ${r.scenario} · seed ${r.seed}`),
    hovertemplate: '%{text}<br>' + x + '=%{x}<br>' + y + '=%{y}<extra>' + c + '=' + k + '</extra>',
    marker: {size: 7, opacity: 0.8}}));
  Plotly.react('plot', traces, {
    template: dark ? 'plotly_dark' : 'plotly_white', paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)', boxmode: 'group', legend: {title: {text: c}},
    xaxis: {title: x, type: $('lx').checked ? 'log' : '-'},
    yaxis: {title: y, type: $('ly').checked ? 'log' : '-'},
    margin: {l: 60, r: 10, t: 10, b: 50}}, {responsive: true});
}
for (const id of ['x','y','c','s','lx','ly','box']) $(id).addEventListener('change', draw);
draw();
</script></body></html>"""


def _records(df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
    def scalar(v: Any) -> bool:
        return v is None or isinstance(v, (str, bool, int, float, np.generic))

    keep = [str(c) for c in df.columns if bool(df[c].map(scalar).all())]
    sub = cast(pd.DataFrame, df[keep].astype(object).where(df[keep].notna(), None))
    records = cast(list[dict[str, Any]], sub.to_dict("records"))
    rows = [
        {k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.items()} for r in records
    ]
    for r in rows:
        for k, v in r.items():
            if isinstance(v, float) and not np.isfinite(v):
                r[k] = None
    return rows, keep


def write_explorer(df: pd.DataFrame, out: str | Path, name: str = "sweep") -> Path | None:
    """Write the explorer HTML; None if plotly is not installed."""
    try:
        from plotly.offline import get_plotlyjs
    except ImportError:
        return None
    rows, cols = _records(df)
    y = next((c for c in _DEFAULT_Y if c in cols), cols[0])
    swept = [c for c in cols if c.startswith(("jan.", "noise.", "schedule.", "realism."))]
    defaults = {
        "x": swept[0] if swept else "label",
        "y": y,
        "c": "label" if "label" in cols else cols[0],
    }
    page = (
        _TEMPLATE.replace("__PLOTLY__", get_plotlyjs().replace("</script>", "<\\/script>"))
        .replace("__NAME__", html.escape(name))
        .replace("__DATA__", json.dumps(rows).replace("</", "<\\/"))
        .replace("__COLS__", json.dumps(cols))
        .replace("__DEFAULTS__", json.dumps(defaults))
    )
    path = Path(out)
    path.write_text(page, encoding="utf-8")
    return path
