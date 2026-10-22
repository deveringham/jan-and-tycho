"""Static HTML reports built from figure manifests, never by globbing (SPEC §9.3)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import jinja2
import numpy as np
import pandas as pd
from markupsafe import Markup

from jt_core import io
from rudolf.data import JanOutputs, RunView, Sealed, list_labels
from rudolf.viz import style
from rudolf.viz.compare_plots import render_comparisons
from rudolf.viz.manifest import FigureEntry, FigureManifest
from rudolf.viz.run_plots import TimeAxis

_CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:0 16px 48px;
     color:#222;background:#fafafa;max-width:1200px;margin:auto;line-height:1.45}
h1{font-size:1.5rem;margin:24px 0 4px} h2{font-size:1.2rem;margin-top:32px;border-bottom:1px solid #ddd}
h3{font-size:1rem;margin:20px 0 6px} .muted{color:#666;font-size:.9rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px}
figure{margin:0;background:#fff;border:1px solid #e3e3e3;border-radius:6px;padding:8px}
figure img,figure video{width:100%;height:auto} figcaption{font-size:.82rem;color:#444;margin-top:4px}
table{border-collapse:collapse;font-size:.85rem;background:#fff;margin:8px 0;max-width:100%;display:block;overflow-x:auto}
th,td{border:1px solid #ddd;padding:4px 8px;text-align:right} th{background:#f0f0f0}
td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){text-align:left}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
pre{white-space:pre-wrap;font-size:.8rem;background:#fff;border:1px solid #eee;padding:8px}
"""

_MEDIA = """{% macro media(e, base) -%}
<figure><a href="{{ base }}/{{ e.paths[0] }}">
{%- set p = e.paths[0] -%}
{%- if p.endswith('.mp4') -%}<video src="{{ base }}/{{ p }}" controls loop muted></video>
{%- else -%}<img src="{{ base }}/{{ p }}" alt="{{ e.title }}" loading="lazy">{%- endif -%}
</a><figcaption><b>{{ e.title }}</b> — {{ e.caption }}</figcaption></figure>
{%- endmacro %}"""

_RUN_TEMPLATE = (
    _MEDIA
    + """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ scenario }} · seed {{ seed }}</title><style>{{ css }}</style></head><body>
<h1>{{ scenario }} — seed {{ seed }}</h1>
<p class="muted">{{ description }}<br>Tier {{ tier }} · {{ n_train }} training / {{ n_holdout }} holdout
observations · holdout starts at t = {{ "%.0f"|format(t_split) }} d of {{ "%.0f"|format(duration) }} d.</p>
<h2>Scores</h2>
<table><thead><tr><th>strategy</th><th>kind</th><th>holdout RMS</th><th>holdout max</th>
<th>train (dense) RMS</th><th>traj. RMS holdout (AU)</th><th>runtime (s)</th></tr></thead><tbody>
{% for r in rows %}<tr><td><span class="swatch" style="background:{{ r.color }}"></span>{{ r.label }}</td>
<td>{{ r.kind }}</td><td>{{ r.rms }}</td><td>{{ r.max }}</td><td>{{ r.train }}</td><td>{{ r.traj }}</td>
<td>{{ r.runtime }}</td></tr>{% endfor %}</tbody></table>
{% if overview %}<div class="grid">{% for e in overview %}{{ media(e, overview_base) }}{% endfor %}</div>{% endif %}
{% for s in sections %}<h2>{{ s.label }}</h2><pre>{{ s.notes }}</pre>
<div class="grid">{% for e in s.entries %}{{ media(e, s.base) }}{% endfor %}</div>{% endfor %}
</body></html>"""
)

_BENCH_TEMPLATE = (
    _MEDIA
    + """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ name }} report</title><style>{{ css }}</style></head><body>
<h1>{{ name }}</h1><p class="muted">{{ n_cells }} cells ({{ n_errors }} failed) · swept: {{ swept or "nothing" }}</p>
{% if explorer %}<p><a href="{{ explorer }}">Interactive explorer</a> — any result column against any other.</p>{% endif %}
<h2>Leaderboard</h2>{{ leaderboard_html }}
<h2>Comparisons</h2><div class="grid">{% for e in comparisons %}{{ media(e, "figures") }}{% endfor %}</div>
{% if skipped %}<p class="muted">Not yet implemented (later phases): {{ skipped|join(", ") }}</p>{% endif %}
{% if errors %}<h2>Failed cells</h2><table><thead><tr><th>scenario</th><th>strategy</th><th>seed</th><th>error</th></tr></thead>
<tbody>{% for e in errors %}<tr><td>{{ e.scenario }}</td><td>{{ e.strategy }}</td><td>{{ e.seed }}</td><td>{{ e.error }}</td></tr>{% endfor %}</tbody></table>{% endif %}
<h2>Per-run figures</h2>
{% for c in cells %}<h3>{{ c.title }}</h3><div class="grid">{% for e in c.entries %}{{ media(e, c.base) }}{% endfor %}</div>{% endfor %}
</body></html>"""
)

_env = jinja2.Environment(
    autoescape=jinja2.select_autoescape(default=True), undefined=jinja2.StrictUndefined
)


def _present(v: Any) -> bool:
    """A swept value worth printing (lists, e.g. a family set, count as present)."""
    if isinstance(v, list | tuple | np.ndarray):
        return len(v) > 0
    return bool(pd.notna(v))


def _fmt(v: Any, unit: str = "") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.3g}{unit}"


def run_overview(run_dir: Path, labels: list[str]) -> FigureManifest:
    """Run-level figure: every strategy's prediction error vs time on one axis."""
    out = run_dir / "rudolf"
    m = FigureManifest.load(out)
    sealed = Sealed.load(run_dir)
    if not sealed.is_sky or not labels:
        return m
    ta = TimeAxis(sealed.duration)
    for body in sealed.body_ids:
        fig, ax = style.new_figure(8.0, 4.0)
        drew = False
        for lab in labels:
            v = RunView(sealed, JanOutputs.load(run_dir, lab), body)
            if v.dense_err_arcsec is not None:
                ax.plot(
                    ta(v.dense_t),
                    np.maximum(v.dense_err_arcsec, 1e-3),
                    **style.strategy_style(lab).line_kwargs(lab),
                )
                drew = True
        if sealed.noise_sigma_arcsec:
            ax.axhline(
                sealed.noise_sigma_arcsec,
                color=style.NOISE_REF_COLOR,
                ls="--",
                lw=1,
                label="noise σ",
            )
        style.mark_holdout(ax, sealed.t_split, sealed.duration, ta.scale)
        if drew:
            ax.set_yscale("log")
        ax.set_xlabel(ta.label)
        ax.set_ylabel("great-circle error (arcsec)")
        ax.legend(fontsize=7)
        ax.set_title(
            f"All strategies — {sealed.scenario_name} (seed {sealed.seed}), body {body}",
            fontsize=10,
        )
        suffix = "" if len(sealed.body_ids) == 1 else f"__{body}"
        style.save_figure(
            fig,
            out / "figures",
            "strategies_residuals" + suffix,
            m,
            title="All strategies: error vs time",
            caption="Prediction error vs noise-free truth for every strategy run on this scenario and seed.",
            context={"scenario": sealed.scenario_name, "seed": sealed.seed, "body_id": body},
        )
    m.save()
    return m


def run_report(run_dir: str | Path, out: str | Path | None = None) -> Path:
    run_dir = Path(run_dir)
    sealed = Sealed.load(run_dir)
    labels = [
        lab for lab in list_labels(run_dir) if (run_dir / "rudolf" / lab / io.SCORE_FILE).exists()
    ]
    overview = run_overview(run_dir, labels)
    rows: list[dict[str, Any]] = []
    sections: list[dict[str, Any]] = []
    target = Path(out) if out else run_dir / "rudolf" / "report.html"
    for lab in labels:
        sc = io.read_json(run_dir / "rudolf" / lab / io.SCORE_FILE)
        for body, b in sc["bodies"].items():
            p, t = b["predictive"], b["trajectory"]
            unit = "″" if sealed.is_sky else " AU"
            key = "arcsec" if sealed.is_sky else "au"
            rows.append(
                {
                    "label": lab if len(sc["bodies"]) == 1 else f"{lab} [{body}]",
                    "color": style.strategy_color(lab),
                    "kind": sc["kind"],
                    "rms": _fmt(p.get(f"holdout_rms_{key}"), unit),
                    "max": _fmt(p.get(f"holdout_max_{key}"), unit),
                    "train": _fmt(p.get("train_dense_rms_arcsec"), unit),
                    "traj": _fmt(t.get("traj_rms_holdout_au")),
                    "runtime": _fmt(sc.get("runtime_s")),
                }
            )
        m = FigureManifest.load(run_dir / "rudolf" / lab)
        hyp = io.read_json(run_dir / "jan" / lab / io.HYPOTHESIS_FILE)
        sections.append(
            {
                "label": lab,
                "notes": hyp.get("notes", ""),
                "entries": m.entries,
                "base": os.path.relpath(run_dir / "rudolf" / lab, target.parent),
            }
        )
    html = _env.from_string(_RUN_TEMPLATE).render(
        css=_CSS,
        scenario=sealed.scenario_name,
        seed=sealed.seed,
        description=sealed.scenario.get("description", ""),
        tier=sealed.info["tier"],
        n_train=sealed.info["n_train"],
        n_holdout=sealed.info["n_holdout"],
        t_split=sealed.t_split,
        duration=sealed.duration,
        rows=rows,
        sections=sections,
        overview=overview.entries,
        overview_base=os.path.relpath(run_dir / "rudolf", target.parent),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return target


def bench_report(
    results_dir: str | Path,
    out: str | Path | None = None,
    requests: list[dict[str, Any]] | None = None,
) -> Path:
    results_dir = Path(results_dir)
    df = io.read_table(results_dir / "results.parquet")
    meta = (
        io.read_json(results_dir / "results_meta.json")
        if (results_dir / "results_meta.json").exists()
        else {}
    )
    if requests is None:
        sweep_yaml = results_dir / "sweep.yaml"
        import yaml

        requests = (
            yaml.safe_load(sweep_yaml.read_text())["viz"]["comparison"]
            if sweep_yaml.exists()
            else []
        )
    fig_dir = results_dir / "figures"
    m = FigureManifest(fig_dir)
    skipped = render_comparisons(df, fig_dir, list(requests or []), m, results_dir=results_dir)
    m.save()
    lb = m.get("leaderboard_table")
    lb_html = (fig_dir / lb.paths[0]).read_text(encoding="utf-8") if lb else "<p>—</p>"
    comps = [e for e in m.entries if e.kind != "table"]
    cells: list[dict[str, Any]] = []
    target = Path(out) if out else results_dir / "report.html"
    ok = io.rows(df, df["status"] == "ok") if "status" in df.columns else df
    for _, r in ok.iterrows():
        cm = FigureManifest.load(results_dir / str(r["run_dir"]) / "rudolf" / str(r["label"]))
        if cm.entries:
            params = ", ".join(
                f"{k}={r[k]}"
                for k in meta.get("swept_parameters", [])
                if k in r.index and _present(r[k])
            )
            cells.append(
                {
                    "title": f"{r['scenario']} · {r['label']} · seed {r['seed']}"
                    + (f" · {params}" if params else ""),
                    "entries": cm.entries,
                    "base": os.path.relpath(
                        results_dir / str(r["run_dir"]) / "rudolf" / str(r["label"]), target.parent
                    ),
                }
            )
    errors = (
        io.rows(df, df["status"] != "ok").to_dict(orient="records")
        if "status" in df.columns
        else []
    )
    from rudolf.explorer import write_explorer

    ex = write_explorer(df, results_dir / "explorer.html", results_dir.name)
    html = _env.from_string(_BENCH_TEMPLATE).render(
        css=_CSS,
        explorer=os.path.relpath(ex, target.parent) if ex else None,
        name=results_dir.name,
        n_cells=len(df),
        n_errors=len(errors),
        swept=", ".join(meta.get("swept_parameters", [])),
        leaderboard_html=Markup(lb_html),
        comparisons=comps,
        skipped=skipped,
        errors=errors,
        cells=cells,
    )
    target.write_text(html, encoding="utf-8")
    return target


__all__ = ["FigureEntry", "bench_report", "run_report"]
