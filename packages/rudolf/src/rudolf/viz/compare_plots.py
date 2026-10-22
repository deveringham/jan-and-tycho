"""Cross-run comparison plots from the bench's aggregated `results.parquet` (SPEC §7.1.4)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from jt_core import io
from jt_core.constants import YEAR_DAYS
from rudolf.viz import style
from rudolf.viz.manifest import FigureManifest

LEADERBOARD_METRIC = "holdout_rms_arcsec"


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    return io.rows(df, df["status"] == "ok") if "status" in df.columns else df


def _q(x: Any, q: float) -> float:
    v = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, q)) if v.size else float("nan")


def leaderboard_table(df: pd.DataFrame, metric: str = LEADERBOARD_METRIC) -> pd.DataFrame:
    """Per scenario × strategy label: median and IQR over seeds (and over Tycho overrides)."""
    d = _ok(df)
    if metric not in d.columns:
        metric = "holdout_rms_au"
    rows: list[dict[str, Any]] = []
    for key, g in d.groupby(["scenario", "label"], sort=True):
        sc, label = cast(tuple[str, str], key)
        rows.append(
            {
                "scenario": sc,
                "strategy": label,
                "n": len(g),
                "metric": metric,
                "median": _q(g[metric], 0.5) if metric in g else np.nan,
                "q25": _q(g[metric], 0.25) if metric in g else np.nan,
                "q75": _q(g[metric], 0.75) if metric in g else np.nan,
                "traj_rms_holdout_au": _q(g["traj_rms_holdout_au"], 0.5)
                if "traj_rms_holdout_au" in g
                else np.nan,
                "runtime_s": _q(g["runtime_s"], 0.5) if "runtime_s" in g else np.nan,
            }
        )
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    return t.sort_values(["scenario", "median"]).reset_index(drop=True)


def _fmt(v: Any, unit: str = "") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.3g}{unit}"


def leaderboard(
    df: pd.DataFrame, out: Path, m: FigureManifest, ctx: dict[str, Any] | None = None
) -> None:
    t = leaderboard_table(df)
    scenarios = sorted(t["scenario"].unique()) if not t.empty else []
    n = max(len(scenarios), 1)
    fig, axes = style.new_figure(7.5, 1.0 + 0.45 * max(len(t), 2) + 0.4 * n, nrows=n, squeeze=False)
    unit = "″" if (not t.empty and t["metric"].iloc[0].endswith("arcsec")) else " AU"
    for ax, sc in zip(axes[:, 0], scenarios, strict=False):
        g = t[t["scenario"] == sc].iloc[::-1]
        ys = np.arange(len(g))
        for y, (_, r) in zip(ys, g.iterrows(), strict=True):
            c = style.strategy_color(str(r["strategy"]))
            ax.plot([r["q25"], r["q75"]], [y, y], color=c, lw=3, alpha=0.4)
            ax.scatter([r["median"]], [y], color=c, zorder=3)
            ax.text(r["median"], y + 0.22, f" {_fmt(r['median'], unit)}", fontsize=7)
        ax.set_yticks(ys, [f"{r['strategy']} (n={r['n']})" for _, r in g.iterrows()])
        ax.set_xscale("log")
        ax.set_title(sc, fontsize=9)
        ax.set_xlabel(
            f"holdout RMS error ({'arcsec' if unit == '″' else 'AU'}); median and IQR over seeds"
        )
    if not scenarios:
        axes[0, 0].text(
            0.5, 0.5, "no successful cells", ha="center", transform=axes[0, 0].transAxes
        )
    fig.suptitle("Leaderboard", fontsize=10)
    style.save_figure(
        fig,
        out,
        "leaderboard",
        m,
        title="Leaderboard",
        caption="Strategies ranked by holdout RMS (median over seeds, bar = interquartile range).",
        context=ctx or {},
    )
    html = [
        "<table class='lb'><thead><tr><th>scenario</th><th>strategy</th><th>n</th>"
        "<th>holdout RMS (median)</th><th>IQR</th><th>trajectory RMS, holdout (AU)</th>"
        "<th>runtime (s)</th></tr></thead><tbody>"
    ]
    for _, r in t.iterrows():
        html.append(
            f"<tr><td>{r['scenario']}</td><td>{r['strategy']}</td><td>{r['n']}</td>"
            f"<td>{_fmt(r['median'], unit)}</td><td>{_fmt(r['q25'])}–{_fmt(r['q75'])}</td>"
            f"<td>{_fmt(r['traj_rms_holdout_au'])}</td><td>{_fmt(r['runtime_s'])}</td></tr>"
        )
    html.append("</tbody></table>")
    p = out / "leaderboard.html"
    p.write_text("\n".join(html), encoding="utf-8")
    m.add(
        name="leaderboard_table",
        kind="table",
        paths=[p],
        title="Leaderboard table",
        caption="Leaderboard as a table.",
        context=ctx or {},
    )


def extrapolation_curves(
    df: pd.DataFrame, out: Path, m: FigureManifest, ctx: dict[str, Any] | None = None
) -> None:
    d = _ok(df)
    if "extrap_rms" not in d.columns:
        return
    scenarios = sorted(d["scenario"].unique())
    fig, axes = style.new_figure(
        7.0, 3.2 * max(len(scenarios), 1), nrows=max(len(scenarios), 1), squeeze=False
    )
    unit = (
        "arcsec"
        if "holdout_rms_arcsec" in d.columns and bool(d["holdout_rms_arcsec"].notna().any())
        else "AU"
    )
    for ax, sc in zip(axes[:, 0], scenarios, strict=False):
        g = io.rows(d, d["scenario"] == sc)
        years = False
        for label, gl in g.groupby("label", sort=True):
            dt = np.asarray(gl["extrap_dt_days"].to_numpy()[0], dtype=float)
            curves = np.array(
                [[np.nan if v is None else v for v in c] for c in gl["extrap_rms"]], dtype=float
            )
            if not np.isfinite(curves).any():
                continue
            years = dt.max() > 3 * YEAR_DAYS
            x = dt / YEAR_DAYS if years else dt
            med = np.nanmedian(curves, axis=0)
            sty = style.strategy_style(str(label))
            ax.plot(x, med, **sty.line_kwargs(f"{label} (n={len(gl)})"), marker="o", ms=3)
            if len(gl) > 1:
                ax.fill_between(
                    x,
                    np.nanquantile(curves, 0.25, 0),
                    np.nanquantile(curves, 0.75, 0),
                    color=sty.color,
                    alpha=0.15,
                    lw=0,
                )
        ax.set_yscale("log")
        ax.set_xlabel("time since end of training (" + ("years" if years else "days") + ")")
        ax.set_ylabel(f"RMS error ({unit})")
        ax.set_title(sc, fontsize=9)
        ax.axvline(0, color=style.HOLDOUT_SHADE, ls="--", lw=0.9)
        ax.legend(fontsize=7)
    fig.suptitle("Extrapolation: error growth after the training window", fontsize=10)
    style.save_figure(
        fig,
        out,
        "extrapolation_curves",
        m,
        title="Extrapolation curves",
        caption=(
            "Holdout prediction error vs time since the end of training; "
            "median over seeds with IQR band."
        ),
        context=ctx or {},
    )


# ----------------------------------------------------------------------------- Phase 2


def _col(df: pd.DataFrame, name: str) -> str:
    """Resolve a swept-parameter column: exact, or `jan.<name>` for Jan hyperparameters."""
    if name in df.columns:
        return name
    if f"jan.{name}" in df.columns:
        return f"jan.{name}"
    raise KeyError(f"no column {name!r} in results")


def _num(series: Any) -> np.ndarray:
    return np.asarray(pd.to_numeric(pd.Series(series), errors="coerce"), dtype=np.float64)


def _median_iqr(
    df: pd.DataFrame, x: str, y: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, med, lo, hi = [], [], [], []
    for xv, g in df.groupby(x, sort=True):
        v = _num(g[y])
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        xs.append(float(cast(float, xv)))
        med.append(float(np.median(v)))
        lo.append(float(np.quantile(v, 0.25)))
        hi.append(float(np.quantile(v, 0.75)))
    return np.array(xs), np.array(med), np.array(lo), np.array(hi)


def _xscale(ax: Any, xs: np.ndarray) -> None:
    if xs.size and np.all(xs > 0) and xs.max() / xs.min() >= 10:
        ax.set_xscale("log")


def _metric_label(metric: str) -> str:
    return metric.replace("_arcsec", " (arcsec)").replace("_au", " (AU)").replace("_", " ")


def _name(base: str, ctx: dict[str, Any], *keys: str) -> str:
    parts = [str(ctx[k]).replace(".", "_") for k in keys if ctx.get(k) is not None]
    return "__".join([base, *parts]) if parts else base


def error_vs_tycho_param(
    df: pd.DataFrame, out: Path, m: FigureManifest, ctx: dict[str, Any]
) -> None:
    d = _ok(df)
    x = _col(d, ctx["x"])
    metric = ctx.get("metric", LEADERBOARD_METRIC)
    scenarios = sorted(d["scenario"].unique())
    fig, axes = style.new_figure(7.0, 3.4 * len(scenarios), nrows=len(scenarios), squeeze=False)
    for ax, sc in zip(axes[:, 0], scenarios, strict=True):
        g = io.rows(d, d["scenario"] == sc)
        by = _col(g, ctx["by"]) if ctx.get("by") else None
        groups = [(None, g)] if by is None else list(g.groupby(by, sort=True))
        for k, (bv, gb) in enumerate(groups):
            for label, gl in gb.groupby("label", sort=True):
                xs, med, lo, hi = _median_iqr(gl, x, metric)
                if not xs.size:
                    continue
                sty = style.strategy_style(str(label))
                tag = str(label) + (f", {by.split('.')[-1]}={bv}" if by else "")
                ax.plot(
                    xs,
                    med,
                    marker="osD^v"[k % 5],
                    ms=3,
                    color=sty.color,
                    ls=sty.linestyle,
                    alpha=1.0 - 0.3 * k / max(len(groups) - 1, 1),
                    label=tag,
                )
                ax.fill_between(xs, lo, hi, color=sty.color, alpha=0.08, lw=0)
        if x.startswith("noise") and "sigma" in x:
            xs_all = np.sort(_num(g[x].unique()))
            ax.plot(xs_all, xs_all, color=style.NOISE_REF_COLOR, ls="--", lw=1, label="y = noise σ")
        ax.set_yscale("log")
        _xscale(ax, _num(g[x]))
        ax.set_xlabel(x)
        ax.set_ylabel(_metric_label(metric))
        ax.set_title(sc, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"{_metric_label(metric)} vs {x}", fontsize=10)
    style.save_figure(
        fig,
        out,
        _name("error_vs_tycho_param", ctx, "x", "by", "metric"),
        m,
        title=f"Error vs {x}",
        caption=f"{_metric_label(metric)} as a function of the swept Tycho parameter {x}; median over seeds with IQR band, one line per strategy.",
        context=ctx,
    )


def error_vs_jan_param(df: pd.DataFrame, out: Path, m: FigureManifest, ctx: dict[str, Any]) -> None:
    d = _ok(df)
    strat = ctx["strategy"]
    d = io.rows(d, d["strategy"] == strat)
    x = _col(d, ctx["x"])
    metric = ctx.get("metric", LEADERBOARD_METRIC)
    train_metric = ctx.get("train_metric", "train_fit_rms_arcsec")
    scenarios = sorted(d["scenario"].unique())
    fig, axes = style.new_figure(
        7.0, 3.4 * max(len(scenarios), 1), nrows=max(len(scenarios), 1), squeeze=False
    )
    sty = style.strategy_style(strat)
    for ax, sc in zip(axes[:, 0], scenarios, strict=False):
        g = io.rows(d, d["scenario"] == sc)
        for met, ls, lab in (
            (metric, "-", "holdout vs truth"),
            (train_metric, ":", "training fit vs data"),
        ):
            if met not in g.columns:
                continue
            xs, med, lo, hi = _median_iqr(g, x, met)
            if xs.size:
                ax.plot(
                    xs,
                    med,
                    color=sty.color,
                    ls=ls,
                    marker="o",
                    ms=3,
                    label=f"{lab}: {_metric_label(met)}",
                )
                ax.fill_between(xs, lo, hi, color=sty.color, alpha=0.12, lw=0)
        ax.set_yscale("log")
        _xscale(ax, _num(g[x]))
        ax.set_xlabel(f"{strat}: {x.removeprefix('jan.')}")
        ax.set_ylabel("error")
        ax.set_title(sc, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"{strat}: error vs {x.removeprefix('jan.')} (training vs holdout)", fontsize=10)
    style.save_figure(
        fig,
        out,
        _name("error_vs_jan_param", ctx, "strategy", "x"),
        m,
        title=f"{strat}: error vs {x.removeprefix('jan.')}",
        caption="Training-window vs holdout error as a Jan hyperparameter varies; a growing gap signals overfitting.",
        context=ctx,
    )


def heatmap_2d(df: pd.DataFrame, out: Path, m: FigureManifest, ctx: dict[str, Any]) -> None:
    d = _ok(df)
    x, y = _col(d, ctx["x"]), _col(d, ctx["y"])
    metric = ctx.get("metric", LEADERBOARD_METRIC)
    labels = sorted(d["label"].unique())
    xs = np.sort(_num(d[x].unique()))
    ys = np.sort(_num(d[y].unique()))
    grids = {}
    for lab in labels:
        g = io.rows(d, d["label"] == lab)
        z = np.full((len(ys), len(xs)), np.nan)
        for key, gg in g.groupby([x, y]):
            xv, yv = cast(tuple[Any, Any], key)
            v = _num(gg[metric])
            v = v[np.isfinite(v)]
            if v.size:
                z[
                    np.searchsorted(ys, float(cast(float, yv))),
                    np.searchsorted(xs, float(cast(float, xv))),
                ] = np.median(v)
        grids[lab] = z
    allv = np.concatenate([z[np.isfinite(z)] for z in grids.values()]) if grids else np.array([1.0])
    vmin, vmax = float(np.min(allv)), float(np.max(allv))
    n = len(labels) + 1
    fig, axes = style.new_figure(3.2 * n, 3.4, ncols=n, squeeze=False)
    from matplotlib.colors import LogNorm

    norm = LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, vmin * 1.01, 1e-6))
    im = None
    for ax, lab in zip(axes[0, :-1], labels, strict=True):
        im = ax.imshow(grids[lab], origin="lower", aspect="auto", norm=norm, cmap="viridis")
        ax.set_title(lab, fontsize=8, color=style.strategy_color(lab))
        ax.set_xticks(range(len(xs)), [f"{v:g}" for v in xs], fontsize=7, rotation=45)
        ax.set_yticks(range(len(ys)), [f"{v:g}" for v in ys], fontsize=7)
        ax.set_xlabel(x, fontsize=8)
        ax.set_ylabel(y, fontsize=8)
        ax.grid(False)
    if im is not None:
        fig.colorbar(im, ax=axes[0, :-1].tolist(), label=_metric_label(metric), shrink=0.85)
    stack = np.stack([grids[lab] for lab in labels])
    win = np.where(
        np.all(np.isnan(stack), 0),
        -1,
        np.nanargmin(np.where(np.isnan(stack), np.inf, stack), axis=0),
    )
    axw = axes[0, -1]
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#eeeeee"] + [style.strategy_color(lab) for lab in labels])
    axw.imshow(win + 1, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=len(labels))
    axw.set_title("best strategy", fontsize=8)
    axw.set_xticks(range(len(xs)), [f"{v:g}" for v in xs], fontsize=7, rotation=45)
    axw.set_yticks(range(len(ys)), [f"{v:g}" for v in ys], fontsize=7)
    axw.set_xlabel(x, fontsize=8)
    axw.grid(False)
    for lab in labels:
        axw.plot([], [], "s", color=style.strategy_color(lab), label=lab)
    axw.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"{_metric_label(metric)} over {x} × {y} (median over seeds)", fontsize=10)
    style.save_figure(
        fig,
        out,
        _name("heatmap_2d", ctx, "x", "y"),
        m,
        title=f"Heatmap {x} × {y}",
        caption="Median error over a 2D grid of swept parameters, one panel per strategy on a shared log colour scale, plus the winning strategy per cell.",
        context=ctx,
    )


_NOMINAL = (0.68, 0.95)


def calibration(df: pd.DataFrame, out: Path, m: FigureManifest, ctx: dict[str, Any]) -> None:
    d = _ok(df)
    labels = sorted(d["label"].unique())
    fig, (ax_c, ax_r) = style.new_figure(10.0, 4.2, ncols=2)
    ax_c.plot([0.5, 1], [0.5, 1], color=style.TRUTH_COLOR, lw=0.8, ls="--", label="ideal")
    any_rank = False
    for lab in labels:
        g = io.rows(d, d["label"] == lab)
        sty = style.strategy_style(lab)
        emp = []
        for tag in ("68", "95"):
            cols = [c for c in g.columns if c.startswith(f"in{tag}_")]
            vals = np.concatenate([_num(g[c]) for c in cols]) if cols else np.array([])
            vals = vals[np.isfinite(vals)]
            emp.append(float(np.mean(vals)) if vals.size else np.nan)
        if np.isfinite(emp).any():
            ax_c.plot(_NOMINAL, emp, marker="o", **sty.line_kwargs(f"{lab}: elements"))
        pc = [
            float(np.nanmean(_num(g[c])))
            if c in g.columns and np.isfinite(_num(g[c])).any()
            else np.nan
            for c in ("pred_cover68", "pred_cover95")
        ]
        if np.isfinite(pc).any():
            ax_c.plot(
                _NOMINAL, pc, marker="s", color=sty.color, ls=":", label=f"{lab}: predictions"
            )
        rank_cols = [c for c in g.columns if c.startswith("rank_")]
        ranks = np.concatenate([_num(g[c]) for c in rank_cols]) if rank_cols else np.array([])
        ranks = ranks[np.isfinite(ranks)]
        if ranks.size:
            any_rank = True
            ax_r.hist(
                ranks,
                bins=10,
                range=(0, 1),
                histtype="step",
                color=sty.color,
                lw=1.4,
                density=True,
                label=f"{lab} (n={ranks.size})",
            )
    ax_c.set_xlabel("nominal credible level")
    ax_c.set_ylabel("empirical coverage")
    ax_c.set_xlim(0.6, 1.0)
    ax_c.set_ylim(0.0, 1.02)
    ax_c.legend(fontsize=6)
    ax_c.set_title("coverage across seeds", fontsize=9)
    if any_rank:
        ax_r.axhline(1.0, color=style.TRUTH_COLOR, ls="--", lw=0.8)
        ax_r.set_xlabel("rank of truth within posterior (fraction)")
        ax_r.set_ylabel("density")
        ax_r.legend(fontsize=6)
        ax_r.set_title("SBC rank histogram (all elements pooled; flat = calibrated)", fontsize=9)
    else:
        ax_r.text(
            0.5, 0.5, "no posterior samples", ha="center", transform=ax_r.transAxes, color="0.4"
        )
    fig.suptitle("Calibration", fontsize=10)
    style.save_figure(
        fig,
        out,
        "calibration",
        m,
        title="Calibration",
        caption="Left: empirical coverage of credible intervals (elements) and predictive ellipses vs nominal level, across seeds. Right: simulation-based-calibration rank histograms of the truth within each posterior.",
        context=ctx,
    )


def model_selection_vs_noise(
    df: pd.DataFrame, out: Path, m: FigureManifest, ctx: dict[str, Any]
) -> None:
    d = _ok(df)
    if "p_true_family" not in d.columns:
        return
    d = io.rows(d, d["p_true_family"].notna())
    if d.empty:
        return
    x = _col(d, ctx.get("x", "noise.0.sigma_arcsec"))
    fam_cols = [c for c in d.columns if c.startswith("p_") and c not in ("p_true_family",)]
    fig, (ax1, ax2) = style.new_figure(10.0, 4.0, ncols=2)
    by = _col(d, ctx["by"]) if ctx.get("by") else None
    groups = [(None, d)] if by is None else list(d.groupby(by, sort=True))
    markers = ["o", "s", "^", "D", "v", "P"]
    for k, (bv, gb) in enumerate(groups):
        for lab, g in gb.groupby("label", sort=True):
            xs, med, lo, hi = _median_iqr(g, x, "p_true_family")
            sty = style.strategy_style(str(lab))
            tag = f"{lab}" + (f", {by.split('.')[-1]}={bv}" if by else "")
            ax1.plot(
                xs,
                med,
                marker=markers[k % len(markers)],
                color=sty.color,
                alpha=1.0 - 0.25 * k / max(len(groups) - 1, 1),
                label=tag,
            )
            ax1.fill_between(xs, lo, hi, color=sty.color, alpha=0.08, lw=0)
    ax1.axhline(0.95, color=style.NOISE_REF_COLOR, ls="--", lw=0.8)
    ax1.set_ylim(-0.02, 1.02)
    _xscale(ax1, _num(d[x]))
    ax1.set_xlabel(x)
    ax1.set_ylabel("posterior probability of the true family")
    ax1.legend(fontsize=7)
    ax1.set_title("median and IQR over seeds", fontsize=9)
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    # Show every family for the group where the true family is least certain.
    focus = min(groups, key=lambda gv: float(np.nanmin(_num(gv[1]["p_true_family"]))))[0]
    d2 = d if by is None else io.rows(d, d[by] == focus)
    for k, c in enumerate(sorted(fam_cols)):
        xs, med, _, _ = _median_iqr(d2, x, c)
        if xs.size and np.nanmax(med) > 1e-3:
            ax2.plot(
                xs, med, marker="o", color=palette[k % len(palette)], label=c.removeprefix("p_")
            )
    _xscale(ax2, _num(d[x]))
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel(x)
    ax2.set_ylabel("median posterior probability")
    ax2.legend(fontsize=7)
    ax2.set_title("every family" + (f" ({by.split('.')[-1]}={focus})" if by else ""), fontsize=9)
    truth = ", ".join(sorted(str(v) for v in d["true_family"].dropna().unique()))
    fig.suptitle(f"Model selection vs {x} (true family: {truth})", fontsize=10)
    style.save_figure(
        fig,
        out,
        _name("model_selection_vs_noise", ctx, "x", "by"),
        m,
        title="Model selection vs noise",
        caption="Posterior probability assigned to the true model family as the swept parameter grows (the “eight arcminutes” curve); right: every family's median probability.",
        context=ctx,
    )


def accuracy_vs_runtime(
    df: pd.DataFrame, out: Path, m: FigureManifest, ctx: dict[str, Any]
) -> None:
    d = _ok(df)
    metric = ctx.get("metric", LEADERBOARD_METRIC)
    fig, ax = style.new_figure(6.5, 4.5)
    for lab, g in d.groupby("label", sort=True):
        sty = style.strategy_style(str(lab))
        rt, er = _num(g["runtime_s"]), _num(g[metric])
        ok = np.isfinite(rt) & np.isfinite(er) & (er > 0)
        ax.scatter(rt[ok], er[ok], s=10, color=sty.color, alpha=0.35, lw=0)
        if ok.any():
            ax.scatter(
                [np.median(rt[ok])],
                [np.median(er[ok])],
                s=70,
                color=sty.color,
                edgecolor="k",
                label=str(lab),
                zorder=3,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall-clock fit time (s)")
    ax.set_ylabel(_metric_label(metric))
    ax.legend(fontsize=7)
    ax.set_title("Accuracy vs runtime (cells faint, medians bold)", fontsize=10)
    style.save_figure(
        fig,
        out,
        "accuracy_vs_runtime",
        m,
        title="Accuracy vs runtime",
        caption="Error vs fit time per strategy: the cost/quality Pareto view.",
        context=ctx,
    )


_VERDICTS = ["exact", "up_to_constants", "field_matches", "black_box", "mismatch", "no law"]
_VERDICT_COLORS = {
    "exact": "#1b9e77",
    "up_to_constants": "#66c2a5",
    "field_matches": "#a6d854",
    "black_box": "#8da0cb",
    "mismatch": "#d95f02",
    "no law": "#cccccc",
}


def _verdict(row: pd.Series) -> str:
    """Symbolic verdict, or — when the truth has no radial expression (e.g. Newton plus a
    hidden perturber) — whether the recovered field / perturber was judged correct."""
    sym = row.get("symbolic")
    if isinstance(sym, str) and sym != "black_box":
        return sym
    answer = row.get("answer")
    if answer == "correct law":
        return "field_matches"
    if answer == "wrong law":
        return "black_box" if sym == "black_box" else "mismatch"
    return "no law"


def structural_success(df: pd.DataFrame, out: Path, m: FigureManifest, ctx: dict[str, Any]) -> None:
    from matplotlib.patches import Patch

    d = _ok(df)
    if "symbolic" not in d.columns:
        return
    d = io.rows(d, d["truth_kind"] == "dynamical") if "truth_kind" in d.columns else d
    scenarios = sorted(d["scenario"].unique())
    labels = sorted(d["label"].unique())
    fig, axes = style.new_figure(
        3.2 * max(len(scenarios), 1),
        0.45 * len(labels) + 1.5,
        ncols=max(len(scenarios), 1),
        squeeze=False,
        sharey=True,
    )
    for ax, sc in zip(axes[0], scenarios, strict=False):
        g = io.rows(d, d["scenario"] == sc)
        for y, lab in enumerate(labels):
            gl = io.rows(g, g["label"] == lab)
            if gl.empty:
                continue
            verdict = gl.apply(_verdict, axis=1)
            left = 0.0
            for v in _VERDICTS:
                frac = float((verdict == v).mean())
                if frac:
                    ax.barh(y, frac, left=left, color=_VERDICT_COLORS[v], edgecolor="white")
                    left += frac
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_title(sc, fontsize=9)
        ax.set_xlabel("fraction of seeds")
    handles = [Patch(color=c, label=v.replace("_", " ")) for v, c in _VERDICT_COLORS.items()]
    fig.legend(
        handles=handles, fontsize=7, loc="outside lower center", ncol=len(handles), frameon=False
    )
    fig.suptitle("Structural success: is the recovered law the true one?", fontsize=10)
    style.save_figure(
        fig,
        out,
        "structural_success",
        m,
        title="Structural success",
        caption="Fraction of seeds where each strategy recovered the true law exactly, up to constants, or got it wrong (dynamical truths). When the truth has no radial law (Newton plus a hidden perturber) the verdict comes from the recovered field or, for a claimed perturber, its localization (field matches). Black box: a law without a formula whose field misses the truth.",
        context=ctx,
    )


_ANSWERS = [
    "correct law",
    "right form",
    "wrong law",
    "spurious law",
    "kinematic",
    "rejected",
    "nonparametric",
]
_ANSWER_COLORS = ["#1b9e77", "#a6d854", "#d95f02", "#e7298a", "#7570b3", "#66a61e", "#bdbdbd"]


def rejection_matrix(df: pd.DataFrame, out: Path, m: FigureManifest, ctx: dict[str, Any]) -> None:
    d = _ok(df)
    if "answer" not in d.columns:
        return
    rows_key = "truth_family" if "truth_family" in d.columns else "scenario"
    truths = sorted(str(v) for v in d[rows_key].dropna().unique())
    labels = sorted(d["label"].unique())
    from matplotlib.colors import ListedColormap

    grid = np.full((len(truths), len(labels)), np.nan)
    text = [["" for _ in labels] for _ in truths]
    for i, tfam in enumerate(truths):
        for j, lab in enumerate(labels):
            g = io.rows(d, (d[rows_key] == tfam) & (d["label"] == lab))
            if g.empty:
                continue
            counts = g["answer"].value_counts()
            top = str(counts.index[0])
            grid[i, j] = _ANSWERS.index(top) if top in _ANSWERS else np.nan
            text[i][j] = f"{top}\n{counts.iloc[0]}/{len(g)}"
    fig, ax = style.new_figure(1.3 * len(labels) + 3, 0.6 * len(truths) + 2)
    ax.imshow(
        grid,
        cmap=ListedColormap(_ANSWER_COLORS),
        vmin=-0.5,
        vmax=len(_ANSWERS) - 0.5,
        aspect="auto",
    )
    for i in range(len(truths)):
        for j in range(len(labels)):
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=6, color="white")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(truths)), truths, fontsize=7)
    ax.grid(False)
    for c, a in zip(_ANSWER_COLORS, _ANSWERS, strict=True):
        ax.plot([], [], "s", color=c, label=a)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("What kind of answer each strategy gave, by truth family", fontsize=10)
    style.save_figure(
        fig,
        out,
        "rejection_matrix",
        m,
        title="Rejection matrix",
        caption="For every truth family (dynamical laws and kinematic paths) and strategy: the most common kind of answer — correct law (explains at least half of the truth's departure from Newton, constants included), right form (correct expression, wrong constants), wrong law, a spurious law for a kinematic truth, an honest kinematic/rejected answer, or a nonparametric fit.",
        context=ctx,
    )


_EFFECT_NAMES = {"light_time": "light-time", "aberration": "aberration", "frame": "frame"}


def _effects(row: pd.Series, cols: list[str]) -> str:
    on = []
    for c in cols:
        v, key = row[c], c.split(".")[-1]
        if key == "frame":
            if str(v) != "ecliptic":
                on.append(str(v))
        elif bool(v):
            on.append(_EFFECT_NAMES.get(key, key))
    return " + ".join(on) if on else "none"


def realism_bias(df: pd.DataFrame, out: Path, m: FigureManifest, ctx: dict[str, Any]) -> None:
    """Holdout error per set of realism effects in the data, modelled vs ignored."""
    d = _ok(df)
    cols = [c for c in d.columns if c.startswith("realism.")]
    if not cols:
        return
    metric = ctx.get("metric", LEADERBOARD_METRIC)
    d = d.assign(_effects=[_effects(r, cols) for _, r in d.iterrows()])
    if "jan.realism" not in d.columns:
        d = d.assign(**{"jan.realism": "model"})
    d["jan.realism"] = d["jan.realism"].fillna("model")
    groups = sorted(d["_effects"].unique(), key=lambda g: (g != "none", g.count("+"), g))
    bars = sorted({(str(s), str(r)) for s, r in zip(d["strategy"], d["jan.realism"], strict=True)})
    fig, ax = style.new_figure(max(7.0, 1.1 * len(groups) * len(bars) / 2 + 2), 4.2)
    width = 0.8 / len(bars)
    for k, (strat, mode) in enumerate(bars):
        g = io.rows(d, (d["strategy"] == strat) & (d["jan.realism"] == mode))
        med, lo, hi = [], [], []
        for grp in groups:
            v = _num(g.loc[g["_effects"] == grp, metric])
            v = v[np.isfinite(v)]
            med.append(float(np.median(v)) if v.size else np.nan)
            lo.append(float(np.quantile(v, 0.25)) if v.size else np.nan)
            hi.append(float(np.quantile(v, 0.75)) if v.size else np.nan)
        x = np.arange(len(groups)) + (k - (len(bars) - 1) / 2) * width
        md = np.array(med)
        ax.bar(
            x,
            md,
            width * 0.92,
            color=style.strategy_color(strat),
            alpha=1.0 if mode == "model" else 0.45,
            hatch=None if mode == "model" else "//",
            edgecolor="white" if mode == "model" else style.strategy_color(strat),
            label=f"{strat} (effects {'modelled' if mode == 'model' else 'ignored'})",
        )
        ax.errorbar(
            x, md, yerr=[md - np.array(lo), np.array(hi) - md], fmt="none", color="0.3", lw=0.8
        )
    ax.set_yscale("log")
    ax.set_xticks(range(len(groups)), groups, fontsize=8)
    ax.set_xlabel("effects in the published directions")
    ax.set_ylabel(_metric_label(metric))
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Holdout error with realism effects modelled or ignored", fontsize=10)
    style.save_figure(
        fig,
        out,
        _name("realism_bias", ctx, "metric"),
        m,
        title="Realism bias",
        caption=f"Median {_metric_label(metric)} over seeds (IQR bars) for each set of effects in Tycho's published directions. Solid bars: Jan reduces the data (frame rotation, aberration with the observer's velocity, light-time by two-pass differential correction) and predicts apparent directions; hatched bars: the same strategy ignoring light-time and aberration (the frame is always converted).",
        context=ctx,
    )


COMPARISON_FIGURES: dict[str, Callable[..., None]] = {
    "leaderboard": leaderboard,
    "extrapolation_curves": extrapolation_curves,
    "error_vs_tycho_param": error_vs_tycho_param,
    "error_vs_jan_param": error_vs_jan_param,
    "heatmap_2d": heatmap_2d,
    "calibration": calibration,
    "model_selection_vs_noise": model_selection_vs_noise,
    "accuracy_vs_runtime": accuracy_vs_runtime,
    "structural_success": structural_success,
    "rejection_matrix": rejection_matrix,
    "realism_bias": realism_bias,
}


def render_comparisons(
    df: pd.DataFrame,
    out_dir: str | Path,
    requests: list[dict[str, Any]],
    manifest: FigureManifest,
    results_dir: str | Path | None = None,
) -> list[str]:
    """Render requested comparison figures; returns names of requests not yet supported."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []
    names = [r.get("figure", "") for r in requests]
    for always in ("leaderboard", "extrapolation_curves"):
        if always not in names:
            requests = [{"figure": always}, *requests]
    from rudolf.viz.sweep_animations import CROSS_RUN_ANIMATIONS

    for req in requests:
        name = str(req.get("figure"))
        ctx = {k: v for k, v in req.items() if k != "figure"}
        if name in CROSS_RUN_ANIMATIONS:
            if results_dir is None:
                skipped.append(name)
                continue
            CROSS_RUN_ANIMATIONS[name](df, out, manifest, ctx, Path(results_dir))
            continue
        fn = COMPARISON_FIGURES.get(name)
        if fn is None:
            skipped.append(name)
            continue
        try:
            fn(df, out, manifest, ctx)
        except KeyError as exc:  # a requested column is absent from these results
            skipped.append(f"{name} ({exc})")
    return skipped
