"""Phase 4 per-run figures and animation (SPEC §7.1.2–7.1.3): force laws, acceleration
fields, symbolic Pareto fronts, conserved quantities, law discovery."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from jt_core import io
from rudolf.data import RunView
from rudolf.metrics.structural import _eval, _radial_fn
from rudolf.viz import style
from rudolf.viz.manifest import FigureManifest
from rudolf.viz.run_plots import TimeAxis, _no_data, _suffix, _title, context
from rudolf.viz.writers import AnimationSettings, save_animation

FloatArray = NDArray[np.float64]


def _radial_points(df: Any) -> tuple[FloatArray, FloatArray]:
    """(r, f = a · r̂) from a field table."""
    x = io.cols(df, ["x", "y", "z"]).to_numpy()
    a = io.cols(df, ["ax", "ay", "az"]).to_numpy()
    r = np.linalg.norm(x, axis=1)
    return r, np.sum(a * x, axis=1) / r


def _local_exponent(r: FloatArray, f: FloatArray) -> FloatArray:
    """n(r) = −d ln|f| / d ln r (n = 2 for Newton)."""
    with np.errstate(all="ignore"):
        return -np.gradient(np.log(np.abs(f)), np.log(r))


def _binned_exponent(r: FloatArray, f: FloatArray, nbins: int = 8) -> tuple[FloatArray, FloatArray]:
    """Local log-log slopes of scattered field samples, in bins of r."""
    ok = np.isfinite(f) & (np.abs(f) > 0)
    r, f = r[ok], f[ok]
    if len(r) < 4 * nbins or np.ptp(np.log(r)) < 1e-6:
        return np.array([]), np.array([])
    edges = np.quantile(r, np.linspace(0, 1, nbins + 1))
    rc, nc = [], []
    for a, b in itertools.pairwise(edges):
        sel = (r >= a) & (r <= b)
        if sel.sum() >= 4 and np.ptp(np.log(r[sel])) > 1e-9:
            slope = np.polyfit(np.log(r[sel]), np.log(np.abs(f[sel])), 1)[0]
            rc.append(float(np.exp(np.mean(np.log(r[sel])))))
            nc.append(-float(slope))
    return np.array(rc), np.array(nc)


def force_law(view: RunView, out: Path, m: FigureManifest) -> None:
    s, sty = view.sealed, style.strategy_style(view.jan.label)
    fig, (ax, axn) = style.new_figure(12.0, 4.5, ncols=2)
    truth = s.body_truth(view.body_id)
    t_expr = _radial_fn(truth.get("law"))
    near = s.field_for(view.body_id, "near")
    r_vis = np.linalg.norm(s.dense_positions(view.body_id), axis=1)
    lo, hi = 0.6 * r_vis.min(), 1.5 * r_vis.max()
    r_plot = np.geomspace(lo, hi, 300)
    notes = []
    if t_expr is not None:
        ft = _eval(t_expr, r_plot)
        ax.plot(r_plot, np.abs(ft), **style.TRUTH_LINE)
        axn.plot(r_plot, _local_exponent(r_plot, ft), **style.TRUTH_LINE)
    elif near is not None and not bool(near["ax"].isna().all()):
        r, f = _radial_points(near)
        ax.scatter(r, np.abs(f), s=4, color=style.TRUTH_COLOR, label="truth (field samples)")
        rb, nb = _binned_exponent(r, f)
        axn.plot(rb, nb, "o-", ms=3, color=style.TRUTH_COLOR, label="truth (field samples)")
    else:
        notes.append("the truth has no force law (kinematic path)")
    r_expr = _radial_fn(view.jan.law())
    if r_expr is not None:
        try:
            fr = _eval(r_expr, r_plot)
            lab = f"{view.jan.label}: {view.jan.law()['radial']}"  # type: ignore[index]
            ax.plot(r_plot, np.abs(fr), **sty.line_kwargs(lab))
            axn.plot(r_plot, _local_exponent(r_plot, fr), **sty.line_kwargs(lab))
        except (TypeError, ValueError):
            notes.append("recovered law not evaluable")
    jf = view.jan.field_for(view.body_id, "near")
    if jf is not None:
        r, f = _radial_points(jf)
        ax.scatter(r, np.abs(f), s=5, color=sty.color, alpha=0.5, label=f"{view.jan.label} field")
        rb, nb = _binned_exponent(r, f)
        if len(rb):
            axn.scatter(rb, nb, s=14, color=sty.color, alpha=0.7, label=f"{view.jan.label} field")
    if r_expr is None and jf is None:
        notes.append("the hypothesis has no force law")
    for a in (ax, axn):
        a.axvspan(
            r_vis.min(),
            r_vis.max(),
            color=style.HOLDOUT_SHADE,
            alpha=0.07,
            lw=0,
            label="radii visited",
        )
        a.set_xscale("log")
        a.set_xlabel("r (AU)")
    ax.set_yscale("log")
    ax.set_ylabel("|radial acceleration| (AU/day²)")
    axn.set_ylabel("local exponent n(r) = −d ln|f| / d ln r")
    axn.axhline(2.0, color="0.6", lw=0.8, ls=":", label="Newton (n = 2)")
    vals = [ln.get_ydata() for ln in axn.get_lines()]
    finite = (
        np.concatenate(
            [np.asarray(v, dtype=float)[np.isfinite(np.asarray(v, dtype=float))] for v in vals]
        )
        if vals
        else np.array([])
    )
    if finite.size:
        lo_n, hi_n = np.percentile(finite, [1, 99])
        pad = max(0.05, 0.15 * (hi_n - lo_n))
        axn.set_ylim(min(lo_n, 2.0) - pad, max(hi_n, 2.0) + pad)
    ax.legend(fontsize=7)
    axn.legend(fontsize=7)
    ax.set_title(_title(view, "Force law"), fontsize=10)
    axn.set_title("Local power-law exponent", fontsize=10)
    cap = "Left: radial acceleration vs heliocentric distance — true law, recovered law (curve) and the recovered field sampled near the orbit (points). Right: the local exponent n(r) = −d ln|f|/d ln r, which separates laws (2 for Newton, 2.05 for the precessing power law, −1 for Hooke) far more sharply than the log-log curve. Shaded: radii the orbit visits."
    if notes:
        cap += " Note: " + "; ".join(notes) + "."
    style.save_figure(
        fig,
        out,
        "force_law" + _suffix(view),
        m,
        title="Force law",
        caption=cap,
        context=context(view),
    )


def _plane(grid: FloatArray) -> tuple[FloatArray, FloatArray]:
    _, _, vt = np.linalg.svd(grid - grid.mean(axis=0), full_matrices=False)
    return vt[0], vt[1]


def acceleration_field(view: RunView, out: Path, m: FigureManifest) -> None:
    s, sty = view.sealed, style.strategy_style(view.jan.label)
    tg = s.field_for(view.body_id, "grid")
    jg = view.jan.field_for(view.body_id, "grid")
    fig, axes = style.new_figure(12.0, 4.4, ncols=3)
    if tg is None or jg is None or bool(tg["ax"].isna().all()):
        for ax in axes:
            _no_data(ax, "no force law (truth or hypothesis)")
        cap = "Omitted: needs both a true and a recovered acceleration field."
    else:
        x = io.cols(tg, ["x", "y", "z"]).to_numpy()
        at = io.cols(tg, ["ax", "ay", "az"]).to_numpy()
        aj = io.cols(jg, ["ax", "ay", "az"]).to_numpy()
        e1, e2 = _plane(x)
        u, v = x @ e1, x @ e2
        tr = view.dense_true_r
        for ax, a, lab, col in (
            (axes[0], at, "truth", style.TRUTH_COLOR),
            (axes[1], aj, view.jan.label, sty.color),
        ):
            au, av = a @ e1, a @ e2
            mag = np.hypot(au, av)
            ax.quiver(
                u, v, au / mag, av / mag, np.log10(mag), cmap="viridis", scale=30, width=0.004
            )
            ax.plot(tr @ e1, tr @ e2, color=style.TRUTH_COLOR, lw=0.8, alpha=0.6)
            ax.scatter([0], [0], s=40, color=style.SUN_COLOR)
            ax.set_aspect("equal")
            ax.set_title(f"{lab} (direction; colour = log |a|)", fontsize=9, color=col)
        rel = np.linalg.norm(aj - at, axis=1) / np.linalg.norm(at, axis=1)
        sc = axes[2].scatter(
            u, v, c=np.log10(np.maximum(rel, 1e-12)), cmap="magma", s=30, marker="s"
        )
        axes[2].plot(tr @ e1, tr @ e2, color="white", lw=0.8)
        axes[2].set_aspect("equal")
        fig.colorbar(sc, ax=axes[2], label="log₁₀ relative error")
        axes[2].set_title("relative field error", fontsize=9)
        cap = "Acceleration field in the orbital plane: truth vs recovered (arrows), and the relative error map; the orbit is overlaid."
    fig.suptitle(_title(view, "Acceleration field"), fontsize=10)
    style.save_figure(
        fig,
        out,
        "acceleration_field" + _suffix(view),
        m,
        title="Acceleration field",
        caption=cap,
        context=context(view),
    )


def symbolic_pareto(view: RunView, out: Path, m: FigureManifest) -> None:
    law = view.jan.law() or {}
    pareto = (law.get("extra") or {}).get("pareto")
    fig, ax = style.new_figure(8.0, 4.5)
    if not pareto:
        _no_data(ax, "no symbolic-regression Pareto front")
        cap = "Omitted: the strategy reports no Pareto front."
    else:
        sty = style.strategy_style(view.jan.label)
        c = np.array([p["complexity"] for p in pareto])
        loss = np.array([p["loss"] for p in pareto])
        ax.plot(c, loss, marker="o", color=sty.color)
        sel = law.get("radial")
        for p in pareto:
            is_sel = p["expression"] == sel
            ax.annotate(
                p["expression"][:40],
                (p["complexity"], p["loss"]),
                fontsize=6,
                xytext=(4, 4),
                textcoords="offset points",
                color=style.HOLDOUT_OBS_COLOR if is_sel else "0.3",
                weight="bold" if is_sel else "normal",
            )
        t_law = view.sealed.body_truth(view.body_id).get("law") or {}
        ax.set_yscale("log")
        ax.set_xlabel("expression complexity")
        ax.set_ylabel("loss (AU²/day⁴)")
        ax.set_title(f"selected: {sel}   ·   truth: {t_law.get('radial')}", fontsize=8)
        cap = "Symbolic regression Pareto front (complexity vs loss); the selected expression is highlighted."
    fig.suptitle(_title(view, "Symbolic Pareto front"), fontsize=10)
    style.save_figure(
        fig,
        out,
        "symbolic_pareto" + _suffix(view),
        m,
        title="Symbolic Pareto front",
        caption=cap,
        context=context(view),
    )


def conserved_quantities(view: RunView, out: Path, m: FigureManifest) -> None:
    """Along the TRUE trajectory, integrate the recovered law's work and torque: a correct
    conservative central law keeps E and h constant; wrong laws drift."""
    s, sty = view.sealed, style.strategy_style(view.jan.label)
    tp = s.field_for(view.body_id, "path")
    jp = view.jan.field_for(view.body_id, "path")
    fig, (ax_e, ax_h) = style.new_figure(10.0, 4.0, ncols=2)
    ta = TimeAxis(s.duration)
    if tp is None:
        _no_data(ax_e, "no path samples")
        _no_data(ax_h, "")
        cap = "Omitted: no trajectory samples."
    else:
        t = tp["t"].to_numpy()
        x = io.cols(tp, ["x", "y", "z"]).to_numpy()
        v = io.cols(tp, ["vx", "vy", "vz"]).to_numpy()
        h0 = np.linalg.norm(np.cross(x[0], v[0]))

        def drift(a: FloatArray) -> tuple[FloatArray, FloatArray]:
            power = np.sum(a * v, axis=1)
            work = np.concatenate([[0], np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(t))])
            ke = 0.5 * np.sum(v * v, axis=1)
            e = ke - work  # = KE + potential for a conservative field
            torque = np.cross(x, a)
            h_int = np.concatenate(
                [
                    np.zeros((1, 3)),
                    np.cumsum(0.5 * (torque[1:] + torque[:-1]) * np.diff(t)[:, None], axis=0),
                ]
            )
            h_pred = np.cross(x[0], v[0]) + h_int
            h_true = np.cross(x, v)
            return (e - e[0]) / max(abs(ke[0]), 1e-30), np.linalg.norm(h_pred - h_true, axis=1) / h0

        if not bool(tp["ax"].isna().all()):
            de, dh = drift(io.cols(tp, ["ax", "ay", "az"]).to_numpy())
            ax_e.plot(ta(t), np.abs(de) + 1e-16, **style.TRUTH_LINE)
            ax_h.plot(ta(t), dh + 1e-16, **style.TRUTH_LINE)
        if jp is not None:
            de, dh = drift(io.cols(jp, ["ax", "ay", "az"]).to_numpy())
            ax_e.plot(ta(t), np.abs(de) + 1e-16, **sty.line_kwargs(view.jan.label))
            ax_h.plot(ta(t), dh + 1e-16, **sty.line_kwargs(view.jan.label))
        for ax in (ax_e, ax_h):
            ax.set_yscale("log")
            ax.set_xlabel(ta.label)
            style.mark_holdout(ax, s.t_split, s.duration, ta.scale)
            ax.legend(fontsize=7)
        ax_e.set_ylabel("|ΔE| / KE₀ along the true path")
        ax_h.set_ylabel("|h_law − h_true| / |h₀|")
        ax_e.set_title("energy implied by the law", fontsize=9)
        ax_h.set_title("angular momentum implied by the law", fontsize=9)
        cap = (
            "Energy and angular momentum along the true trajectory, evaluated with each law's acceleration "
            "(work and torque integrated along the path): the true law keeps them constant up to quadrature "
            "error; a wrong or non-central law drifts."
        )
        if jp is None:
            cap += " Omitted: the hypothesis has no force law."
    fig.suptitle(_title(view, "Conserved quantities"), fontsize=10)
    style.save_figure(
        fig,
        out,
        "conserved_quantities" + _suffix(view),
        m,
        title="Conserved quantities",
        caption=cap,
        context=context(view),
    )


def law_discovery(view: RunView, out: Path, m: FigureManifest, settings: AnimationSettings) -> None:
    """The recovered force law morphing across a discovery fit's trace (truth fixed)."""
    tf = view.jan.trace_field_df
    if tf is None:
        return
    tf = io.rows(tf, tf["body_id"] == view.body_id)
    if tf.empty:
        return
    frames = sorted(int(f) for f in tf["frame"].unique())
    labels = view.jan.hypothesis.get("trace_labels") or []
    s, sty = view.sealed, style.strategy_style(view.jan.label)
    style.apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t_expr = _radial_fn(s.body_truth(view.body_id).get("law"))
    r_vis = np.linalg.norm(s.dense_positions(view.body_id), axis=1)
    r_plot = np.geomspace(0.7 * r_vis.min(), 1.4 * r_vis.max(), 200)
    all_f = []
    per_frame = []
    for f in frames:
        r, fr = _radial_points(io.rows(tf, tf["frame"] == f))
        order = np.argsort(r)
        per_frame.append((r[order], fr[order]))
        all_f.append(np.abs(fr))
    if t_expr is not None:
        ft = np.abs(_eval(t_expr, r_plot))
        ax.plot(r_plot, ft, **style.TRUTH_LINE)
        all_f.append(ft)
    vals = np.concatenate(all_f)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size:
        ax.set_ylim(vals.min() / 3, vals.max() * 3)
    ax.axvspan(r_vis.min(), r_vis.max(), color=style.HOLDOUT_SHADE, alpha=0.07, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(r_plot[0], r_plot[-1])
    ax.set_xlabel("r (AU)")
    ax.set_ylabel("|radial acceleration| (AU/day²)")
    pts = ax.scatter([], [], s=8, color=sty.color, label=view.jan.label)
    label = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"Law discovery — {view.jan.label} on {s.scenario_name}", fontsize=10)

    def update(k: int) -> tuple[Any, ...]:
        r, fr = per_frame[k]
        pts.set_offsets(np.c_[r, np.abs(fr)])
        f = frames[k]
        label.set_text(labels[f] if f < len(labels) else f"frame {f}")
        return pts, label

    settings = AnimationSettings(
        fps=min(settings.fps, 8), max_seconds=settings.max_seconds, quick=settings.quick
    )
    idx = np.unique(
        np.linspace(0, len(frames) - 1, settings.frames(len(frames))).round().astype(int)
    )
    anim = FuncAnimation(fig, update, frames=idx, blit=False)
    path = save_animation(anim, out / ("law_discovery" + _suffix(view)), settings)
    m.add(
        name="law_discovery" + _suffix(view),
        kind="animation",
        paths=[path],
        title="Law discovery",
        caption="The recovered force law (field sampled near the orbit) evolving across the fit, against the true law.",
        context=context(view),
    )


PHASE4_FIGURES = [force_law, acceleration_field, symbolic_pareto, conserved_quantities]
PHASE4_ANIMATIONS = [law_discovery]
