"""Phase 3 per-run figures (SPEC §7.1.2–7.1.3): noise diagnostics and Kepler's
triangulation (static geometry and step-by-step animation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray
from scipy import stats

from jt_core import io
from jt_core.angles import tangent_residuals
from jt_core.constants import ARCSEC
from jt_core.frames import lonlat_to_unit
from rudolf.data import RunView
from rudolf.viz import style
from rudolf.viz.manifest import FigureManifest
from rudolf.viz.run_plots import TimeAxis, _no_data, _suffix, _title, context
from rudolf.viz.writers import AnimationSettings, save_animation

FloatArray = NDArray[np.float64]

OUTLIER_SIGMAS = 4.0
"""|residual| / σ̂ above which an observation counts as detected as an outlier."""


def _train_residuals(view: RunView) -> tuple[FloatArray, FloatArray, FloatArray] | None:
    """(t, tangent residuals (N, 2) in arcsec of noisy obs vs prediction, obs_id)."""
    pred = view.jan.sky("train", view.body_id)
    pub = view.sealed.public_body(view.body_id)
    if pred is None or len(pred) != len(pub):
        return None
    u_obs = lonlat_to_unit(pub["lon"].to_numpy(), pub["lat"].to_numpy())
    r = tangent_residuals(u_obs, pred) / ARCSEC
    return pub["t"].to_numpy(), r, pub["obs_id"].to_numpy()


def noise_diagnostics(view: RunView, out: Path, m: FigureManifest) -> None:
    s = view.sealed
    sty = style.strategy_style(view.jan.label)
    res = _train_residuals(view) if s.is_sky else None
    sig = view.jan.obs_sigma(view.body_id)
    fig, axes = style.new_figure(12.0, 3.8, ncols=3, gridspec_kw={"width_ratios": [1, 1, 1.6]})
    if res is None:
        for ax in axes:
            _no_data(ax, "no predictions at the training times")
        cap = "Omitted: no predictions at the training observation times."
    else:
        t, r, obs_id = res
        sig_as = sig / ARCSEC if sig else (s.noise_sigma_arcsec or float(np.std(r)))
        z = r / sig_as
        cfg = view.jan.hypothesis.get("config") or {}
        robust = cfg.get("likelihood") == "student_t" or cfg.get("loss", "linear") != "linear"
        nu = float(cfg.get("nu", 1.0 if cfg.get("loss") == "cauchy" else 4.0))
        ax = axes[0]
        bins = np.linspace(-8, 8, 81)
        ax.hist(
            np.clip(z.ravel(), -8, 8),
            bins=bins,
            density=True,
            color=sty.color,
            alpha=0.6,
            label="residuals / σ̂",
        )
        xs = np.linspace(-8, 8, 400)
        ax.plot(xs, stats.norm.pdf(xs), color=style.TRUTH_COLOR, lw=1, label="N(0, 1)")
        if robust:
            ax.plot(
                xs,
                stats.t.pdf(xs, df=nu),
                color=style.TRUTH_COLOR,
                ls="--",
                lw=1,
                label=f"Student-t (ν={nu:g})",
            )
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1)
        ax.set_xlabel("normalized residual (per axis)")
        ax.legend(fontsize=7)
        ax.set_title("residuals vs the assumed likelihood", fontsize=9)
        ax = axes[1]
        osm, osr = stats.probplot(z.ravel(), dist="norm", fit=False)
        ax.scatter(osm, osr, s=4, color=sty.color)
        lim = max(4.0, float(np.max(np.abs(osm))))
        ax.plot([-lim, lim], [-lim, lim], color=style.TRUTH_COLOR, lw=0.8)
        ax.set_ylim(-max(lim, 8), max(lim, 8))
        ax.set_xlabel("normal quantile")
        ax.set_ylabel("sample quantile")
        ax.set_title("Q–Q plot", fontsize=9)
        ax = axes[2]
        ta = TimeAxis(s.duration)
        mag = np.linalg.norm(z, axis=1)
        nz = s.noise
        flags = (
            dict(zip(nz["obs_id"].to_numpy(), nz["outlier"].to_numpy(), strict=True))
            if "outlier" in nz
            else {}
        )
        true_out = np.array([bool(flags.get(int(i), False)) for i in obs_id])
        detected = mag > OUTLIER_SIGMAS
        ax.scatter(ta(t), mag, s=5, color="0.6", lw=0, label="observations")
        hit = true_out & detected
        miss = true_out & ~detected
        false_alarm = ~true_out & detected
        ax.scatter(
            ta(t[hit]),
            mag[hit],
            s=30,
            marker="o",
            facecolors="none",
            edgecolors=style.HOLDOUT_OBS_COLOR,
            label=f"true outlier, detected ({hit.sum()})",
        )
        ax.scatter(
            ta(t[miss]),
            mag[miss],
            s=30,
            marker="x",
            color=style.HOLDOUT_OBS_COLOR,
            label=f"true outlier, missed ({miss.sum()})",
        )
        ax.scatter(
            ta(t[false_alarm]),
            mag[false_alarm],
            s=30,
            marker="^",
            facecolors="none",
            edgecolors=sty.color,
            label=f"false alarm ({false_alarm.sum()})",
        )
        ax.axhline(OUTLIER_SIGMAS, color=style.NOISE_REF_COLOR, ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(ta.label)
        ax.set_ylabel("|residual| / σ̂")
        ax.legend(fontsize=6, loc="upper left")
        ax.set_title(
            f"outliers (true flags from Tycho; detected: > {OUTLIER_SIGMAS:g} σ̂)", fontsize=9
        )
        cap = (
            f"Training residuals against the strategy's prediction, normalized by its noise estimate σ̂ = {sig_as:.1f}″: "
            "histogram vs the assumed likelihood, Q–Q plot, and outliers (Tycho's true flags vs detection)."
        )
        if not sig:
            cap += " The strategy reports no σ̂; Tycho's nominal σ is used."
    fig.suptitle(_title(view, "Noise diagnostics"), fontsize=10)
    style.save_figure(
        fig,
        out,
        "noise_diagnostics" + _suffix(view),
        m,
        title="Noise diagnostics",
        caption=cap,
        context=context(view),
    )


def _triangulation(view: RunView) -> dict[str, Any] | None:
    hyp = view.jan.hypothesis.get("hypothesis") or {}
    body = (hyp.get("bodies") or {}).get(view.body_id) or hyp
    tri = body.get("triangulation")
    if not tri:
        return None
    return {"points": tri, "period": body.get("period_days")}


def _true_at(view: RunView, body: str, t: FloatArray) -> FloatArray:
    tr = view.sealed.traj(body)
    tt = tr["t"].to_numpy()
    xyz = io.cols(tr, ["x", "y", "z"]).to_numpy()
    return np.stack([np.interp(t, tt, xyz[:, k]) for k in range(3)], axis=1)


def triangulation_geometry(view: RunView, out: Path, m: FigureManifest) -> None:
    tri = _triangulation(view)
    sty = style.strategy_style(view.jan.label)
    fig, (ax, ax_e) = style.new_figure(11.0, 5.4, ncols=2, gridspec_kw={"width_ratios": [1.2, 1]})
    if tri is None:
        _no_data(ax, "strategy did not triangulate")
        _no_data(ax_e, "")
        cap = "Omitted: this strategy does not use periodic triangulation."
    else:
        pts = tri["points"]
        x = np.array([p["x"] for p in pts])
        taus = np.array([p["tau"] for p in pts])
        truth_x = _true_at(view, view.body_id, taus)
        tr, ob = view.dense_true_r, view.observer_r
        ax.plot(tr[:, 0], tr[:, 1], color=style.TRUTH_COLOR, lw=1.0, alpha=0.6, label="true orbit")
        ax.plot(
            ob[:, 0],
            ob[:, 1],
            color=style.OBSERVER_COLOR,
            lw=0.8,
            alpha=0.6,
            label="true observer orbit",
        )
        for p in pts[:: max(len(pts) // 4, 1)]:
            for o, d in zip(np.array(p["origins"]), np.array(p["dirs"]), strict=True):
                rng = np.linalg.norm(np.array(p["x"]) - o) * 1.15
                ax.plot(
                    [o[0], o[0] + rng * d[0]],
                    [o[1], o[1] + rng * d[1]],
                    color=sty.color,
                    lw=0.4,
                    alpha=0.6,
                )
        ax.scatter(x[:, 0], x[:, 1], s=12, color=sty.color, zorder=4, label="triangulated points")
        obs_pts = [np.array(p["observer"]) for p in pts if p.get("observer")]
        if obs_pts:
            op = np.concatenate(obs_pts)
            ax.scatter(
                op[:, 0],
                op[:, 1],
                s=6,
                marker="s",
                color=sty.color,
                alpha=0.5,
                zorder=3,
                label="reconstructed observer",
            )
        ax.scatter([0], [0], s=60, color=style.SUN_COLOR, zorder=5)
        ax.set_aspect("equal", "datalim")
        ax.set_xlabel("x (AU)")
        ax.set_ylabel("y (AU)")
        ax.legend(fontsize=7, loc="upper right")
        err = np.linalg.norm(x - truth_x, axis=1)
        phase = np.mod(taus - taus.min(), tri["period"] or 1.0) / (tri["period"] or 1.0)
        ax_e.scatter(phase, err, s=12, color=sty.color, label="target points")
        if obs_pts:
            o_t = np.concatenate([np.array(p["times"]) for p in pts if p.get("observer")])
            o_err = np.linalg.norm(
                np.concatenate(obs_pts) - _true_at(view, "observer", o_t), axis=1
            )
            ax_e.scatter(
                np.mod(o_t - taus.min(), tri["period"]) / tri["period"],
                o_err,
                s=5,
                marker="s",
                color=style.OBSERVER_COLOR,
                alpha=0.6,
                label="observer points",
            )
        ax_e.set_yscale("log")
        ax_e.set_xlabel("orbital phase (fraction of the period)")
        ax_e.set_ylabel("error vs truth (AU)")
        ax_e.legend(fontsize=7)
        ax_e.set_title(
            f"P = {tri['period']:.2f} d; median error {np.median(err):.2e} AU", fontsize=9
        )
        cap = (
            "Kepler's trick: lines of sight at epochs one period apart (thin lines, every few groups) "
            "intersect at the body's position; triangulated points vs the true orbit, and their errors."
        )
        if obs_pts:
            cap += " The observer's positions were reconstructed from the Sun stream (squares)."
    fig.suptitle(_title(view, "Triangulation geometry"), fontsize=10)
    style.save_figure(
        fig,
        out,
        "triangulation_geometry" + _suffix(view),
        m,
        title="Triangulation geometry",
        caption=cap,
        context=context(view),
    )


def triangulation_animation(
    view: RunView, out: Path, m: FigureManifest, settings: AnimationSettings
) -> None:
    """Group by group, lines of sight one period apart meet and the orbit(s) appear."""
    tri = _triangulation(view)
    if tri is None:
        return
    pts = sorted(tri["points"], key=lambda p: p["tau"])
    sty = style.strategy_style(view.jan.label)
    style.apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    tr, ob = view.dense_true_r, view.observer_r
    ax.plot(tr[:, 0], tr[:, 1], color=style.TRUTH_COLOR, lw=0.8, alpha=0.35)
    ax.plot(ob[:, 0], ob[:, 1], color=style.OBSERVER_COLOR, lw=0.8, alpha=0.35)
    ax.scatter([0], [0], s=60, color=style.SUN_COLOR, zorder=5)
    lim = 1.25 * float(np.max(np.abs(np.concatenate([tr[:, :2], ob[:, :2]]))))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    max_lines = max(len(p["dirs"]) for p in pts)
    lines = [ax.plot([], [], color=sty.color, lw=0.8)[0] for _ in range(max_lines)]
    found = ax.scatter([], [], s=14, color=sty.color, zorder=4, label="target (triangulated)")
    found_obs = ax.scatter(
        [],
        [],
        s=8,
        marker="s",
        color=style.OBSERVER_COLOR,
        zorder=3,
        label="observer (reconstructed)" if pts[0].get("observer") else None,
    )
    label = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"Triangulation — {view.jan.label} on {view.sealed.scenario_name}", fontsize=10)
    xs = np.array([p["x"] for p in pts])

    def update(k: int) -> tuple[Any, ...]:
        p = pts[k]
        o = np.array(p["origins"])
        d = np.array(p["dirs"])
        for j, ln in enumerate(lines):
            if j < len(o):
                rng = np.linalg.norm(np.array(p["x"]) - o[j]) * 1.15
                ln.set_data([o[j, 0], o[j, 0] + rng * d[j, 0]], [o[j, 1], o[j, 1] + rng * d[j, 1]])
            else:
                ln.set_data([], [])
        found.set_offsets(xs[: k + 1, :2])
        obs_seen = [np.array(q["observer"])[:, :2] for q in pts[: k + 1] if q.get("observer")]
        if obs_seen:
            found_obs.set_offsets(np.concatenate(obs_seen))
        label.set_text(
            f"group {k + 1}/{len(pts)}: {len(o)} lines of sight, {tri['period']:.1f} d apart"
        )
        return (*lines, found, found_obs, label)

    settings = AnimationSettings(
        fps=min(settings.fps, 8), max_seconds=settings.max_seconds, quick=settings.quick
    )
    idx = np.unique(np.linspace(0, len(pts) - 1, settings.frames(len(pts))).round().astype(int))
    anim = FuncAnimation(fig, update, frames=idx, blit=False)
    path = save_animation(anim, out / ("triangulation_animation" + _suffix(view)), settings)
    cap = "Kepler's trick step by step: lines of sight one period apart intersect, revealing the orbit point by point."
    if pts[0].get("observer"):
        cap += (
            " The observer's orbit is reconstructed at the same time from the Sun stream (squares)."
        )
    m.add(
        name="triangulation_animation" + _suffix(view),
        kind="animation",
        paths=[path],
        title="Triangulation animation",
        caption=cap,
        context=context(view),
    )


PHASE3_FIGURES = [noise_diagnostics, triangulation_geometry]
PHASE3_ANIMATIONS = [triangulation_animation]
