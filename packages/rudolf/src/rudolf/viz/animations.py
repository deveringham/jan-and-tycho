"""Per-run animations (SPEC §7.1.3)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from rudolf.data import RunView, sep_arcsec
from rudolf.viz import style
from rudolf.viz.manifest import FigureManifest
from rudolf.viz.run_plots import TimeAxis, _suffix, context, lonlat_deg
from rudolf.viz.writers import AnimationSettings, save_animation

FloatArray = NDArray[np.float64]


def _frame_indices(n_dense: int, settings: AnimationSettings) -> NDArray[np.int_]:
    return np.unique(np.linspace(0, n_dense - 1, settings.frames()).round().astype(int))


def orbit_animation(
    view: RunView,
    out: Path,
    m: FigureManifest,
    settings: AnimationSettings,
    trail_days: float = 400.0,
) -> None:
    """Top-down: Sun, observer, true body, predicted body (+cloud), line of sight, clock."""
    s, sty = view.sealed, style.strategy_style(view.jan.label)
    t = view.dense_t
    tr, ob = view.dense_true_r, view.observer_r
    pred = view.dense_pred_r
    pred_u = view.dense_pred_u
    samples = view.jan.sample_array("dense", view.body_id, "p")
    style.apply_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(tr[:, 0], tr[:, 1], color=style.TRUTH_COLOR, lw=0.5, alpha=0.25)
    ax.plot(ob[:, 0], ob[:, 1], color=style.OBSERVER_COLOR, lw=0.5, alpha=0.25)
    ob_fit = view.jan.position("dense", "observer")  # T2/T3: the observer as Jan inferred it
    if ob_fit is not None:
        ax.plot(ob_fit[:, 0], ob_fit[:, 1], color=sty.color, lw=0.8, ls=":", alpha=0.8)
    ax.scatter([0], [0], s=80, color=style.SUN_COLOR, zorder=5)
    hidden_true = [s.dense_positions(h) for h in s.hidden_ids]
    hidden_pred = view.jan.position("dense", "hidden:perturber")
    extra_xy = [tr[:, :2], ob[:, :2], *(h[:, :2] for h in hidden_true)]
    if hidden_pred is not None:
        extra_xy.append(hidden_pred[:, :2])
    for h in hidden_true:
        ax.plot(h[:, 0], h[:, 1], color=style.TRUTH_COLOR, lw=0.5, ls=":", alpha=0.4)
    (dot_hid,) = ax.plot(
        [],
        [],
        "*",
        color=style.TRUTH_COLOR,
        ms=9,
        label="hidden body (true)" if hidden_true else None,
    )
    (dot_hid_pred,) = ax.plot(
        [],
        [],
        "*",
        color=sty.color,
        ms=9,
        mfc="none",
        label="hidden body (inferred)" if hidden_pred is not None else None,
    )
    lim = 1.15 * float(np.max(np.abs(np.concatenate(extra_xy))))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    (trail_true,) = ax.plot([], [], color=style.TRUTH_COLOR, lw=1.2)
    (trail_pred,) = ax.plot([], [], color=sty.color, lw=1.2, ls=sty.linestyle)
    (dot_true,) = ax.plot([], [], "o", color=style.TRUTH_COLOR, ms=6, label="truth")
    (dot_pred,) = ax.plot(
        [], [], "o", color=sty.color, ms=6, mfc="none", mew=1.5, label=view.jan.label
    )
    (dot_obs,) = ax.plot([], [], "o", color=style.OBSERVER_COLOR, ms=5, label="observer")
    (dot_obs_fit,) = ax.plot(
        [],
        [],
        "s",
        color=sty.color,
        ms=4,
        mfc="none",
        label="observer (recovered)" if ob_fit is not None else None,
    )
    (los_true,) = ax.plot([], [], color=style.TRUTH_COLOR, lw=0.6, alpha=0.6)
    (los_pred,) = ax.plot([], [], color=sty.color, lw=0.6, ls="--", alpha=0.9)
    cloud = ax.scatter([], [], s=4, color=sty.color, alpha=0.3)
    clock = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9)
    phase = ax.text(
        0.98, 0.97, "", transform=ax.transAxes, va="top", ha="right", fontsize=9, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title(f"{view.jan.label} on {s.scenario_name} (seed {s.seed})", fontsize=10)
    ray_len = 2.0 * lim

    def update(k: int) -> tuple[Any, ...]:
        tk = t[k]
        w = (t >= tk - trail_days) & (t <= tk)
        trail_true.set_data(tr[w, 0], tr[w, 1])
        dot_true.set_data([tr[k, 0]], [tr[k, 1]])
        dot_obs.set_data([ob[k, 0]], [ob[k, 1]])
        if hidden_true:
            dot_hid.set_data([hidden_true[0][k, 0]], [hidden_true[0][k, 1]])
        if hidden_pred is not None:
            dot_hid_pred.set_data([hidden_pred[k, 0]], [hidden_pred[k, 1]])
        if ob_fit is not None:
            dot_obs_fit.set_data([ob_fit[k, 0]], [ob_fit[k, 1]])
        los_true.set_data([ob[k, 0], tr[k, 0]], [ob[k, 1], tr[k, 1]])
        if pred is not None:
            trail_pred.set_data(pred[w, 0], pred[w, 1])
            dot_pred.set_data([pred[k, 0]], [pred[k, 1]])
        if pred_u is not None:
            end = ob[k] + ray_len * pred_u[k] if pred is None else pred[k]
            los_pred.set_data([ob[k, 0], end[0]], [ob[k, 1], end[1]])
        if samples is not None:
            cloud.set_offsets(samples[:, k, :2])
        clock.set_text(f"t = {tk:7.1f} d ({tk / 365.25:5.2f} yr)")
        holdout = tk >= s.t_split
        phase.set_text("HOLDOUT" if holdout else "TRAINING")
        phase.set_color(style.HOLDOUT_OBS_COLOR if holdout else "0.3")
        return (
            trail_true,
            trail_pred,
            dot_true,
            dot_pred,
            dot_obs,
            dot_obs_fit,
            los_true,
            los_pred,
            cloud,
            clock,
            phase,
        )

    frames = _frame_indices(t.size, settings)
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    path = save_animation(anim, out / ("orbit_animation" + _suffix(view)), settings)
    cap = (
        "Top-down heliocentric view over time: Sun, observer, true body (black) and "
        "predicted body, with lines of sight from the observer."
    )
    if pred is None:
        cap += (
            " The hypothesis has no heliocentric position, so only its predicted line of"
            " sight is drawn."
        )
    if samples is not None:
        cap += " The cloud shows draws from the reported uncertainty."
    m.add(
        name="orbit_animation" + _suffix(view),
        kind="animation",
        paths=[path],
        title="Orbit animation",
        caption=cap,
        context=context(view),
    )


def sky_animation(view: RunView, out: Path, m: FigureManifest, settings: AnimationSettings) -> None:
    """Apparent motion on the sky building up over time; observations pop in as they happen."""
    s, sty = view.sealed, style.strategy_style(view.jan.label)
    t = view.dense_t
    ta = TimeAxis(s.duration)
    lon_t, lat_t = lonlat_deg(view.dense_true_u)
    pu = view.dense_pred_u
    lon_p, lat_p = lonlat_deg(pu) if pu is not None else (None, None)
    pub, hold = s.public_body(view.body_id), s.holdout_body(view.body_id)
    o_t = np.concatenate([pub["t"], hold["t"]])
    o_lon = np.degrees(np.concatenate([pub["lon"], hold["lon"]]))
    o_lat = np.degrees(np.concatenate([pub["lat"], hold["lat"]]))
    o_hold = np.concatenate([np.zeros(len(pub), bool), np.ones(len(hold), bool)])
    style.apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(360, 0)
    pad = 0.5 + 0.1 * float(np.ptp(lat_t))
    ax.set_ylim(float(lat_t.min()) - pad, float(lat_t.max()) + pad)
    ax.set_xlabel("λ (deg)")
    ax.set_ylabel("β (deg)")
    (line_t,) = ax.plot([], [], color=style.TRUTH_COLOR, lw=1.0, label="truth")
    (line_p,) = ax.plot([], [], color=sty.color, lw=1.0, ls=sty.linestyle, label=view.jan.label)
    obs_sc = ax.scatter([], [], s=6, color=style.OBS_COLOR)
    hold_sc = ax.scatter(
        [], [], s=10, facecolors="none", edgecolors=style.HOLDOUT_OBS_COLOR, marker="D"
    )
    (head_t,) = ax.plot([], [], "o", color=style.TRUTH_COLOR, ms=5)
    (head_p,) = ax.plot([], [], "o", color=sty.color, ms=5, mfc="none")
    clock = ax.text(0.01, 0.95, "", transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(f"Sky motion — {view.jan.label} on {s.scenario_name} (seed {s.seed})", fontsize=10)

    def _broken(lon: FloatArray, lat: FloatArray) -> tuple[FloatArray, FloatArray]:
        j = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
        return np.insert(lon, j, np.nan), np.insert(lat, j, np.nan)

    def update(k: int) -> tuple[Any, ...]:
        line_t.set_data(*_broken(lon_t[: k + 1], lat_t[: k + 1]))
        head_t.set_data([lon_t[k]], [lat_t[k]])
        if lon_p is not None and lat_p is not None:
            line_p.set_data(*_broken(lon_p[: k + 1], lat_p[: k + 1]))
            head_p.set_data([lon_p[k]], [lat_p[k]])
        seen = o_t <= t[k]
        obs_sc.set_offsets(np.c_[o_lon[seen & ~o_hold], o_lat[seen & ~o_hold]])
        hold_sc.set_offsets(np.c_[o_lon[seen & o_hold], o_lat[seen & o_hold]])
        tag = "HOLDOUT" if t[k] >= s.t_split else "training"
        clock.set_text(
            f"{ta.label.split(' (')[0]} = {ta(t[k]):.2f} {'yr' if ta.years else 'd'}  [{tag}]"
        )
        return line_t, line_p, obs_sc, hold_sc, head_t, head_p, clock

    anim = FuncAnimation(fig, update, frames=_frame_indices(t.size, settings), blit=False)
    path = save_animation(anim, out / ("sky_animation" + _suffix(view)), settings)
    cap = (
        "Apparent motion on the sky as seen from the observer: truth vs prediction; "
        "observations appear as they are made."
    )
    if pu is None:
        cap += " Omitted: no sky prediction."
    m.add(
        name="sky_animation" + _suffix(view),
        kind="animation",
        paths=[path],
        title="Sky animation",
        caption=cap,
        context=context(view),
    )


def fit_progress(view: RunView, out: Path, m: FigureManifest, settings: AnimationSettings) -> None:
    """How the estimate evolves over the fit's trace frames (optimizer or sampler steps)."""
    frames = view.jan.trace_frames()
    if not frames:
        return
    s, sty = view.sealed, style.strategy_style(view.jan.label)
    labels = view.jan.hypothesis.get("trace_labels") or [str(f) for f in frames]
    first = view.jan.trace_array(frames[0], view.body_id, "u")
    has_pos = view.jan.trace_array(frames[0], view.body_id, "p") is not None
    if first is None and not has_pos:
        return
    t_grid = first[0] if first is not None else view.dense_t
    tr_u = np.stack([np.interp(t_grid, view.dense_t, view.dense_true_u[:, k]) for k in range(3)], 1)
    tr_u /= np.linalg.norm(tr_u, axis=1, keepdims=True)
    ta = TimeAxis(s.duration)
    style.apply_style()
    fig, (ax_o, ax_e) = plt.subplots(
        1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1, 1.3]}
    )
    tr = view.dense_true_r
    ax_o.plot(tr[:, 0], tr[:, 1], color=style.TRUTH_COLOR, lw=2.0, alpha=0.8, label="truth")
    ax_o.plot(
        view.observer_r[:, 0], view.observer_r[:, 1], color=style.OBSERVER_COLOR, lw=0.6, alpha=0.5
    )
    ax_o.scatter([0], [0], s=60, color=style.SUN_COLOR)
    lim = 1.4 * float(np.max(np.abs(tr[:, :2])))
    ax_o.set_xlim(-lim, lim)
    ax_o.set_ylim(-lim, lim)
    ax_o.set_aspect("equal")
    ax_o.set_xlabel("x (AU)")
    ax_o.set_ylabel("y (AU)")
    (orbit_line,) = ax_o.plot([], [], **sty.line_kwargs(view.jan.label))
    if not has_pos:
        ax_o.text(
            0.5,
            0.05,
            "no heliocentric positions",
            transform=ax_o.transAxes,
            ha="center",
            color="0.4",
        )
    ax_o.legend(fontsize=7, loc="upper right")
    (err_line,) = ax_e.plot([], [], color=sty.color, lw=1.0)
    style.mark_holdout(ax_e, s.t_split, s.duration, ta.scale)
    if s.noise_sigma_arcsec:
        ax_e.axhline(s.noise_sigma_arcsec, color=style.NOISE_REF_COLOR, ls="--", lw=1)
    ax_e.set_yscale("log")
    ax_e.set_xlim(0, ta(s.duration))
    errs = []
    for f in frames:
        a = view.jan.trace_array(f, view.body_id, "u")
        errs.append(None if a is None else sep_arcsec(a[1], tr_u))
    finite = [e for e in errs if e is not None]
    if finite:
        lo = max(min(float(np.min(e)) for e in finite), 1e-2)
        hi = max(float(np.max(e)) for e in finite)
        ax_e.set_ylim(lo / 2, hi * 2)
    ax_e.set_xlabel(ta.label)
    ax_e.set_ylabel("sky error vs truth (arcsec)")
    title = fig.suptitle("", fontsize=10)

    def update(k: int) -> tuple[Any, ...]:
        f = frames[k]
        p = view.jan.trace_array(f, view.body_id, "p")
        if p is not None:
            orbit_line.set_data(p[1][:, 0], p[1][:, 1])
        if errs[k] is not None:
            err_line.set_data(ta(t_grid), np.maximum(errs[k], 1e-3))
        title.set_text(
            f"Fit progress — {view.jan.label} on {s.scenario_name}: {labels[min(k, len(labels) - 1)]}"
        )
        return orbit_line, err_line, title

    n = len(frames)
    idx = np.unique(np.linspace(0, n - 1, settings.frames(max(n, 2))).round().astype(int))
    anim = FuncAnimation(fig, update, frames=idx, blit=False)
    path = save_animation(anim, out / ("fit_progress" + _suffix(view)), _trace_settings(settings))
    m.add(
        name="fit_progress" + _suffix(view),
        kind="animation",
        paths=[path],
        title="Fit progress",
        caption="The estimate across the fit's trace (optimizer iterations or sampler steps): "
        "recovered orbit (left) and sky error vs the noise-free truth (right).",
        context=context(view),
    )


def _trace_settings(settings: AnimationSettings) -> AnimationSettings:
    """Trace frames are few (≤ ~100): play them slowly enough to follow."""
    return dataclasses.replace(settings, fps=min(settings.fps, 10))


def posterior_samples(
    view: RunView, out: Path, m: FigureManifest, settings: AnimationSettings
) -> None:
    """Posterior orbits: top-down spaghetti plus each draw's deviation from the truth
    (the only place a tight posterior is visible), over sampler progress if traced, else
    over time."""
    s, sty = view.sealed, style.strategy_style(view.jan.label)
    frames = [
        f
        for f in view.jan.trace_frames()
        if view.jan.trace_sample_array(f, view.body_id, "p") is not None
    ]
    final = view.jan.sample_array("dense", view.body_id, "p")
    if not frames and final is None:
        return
    ta = TimeAxis(s.duration)
    style.apply_style()
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.3]})
    tr = view.dense_true_r
    ax.plot(tr[:, 0], tr[:, 1], color=style.TRUTH_COLOR, lw=1.0, alpha=0.8, label="truth")
    ax.scatter([0], [0], s=60, color=style.SUN_COLOR)
    lim = 1.4 * float(np.max(np.abs(tr[:, :2])))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    n_lines = 16
    lines = [ax.plot([], [], color=sty.color, lw=0.6, alpha=0.6)[0] for _ in range(n_lines)]
    dev_lines = [axd.plot([], [], color=sty.color, lw=0.6, alpha=0.6)[0] for _ in range(n_lines)]
    dots = ax.scatter([], [], s=8, color=sty.color, zorder=4)
    (true_dot,) = ax.plot([], [], "o", color=style.TRUTH_COLOR, ms=5, zorder=5)
    marker = axd.axvline(0, color=style.TRUTH_COLOR, lw=0.8)
    label = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    labels = view.jan.hypothesis.get("trace_labels") or []
    style.mark_holdout(axd, s.t_split, s.duration, ta.scale)
    axd.set_yscale("log")
    axd.set_xlim(0, ta(s.duration))
    axd.set_xlabel(ta.label)
    axd.set_ylabel("|draw − truth| (AU)")
    axd.set_title("each posterior draw's 3D distance from the truth", fontsize=9)

    def truth_on(t_grid: FloatArray) -> FloatArray:
        return np.stack([np.interp(t_grid, view.dense_t, tr[:, k]) for k in range(3)], 1)

    if frames:
        first = view.jan.trace_array(frames[0], view.body_id, "p")
        t_grid = first[0] if first is not None else view.dense_t
        tr_g = truth_on(t_grid)
        arrs = [view.jan.trace_sample_array(f, view.body_id, "p") for f in frames]
        devs = [np.linalg.norm(a - tr_g[None], axis=2) for a in arrs if a is not None]
        allv = np.concatenate([d.ravel() for d in devs])
        allv = allv[allv > 0]
        if allv.size:
            axd.set_ylim(np.quantile(allv, 0.01) / 2, allv.max() * 2)

        def update(k: int) -> tuple[Any, ...]:
            arr = arrs[k]
            assert arr is not None
            dev = devs[k]
            for j in range(n_lines):
                if j < len(arr):
                    lines[j].set_data(arr[j, :, 0], arr[j, :, 1])
                    dev_lines[j].set_data(ta(t_grid), np.maximum(dev[j], 1e-12))
                else:
                    lines[j].set_data([], [])
                    dev_lines[j].set_data([], [])
            label.set_text(labels[frames[k]] if frames[k] < len(labels) else f"frame {frames[k]}")
            return (*lines, *dev_lines, label)

        idx = np.unique(
            np.linspace(0, len(frames) - 1, settings.frames(len(frames))).round().astype(int)
        )
        settings = _trace_settings(settings)
        cap = "Posterior draws as the sampler progresses: orbits (left) and each draw's distance from the true trajectory over time (right, log scale)."
    else:
        assert final is not None
        t = view.dense_t
        dev = np.linalg.norm(final - tr[None], axis=2)
        for j in range(min(n_lines, len(final))):
            lines[j].set_data(final[j, :, 0], final[j, :, 1])
            dev_lines[j].set_data(ta(t), np.maximum(dev[j], 1e-12))
        pos = dev[dev > 0]
        if pos.size:
            axd.set_ylim(np.quantile(pos, 0.01) / 2, pos.max() * 2)

        def update(k: int) -> tuple[Any, ...]:
            dots.set_offsets(final[:, k, :2])
            true_dot.set_data([tr[k, 0]], [tr[k, 1]])
            marker.set_xdata([ta(t[k]), ta(t[k])])
            phase = "HOLDOUT" if t[k] >= s.t_split else "training"
            label.set_text(f"t = {t[k]:.0f} d [{phase}]")
            return dots, true_dot, marker, label

        idx = np.unique(np.linspace(0, len(t) - 1, settings.frames()).round().astype(int))
        cap = "Posterior draws: orbits and the posterior cloud of the body's position over time (left), and each draw's distance from the true trajectory (right, log scale)."
    fig.suptitle(
        f"Posterior samples — {view.jan.label} on {s.scenario_name} (seed {s.seed})", fontsize=10
    )
    anim = FuncAnimation(fig, update, frames=idx, blit=False)
    path = save_animation(anim, out / ("posterior_samples" + _suffix(view)), settings)
    m.add(
        name="posterior_samples" + _suffix(view),
        kind="animation",
        paths=[path],
        title="Posterior samples",
        caption=cap,
        context=context(view),
    )


def _triangulation_animation(
    view: RunView, out: Path, m: FigureManifest, settings: AnimationSettings
) -> None:
    from rudolf.viz.phase3_plots import triangulation_animation

    triangulation_animation(view, out, m, settings)


_triangulation_animation.__name__ = "triangulation_animation"


def _law_discovery(
    view: RunView, out: Path, m: FigureManifest, settings: AnimationSettings
) -> None:
    from rudolf.viz.phase4_plots import law_discovery

    law_discovery(view, out, m, settings)


_law_discovery.__name__ = "law_discovery"

PER_RUN_ANIMATIONS: list[Callable[[RunView, Path, FigureManifest, AnimationSettings], None]] = [
    orbit_animation,
    sky_animation,
    fit_progress,
    posterior_samples,
    _triangulation_animation,
    _law_discovery,
]
