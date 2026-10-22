"""Per-run static diagnostics (SPEC §7.1.2).

Every figure shows the truth; missing optional data (no positions, no elements, no
uncertainty) drops that layer and says so in the caption, never crashes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from jt_core import io
from jt_core.constants import YEAR_DAYS
from jt_core.frames import unit_to_lonlat
from rudolf.data import JanOutputs, RunView, Sealed
from rudolf.metrics.parameters import parameter_metrics
from rudolf.metrics.predictive import holdout_errors
from rudolf.viz import style
from rudolf.viz.manifest import FigureManifest

FloatArray = NDArray[np.float64]


# ----------------------------------------------------------------------------- helpers


class TimeAxis:
    """Days for short spans, years for long ones."""

    def __init__(self, duration: float) -> None:
        self.years = duration > 3 * YEAR_DAYS
        self.scale = 1.0 / YEAR_DAYS if self.years else 1.0
        self.label = "t (years since epoch)" if self.years else "t (days since epoch)"

    def __call__(self, t: Any) -> FloatArray:
        return np.asarray(t, dtype=np.float64) * self.scale


def break_wraps(
    x: FloatArray, y: FloatArray, period: float = 360.0
) -> tuple[FloatArray, FloatArray]:
    """Insert NaNs where x jumps by more than half a period (longitude wraparound)."""
    if x.size < 2:
        return x, y
    jumps = np.where(np.abs(np.diff(x)) > period / 2)[0] + 1
    return np.insert(x.astype(float), jumps, np.nan), np.insert(y.astype(float), jumps, np.nan)


def lonlat_deg(u: FloatArray) -> tuple[FloatArray, FloatArray]:
    lon, lat = unit_to_lonlat(u)
    return np.degrees(lon), np.degrees(lat)


def context(view: RunView) -> dict[str, Any]:
    return {
        "scenario": view.sealed.scenario_name,
        "seed": view.sealed.seed,
        "strategy": view.jan.strategy,
        "label": view.jan.label,
        "body_id": view.body_id,
    }


def _title(view: RunView, what: str) -> str:
    return f"{what} — {view.jan.label} on {view.sealed.scenario_name} (seed {view.sealed.seed})"


def _no_data(ax: Any, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="0.4")
    ax.set_xticks([])
    ax.set_yticks([])


def _suffix(view: RunView) -> str:
    return "" if len(view.sealed.body_ids) == 1 else f"__{view.body_id}"


# ----------------------------------------------------------------------------- figures


def sky_track(view: RunView, out: Path, m: FigureManifest) -> None:
    s, sty = view.sealed, style.strategy_style(view.jan.label)
    ta = TimeAxis(s.duration)
    fig, (ax_t, ax_s) = style.new_figure(8.0, 6.5, nrows=2, gridspec_kw={"height_ratios": [1, 1.1]})
    notes: list[str] = []
    pub = s.public_body(view.body_id)
    hold = s.holdout_body(view.body_id)
    lon_tr, lat_tr = lonlat_deg(view.dense_true_u)
    t_tr = view.dense_t
    ax_t.plot(*_wrap_series(ta(t_tr), lon_tr), **style.TRUTH_LINE)
    if view.dense_pred_u is not None:
        lon_p, _ = lonlat_deg(view.dense_pred_u)
        ax_t.plot(*_wrap_series(ta(t_tr), lon_p), **sty.line_kwargs(view.jan.label))
    else:
        notes.append("no sky prediction")
    ax_t.scatter(ta(pub["t"]), np.degrees(pub["lon"]), label="observations", **style.OBS_MARKER)
    ax_t.scatter(
        ta(hold["t"]), np.degrees(hold["lon"]), label="holdout obs.", **style.HOLDOUT_MARKER
    )
    style.mark_holdout(ax_t, s.t_split, s.duration, ta.scale)
    ax_t.set_xlabel(ta.label)
    ax_t.set_ylabel("λ (deg)")
    ax_t.set_ylim(0, 360)
    ax_t.legend(loc="upper right", ncols=4, fontsize=7)

    ax_s.plot(*break_wraps(lon_tr, lat_tr), **style.TRUTH_LINE)
    if view.dense_pred_u is not None:
        lon_p, lat_p = lonlat_deg(view.dense_pred_u)
        ax_s.plot(*break_wraps(lon_p, lat_p), **sty.line_kwargs(view.jan.label))
    ax_s.scatter(np.degrees(pub["lon"]), np.degrees(pub["lat"]), **style.OBS_MARKER)
    ax_s.scatter(np.degrees(hold["lon"]), np.degrees(hold["lat"]), **style.HOLDOUT_MARKER)
    ax_s.set_xlim(360, 0)  # sky convention: longitude increases to the left
    ax_s.set_xlabel("λ (deg)")
    ax_s.set_ylabel("β (deg)")
    ax_s.set_title("track on the sky (retrograde loops)", fontsize=9)
    fig.suptitle(_title(view, "Sky track"), fontsize=10)
    cap = (
        "Geocentric ecliptic longitude vs time and the (λ, β) track: truth (black), "
        "prediction, observations."
    )
    if notes:
        cap += " Omitted: " + ", ".join(notes) + "."
    style.save_figure(
        fig,
        out,
        "sky_track" + _suffix(view),
        m,
        title="Sky track",
        caption=cap,
        context=context(view),
    )


def _wrap_series(t: FloatArray, lon: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Break a λ(t) series where λ wraps through 0/360."""
    jumps = np.where(np.abs(np.diff(lon)) > 180.0)[0] + 1
    return np.insert(t.astype(float), jumps, np.nan), np.insert(lon.astype(float), jumps, np.nan)


def residuals_vs_time(view: RunView, out: Path, m: FigureManifest) -> None:
    s, sty = view.sealed, style.strategy_style(view.jan.label)
    ta = TimeAxis(s.duration)
    fig, ax = style.new_figure(8.0, 4.0)
    notes: list[str] = []
    err = view.dense_err_arcsec
    if err is not None:
        ax.plot(
            ta(view.dense_t), np.maximum(err, 1e-3), **sty.line_kwargs("prediction − truth (dense)")
        )
        he = holdout_errors(view)
        if he is not None and he.size:
            ax.scatter(
                ta(s.holdout_body(view.body_id)["t"]),
                np.maximum(he, 1e-3),
                s=10,
                color=sty.color,
                marker="D",
                label="at holdout obs.",
                zorder=3,
            )
        cov = view.jan.sky_cov("dense", view.body_id)
        if cov is not None:
            sig = np.sqrt(np.clip(cov[:, 0, 0] + cov[:, 1, 1], 0, None)) * 206264.80624709636
            ax.plot(
                ta(view.dense_t),
                np.maximum(sig, 1e-3),
                color=sty.color,
                ls=":",
                lw=1,
                label="reported 1σ (radial)",
            )
        else:
            notes.append("no predictive uncertainty")
    else:
        _no_data(ax, "strategy produced no sky prediction")
        notes.append("no sky prediction")
    sig_n = s.noise_sigma_arcsec
    if sig_n:
        ax.axhline(
            sig_n, color=style.NOISE_REF_COLOR, ls="--", lw=1, label=f"noise σ = {sig_n:.0f}″"
        )
    style.mark_holdout(ax, s.t_split, s.duration, ta.scale)
    if err is not None:
        ax.set_yscale("log")
        ax.legend(loc="upper left", fontsize=7)
    ax.set_xlabel(ta.label)
    ax.set_ylabel("great-circle error (arcsec)")
    ax.set_title(_title(view, "Prediction error vs time"), fontsize=10)
    cap = (
        "Great-circle error of the predicted direction vs the noise-free truth; "
        "the shaded region is the sealed holdout."
    )
    if notes:
        cap += " Omitted: " + ", ".join(notes) + "."
    style.save_figure(
        fig,
        out,
        "residuals_vs_time" + _suffix(view),
        m,
        title="Residuals vs time",
        caption=cap,
        context=context(view),
    )


def _orbit_layers(view: RunView) -> tuple[FloatArray | None, FloatArray | None]:
    pos_s = view.jan.sample_array("dense", view.body_id, "p")
    return view.dense_pred_r, pos_s


def orbit_topdown(view: RunView, out: Path, m: FigureManifest) -> None:
    sty = style.strategy_style(view.jan.label)
    fig, (ax, ins) = style.new_figure(6.5, 8.0, nrows=2, gridspec_kw={"height_ratios": [4, 1]})
    notes: list[str] = []
    tr = view.dense_true_r
    ob = view.observer_r
    pred, samples = _orbit_layers(view)
    ax.plot(ob[:, 0], ob[:, 1], color=style.OBSERVER_COLOR, lw=0.8, alpha=0.7, label="observer")
    ob_fit = view.jan.position("dense", "observer")
    if ob_fit is not None:
        ax.plot(
            ob_fit[:, 0],
            ob_fit[:, 1],
            color=sty.color,
            lw=1.0,
            ls=":",
            label="observer (recovered)",
        )
    ax.plot(tr[:, 0], tr[:, 1], **style.TRUTH_LINE)
    if samples is not None:
        for k in range(min(20, samples.shape[0])):
            ax.plot(samples[k, :, 0], samples[k, :, 1], color=sty.color, lw=0.4, alpha=0.15)
    if pred is not None:
        ax.plot(pred[:, 0], pred[:, 1], **sty.line_kwargs(view.jan.label))
    else:
        notes.append("no heliocentric trajectory (sky-only hypothesis)")
    ax.scatter([0], [0], s=80, color=style.SUN_COLOR, zorder=5, label="Sun")
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.legend(loc="upper right", fontsize=7)
    ins.plot(tr[:, 0], tr[:, 2], color=style.TRUTH_COLOR, lw=2.0, alpha=0.8)
    if pred is not None:
        ins.plot(pred[:, 0], pred[:, 2], color=sty.color, lw=1.0, ls=sty.linestyle)
    ins.plot(ob[:, 0], ob[:, 2], color=style.OBSERVER_COLOR, lw=0.8, alpha=0.7)
    ins.set_xlabel("x (AU)")
    ins.set_ylabel("z (AU)")
    ins.set_title("side view (x–z, z exaggerated)", fontsize=8)
    ax.set_title(_title(view, "Orbit (top-down)"), fontsize=10)
    cap = (
        "Heliocentric x–y view: true orbit (black), recovered orbit, observer orbit, Sun; "
        "below, the x–z side view."
    )
    if samples is not None:
        cap += " Thin lines are draws from the reported uncertainty."
    if notes:
        cap += " Omitted: " + ", ".join(notes) + "."
    style.save_figure(
        fig,
        out,
        "orbit_topdown" + _suffix(view),
        m,
        title="Orbit top-down",
        caption=cap,
        context=context(view),
    )


def orbit_3d(view: RunView, out: Path, m: FigureManifest) -> None:
    sty = style.strategy_style(view.jan.label)
    style.apply_style()
    fig = plt.figure(figsize=(6.5, 5.5))
    ax: Any = fig.add_subplot(projection="3d")
    tr = view.dense_true_r
    ob = view.observer_r
    pred = view.dense_pred_r
    ax.plot(
        ob[:, 0],
        ob[:, 1],
        ob[:, 2],
        color=style.OBSERVER_COLOR,
        lw=0.7,
        alpha=0.7,
        label="observer",
    )
    ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=style.TRUTH_COLOR, lw=1.2, label="truth")
    if pred is not None:
        ax.plot(
            pred[:, 0],
            pred[:, 1],
            pred[:, 2],
            color=sty.color,
            ls=sty.linestyle,
            lw=1.1,
            label=view.jan.label,
        )
    ax.scatter([0], [0], [0], s=60, color=style.SUN_COLOR)
    lim = float(np.max(np.abs(np.concatenate([tr, ob]))))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim / 2, lim / 2)
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.set_zlabel("z (AU)")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(_title(view, "Orbit (3D)"), fontsize=10)
    cap = "3D heliocentric trajectories: truth vs recovered (z axis compressed)."
    if pred is None:
        cap += " Omitted: no heliocentric trajectory (sky-only hypothesis)."
    style.save_figure(
        fig,
        out,
        "orbit_3d" + _suffix(view),
        m,
        title="Orbit 3D",
        caption=cap,
        context=context(view),
    )


_ELEMENT_ROWS = [
    ("a_rel", "a", "relative", "a", 1.0),
    ("q_rel", "q", "relative", "q", 1.0),
    ("e", "e", "absolute", "e", 1.0),
    ("i_deg", "i", "deg", "i", 1.0),
    ("Omega_deg", "Ω", "deg", "Omega", 1.0),
    ("omega_deg", "ω", "deg", "omega", 1.0),
    ("varpi_deg", "ϖ = Ω + ω", "deg", None, 1.0),
    ("lambda_deg", "λ (mean lon. at t=0)", "deg", None, 1.0),
    ("t_p_days", "t_p", "days", "t_p", 1.0),
]


def element_errors(view: RunView, out: Path, m: FigureManifest) -> None:
    sty = style.strategy_style(view.jan.label)
    pm = parameter_metrics(view.sealed.body_truth(view.body_id), view.jan.elements(view.body_id))
    fig, (ax_z, ax_r) = style.new_figure(9.0, 4.0, ncols=2, gridspec_kw={"width_ratios": [1, 1.2]})
    errs = pm["errors"]
    zs = pm["zscores"] or {}
    if errs is None:
        _no_data(ax_z, "strategy reports no orbital elements")
        _no_data(ax_r, "—")
        cap = "Omitted: the hypothesis reports no orbital elements."
    else:
        rows = [
            (k, lab, unit, zkey)
            for k, lab, unit, zkey, _ in _ELEMENT_ROWS
            if errs.get(k) is not None
        ]
        ys = np.arange(len(rows))[::-1]
        zvals = [zs.get(zk) if zk else None for _, _, _, zk in rows]
        if any(z is not None for z in zvals):
            for band, a in ((3, 0.06), (2, 0.1), (1, 0.16)):
                ax_z.axvspan(-band, band, color=sty.color, alpha=a, lw=0)
            for y, z in zip(ys, zvals, strict=True):
                if z is not None:
                    ax_z.scatter([z], [y], color=sty.color, zorder=3)
            ax_z.axvline(0, color=style.TRUTH_COLOR, lw=0.8)
            ax_z.set_xlabel("(estimate − truth) / reported σ")
            lim = max(4.0, *(abs(z) * 1.1 for z in zvals if z is not None))
            ax_z.set_xlim(-lim, lim)
        else:
            _no_data(ax_z, "no reported uncertainty")
        ax_z.set_yticks(ys, [lab for _, lab, _, _ in rows])
        ax_z.set_ylim(-0.5, len(rows) - 0.5)
        vals = [abs(float(errs[k])) for k, *_ in rows]
        ax_r.barh(ys, np.maximum(vals, 1e-12), color=sty.color, alpha=0.8)
        ax_r.set_xscale("log")
        ax_r.set_yticks(ys, [f"{lab} ({unit})" for _, lab, unit, _ in rows])
        for y, (k, *_) in zip(ys, rows, strict=True):
            ax_r.text(
                max(abs(float(errs[k])), 1e-12),
                y,
                f"  {float(errs[k]):+.3g}",
                va="center",
                fontsize=7,
            )
        ax_r.set_xlabel("|error| (unit per row)")
        cap = (
            "Per-element errors (estimate − truth). Left: normalized by the strategy's reported σ "
            "(shaded ±1/2/3σ). Right: raw magnitudes. Degenerate elements (Ω at i≈0, ω and t_p "
            "at e≈0) are skipped in favour of ϖ and mean longitude."
        )
        if not any(z is not None for z in zvals):
            cap += " Omitted: normalized errors (no reported uncertainty)."
    fig.suptitle(_title(view, "Element errors"), fontsize=10)
    style.save_figure(
        fig,
        out,
        "element_errors" + _suffix(view),
        m,
        title="Element errors",
        caption=cap,
        context=context(view),
    )


_REASON_COLORS = {
    "observed": style.OBS_COLOR,
    "weather": "#6baed6",
    "elongation": style.SUN_COLOR,
    "magnitude": "#9e9ac8",
    "season": "#bdbdbd",
}


def observation_coverage(view: RunView, out: Path, m: FigureManifest) -> None:
    s = view.sealed
    ta = TimeAxis(s.duration)
    fig, (ax_t, ax_p) = style.new_figure(8.0, 5.0, nrows=2, gridspec_kw={"height_ratios": [1, 1]})
    sched = (
        io.rows(s.schedule, s.schedule["body_id"] == view.body_id) if not s.schedule.empty else None
    )
    if sched is not None and len(sched):
        rows = []
        for k, reason in enumerate(_REASON_COLORS):
            sel = io.rows(sched, sched["reason"] == reason)
            if reason == "observed":
                tr_ = io.rows(sel, sel["split"] == "train")["t"].to_numpy()
                ho_ = io.rows(sel, sel["split"] == "holdout")["t"].to_numpy()
                ax_t.vlines(ta(tr_), k - 0.4, k + 0.4, color=style.OBS_COLOR, lw=0.7)
                ax_t.vlines(ta(ho_), k - 0.4, k + 0.4, color=style.HOLDOUT_OBS_COLOR, lw=0.7)
            else:
                ax_t.vlines(
                    ta(sel["t"].to_numpy()), k - 0.4, k + 0.4, color=_REASON_COLORS[reason], lw=0.7
                )
            rows.append(f"{reason} ({len(sel)})")
        ax_t.set_yticks(range(len(rows)), rows)
        style.mark_holdout(ax_t, s.t_split, s.duration, ta.scale)
        ax_t.set_xlabel(ta.label)
        ax_t.set_title("candidate nights and why observations are missing", fontsize=9)
    else:
        _no_data(ax_t, "no schedule recorded")
    # Orbital-phase coverage: heliocentric longitude of the body at observation times.
    t_obs = np.concatenate([s.public_body(view.body_id)["t"], s.holdout_body(view.body_id)["t"]])
    split = np.concatenate(
        [np.zeros(len(s.public_body(view.body_id))), np.ones(len(s.holdout_body(view.body_id)))]
    )
    r = view.dense_true_r
    x = np.interp(t_obs, view.dense_t, r[:, 0])
    y = np.interp(t_obs, view.dense_t, r[:, 1])
    lon = np.degrees(np.arctan2(y, x)) % 360
    bins = np.linspace(0, 360, 37)
    ax_p.hist(
        [lon[split == 0], lon[split == 1]],
        bins=bins,
        stacked=True,
        color=[style.OBS_COLOR, style.HOLDOUT_OBS_COLOR],
        label=["train", "holdout"],
    )
    covered = np.mean(np.histogram(lon[split == 0], bins=bins)[0] > 0)
    ax_p.set_xlabel("heliocentric longitude of the body at observation (deg)")
    ax_p.set_ylabel("observations")
    ax_p.set_title(
        f"orbital-phase coverage (training covers {covered:.0%} of 10° bins)", fontsize=9
    )
    ax_p.legend(fontsize=7)
    fig.suptitle(f"Observation coverage — {s.scenario_name} (seed {s.seed})", fontsize=10)
    style.save_figure(
        fig,
        out,
        "observation_coverage" + _suffix(view),
        m,
        title="Observation coverage",
        caption=(
            "When observations exist and why gaps exist (weather, solar elongation, "
            "magnitude); and which parts of the orbit were observed."
        ),
        context=context(view),
    )


STATIC_FIGURES: list[Callable[[RunView, Path, FigureManifest], None]] = [
    sky_track,
    residuals_vs_time,
    orbit_topdown,
    orbit_3d,
    element_errors,
    observation_coverage,
]


def render_run(
    sealed: Sealed,
    jan: JanOutputs,
    out_dir: str | Path,
    *,
    static: bool = True,
    animate: bool = False,
    quick: bool = False,
) -> FigureManifest:
    """Render every per-run figure (and optionally animation) into `out_dir/figures`."""
    out = Path(out_dir)
    fig_dir = out / "figures"
    m = FigureManifest.load(out)
    for body in sealed.body_ids:
        view = RunView(sealed, jan, body)
        if static:
            from rudolf.viz.phase3_plots import PHASE3_FIGURES
            from rudolf.viz.phase4_plots import PHASE4_FIGURES
            from rudolf.viz.phase5_plots import PHASE5_FIGURES
            from rudolf.viz.posterior_plots import POSTERIOR_FIGURES

            figures = STATIC_FIGURES + POSTERIOR_FIGURES + PHASE3_FIGURES + PHASE4_FIGURES
            for f in figures + PHASE5_FIGURES:
                if not sealed.is_sky and f is sky_track:
                    continue
                f(view, fig_dir, m)
        if animate:
            from rudolf.viz.animations import PER_RUN_ANIMATIONS
            from rudolf.viz.writers import AnimationSettings

            settings = AnimationSettings(quick=quick)
            for a in PER_RUN_ANIMATIONS:
                a(view, fig_dir, m, settings)
    m.save()
    return m
