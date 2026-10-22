"""Per-run posterior figures (SPEC §7.1.2, Phase 2): corner, posterior predictive, model
probabilities. Strategies without posterior samples fall back to a Gaussian approximation
from their covariance (and say so), or the figure notes that the layer is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import corner
import numpy as np
from numpy.typing import NDArray

from jt_core.angles import tangent_residuals
from jt_core.constants import ARCSEC, DEG
from rudolf.data import RunView, sep_arcsec
from rudolf.metrics.parameters import element_errors, sample_errors
from rudolf.metrics.selection import selection_metrics
from rudolf.viz import style
from rudolf.viz.manifest import FigureManifest
from rudolf.viz.run_plots import TimeAxis, _no_data, _suffix, _title, context

FloatArray = NDArray[np.float64]

_LABELS = {
    "q": ("q", "AU", 1.0),
    "e": ("e", "", 1.0),
    "i": ("i", "deg", 1 / DEG),
    "Omega": ("Ω", "deg", 1 / DEG),
    "omega": ("ω", "deg", 1 / DEG),
    "t_p": ("t_p", "d", 1.0),
}


def _element_draws(view: RunView, n: int = 2000) -> tuple[dict[str, FloatArray], str] | None:
    """Element samples (errors relative to truth, wrapped) and their provenance."""
    est = view.jan.elements(view.body_id)
    tb = view.sealed.body_truth(view.body_id)
    if est is None or not tb.get("elements"):
        return None
    truth = tb["elements"]
    gm = float(tb.get("gm", 0.01720209895**2))
    if est.get("samples"):
        return sample_errors(truth, est["samples"], gm), "posterior samples"
    if est.get("cov") is None:
        return None
    names = est["cov_names"]
    g = np.random.default_rng(0)
    pt = np.array([est["point"][k] for k in names])
    cov = np.asarray(est["cov"], dtype=np.float64)
    w, v = np.linalg.eigh(0.5 * (cov + cov.T))
    draws = pt + g.standard_normal((n, len(pt))) @ (v * np.sqrt(np.clip(w, 0, None))).T
    samp = {k: draws[:, j].tolist() for j, k in enumerate(names)}
    return sample_errors(truth, samp, gm), "Gaussian approximation from the reported covariance"


def corner_plot(view: RunView, out: Path, m: FigureManifest) -> None:
    res = _element_draws(view)
    errs = None
    if res is not None:
        tb = view.sealed.body_truth(view.body_id)
        errs = element_errors(
            tb["elements"],
            view.jan.elements(view.body_id)["point"],  # type: ignore[index]
            float(tb.get("gm", 0.01720209895**2)),
        )
    skip = set()
    if errs is not None:
        if errs["Omega_deg"] is None:
            skip.add("Omega")
        if errs["omega_deg"] is None:
            skip.add("omega")
        if errs["t_p_days"] is None:
            skip.add("t_p")
    keys = [k for k in _LABELS if res is not None and k in res[0] and k not in skip]
    if res is None or len(keys) < 2:
        fig, ax = style.new_figure(5, 3)
        _no_data(ax, "no posterior or covariance for orbital elements")
        cap = "Omitted: the hypothesis reports no element uncertainty."
    else:
        draws, source = res
        data = np.column_stack([np.asarray(draws[k]) * _LABELS[k][2] for k in keys])
        labels = [
            f"Δ{_LABELS[k][0]}" + (f" ({_LABELS[k][1]})" if _LABELS[k][1] else "") for k in keys
        ]
        style.apply_style()
        fig = corner.corner(
            data,
            labels=labels,
            truths=[0.0] * len(keys),
            truth_color=style.TRUTH_COLOR,
            color=style.strategy_color(view.jan.label),
            quantiles=[0.025, 0.5, 0.975],
            show_titles=True,
            title_fmt=".2g",
            title_kwargs={"fontsize": 7},
            label_kwargs={"fontsize": 8},
            plot_datapoints=len(data) < 3000,
            plot_contours=len(data) >= 500,
            plot_density=len(data) >= 500,
            fill_contours=False,
        )
        fig.suptitle(_title(view, "Posterior (element errors)"), fontsize=10, y=1.02)
        cap = (
            f"Posterior of the orbital-element errors (estimate − truth; truth = 0, black lines) "
            f"from {source}. Dashed: 2.5/50/97.5% quantiles."
        )
        if skip:
            cap += " Degenerate elements omitted: " + ", ".join(sorted(skip)) + "."
    style.save_figure(
        fig,
        out,
        "corner" + _suffix(view),
        m,
        title="Posterior corner plot",
        caption=cap,
        context=context(view),
    )


def posterior_predictive(view: RunView, out: Path, m: FigureManifest) -> None:
    s = view.sealed
    sty = style.strategy_style(view.jan.label)
    ta = TimeAxis(s.duration)
    samples = view.jan.sample_array("dense", view.body_id, "u")
    fig, (ax1, ax2) = style.new_figure(8.0, 6.0, nrows=2, sharex=True)
    if not s.is_sky or view.dense_pred_u is None:
        _no_data(ax1, "no sky prediction")
        _no_data(ax2, "")
        cap = "Omitted: no sky prediction."
    else:
        t = ta(view.dense_t)
        truth = view.dense_true_u
        if samples is not None:
            sep = np.stack([sep_arcsec(x, truth) for x in samples])
            q = np.quantile(sep, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0)
            ax1.fill_between(
                t, np.maximum(q[0], 1e-3), q[4], color=sty.color, alpha=0.15, lw=0, label="95% band"
            )
            ax1.fill_between(
                t, np.maximum(q[1], 1e-3), q[3], color=sty.color, alpha=0.3, lw=0, label="68% band"
            )
            east = np.stack([tangent_residuals(x, truth)[:, 0] for x in samples]) / ARCSEC
            qe = np.quantile(east, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0)
            ax2.fill_between(t, qe[0], qe[4], color=sty.color, alpha=0.15, lw=0)
            ax2.fill_between(t, qe[1], qe[3], color=sty.color, alpha=0.3, lw=0)
            ax2.plot(t, qe[2], color=sty.color, lw=0.8)
            cap = (
                "Posterior predictive vs the noise-free truth: top, great-circle distance of "
                "predictive draws from the truth (68/95% bands, log scale); bottom, their signed "
                "east (≈ along-track) offset."
            )
        else:
            cov = view.jan.sky_cov("dense", view.body_id)
            mean_e = tangent_residuals(view.dense_pred_u, truth)[:, 0] / ARCSEC
            ax2.plot(t, mean_e, color=sty.color, lw=0.8)
            if cov is not None:
                sd = np.sqrt(np.clip(cov[:, 0, 0], 0, None)) / ARCSEC
                ax2.fill_between(
                    t, mean_e - 1.96 * sd, mean_e + 1.96 * sd, color=sty.color, alpha=0.15, lw=0
                )
                ax2.fill_between(t, mean_e - sd, mean_e + sd, color=sty.color, alpha=0.3, lw=0)
            cap = (
                "Predictive error vs the noise-free truth with Gaussian bands from the reported "
                "covariance (no predictive samples)."
                if cov is not None
                else "Predictive error vs truth. Omitted: predictive uncertainty (none reported)."
            )
        ax1.plot(
            t,
            np.maximum(view.dense_err_arcsec, 1e-3),
            color=sty.color,
            lw=1.0,
            label="predictive mean",
        )  # type: ignore[arg-type]
        if s.noise_sigma_arcsec:
            ax1.axhline(
                s.noise_sigma_arcsec, color=style.NOISE_REF_COLOR, ls="--", lw=1, label="noise σ"
            )
        ax1.set_yscale("log")
        ax1.set_ylabel("distance from truth (arcsec)")
        ax1.legend(fontsize=7, loc="upper left")
        ax2.axhline(0, color=style.TRUTH_COLOR, lw=0.8)
        lin = max(s.noise_sigma_arcsec or 1.0, 1.0)
        lo_y, hi_y = ax2.get_ylim()
        if max(abs(lo_y), abs(hi_y)) > 20 * lin:  # only compress when offsets dwarf the noise
            ax2.set_yscale("symlog", linthresh=lin)
        ax2.set_ylabel("east offset (arcsec)")
        ax2.set_xlabel(ta.label)
        for ax in (ax1, ax2):
            style.mark_holdout(ax, s.t_split, s.duration, ta.scale)
    fig.suptitle(_title(view, "Posterior predictive"), fontsize=10)
    style.save_figure(
        fig,
        out,
        "posterior_predictive" + _suffix(view),
        m,
        title="Posterior predictive",
        caption=cap,
        context=context(view),
    )


def model_probabilities(view: RunView, out: Path, m: FigureManifest) -> None:
    comp = view.jan.model_comparison()
    sel = selection_metrics(view.sealed.body_truth(view.body_id), comp)
    fig, ax = style.new_figure(6.5, 3.2)
    if not comp or all(c.get("posterior_prob") is None for c in comp):
        _no_data(ax, "strategy does not compare model families")
        cap = "Omitted: no model comparison reported."
    else:
        names = [c["model_name"] for c in comp]
        probs = np.array([c.get("posterior_prob") or 0.0 for c in comp])
        lnz = [c.get("log_evidence") for c in comp]
        ys = np.arange(len(names))[::-1]
        true_fam = sel["true_family"]
        colors = [
            style.TRUTH_COLOR if n == true_fam else style.strategy_color(view.jan.label)
            for n in names
        ]
        ax.barh(ys, np.maximum(probs, 1e-300), color=colors, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlim(1e-12, 2)
        for y, p, z in zip(ys, probs, lnz, strict=True):
            txt = f" p={p:.3g}" + (f", ln Z={z:.1f}" if z is not None else "")
            ax.text(max(p, 1.5e-12), y, txt, va="center", fontsize=7)
        ax.set_yticks(ys, [n + (" (true)" if n == true_fam else "") for n in names])
        ax.set_xlabel("posterior model probability (log scale)")
        cap = "Posterior probability of each candidate family (equal prior odds); the true family is drawn in black."
    ax.set_title(_title(view, "Model probabilities"), fontsize=10)
    style.save_figure(
        fig,
        out,
        "model_probabilities" + _suffix(view),
        m,
        title="Model probabilities",
        caption=cap,
        context=context(view),
    )


POSTERIOR_FIGURES: list[Any] = [corner_plot, posterior_predictive, model_probabilities]
