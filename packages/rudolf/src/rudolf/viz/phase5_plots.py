"""Phase 5 per-run figures: `sbi.npe` posteriors round by round (amortized vs
per-scenario)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.ticker import SymmetricalLogLocator
from numpy.typing import NDArray

from rudolf.data import RunView
from rudolf.metrics.parameters import sample_errors
from rudolf.viz import style
from rudolf.viz.manifest import FigureManifest
from rudolf.viz.posterior_plots import _LABELS
from rudolf.viz.run_plots import _suffix, _title, context

FloatArray = NDArray[np.float64]


def _info(view: RunView) -> dict[str, Any]:
    bodies = (view.jan.hypothesis.get("hypothesis") or {}).get("bodies") or {}
    return (bodies.get(view.body_id) or {}).get("info") or {}


def _rounds(view: RunView) -> list[dict[str, Any]] | None:
    rounds = _info(view).get("rounds")
    return rounds if rounds and all(r.get("samples") for r in rounds) else None


def npe_rounds(view: RunView, out: Path, m: FigureManifest) -> None:
    """Forest plot of each NPE round's marginal posteriors, as errors from the truth."""
    rounds = _rounds(view)
    tb = view.sealed.body_truth(view.body_id)
    if rounds is None or not tb.get("elements"):
        return
    gm = float(tb.get("gm", 0.01720209895**2))
    errs = [sample_errors(tb["elements"], r["samples"], gm) for r in rounds]
    names = [k for k in _LABELS if all(k in e for e in errs)]
    sty = style.strategy_style(view.jan.label)
    fig, axes = style.new_figure(11.0, 1.2 + 0.45 * len(rounds) * 2, ncols=3, nrows=2)
    ys = np.arange(len(rounds))[::-1]
    for ax, k in zip(axes.ravel(), names, strict=False):
        sym, unit, scale = _LABELS[k]
        q = [np.quantile(np.asarray(e[k]) * scale, [0.025, 0.16, 0.5, 0.84, 0.975]) for e in errs]
        # Errors in units of the final round's σ (68 % half-width): one symlog axis for all.
        sig = max(float(q[-1][3] - q[-1][1]) / 2, 1e-300)
        for y, qq in zip(ys, q, strict=True):
            z = qq / sig
            ax.plot([z[0], z[4]], [y, y], color=sty.color, lw=1)
            ax.plot([z[1], z[3]], [y, y], color=sty.color, lw=4, solid_capstyle="butt")
            ax.plot(z[2], y, "o", color="white", mec=sty.color, ms=4)
        ax.axvline(0.0, color=style.TRUTH_COLOR, lw=1.2)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.xaxis.set_major_locator(SymmetricalLogLocator(base=100, linthresh=1.0))
        ax.set_xlabel(
            f"({sym} − truth) / σ_final, σ_final = {sig:.2g}" + (f" {unit}" if unit else ""),
            fontsize=8,
        )
        ax.set_yticks(ys, [r["label"] for r in rounds], fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", visible=False)
    for ax in axes.ravel()[len(names) :]:
        ax.set_visible(False)
    fig.suptitle(_title(view, "NPE posterior by round"), fontsize=10)
    n_sims = sum(int(r.get("n_sims", 0)) for r in rounds)
    style.save_figure(
        fig,
        out,
        "npe_rounds" + _suffix(view),
        m,
        title="NPE: amortized vs per-scenario posterior",
        caption=f"Marginal posteriors of the elements after each round of neural posterior estimation, as errors from the truth in units of the final round's σ (68 % half-width; median, 68 % and 95 % intervals on a symmetric-log axis, linear within ±1). Round 1 is trained on the whole prior (amortized: valid for any data on this schedule); later rounds retrain on the prior truncated to the previous round's posterior, focusing on these data. {len(rounds)} rounds, {n_sims} simulations from Jan's own family in total.",
        context=context(view),
    )


def hypothesis_history(view: RunView, out: Path, m: FigureManifest) -> None:
    """`llm.hypothesize`: every family fitted, by the iteration that proposed it, with its
    log-evidence relative to the best; refused proposals listed with the reason."""
    info = _info(view)
    cands = info.get("candidates")
    if not cands:
        return
    sty = style.strategy_style(view.jan.label)
    best = max(c["ln_evidence"] for c in cands)
    fig, (ax, tx) = style.new_figure(12.0, 4.6, ncols=2, gridspec_kw={"width_ratios": [1.3, 1]})
    for c in cands:
        dz = c["ln_evidence"] - best
        nested = c["method"] == "nested"
        chosen = c["family"] == info.get("best_family")
        ax.plot(
            c["iteration"],
            dz,
            "*" if chosen else ("o" if nested else "s"),
            ms=13 if chosen else 7,
            color=sty.color if nested or chosen else "white",
            mec=sty.color,
        )
        ax.annotate(
            c["family"], (c["iteration"], dz), xytext=(6, 3), textcoords="offset points", fontsize=7
        )
    ax.set_yscale("symlog", linthresh=1.0)
    n_it = max(c["iteration"] for c in cands)
    ax.set_xticks(range(n_it + 1), ["start", *[str(k) for k in range(1, n_it + 1)]])
    ax.set_xlim(-0.4, n_it + 1.2)
    ax.set_xlabel("iteration that proposed the family")
    ax.set_ylabel("ln Z − ln Z_best")
    ax.plot([], [], "*", color=sty.color, ms=10, label="chosen")
    ax.plot([], [], "o", color=sty.color, label="nested-sampling evidence")
    ax.plot([], [], "s", color="white", mec=sty.color, label="Laplace evidence")
    ax.legend(fontsize=7, loc="center right")
    lines: list[str] = []
    for h in info.get("history", []):
        if "analysis" not in h:
            continue
        lines.append(f"iteration {h['iteration']}" + (" (stop)" if h.get("stop") else ""))
        for p in h.get("proposals", []):
            prop = p.get("proposal", {})
            what = prop.get("family") or prop.get("radial") or "?"
            tag = p.get("family") or "refused: " + str(p.get("error", ""))[:60]
            lines.append(f"  • {what[:48]} → {tag}")
    tx.axis("off")
    tx.text(
        0,
        1,
        "\n".join(lines) or "no LLM iterations",
        va="top",
        family="monospace",
        fontsize=7,
        transform=tx.transAxes,
    )
    fig.suptitle(_title(view, "LLM hypothesis history"), fontsize=10)
    use = info.get("usage") or {}
    style.save_figure(
        fig,
        out,
        "hypothesis_history" + _suffix(view),
        m,
        title="LLM hypothesis history",
        caption=f"Families proposed by {info.get('llm_model', 'the LLM')} at each iteration of the propose → fit → compare loop, with their log-evidence relative to the best (filled: nested sampling; hollow: Laplace approximation; star: chosen). Right: each iteration's proposals and what became of them (fitted family, or why it was refused). {use.get('calls', 0)} LLM calls, all cached for replay.",
        context=context(view),
    )


PHASE5_FIGURES = [npe_rounds, hypothesis_history]
