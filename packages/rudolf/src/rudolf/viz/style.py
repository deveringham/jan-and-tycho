"""Shared palette, strategy → color map and figure conventions (SPEC §7.1.1)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from rudolf.viz.manifest import FigureManifest

TRUTH_COLOR = "#222222"
OBSERVER_COLOR = "#2a7fbf"
SUN_COLOR = "#f2b01e"
OBS_COLOR = "#7a7a7a"
HOLDOUT_OBS_COLOR = "#d1495b"
HOLDOUT_SHADE = "#d1495b"
NOISE_REF_COLOR = "#999999"

OBS_MARKER: dict[str, Any] = {"marker": "o", "s": 6, "color": OBS_COLOR, "alpha": 0.6, "lw": 0}
HOLDOUT_MARKER: dict[str, Any] = {
    "marker": "D",
    "s": 10,
    "facecolors": "none",
    "edgecolors": HOLDOUT_OBS_COLOR,
    "linewidths": 0.7,
}
TRUTH_LINE: dict[str, Any] = {"color": TRUTH_COLOR, "lw": 2.4, "alpha": 0.8, "label": "truth"}

# Stable colors for the catalogue (SPEC §6.5). Unknown strategies get a hashed fallback.
_FIXED: dict[str, tuple[str, str]] = {
    "ptolemy.fourier": ("#e07a1f", "--"),
    "gp.sky": ("#c9a227", "--"),
    "classical.gauss_iod": ("#8fb339", ":"),
    "classical.lsq": ("#1b9e77", "-"),
    "bayes.mcmc": ("#7570b3", "-"),
    "bayes.hmc": ("#5e3c99", "-"),
    "bayes.nested": ("#b2abd2", "-."),
    "kepler.triangulate": ("#e7298a", "-"),
    "discovery.sindy": ("#66a61e", "-."),
    "discovery.pysr": ("#a6761d", "-."),
    "piml.neural_ode": ("#1f78b4", ":"),
    "piml.hnn": ("#a6cee3", ":"),
    "piml.ude": ("#fb9a99", "-"),
    "sbi.npe": ("#6a3d9a", "--"),
    "llm.hypothesize": ("#b15928", "--"),
}
_FALLBACK = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#76b7b2",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
]


@dataclass(frozen=True)
class StrategyStyle:
    color: str
    linestyle: str

    def line_kwargs(self, label: str | None = None) -> dict[str, Any]:
        kw: dict[str, Any] = {"color": self.color, "ls": self.linestyle, "lw": 1.1}
        if label is not None:
            kw["label"] = label
        return kw


def strategy_style(name: str) -> StrategyStyle:
    """Stable style for a strategy name, identical in every figure and animation."""
    if name in _FIXED:
        c, ls = _FIXED[name]
        return StrategyStyle(c, ls)
    h = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    return StrategyStyle(_FALLBACK[h % len(_FALLBACK)], ["-", "--", "-.", ":"][(h >> 8) % 4])


def strategy_color(name: str) -> str:
    return strategy_style(name).color


RC: dict[str, Any] = {
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.hashsalt": "jan-and-tycho",  # deterministic SVG ids
}


def apply_style() -> None:
    plt.style.use(RC)


def new_figure(width: float = 7.0, height: float = 4.0, **kw: Any) -> tuple[Figure, Any]:
    apply_style()
    kw.setdefault("layout", "constrained")
    fig, ax = plt.subplots(figsize=(width, height), **kw)
    return fig, ax


def mark_holdout(ax: Axes, t_train_end: float, t_end: float, scale: float = 1.0) -> None:
    """Vertical line at the end of training and shaded holdout region (SPEC §7.1.1)."""
    ax.axvline(t_train_end * scale, color=HOLDOUT_SHADE, lw=0.9, ls="--", alpha=0.8)
    ax.axvspan(t_train_end * scale, t_end * scale, color=HOLDOUT_SHADE, alpha=0.07, lw=0)


def save_figure(
    fig: Figure,
    out_dir: str | Path,
    name: str,
    manifest: FigureManifest | None = None,
    *,
    title: str = "",
    caption: str = "",
    context: dict[str, Any] | None = None,
    formats: tuple[str, ...] = ("png", "svg"),
) -> list[Path]:
    """Save a figure as PNG + SVG and record it in the manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        p = out / f"{name}.{fmt}"
        meta = {"Date": None} if fmt == "svg" else {"Software": None}
        fig.savefig(p, format=fmt, bbox_inches="tight", metadata=meta)
        paths.append(p)
    plt.close(fig)
    if manifest is not None:
        manifest.add(
            name=name,
            kind="figure",
            paths=paths,
            title=title or name,
            caption=caption,
            context=context or {},
        )
    return paths
