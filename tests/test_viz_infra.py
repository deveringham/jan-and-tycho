"""Phase 0 visualization infrastructure: style, manifest, animation writer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation

from rudolf.viz import style, writers
from rudolf.viz.manifest import FigureManifest


def test_strategy_colors_stable_and_distinct() -> None:
    assert style.strategy_style("classical.lsq") == style.strategy_style("classical.lsq")
    assert style.strategy_color("classical.lsq") != style.strategy_color("ptolemy.fourier")
    assert style.strategy_style("some.unknown") == style.strategy_style("some.unknown")
    assert style.strategy_color("classical.lsq") != style.TRUTH_COLOR


def test_save_figure_and_manifest(tmp_path: Path) -> None:
    m = FigureManifest(tmp_path)
    fig, ax = style.new_figure()
    ax.plot([0, 1], [0, 1])
    style.mark_holdout(ax, 0.7, 1.0)
    paths = style.save_figure(
        fig, tmp_path / "figs", "demo", m, title="Demo", caption="c", context={"strategy": "x"}
    )
    m.save()
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    m2 = FigureManifest.load(tmp_path)
    entry = m2.get("demo")
    assert entry is not None and entry.paths == ["figs/demo.png", "figs/demo.svg"]
    assert entry.context == {"strategy": "x"}


def _tiny_animation() -> FuncAnimation:
    fig, ax = plt.subplots(figsize=(2, 2))
    (line,) = ax.plot([], [])
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    xs = np.linspace(0, 1, 20)

    def update(k: int):
        line.set_data(xs, np.sin(xs * 6 + k))
        return (line,)

    return FuncAnimation(fig, update, frames=4, blit=False)


def test_animation_gif_fallback(tmp_path: Path) -> None:
    s = writers.AnimationSettings(quick=True, force_gif=True)
    p = writers.save_animation(_tiny_animation(), tmp_path / "anim", s)
    assert p.suffix == ".gif" and p.stat().st_size > 0


def test_animation_prefers_mp4_when_ffmpeg_present(tmp_path: Path) -> None:
    p = writers.save_animation(
        _tiny_animation(), tmp_path / "anim", writers.AnimationSettings(quick=True)
    )
    expected = ".mp4" if writers.ffmpeg_available() else ".gif"
    assert p.suffix == expected and p.stat().st_size > 0


@pytest.mark.skipif(not writers.ffmpeg_available(), reason="ffmpeg not installed")
def test_animation_mp4(tmp_path: Path) -> None:
    p = writers.save_animation(
        _tiny_animation(), tmp_path / "anim", writers.AnimationSettings(quick=True)
    )
    assert p.suffix == ".mp4"
