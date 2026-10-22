"""Cross-run animations (SPEC §7.1.5): `sweep_animation` and `strategy_race`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

from jt_core import io
from rudolf.data import JanOutputs, RunView, Sealed
from rudolf.viz import style
from rudolf.viz.manifest import FigureManifest
from rudolf.viz.run_plots import TimeAxis, _wrap_series, lonlat_deg
from rudolf.viz.writers import AnimationSettings, save_animation


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    return io.rows(df, df["status"] == "ok") if "status" in df.columns else df


def _views(results_dir: Path, rows: pd.DataFrame) -> list[RunView]:
    out: list[RunView] = []
    for _, r in rows.iterrows():
        run_dir = results_dir / str(r["run_dir"])
        sealed = Sealed.load(run_dir)
        out.append(RunView(sealed, JanOutputs.load(run_dir, str(r["label"])), sealed.body_ids[0]))
    return out


def _select(df: pd.DataFrame, ctx: dict[str, Any]) -> pd.DataFrame:
    d = _ok(df)
    scenario = ctx.get("scenario") or sorted(d["scenario"].unique())[0]
    d = io.rows(d, d["scenario"] == scenario)
    seed = ctx.get("seed", int(np.min(d["seed"].to_numpy())))
    return io.rows(d, d["seed"] == seed)


def sweep_animation(
    df: pd.DataFrame, out: Path, m: FigureManifest, ctx: dict[str, Any], results_dir: Path
) -> None:
    """Step a per-run figure (orbit_topdown or sky_track) through one swept parameter."""
    d = _select(df, ctx)
    x = ctx["x"] if ctx["x"] in d.columns else f"jan.{ctx['x']}"
    base = ctx.get("base_figure", "orbit_topdown")
    values = sorted(v for v in d[x].dropna().unique())
    if not values:
        return
    frames = []
    for v in values:
        rows = io.rows(d, d[x] == v)
        # Keep one variant per label (other swept parameters at their first value).
        rows = rows.drop_duplicates(subset=["label"])
        frames.append((v, _views(results_dir, rows)))
    style.apply_style()
    fig, ax = plt.subplots(figsize=(7, 6) if base == "orbit_topdown" else (9, 4.5))

    def draw(k: int) -> tuple[Any, ...]:
        v, views = frames[k]
        ax.cla()
        s = views[0].sealed
        if base == "orbit_topdown":
            tr, ob = views[0].dense_true_r, views[0].observer_r
            ax.plot(ob[:, 0], ob[:, 1], color=style.OBSERVER_COLOR, lw=0.7, alpha=0.6)
            ax.plot(tr[:, 0], tr[:, 1], **style.TRUTH_LINE)
            for view in views:
                p = view.dense_pred_r
                if p is not None:
                    ax.plot(
                        p[:, 0],
                        p[:, 1],
                        **style.strategy_style(view.jan.label).line_kwargs(view.jan.label),
                    )
            ax.scatter([0], [0], s=60, color=style.SUN_COLOR)
            lim = 1.5 * float(np.max(np.abs(tr[:, :2])))
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xlabel("x (AU)")
            ax.set_ylabel("y (AU)")
        else:
            ta = TimeAxis(s.duration)
            lon_t, _ = lonlat_deg(views[0].dense_true_u)
            ax.plot(*_wrap_series(ta(views[0].dense_t), lon_t), **style.TRUTH_LINE)
            for view in views:
                if view.dense_pred_u is not None:
                    lon_p, _ = lonlat_deg(view.dense_pred_u)
                    ax.plot(
                        *_wrap_series(ta(view.dense_t), lon_p),
                        **style.strategy_style(view.jan.label).line_kwargs(view.jan.label),
                    )
            pub = s.public_body(views[0].body_id)
            ax.scatter(ta(pub["t"]), np.degrees(pub["lon"]), **style.OBS_MARKER)
            style.mark_holdout(ax, s.t_split, s.duration, ta.scale)
            ax.set_ylim(0, 360)
            ax.set_xlabel(ta.label)
            ax.set_ylabel("λ (deg)")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"{s.scenario_name} (seed {s.seed}) — {x} = {v:g}", fontsize=10)
        return ()

    settings = AnimationSettings(fps=2, quick=bool(ctx.get("quick", False)))
    reps = 1 if settings.quick else 2  # hold each value for about a second
    seq = [k for k in range(len(frames)) for _ in range(reps)]
    anim = FuncAnimation(fig, draw, frames=seq, blit=False)
    name = f"sweep_animation__{base}__{x.replace('.', '_')}"
    path = save_animation(anim, out / name, settings)
    m.add(
        name=name,
        kind="animation",
        paths=[path],
        title=f"Sweep animation: {base} vs {x}",
        caption=f"{base} for every strategy as {x} steps through {', '.join(f'{v:g}' for v in values)} (truth in black).",
        context=ctx,
    )


def strategy_race(
    df: pd.DataFrame, out: Path, m: FigureManifest, ctx: dict[str, Any], results_dir: Path
) -> None:
    """Several strategies' orbit animations in a synchronized grid (same scenario and seed)."""
    d = _select(df, ctx).drop_duplicates(subset=["label"])
    views = _views(results_dir, d)
    if not views:
        return
    n = len(views)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    style.apply_style()
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.8 * rows), squeeze=False)
    s = views[0].sealed
    t = views[0].dense_t
    tr = views[0].dense_true_r
    ob = views[0].observer_r
    lim = 1.3 * float(np.max(np.abs(np.concatenate([tr[:, :2], ob[:, :2]]))))
    artists = []
    for ax, view in zip(axes.ravel(), views, strict=False):
        sty = style.strategy_style(view.jan.label)
        ax.plot(tr[:, 0], tr[:, 1], color=style.TRUTH_COLOR, lw=0.5, alpha=0.3)
        ax.plot(ob[:, 0], ob[:, 1], color=style.OBSERVER_COLOR, lw=0.5, alpha=0.3)
        ax.scatter([0], [0], s=30, color=style.SUN_COLOR)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(view.jan.label, fontsize=9, color=sty.color)
        (dt,) = ax.plot([], [], "o", color=style.TRUTH_COLOR, ms=5)
        (dp,) = ax.plot([], [], "o", color=sty.color, ms=5, mfc="none", mew=1.5)
        (do,) = ax.plot([], [], "o", color=style.OBSERVER_COLOR, ms=4)
        (los,) = ax.plot([], [], color=sty.color, lw=0.6, ls="--")
        artists.append((view, dt, dp, do, los))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    clock = fig.suptitle("", fontsize=10)

    def update(k: int) -> tuple[Any, ...]:
        for view, dt, dp, do, los in artists:
            dt.set_data([tr[k, 0]], [tr[k, 1]])
            do.set_data([ob[k, 0]], [ob[k, 1]])
            p = view.dense_pred_r
            if p is not None:
                dp.set_data([p[k, 0]], [p[k, 1]])
                los.set_data([ob[k, 0], p[k, 0]], [ob[k, 1], p[k, 1]])
            elif view.dense_pred_u is not None:
                end = ob[k] + 2 * lim * view.dense_pred_u[k]
                los.set_data([ob[k, 0], end[0]], [ob[k, 1], end[1]])
        phase = "HOLDOUT" if t[k] >= s.t_split else "training"
        clock.set_text(
            f"Strategy race — {s.scenario_name} (seed {s.seed})   t = {t[k]:.0f} d [{phase}]"
        )
        return ()

    settings = AnimationSettings(quick=bool(ctx.get("quick", False)))
    idx = np.unique(np.linspace(0, len(t) - 1, settings.frames()).round().astype(int))
    anim = FuncAnimation(fig, update, frames=idx, blit=False)
    path = save_animation(anim, out / "strategy_race", settings)
    m.add(
        name="strategy_race",
        kind="animation",
        paths=[path],
        title="Strategy race",
        caption="Every strategy's predicted body (open circle) against the truth (black) on the same scenario and seed, synchronized in time; dashed: predicted line of sight.",
        context=ctx,
    )


CROSS_RUN_ANIMATIONS: dict[
    str, Callable[[pd.DataFrame, Path, FigureManifest, dict[str, Any], Path], None]
] = {
    "sweep_animation": sweep_animation,
    "strategy_race": strategy_race,
}
