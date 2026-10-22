"""`jt` — the single command-line entry point (SPEC §9.1)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

os.environ.setdefault("MPLBACKEND", "Agg")

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="Jan and Tycho: law discovery testbed."
)
tycho_app = typer.Typer(no_args_is_help=True, help="Tycho: generate observations.")
jan_app = typer.Typer(no_args_is_help=True, help="Jan: fit strategies to public data.")
rudolf_app = typer.Typer(no_args_is_help=True, help="Rudolf: score, plot, animate, report.")
bench_app = typer.Typer(no_args_is_help=True, help="Bench: sweeps and comparison reports.")
list_app = typer.Typer(no_args_is_help=True, help="List scenarios or strategies.")
app.add_typer(tycho_app, name="tycho")
app.add_typer(jan_app, name="jan")
app.add_typer(rudolf_app, name="rudolf")
app.add_typer(bench_app, name="bench")
app.add_typer(list_app, name="list")

SCENARIO_DIR = Path("scenarios")


def _config(path: Path | None) -> dict[str, Any] | None:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path else None


# ----------------------------------------------------------------------------- list


@list_app.command("scenarios")
def list_scenarios(
    directory: Annotated[Path, typer.Option(help="Scenario directory.")] = SCENARIO_DIR,
) -> None:
    """Scenario YAMLs in the library."""
    for p in sorted(directory.glob("*.yaml")):
        desc = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("description", "")
        typer.echo(f"{p.stem:24s} {desc}")


@list_app.command("strategies")
def list_strategies() -> None:
    """Registered Jan strategies and their supported tiers."""
    import jan

    for name in jan.list_strategies():
        s = jan.get_strategy(name)
        cls = type(getattr(s, "inner", s))
        doc = (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
        typer.echo(f"{name:22s} tiers={','.join(sorted(s.supported_tiers)):12s} {doc}")


# ----------------------------------------------------------------------------- tycho


@tycho_app.command("generate")
def tycho_generate(
    scenario: Path,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 1,
    out: Annotated[Path, typer.Option(help="Root output directory.")] = Path("runs"),
    force: Annotated[bool, typer.Option(help="Regenerate even if the run exists.")] = False,
) -> None:
    """Simulate a scenario into runs/<scenario>/<seed>/{public,sealed}."""
    from jt_bench.harness import ensure_run
    from tycho import load_scenario

    d = ensure_run(load_scenario(scenario), seed, out, force=force)
    typer.echo(str(d))


# ----------------------------------------------------------------------------- jan


@jan_app.command("fit")
def jan_fit(
    run_dir: Path,
    strategy: Annotated[str, typer.Option(help="Registry name, e.g. classical.lsq.")],
    config: Annotated[Path | None, typer.Option(help="Strategy config YAML.")] = None,
    label: Annotated[
        str | None, typer.Option(help="Output label (default: strategy[@hash]).")
    ] = None,
) -> None:
    """Fit a strategy on a run's public bundle and write predictions."""
    from jt_bench.harness import fit

    res = fit(run_dir, strategy, _config(config), label=label)
    typer.echo(f"{res.label}: fitted in {res.runtime_s:.1f}s → {res.jan_dir}")


# ----------------------------------------------------------------------------- rudolf


@rudolf_app.command("score")
def rudolf_score(
    run_dir: Path,
    strategy: Annotated[str, typer.Option(help="Strategy label under jan/.")],
    animate: Annotated[bool, typer.Option(help="Also render animations.")] = False,
    no_plots: Annotated[bool, typer.Option("--no-plots", help="Skip all visualization.")] = False,
    quick: Annotated[bool, typer.Option(help="Short, low-resolution animations.")] = False,
) -> None:
    """Score a fitted strategy against the sealed truth (+ figures)."""
    from rudolf.score import score_run

    sc = score_run(
        run_dir, strategy, plots=not no_plots, animate=animate and not no_plots, quick=quick
    )
    _print_score(sc)


@rudolf_app.command("report")
def rudolf_report(run_dir: Path, out: Annotated[Path | None, typer.Option()] = None) -> None:
    """Single-run HTML report across all scored strategies."""
    from rudolf.report import run_report

    typer.echo(str(run_report(run_dir, out)))


@rudolf_app.command("animate")
def rudolf_animate(
    run_dir: Path,
    strategy: Annotated[str, typer.Option(help="Strategy label under jan/.")],
    which: Annotated[str, typer.Option(help="orbit_animation | sky_animation")] = "orbit_animation",
    quick: Annotated[bool, typer.Option()] = False,
) -> None:
    """Render a single animation."""
    from rudolf.data import JanOutputs, RunView, Sealed
    from rudolf.viz import animations
    from rudolf.viz.manifest import FigureManifest
    from rudolf.viz.writers import AnimationSettings

    fn = {f.__name__: f for f in animations.PER_RUN_ANIMATIONS}.get(which)
    if fn is None:
        raise typer.BadParameter(f"unknown animation {which!r}")
    sealed, jan_out = Sealed.load(run_dir), JanOutputs.load(run_dir, strategy)
    out = Path(run_dir) / "rudolf" / strategy
    m = FigureManifest.load(out)
    for body in sealed.body_ids:
        fn(RunView(sealed, jan_out, body), out / "figures", m, AnimationSettings(quick=quick))
    m.save()
    typer.echo(str(out / "figures"))


def _print_score(sc: dict[str, Any]) -> None:
    for body, b in sc["bodies"].items():
        p = b["predictive"]
        rms = p.get("holdout_rms_arcsec", p.get("holdout_rms_au"))
        unit = "″" if "holdout_rms_arcsec" in p else " AU"
        line = (
            f"{sc['label']} [{body}] kind={sc['kind']} holdout RMS={rms:.3g}{unit}"
            if rms is not None
            else (f"{sc['label']} [{body}] kind={sc['kind']} (no holdout prediction)")
        )
        tr = b["trajectory"].get("traj_rms_holdout_au")
        if tr is not None:
            line += f"  traj RMS (holdout)={tr:.3g} AU"
        errs = (b["parameters"] or {}).get("errors")
        if errs:
            show = {
                k: errs[k]
                for k in ("a_rel", "e", "i_deg", "Omega_deg", "omega_deg")
                if errs.get(k) is not None
            }
            line += "  " + " ".join(f"{k}={v:+.2g}" for k, v in show.items())
        typer.echo(line)


# ----------------------------------------------------------------------------- run


@app.command("run")
def run(
    scenario: Path,
    strategy: Annotated[list[str], typer.Option(help="Strategy (repeatable).")],
    seed: Annotated[int, typer.Option()] = 1,
    config: Annotated[
        Path | None, typer.Option(help="Strategy config YAML (single strategy).")
    ] = None,
    out: Annotated[Path, typer.Option()] = Path("runs"),
    animate: Annotated[bool, typer.Option()] = False,
    no_plots: Annotated[bool, typer.Option("--no-plots")] = False,
    quick: Annotated[bool, typer.Option()] = False,
    report: Annotated[bool, typer.Option(help="Write the single-run HTML report.")] = True,
) -> None:
    """Generate + fit + score + plots for one scenario and seed."""
    from jt_bench.harness import ensure_run, fit, score
    from tycho import load_scenario

    d = ensure_run(load_scenario(scenario), seed, out)
    cfg = _config(config)
    for s in strategy:
        fit_cfg = cfg if len(strategy) == 1 else None
        if animate:  # animations need the fit trace
            fit_cfg = {**(fit_cfg or {}), "record_trace": True}
        res = fit(d, s, fit_cfg)
        sc = score(d, res.label, plots=not no_plots, animate=animate and not no_plots, quick=quick)
        _print_score(sc)
    if report and not no_plots:
        from rudolf.report import run_report

        typer.echo(f"report: {run_report(d)}")


# ----------------------------------------------------------------------------- bench


@bench_app.command("sweep")
def bench_sweep(
    config: Path,
    workers: Annotated[
        int, typer.Option(help="Parallel worker processes (0 = inline, no timeouts).")
    ] = 1,
    out: Annotated[
        Path | None, typer.Option(help="Results directory (default runs/sweeps/<name>).")
    ] = None,
    quick: Annotated[bool, typer.Option(help="Quick animations.")] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Recompute every cell.")] = False,
) -> None:
    """Run a sweep (cached per cell) and aggregate results.parquet."""
    from jt_bench.sweep import load_sweep, run_sweep

    cfg = load_sweep(config)
    results = out or Path("runs") / "sweeps" / cfg.name
    run_sweep(cfg, results, workers=workers, quick=quick, log=typer.echo, use_cache=not no_cache)


@bench_app.command("report")
def bench_report(results_dir: Path, out: Annotated[Path | None, typer.Option()] = None) -> None:
    """Comparison figures + static HTML report for a sweep."""
    from rudolf.report import bench_report as _bench_report

    typer.echo(str(_bench_report(results_dir, out)))


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
