"""Grid sweeps over scenarios × Tycho overrides × seeds × strategies × Jan overrides (SPEC §9.2).

Each (run, strategy) cell executes in a separate process. Runs (Tycho output) are generated
first, once per (scenario variant, seed), so strategies on the same run never race.
Every swept parameter becomes its own column in `results.parquet`.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import pickle
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from jt_core import io
from tycho.scenario import Scenario, apply_overrides

VizScope = Literal["none", "first_seed", "all"]


class SeedRange(BaseModel):
    start: int = 0
    count: int = 1


class VizConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_run: VizScope = "first_seed"
    animations: VizScope = "none"
    comparison: list[dict[str, Any]] = Field(
        default_factory=lambda: [{"figure": "leaderboard"}, {"figure": "extrapolation_curves"}]
    )


class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    scenarios: list[str]
    overrides: dict[str, list[Any]] = Field(default_factory=dict[str, list[Any]])
    strategies: list[str]
    strategy_overrides: dict[str, dict[str, list[Any]]] = Field(
        default_factory=dict[str, dict[str, list[Any]]]
    )
    seeds: SeedRange = SeedRange()
    timeout_minutes: float = 30.0
    viz: VizConfig = VizConfig()


def load_sweep(path: str | Path) -> SweepConfig:
    return SweepConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _grid(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not axes:
        return [{}]
    keys = list(axes)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*axes.values())]


def _variant_key(overrides: dict[str, Any]) -> str:
    if not overrides:
        return "base"
    readable = "_".join(f"{k.split('.')[-1]}={v}" for k, v in overrides.items())
    digest = hashlib.sha256(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:6]
    safe = "".join(c if c.isalnum() or c in "=._-" else "-" for c in readable)[:60]
    return f"{safe}__{digest}"


@dataclass
class RunSpec:
    scenario: Scenario
    seed: int
    overrides: dict[str, Any]
    run_dir: Path


@dataclass
class CellSpec:
    run: RunSpec
    strategy: str
    config: dict[str, Any]
    plots: bool
    animate: bool
    quick: bool = False
    params: dict[str, Any] = field(default_factory=dict[str, Any])
    key: str = ""


def _in_scope(scope: VizScope, is_first_seed: bool) -> bool:
    return scope == "all" or (scope == "first_seed" and is_first_seed)


def expand(
    cfg: SweepConfig, results_dir: Path, root: Path = Path(".")
) -> tuple[list[RunSpec], list[CellSpec]]:
    runs: list[RunSpec] = []
    cells: list[CellSpec] = []
    first_seed = cfg.seeds.start
    for sc_path in cfg.scenarios:
        raw = yaml.safe_load((root / sc_path).read_text(encoding="utf-8"))
        for ov in _grid(cfg.overrides):
            sc = Scenario.model_validate(apply_overrides(raw, ov))
            for seed in range(cfg.seeds.start, cfg.seeds.start + cfg.seeds.count):
                per_run = _in_scope(cfg.viz.per_run, seed == first_seed)
                animate = _in_scope(cfg.viz.animations, seed == first_seed)
                rd = results_dir / "cells" / sc.name / _variant_key(ov) / str(seed)
                run = RunSpec(sc, seed, ov, rd)
                runs.append(run)
                for strat in cfg.strategies:
                    for sov in _grid(cfg.strategy_overrides.get(strat, {})):
                        # Animated cells record fit traces (fit_progress, posterior_samples).
                        fit_cfg = {**sov, "record_trace": True} if animate else sov
                        cells.append(
                            CellSpec(
                                run,
                                strat,
                                fit_cfg,
                                per_run,
                                animate,
                                params={**ov, **{f"jan.{k}": v for k, v in sov.items()}},
                            )
                        )
    return runs, cells


def _generate(run: RunSpec) -> dict[str, Any]:
    from jt_bench.harness import ensure_run

    ensure_run(run.scenario, run.seed, run_dir=run.run_dir)
    return {"status": "ok", "run_dir": str(run.run_dir)}


def _execute(cell: CellSpec) -> dict[str, Any]:
    from jt_bench.harness import fit, score

    row: dict[str, Any] = {"run_dir": str(cell.run.run_dir), "strategy": cell.strategy}
    try:
        res = fit(cell.run.run_dir, cell.strategy, cell.config or None)
        sc = score(
            cell.run.run_dir, res.label, plots=cell.plots, animate=cell.animate, quick=cell.quick
        )
        (res.jan_dir / CACHE_FILE).write_text(cell.key, encoding="utf-8")
        row.update({"status": "ok", "label": res.label, "score": sc})
    except Exception as exc:  # a failing cell must not sink the sweep
        row.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    return row


# ----------------------------------------------------------------------------- caching

CACHE_FILE = "cache_key.txt"
_PACKAGES = ("jt_core", "tycho", "jan", "rudolf", "jt_bench")


def code_version(packages: tuple[str, ...] = _PACKAGES) -> str:
    """Hash of the source of the given packages: any code change invalidates caches."""
    import importlib

    h = hashlib.sha256()
    for name in packages:
        mod = importlib.import_module(name)
        root = Path(str(mod.__file__)).parent
        for f in sorted(root.rglob("*.py")):
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def cell_key(cell: CellSpec, code: str) -> str:
    blob = json.dumps(
        {
            "scenario": cell.run.scenario.model_dump(mode="json"),
            "seed": cell.run.seed,
            "strategy": cell.strategy,
            "config": cell.config,
            "code": code,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def _cached(cell: CellSpec) -> dict[str, Any] | None:
    from jt_bench.harness import strategy_label

    label = strategy_label(cell.strategy, cell.config or None)
    jan_dir = cell.run.run_dir / io.JAN_DIR / label
    rud = cell.run.run_dir / io.RUDOLF_DIR / label
    key_file = jan_dir / CACHE_FILE
    if not key_file.exists() or key_file.read_text(encoding="utf-8") != cell.key:
        return None
    if not (rud / io.SCORE_FILE).exists():
        return None
    if cell.plots and not (rud / io.FIGURES_MANIFEST).exists():
        return None
    if cell.animate:
        from rudolf.viz.manifest import FigureManifest

        if FigureManifest.load(rud).get("orbit_animation") is None:
            return None
    return {
        "run_dir": str(cell.run.run_dir),
        "strategy": cell.strategy,
        "status": "ok",
        "label": label,
        "score": io.read_json(rud / io.SCORE_FILE),
        "cached": True,
    }


# ----------------------------------------------------------------------------- execution


def _child(fn: Callable[[Any], dict[str, Any]], item: Any, path: str) -> None:
    out = fn(item)
    Path(path).write_bytes(pickle.dumps(out))


@contextlib.contextmanager
def _single_threaded(enabled: bool) -> Iterator[None]:
    """Spawned workers inherit the environment: one compute thread each, so N workers use
    N cores instead of oversubscribing (each XLA/BLAS runtime would start its own pool)."""
    if not enabled:
        yield
        return
    keys = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    }
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_pool(
    fn: Callable[[Any], dict[str, Any]],
    items: list[Any],
    workers: int,
    timeout_s: float | None,
    on_done: Callable[[int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run `fn` over items, each in its own spawned process, killing any that time out.

    `workers=0` runs everything inline (no isolation, no timeouts; for debugging/tests).
    """
    out: list[dict[str, Any] | None] = [None] * len(items)
    if workers <= 0:
        for k, it in enumerate(items):
            out[k] = fn(it)
            if on_done:
                on_done(k, out[k])  # type: ignore[arg-type]
        return out  # type: ignore[return-value]
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="jt-sweep-") as tmp, _single_threaded(workers > 1):
        pending = list(range(len(items)))
        active: dict[int, tuple[Any, float, str]] = {}
        while pending or active:
            while pending and len(active) < workers:
                k = pending.pop(0)
                path = str(Path(tmp) / f"{k}.pkl")
                proc = ctx.Process(target=_child, args=(fn, items[k], path))
                proc.start()
                active[k] = (proc, time.monotonic(), path)
            time.sleep(0.2)
            for k, (proc, t0, path) in list(active.items()):
                if proc.is_alive():
                    if timeout_s is not None and time.monotonic() - t0 > timeout_s:
                        proc.kill()
                        proc.join()
                        out[k] = {"status": "timeout", "error": f"exceeded {timeout_s:.0f} s"}
                    else:
                        continue
                else:
                    proc.join()
                    if Path(path).exists():
                        out[k] = pickle.loads(Path(path).read_bytes())
                    else:
                        out[k] = {"status": "error", "error": f"worker died (exit {proc.exitcode})"}
                del active[k]
                if on_done:
                    on_done(k, out[k])  # type: ignore[arg-type]
    return out  # type: ignore[return-value]


def run_sweep(
    cfg: SweepConfig,
    results_dir: str | Path,
    workers: int = 1,
    root: Path = Path("."),
    quick: bool = False,
    log: Any = print,
    use_cache: bool = True,
) -> pd.DataFrame:
    from rudolf.score import flatten_score

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "sweep.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    )
    runs, cells = expand(cfg, results_dir, root)
    code = code_version()
    for c in cells:
        c.quick = quick
        c.key = cell_key(c, code)
    timeout = cfg.timeout_minutes * 60 if cfg.timeout_minutes else None
    log(
        f"[sweep {cfg.name}] {len(runs)} runs, {len(cells)} cells, {workers} worker(s), code {code}"
    )

    run_out = run_pool(_generate, runs, workers, timeout)
    bad_runs = {
        str(r.run_dir) for r, o in zip(runs, run_out, strict=True) if o.get("status") != "ok"
    }

    outcomes: list[dict[str, Any] | None] = [None] * len(cells)
    todo: list[int] = []
    for k, c in enumerate(cells):
        if str(c.run.run_dir) in bad_runs:
            outcomes[k] = {"status": "error", "error": "scenario generation failed"}
        elif use_cache and (hit := _cached(c)) is not None:
            outcomes[k] = hit
        else:
            todo.append(k)
    log(f"[sweep {cfg.name}] {len(cells) - len(todo)} cached, {len(todo)} to run")
    t_start = time.monotonic()

    def done(j: int, oc: dict[str, Any]) -> None:
        c = cells[todo[j]]
        log(
            f"  [{sum(o is not None for o in outcomes) + 1}/{len(cells)}] {c.strategy:20s} "
            f"{c.run.scenario.name} seed={c.run.seed} {oc.get('status')} "
            f"({time.monotonic() - t_start:.0f}s elapsed)"
        )
        outcomes[todo[j]] = oc

    run_pool(_execute, [cells[k] for k in todo], workers, timeout, on_done=done)

    rows: list[dict[str, Any]] = []
    for cell, oc in zip(cells, outcomes, strict=True):
        assert oc is not None
        base = {
            "scenario": cell.run.scenario.name,
            "seed": cell.run.seed,
            "strategy": cell.strategy,
            "run_dir": str(cell.run.run_dir.relative_to(results_dir)),
            "status": oc["status"],
            "cached": bool(oc.get("cached", False)),
            **cell.params,
        }
        if oc["status"] == "ok":
            row = {**flatten_score(oc["score"]), **base}
        else:
            row = {**base, "error": oc.get("error")}
            log(f"  ! {cell.strategy} on {cell.run.run_dir}: {oc.get('error')}")
        rows.append(row)
    df = pd.DataFrame(rows)
    swept = list(cfg.overrides) + [f"jan.{k}" for ov in cfg.strategy_overrides.values() for k in ov]
    io.write_table(df, results_dir / "results.parquet")
    io.write_json(
        {
            "swept_parameters": swept,
            "n_cells": len(df),
            "n_errors": int((df["status"] != "ok").sum()),
            "code_version": code,
        },
        results_dir / "results_meta.json",
    )
    log(f"[sweep {cfg.name}] wrote {results_dir / 'results.parquet'} ({len(df)} rows)")
    return df
