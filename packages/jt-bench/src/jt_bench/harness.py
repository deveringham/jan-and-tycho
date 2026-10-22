"""Run one (scenario, seed, strategy) cell end to end (SPEC §6.2 scoring flow, §9).

The harness enforces the runtime information boundary (SPEC §3.3): Jan receives only the
in-memory public bundle and a private scratch `workdir`; prediction times come from a
query sheet that carries times only, and Rudolf scores the serialized predictions.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import jan
from jan.base import BaseHypothesis, make_config
from jt_core import io
from jt_core import rng as jrng
from jt_core.schema import QuerySheet
from tycho import Scenario, generate, write_run
from tycho.generate import default_run_dir

TRACE_GRID_POINTS = 400
OBSERVER_ID = "observer"


PRESENTATION_KEYS = frozenset({"record_trace", "trace_max_frames"})
"""Config keys that change what is recorded, not the fit: excluded from labels."""


def strategy_label(strategy: str, config: dict[str, Any] | None) -> str:
    """Directory label: the strategy name, plus its overrides (`[k=v,…]`, or a short hash when
    long)."""
    semantic = {k: v for k, v in (config or {}).items() if k not in PRESENTATION_KEYS}
    if not semantic:
        return strategy

    def show(v: Any) -> str:
        return "+".join(map(str, v)) if isinstance(v, list | tuple) else str(v)

    readable = ",".join(f"{k}={show(v)}" for k, v in sorted(semantic.items()))
    if len(readable) <= 72 and all(c.isalnum() or c in "=,._-+" for c in readable):
        return f"{strategy}[{readable}]"
    h = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()[:8]
    return f"{strategy}@{h}"


def ensure_run(
    scenario: Scenario,
    seed: int,
    out_root: str | Path = "runs",
    run_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Generate the public/sealed bundles unless they already exist."""
    d = Path(run_dir) if run_dir is not None else default_run_dir(scenario, seed, out_root)
    key = run_key(scenario, seed)
    key_file = d / io.SEALED_DIR / "cache_key.txt"
    fresh = (d / io.SEALED_DIR / "run.json").exists() and key_file.exists()
    if force or not fresh or key_file.read_text(encoding="utf-8") != key:
        for sub in (io.JAN_DIR, io.RUDOLF_DIR):  # outputs of a stale run are invalid
            if (d / sub).exists():
                shutil.rmtree(d / sub)
        write_run(generate(scenario, seed), d)
        key_file.write_text(key, encoding="utf-8")
    return d


def run_key(scenario: Scenario, seed: int) -> str:
    """Hash of scenario + seed + Tycho/jt_core source: changes force regeneration."""
    from jt_bench.sweep import code_version

    blob = json.dumps(
        {
            "scenario": scenario.model_dump(mode="json"),
            "seed": seed,
            "code": code_version(("jt_core", "tycho")),
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def query_sheet(run_dir: str | Path) -> QuerySheet:
    """Times only: sealed holdout times, the dense truth grid, and public training times."""
    s = Path(run_dir) / io.SEALED_DIR
    hold = io.cols(io.read_table(s / "holdout.parquet"), ["obs_id", "body_id", "t"])
    traj = io.read_table(s / "trajectory.parquet")
    dense_t = io.rows(traj, traj["body_id"] == "observer")["t"].to_numpy()
    info = io.read_json(s / "run.json")
    train = io.cols(
        io.read_table(Path(run_dir) / io.PUBLIC_DIR / io.OBSERVATIONS_FILE),
        ["obs_id", "body_id", "t"],
    )
    return QuerySheet(hold, dense_t, list(info["body_ids"]), train)


def _prediction_rows(
    hyp: BaseHypothesis, query: QuerySheet, sky: bool, with_samples: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    rows: list[pd.DataFrame] = []
    srows: list[pd.DataFrame] = []
    for bid in query.body_ids:
        hold = io.rows(query.holdout, query.holdout["body_id"] == bid)
        queries = [
            ("holdout", hold["t"].to_numpy(), hold["obs_id"].to_numpy()),
            ("dense", query.dense_t, np.full(query.dense_t.size, -1)),
        ]
        if query.train is not None:
            tr = io.rows(query.train, query.train["body_id"] == bid)
            queries.append(("train", tr["t"].to_numpy(), tr["obs_id"].to_numpy()))
        for qname, t, obs_id in queries:
            n = t.size
            df = pd.DataFrame(
                {"query": qname, "obs_id": obs_id, "body_id": bid, "t": t, "idx": np.arange(n)}
            )
            nan3 = np.full((n, 3), np.nan)
            u = cov = None
            u_samp = p_samp = None
            if sky and n:
                sp = hyp.predict_sky(t, bid)
                u, cov, u_samp = sp.mean, sp.cov, sp.samples
            pp = hyp.heliocentric_position(t, bid) if n else None
            p = pp.mean if pp is not None else None
            p_samp = pp.samples if pp is not None else None
            df[["ux", "uy", "uz"]] = u if u is not None else nan3
            if cov is not None:
                df["cov_ee"], df["cov_en"], df["cov_nn"] = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
            else:
                df["cov_ee"] = df["cov_en"] = df["cov_nn"] = np.nan
            df[["px", "py", "pz"]] = p if p is not None else nan3
            rows.append(df)
            if with_samples and (u_samp is not None or p_samp is not None):
                n_s = (u_samp if u_samp is not None else p_samp).shape[0]  # type: ignore[union-attr]
                sd = pd.DataFrame(
                    {
                        "query": qname,
                        "body_id": bid,
                        "sample": np.repeat(np.arange(n_s), n),
                        "idx": np.tile(np.arange(n), n_s),
                    }
                )
                sd[["ux", "uy", "uz"]] = (
                    u_samp.reshape(-1, 3) if u_samp is not None else np.full((n_s * n, 3), np.nan)
                )
                sd[["px", "py", "pz"]] = (
                    p_samp.reshape(-1, 3) if p_samp is not None else np.full((n_s * n, 3), np.nan)
                )
                srows.append(sd)
    for name, hp in (hyp.hidden_bodies(query.dense_t) or {}).items():  # inferred unseen bodies
        n = len(query.dense_t)
        df = pd.DataFrame(
            {
                "query": "dense",
                "obs_id": -1,
                "body_id": f"hidden:{name}",
                "t": query.dense_t,
                "idx": np.arange(n),
            }
        )
        df[["ux", "uy", "uz"]] = np.nan
        df["cov_ee"] = df["cov_en"] = df["cov_nn"] = np.nan
        df[["px", "py", "pz"]] = hp.mean
        rows.append(df)
    op = hyp.observer_position(query.dense_t)
    if op is not None:  # the observer's orbit as inferred (T2/T3)
        n = len(query.dense_t)
        df = pd.DataFrame(
            {
                "query": "dense",
                "obs_id": -1,
                "body_id": OBSERVER_ID,
                "t": query.dense_t,
                "idx": np.arange(n),
            }
        )
        df[["ux", "uy", "uz"]] = np.nan
        df["cov_ee"] = df["cov_en"] = df["cov_nn"] = np.nan
        df[["px", "py", "pz"]] = op.mean
        rows.append(df)
    preds = pd.concat(rows, ignore_index=True)
    samples = pd.concat(srows, ignore_index=True) if srows else None
    return preds, samples


def _near_points(run_dir: Path) -> pd.DataFrame | None:
    fp = run_dir / io.SEALED_DIR / io.FIELD_FILE
    if not fp.exists():
        return None
    f = io.read_table(fp)
    return io.rows(f, f["kind"] == "near")


def _field_rows(hyp: BaseHypothesis, field: pd.DataFrame) -> pd.DataFrame | None:
    out: list[pd.DataFrame] = []
    for bid in sorted(field["body_id"].unique()):
        f = io.rows(field, field["body_id"] == bid)
        x = io.cols(f, ["x", "y", "z"]).to_numpy()
        v = io.cols(f, ["vx", "vy", "vz"]).to_numpy()
        a = hyp.acceleration(x, v, f["t"].to_numpy(), bid)
        if a is None:
            continue
        df = io.cols(f, ["kind", "body_id", "t", "x", "y", "z"]).copy()
        df[["ax", "ay", "az"]] = np.asarray(a)
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else None


def _write_field(hyp: BaseHypothesis, field: pd.DataFrame, path: Path) -> None:
    """Jan's acceleration field at the sealed field points (positions, velocities, times
    only — the true accelerations stay sealed)."""
    df = _field_rows(hyp, field)
    if df is not None:
        io.write_table(df, path)


@dataclass
class FitResult:
    label: str
    jan_dir: Path
    runtime_s: float


def fit(
    run_dir: str | Path,
    strategy: str,
    config: dict[str, Any] | None = None,
    label: str | None = None,
    rng_seed: int | None = None,
) -> FitResult:
    """Fit a strategy on the public bundle and serialize predictions on the query sheet."""
    run_dir = Path(run_dir)
    obs, meta = io.read_public(run_dir)
    strat = jan.get_strategy(strategy)
    if meta.tier not in strat.supported_tiers:
        raise ValueError(f"{strategy} does not support tier {meta.tier}")
    cfg = make_config(strat, config)
    label = label or strategy_label(strategy, config)
    jan_dir = run_dir / io.JAN_DIR / label
    workdir = jan_dir / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    seed = rng_seed if rng_seed is not None else jrng.derive_int(0, "jan", strategy) % (2**31)

    t0 = time.perf_counter()
    hyp = strat.fit(obs, meta, cfg, seed, workdir)
    runtime = time.perf_counter() - t0

    query = query_sheet(run_dir)
    io.write_query(jan_dir / io.QUERY_FILE, query)
    sky = meta.observed == "sky"
    preds, samples = _prediction_rows(hyp, query, sky)
    io.write_table(preds, jan_dir / io.PREDICTIONS_FILE)
    if samples is not None:
        io.write_table(samples, jan_dir / io.SAMPLES_FILE)
    trace = hyp.trace()
    trace_labels: list[str] = []
    if trace:
        # Frames are evaluated on a thinned copy of the dense grid to keep files small.
        step = max(1, len(query.dense_t) // TRACE_GRID_POINTS)
        grid = QuerySheet(query.holdout.iloc[0:0], query.dense_t[::step], query.body_ids)
        frames: list[pd.DataFrame] = []
        frame_samples: list[pd.DataFrame] = []
        frame_fields: list[pd.DataFrame] = []
        near = _near_points(run_dir)
        for k, fr in enumerate(trace):
            h = hyp.at_frame(fr)
            if h is None:
                continue
            p, smp = _prediction_rows(h, grid, sky, with_samples=True)
            frames.append(p.assign(frame=k, frame_label=fr.label))
            if near is not None:
                ff = _field_rows(h, near)
                if ff is not None:
                    frame_fields.append(ff.assign(frame=k))
            if smp is not None:
                frame_samples.append(smp.assign(frame=k))
            trace_labels.append(fr.label)
        if frames:
            io.write_table(
                pd.concat(frames, ignore_index=True), jan_dir / io.TRACE_PREDICTIONS_FILE
            )
        if frame_fields:
            io.write_table(
                pd.concat(frame_fields, ignore_index=True), jan_dir / io.TRACE_FIELD_FILE
            )
        if frame_samples:
            io.write_table(
                pd.concat(frame_samples, ignore_index=True), jan_dir / io.TRACE_SAMPLES_FILE
            )

    elements = {}
    for bid in query.body_ids:
        est = hyp.elements(bid)
        elements[bid] = est.to_dict() if est is not None else None
    field_path = run_dir / io.SEALED_DIR / io.FIELD_FILE
    if field_path.exists():
        _write_field(hyp, io.read_table(field_path), jan_dir / io.FIELD_FILE)

    obs_sigma = {bid: hyp.observation_sigma(bid) for bid in query.body_ids}
    obs_el = hyp.observer_elements()
    law = hyp.law()
    mc = hyp.model_comparison()
    io.write_json(
        {
            "strategy": strategy,
            "label": label,
            "config": cfg.model_dump(mode="json"),
            "config_overrides": config or {},
            "rng_seed": seed,
            "runtime_s": runtime,
            "kind": hyp.kind,
            "notes": hyp.notes(),
            "elements": elements,
            "obs_sigma_rad": obs_sigma,
            "observer_elements": obs_el.to_dict() if obs_el is not None else None,
            "law": law.to_dict() if law is not None else None,
            "model_comparison": [m.to_dict() for m in mc] if mc is not None else None,
            "trace_labels": trace_labels,
            "hypothesis": hyp.to_dict(),
        },
        jan_dir / io.HYPOTHESIS_FILE,
    )
    return FitResult(label, jan_dir, runtime)


def score(
    run_dir: str | Path,
    label: str,
    *,
    plots: bool = True,
    animate: bool = False,
    quick: bool = False,
) -> dict[str, Any]:
    from rudolf.score import score_run

    return score_run(run_dir, label, plots=plots, animate=animate, quick=quick)


def run_cell(
    scenario: Scenario,
    seed: int,
    strategy: str,
    config: dict[str, Any] | None = None,
    out_root: str | Path = "runs",
    run_dir: str | Path | None = None,
    *,
    plots: bool = True,
    animate: bool = False,
    quick: bool = False,
) -> dict[str, Any]:
    """generate (if needed) → fit → score (+ plots)."""
    d = ensure_run(scenario, seed, out_root, run_dir)
    res = fit(d, strategy, config)
    return score(d, res.label, plots=plots, animate=animate, quick=quick)
