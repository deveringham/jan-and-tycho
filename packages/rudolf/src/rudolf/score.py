"""Rudolf's entry point: score one strategy's outputs on one run (SPEC §7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jt_core import io
from rudolf.data import JanOutputs, RunView, Sealed
from rudolf.metrics.parameters import parameter_metrics
from rudolf.metrics.predictive import predictive_metrics
from rudolf.metrics.probabilistic import probabilistic_metrics
from rudolf.metrics.selection import selection_metrics
from rudolf.metrics.structural import hidden_body_metrics, structural_metrics, vertex_metrics
from rudolf.metrics.trajectory import trajectory_metrics


def score_view(view: RunView) -> dict[str, Any]:
    pred = predictive_metrics(view)
    traj = trajectory_metrics(view)
    hidden = hidden_body_metrics(view.sealed, view.jan, view.body_id)
    return {
        "predictive": pred,
        "structural": structural_metrics(view.sealed, view.jan, view.body_id, pred, traj, hidden),
        "hidden": hidden,
        "vertices": vertex_metrics(view.sealed, view.jan, view.body_id),
        "trajectory": traj,
        "parameters": parameter_metrics(
            view.sealed.body_truth(view.body_id), view.jan.elements(view.body_id)
        ),
        "probabilistic": probabilistic_metrics(view, rng_seed=view.sealed.seed),
        "model_selection": selection_metrics(
            view.sealed.body_truth(view.body_id), view.jan.model_comparison()
        ),
    }


def observer_metrics(sealed: Sealed, jan: JanOutputs) -> dict[str, Any] | None:
    """How well the observer's own orbit was recovered (T2/T3), when Jan reports it."""
    est = jan.hypothesis.get("observer_elements")
    pred = jan.position("dense", "observer")
    if est is None and pred is None:
        return None
    out: dict[str, Any] = {}
    truth = sealed.truth.get("observer", {})
    if est is not None and truth.get("elements"):
        out.update(parameter_metrics(truth, est))
    if pred is not None:
        err = np.linalg.norm(pred - sealed.dense_positions("observer"), axis=1)
        out["traj_rms_au"] = float(np.sqrt(np.mean(err**2)))
        out["traj_max_au"] = float(np.max(err))
    return out


def flatten_score(score: dict[str, Any]) -> dict[str, Any]:
    """One flat row per (cell, body) for results tables. Curves stay as lists."""
    row: dict[str, Any] = {
        k: score[k] for k in ("scenario", "seed", "strategy", "label", "kind", "runtime_s")
    }
    for body, s in score["bodies"].items():
        pre = "" if len(score["bodies"]) == 1 else f"{body}."
        for k, v in s["predictive"].items():
            if k == "extrapolation":
                row[f"{pre}extrap_dt_days"] = v["dt_days"]
                row[f"{pre}extrap_rms"] = v["rms"]
            else:
                row[f"{pre}{k}"] = v
        row.update({f"{pre}{k}": v for k, v in s["trajectory"].items()})
        for k, v in (s["parameters"]["errors"] or {}).items():
            row[f"{pre}err_{k}"] = v
        for k, v in (s["parameters"]["zscores"] or {}).items():
            row[f"{pre}z_{k}"] = v
        for k, v in (s["parameters"].get("coverage") or {}).items():
            row[f"{pre}{k}"] = v
        row.update({f"{pre}{k}": v for k, v in s.get("probabilistic", {}).items()})
        ms = s.get("model_selection") or {}
        for k in ("true_family", "p_true_family", "selected_family", "selected_correct"):
            row[f"{pre}{k}"] = ms.get(k)
        for fam, p in (ms.get("family_probs") or {}).items():
            row[f"{pre}p_{fam}"] = p
        for k, v in (s.get("structural") or {}).items():
            row[f"{pre}{k}"] = v
        for k, v in (s.get("hidden") or {}).items():
            row[f"{pre}hidden_{k}"] = v
        for k, v in (s.get("vertices") or {}).items():
            row[f"{pre}{k}"] = v
        row[f"{pre}truth_family"] = s.get("model_selection", {}).get("true_family")
    obs = score.get("observer") or {}
    for k, v in (obs.get("errors") or {}).items():
        row[f"observer.err_{k}"] = v
    for k in ("traj_rms_au", "traj_max_au"):
        if k in obs:
            row[f"observer.{k}"] = obs[k]
    return row


def score_run(
    run_dir: str | Path,
    label: str,
    *,
    plots: bool = True,
    animate: bool = False,
    quick: bool = False,
) -> dict[str, Any]:
    """Score `jan/<label>` against the sealed truth; write score.json (+ figures)."""
    run_dir = Path(run_dir)
    sealed = Sealed.load(run_dir)
    jan = JanOutputs.load(run_dir, label)
    bodies = {b: score_view(RunView(sealed, jan, b)) for b in sealed.body_ids}
    score: dict[str, Any] = {
        "scenario": sealed.scenario_name,
        "seed": sealed.seed,
        "tier": sealed.info["tier"],
        "strategy": jan.strategy,
        "label": label,
        "kind": jan.kind,
        "runtime_s": jan.hypothesis.get("runtime_s"),
        "config": jan.hypothesis.get("config"),
        "bodies": bodies,
        "observer": observer_metrics(sealed, jan),
    }
    out = run_dir / io.RUDOLF_DIR / label
    io.write_json(score, out / io.SCORE_FILE)
    if plots or animate:
        from rudolf.viz.run_plots import render_run

        render_run(sealed, jan, out, static=plots, animate=animate, quick=quick)
    return score
