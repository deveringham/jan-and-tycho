"""Predictive metrics on the sealed holdout window (SPEC §7, level "Predictive")."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jt_core.frames import lonlat_to_unit
from rudolf.data import RunView, sep_arcsec

FloatArray = NDArray[np.float64]


def _summary(err: FloatArray, prefix: str, unit: str) -> dict[str, float | None]:
    if err.size == 0:
        return {f"{prefix}_{k}_{unit}": None for k in ("rms", "median", "p95", "max")}
    return {
        f"{prefix}_rms_{unit}": float(np.sqrt(np.mean(err**2))),
        f"{prefix}_median_{unit}": float(np.median(err)),
        f"{prefix}_p95_{unit}": float(np.percentile(err, 95)),
        f"{prefix}_max_{unit}": float(np.max(err)),
    }


def holdout_errors(view: RunView) -> FloatArray | None:
    """Great-circle error (″) of predictions vs the noise-free holdout directions."""
    pred = view.jan.sky("holdout", view.body_id)
    if pred is None or not view.sealed.is_sky:
        return None
    return sep_arcsec(pred, view.sealed.holdout_true_directions(view.body_id))


def holdout_position_errors(view: RunView) -> FloatArray | None:
    """T0: 3D error (AU) vs the noise-free holdout positions."""
    if view.sealed.is_sky:
        return None
    pred = view.jan.position("holdout", view.body_id)
    if pred is None:
        return None
    h = view.sealed.holdout_body(view.body_id)
    return np.linalg.norm(pred - h[["x_true", "y_true", "z_true"]].to_numpy(), axis=1)


def extrapolation_curve(view: RunView, n_bins: int = 10) -> dict[str, list[float | None]]:
    """RMS dense-grid error vs time since the end of training (holdout window only)."""
    t = view.dense_t
    t0 = view.sealed.t_split
    if view.sealed.is_sky:
        err = view.dense_err_arcsec
    else:
        err = (
            None
            if view.dense_pred_r is None
            else np.linalg.norm(view.dense_pred_r - view.dense_true_r, axis=1)
        )
    edges = np.linspace(0.0, view.sealed.duration - t0, n_bins + 1)
    mids = 0.5 * (edges[1:] + edges[:-1])
    if err is None:
        return {"dt_days": mids.tolist(), "rms": [None] * n_bins}
    dt = t - t0
    rms: list[float | None] = []
    for lo, hi in itertools.pairwise(edges):
        m = (dt >= lo) & (dt <= hi)
        rms.append(float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else None)
    return {"dt_days": mids.tolist(), "rms": rms}


def predictive_metrics(view: RunView) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sky = view.sealed.is_sky
    if sky:
        err = holdout_errors(view)
        out.update(_summary(err if err is not None else np.array([]), "holdout", "arcsec"))
        d = view.dense_err_arcsec
        train = view.dense_t < view.sealed.t_split
        out["train_dense_rms_arcsec"] = (
            None if d is None else float(np.sqrt(np.mean(d[train] ** 2)))
        )
        out["holdout_dense_rms_arcsec"] = (
            None if d is None else float(np.sqrt(np.mean(d[~train] ** 2)))
        )
        out["noise_sigma_arcsec"] = view.sealed.noise_sigma_arcsec
        tr = view.jan.sky("train", view.body_id)
        pub = view.sealed.public_body(view.body_id)
        if tr is not None and len(tr) == len(pub):
            u_obs = lonlat_to_unit(pub["lon"].to_numpy(), pub["lat"].to_numpy())
            out["train_fit_rms_arcsec"] = float(np.sqrt(np.mean(sep_arcsec(tr, u_obs) ** 2)))
        else:
            out["train_fit_rms_arcsec"] = None
    else:
        perr = holdout_position_errors(view)
        out.update(_summary(perr if perr is not None else np.array([]), "holdout", "au"))
    out["extrapolation"] = extrapolation_curve(view)
    out["n_holdout"] = len(view.sealed.holdout_body(view.body_id))
    return out
