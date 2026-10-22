"""Orbital-element errors (SPEC §7, level "Parameters").

Degeneracies (SPEC §12): Ω is undefined at i = 0 and ω, t_p at e = 0. Such elements are
reported as None and the well-defined combinations ϖ = Ω + ω (longitude of perihelion)
and λ (mean longitude at t = 0) are compared instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from jt_core.angles import wrap_pi
from jt_core.constants import DEG
from jt_core.schema import ELEMENT_NAMES

E_DEGENERATE = 1e-3
I_DEGENERATE = 1e-3  # rad


def _mean_longitude(el: dict[str, float], gm: float, t: float = 0.0) -> float | None:
    if el["e"] >= 1:
        return None
    a = el["q"] / (1 - el["e"])
    n = np.sqrt(gm / a**3)
    return el["Omega"] + el["omega"] + n * (t - el["t_p"])


def element_errors(truth: dict[str, float], est: dict[str, float], gm: float) -> dict[str, Any]:
    """Signed errors (estimate − truth). Angles in degrees, t_p in days."""
    e_t, i_t = truth["e"], truth["i"]
    ecc_ok = e_t > E_DEGENERATE
    inc_ok = i_t > I_DEGENERATE and abs(i_t - np.pi) > I_DEGENERATE
    out: dict[str, Any] = {
        "q_rel": (est["q"] - truth["q"]) / truth["q"],
        "e": est["e"] - e_t,
        "i_deg": (est["i"] - i_t) / DEG,
        "Omega_deg": float(wrap_pi(est["Omega"] - truth["Omega"])) / DEG if inc_ok else None,
        "omega_deg": float(wrap_pi(est["omega"] - truth["omega"])) / DEG
        if (inc_ok and ecc_ok)
        else None,
        "varpi_deg": (
            float(wrap_pi(est["Omega"] + est["omega"] - truth["Omega"] - truth["omega"])) / DEG
            if ecc_ok
            else None
        ),
    }
    a_t = truth["q"] / (1 - e_t) if e_t < 1 else None
    a_e = est["q"] / (1 - est["e"]) if est["e"] < 1 else None
    out["a_rel"] = (a_e - a_t) / a_t if (a_t is not None and a_e is not None) else None
    lam_t, lam_e = _mean_longitude(truth, gm), _mean_longitude(est, gm)
    out["lambda_deg"] = (
        float(wrap_pi(lam_e - lam_t)) / DEG if (lam_t is not None and lam_e is not None) else None
    )
    if ecc_ok:
        dtp = est["t_p"] - truth["t_p"]
        if e_t < 1 and a_t is not None:
            period = 2 * np.pi * np.sqrt(a_t**3 / gm)
            dtp = (dtp + period / 2) % period - period / 2
        out["t_p_days"] = float(dtp)
    else:
        out["t_p_days"] = None
    return out


def element_zscores(
    truth: dict[str, float], est: dict[str, Any], errors: dict[str, Any]
) -> dict[str, float | None]:
    """(estimate − truth)/σ using the hypothesis's reported covariance, where defined."""
    cov = est.get("cov")
    names = est.get("cov_names")
    out: dict[str, float | None] = dict.fromkeys(ELEMENT_NAMES)
    if cov is None or names is None:
        return out
    sig = dict(
        zip(names, np.sqrt(np.clip(np.diag(np.asarray(cov, dtype=float)), 0, None)), strict=True)
    )
    raw: dict[str, float | None] = {
        "q": errors["q_rel"] * truth["q"],
        "e": errors["e"],
        "i": errors["i_deg"] * DEG,
        "Omega": None if errors["Omega_deg"] is None else errors["Omega_deg"] * DEG,
        "omega": None if errors["omega_deg"] is None else errors["omega_deg"] * DEG,
        "t_p": errors["t_p_days"],
    }
    for k, v in raw.items():
        s = sig.get(k)
        out[k] = None if (v is None or s is None or s <= 0) else float(v / s)
    return out


def sample_errors(
    truth: dict[str, float], samples: dict[str, list[float]], gm: float
) -> dict[str, Any]:
    """Per-sample signed errors for each element (angles wrapped, t_p modulo the period)."""
    out: dict[str, Any] = {}
    for k in ELEMENT_NAMES:
        v = np.asarray(samples.get(k, []), dtype=np.float64)
        if v.size == 0:
            continue
        d = v - truth[k]
        if k in ("Omega", "omega"):
            d = np.asarray(wrap_pi(d))
        if k == "t_p" and truth["e"] < 1:
            period = 2 * np.pi * np.sqrt((truth["q"] / (1 - truth["e"])) ** 3 / gm)
            d = np.mod(d + period / 2, period) - period / 2
        out[k] = d
    return out


def credible_coverage(
    truth: dict[str, float], est: dict[str, Any], errors: dict[str, Any], gm: float
) -> dict[str, Any]:
    """Whether the truth lies in the central 68% / 95% intervals, and its posterior rank.

    Uses posterior samples when present (also giving the SBC rank fraction), else the
    reported covariance (Gaussian intervals). Degenerate elements are skipped.
    """
    skip = {
        "Omega": errors["Omega_deg"] is None,
        "omega": errors["omega_deg"] is None,
        "t_p": errors["t_p_days"] is None,
    }
    out: dict[str, Any] = {}
    samples = est.get("samples")
    if samples:
        se = sample_errors(truth, samples, gm)
        for k, d in se.items():
            if skip.get(k):
                continue
            q = np.quantile(d, [0.025, 0.16, 0.84, 0.975])
            out[f"in68_{k}"] = bool(q[1] <= 0 <= q[2])
            out[f"in95_{k}"] = bool(q[0] <= 0 <= q[3])
            out[f"rank_{k}"] = float(np.mean(d < 0))
        return out
    z = element_zscores(truth, est, errors)
    for k, v in z.items():
        if v is None or skip.get(k):
            continue
        out[f"in68_{k}"] = bool(abs(v) <= 1.0)
        out[f"in95_{k}"] = bool(abs(v) <= 1.96)
    return out


def parameter_metrics(truth_body: dict[str, Any], est: dict[str, Any] | None) -> dict[str, Any]:
    if est is None or not truth_body.get("elements"):
        return {"errors": None, "zscores": None, "coverage": None}
    truth = truth_body["elements"]
    gm = float(truth_body.get("gm", 0.01720209895**2))
    errs = element_errors(truth, est["point"], gm)
    return {
        "errors": errs,
        "zscores": element_zscores(truth, est, errs),
        "coverage": credible_coverage(truth, est, errs, gm),
    }
