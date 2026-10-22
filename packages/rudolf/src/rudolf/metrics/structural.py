"""Structural and rejection metrics (SPEC §7, Phase 4).

* Symbolic: compare the recovered radial law f(r) with the truth's over the radii the orbit
  visits — "exact" (same function, constants included), "up_to_constants" (same form once
  the recovered expression's numeric constants are refitted, within 1e-6), or "mismatch".
  "Exact" means a relative error within 1e-3 *and* within half of the truth's own
  deviation from Newton: a law that departs from Newton by 2e-4 (GR-like precession) is
  only recovered by a hypothesis that explains at least half of that departure — plain
  Newton does not count. Non-central or missing laws are not compared symbolically.
* Numerical: relative L2 error of the recovered acceleration field at sealed points near the
  true trajectory (works for black-box laws).
* Rejection (kinematic truths): credit for answering kinematic/rejected with good holdout
  predictions; a penalty for claiming a dynamical law that fails.
* Hidden bodies: angular error of an inferred perturber's heliocentric longitude at the
  epoch of maximum perturbation. For such truths a claimed perturber within 10° counts as
  the correct law (mass and distance are degenerate over a short arc).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from scipy.optimize import least_squares
from sympy.parsing.sympy_parser import parse_expr

from jt_core import io
from jt_core.angles import wrap_pi
from jt_core.constants import GM_SUN
from rudolf.data import JanOutputs, Sealed

FloatArray = NDArray[np.float64]
_R = sp.Symbol("r", positive=True)
EXACT_TOL = 1e-3
FORM_TOL = 1e-6
HIDDEN_LONGITUDE_TOL_DEG = 10.0


def _radial_fn(law: dict[str, Any] | None) -> sp.Expr | None:
    if not law or not law.get("radial"):
        return None
    consts = law.get("constants") or {}
    expr = parse_expr(
        law["radial"],
        local_dict={
            "r": _R,
            "lam": sp.Symbol("lam"),
            "C": sp.Symbol("C"),
            "n": sp.Symbol("n"),
            "k": sp.Symbol("k"),
            "GM": sp.Symbol("GM"),
            "epsilon": sp.Symbol("epsilon"),
        },
    )
    subs = {s: consts[s.name] for s in expr.free_symbols if s.name in consts}
    expr = expr.subs(subs)
    if expr.free_symbols - {_R}:
        expr = expr.subs(
            {s: sp.Symbol("r", positive=True) for s in expr.free_symbols if s.name == "r"}
        )
    return expr if not (expr.free_symbols - {_R}) else None


def _eval(expr: sp.Expr, r: FloatArray) -> FloatArray:
    f = sp.lambdify(_R, expr, "numpy")
    return np.asarray(f(r), dtype=np.float64) * np.ones_like(r)


def compare_radial(
    truth: sp.Expr, rec: sp.Expr, r: FloatArray, exact_tol: float = EXACT_TOL
) -> tuple[str, float, float]:
    """(verdict, relative error as given, relative error after refitting constants)."""
    ft = _eval(truth, r)
    norm = float(np.linalg.norm(ft)) or 1.0
    try:
        fr = _eval(rec, r)
    except (TypeError, ValueError, ZeroDivisionError):
        return "mismatch", np.inf, np.inf
    err = float(np.linalg.norm(fr - ft) / norm) if np.all(np.isfinite(fr)) else np.inf
    floats = sorted(rec.atoms(sp.Float), key=str)
    refit = err
    if floats:
        syms = sp.symbols(f"p0:{len(floats)}")
        templ = rec.subs(dict(zip(floats, syms, strict=True)))
        g = sp.lambdify((_R, *syms), templ, "numpy")
        x0 = np.array([float(v) for v in floats])

        def res(p: FloatArray) -> FloatArray:
            with np.errstate(all="ignore"):
                v = np.asarray(g(r, *p), dtype=np.float64) * np.ones_like(r) - ft
            return np.where(np.isfinite(v), v, 1e3 * norm) / norm

        try:
            refit = float(np.linalg.norm(least_squares(res, x0, x_scale=np.abs(x0) + 1e-30).fun))
        except (ValueError, FloatingPointError):
            pass
    if err < exact_tol:
        return "exact", err, refit
    if refit < FORM_TOL:
        return "up_to_constants", err, refit
    return "mismatch", err, refit


def field_error(sealed: Sealed, jan: JanOutputs, body_id: str) -> float | None:
    """Relative L2 error of Jan's acceleration field at sealed points near the trajectory."""
    t = sealed.field_for(body_id, "near")
    j = jan.field_for(body_id, "near")
    if t is None or j is None or bool(t["ax"].isna().all()):
        return None
    at = io.cols(t, ["ax", "ay", "az"]).to_numpy()
    aj = io.cols(j, ["ax", "ay", "az"]).to_numpy()
    if aj.shape != at.shape or not np.all(np.isfinite(aj)):
        return float("inf")
    return float(np.linalg.norm(aj - at) / np.linalg.norm(at))


def newton_deviation(sealed: Sealed, body_id: str) -> float | None:
    """Relative L2 departure of the true field from Newton's −GM x/r³ near the orbit."""
    t = sealed.field_for(body_id, "near")
    if t is None or bool(t["ax"].isna().all()):
        return None
    gm = float(sealed.body_truth(body_id).get("gm") or GM_SUN)
    x = io.cols(t, ["x", "y", "z"]).to_numpy()
    at = io.cols(t, ["ax", "ay", "az"]).to_numpy()
    an = -gm * x / np.linalg.norm(x, axis=1, keepdims=True) ** 3
    return float(np.linalg.norm(at - an) / np.linalg.norm(at))


def law_tolerance(dev: float | None) -> float:
    """Relative error within which a recovered law counts as the true one."""
    return EXACT_TOL if not dev else min(EXACT_TOL, 0.5 * dev)


def visited_radii(sealed: Sealed, body_id: str) -> FloatArray:
    r = np.linalg.norm(sealed.dense_positions(body_id), axis=1)
    return np.linspace(0.9 * r.min(), 1.1 * r.max(), 200)


def structural_metrics(
    sealed: Sealed,
    jan: JanOutputs,
    body_id: str,
    predictive: dict[str, Any],
    trajectory: dict[str, Any] | None = None,
    hidden: dict[str, Any] | None = None,
) -> dict[str, Any]:
    truth = sealed.body_truth(body_id)
    t_law = truth.get("law")
    truth_kinematic = truth.get("kind") == "kinematic"
    out: dict[str, Any] = {
        "truth_kind": truth.get("kind"),
        "claimed_kind": jan.kind,
        "field_rel_error": field_error(sealed, jan, body_id),
        "symbolic": None,
        "symbolic_rel_error": None,
        "symbolic_refit_error": None,
    }
    dev = None if truth_kinematic else newton_deviation(sealed, body_id)
    tol = law_tolerance(dev)
    out["newton_deviation"] = dev
    t_expr, r_expr = _radial_fn(t_law), _radial_fn(jan.law())
    if t_expr is not None and r_expr is not None:
        radii = visited_radii(sealed, body_id)
        verdict, err, refit = compare_radial(t_expr, r_expr, radii, tol)
        out.update({"symbolic": verdict, "symbolic_rel_error": err, "symbolic_refit_error": refit})
    elif t_expr is not None and jan.law() is None and jan.kind == "dynamical":
        out["symbolic"] = "black_box"
    # Answer category (rejection matrix) and rejection scoring.
    # A good prediction: typical holdout sky error within 5σ, or a holdout path within 1%
    # of the orbit's size (sky errors of a nearby body blow up near close approaches
    # however good its 3D path is).
    sigma = sealed.noise_sigma_arcsec or 60.0
    med = predictive.get("holdout_median_arcsec")
    traj_rms = (trajectory or {}).get("traj_rms_holdout_au")
    x = sealed.dense_positions(body_id)
    size = float(np.max(np.linalg.norm(x - x.mean(axis=0), axis=1)))
    good_pred = (med is not None and med < 5 * sigma) or (
        traj_rms is not None and traj_rms < 0.01 * size
    )
    fe = out["field_rel_error"]
    sym = out["symbolic"]
    law_ok = sym == "exact" or (sym in (None, "black_box") and fe is not None and fe < tol)
    if hidden and hidden.get("longitude_error_deg") is not None:
        # Newton + an unseen body: the structural question is whether the claimed
        # perturber is where the real one is (its mass and distance trade off).
        law_ok = law_ok or hidden["longitude_error_deg"] < HIDDEN_LONGITUDE_TOL_DEG
    if jan.kind in ("kinematic", "rejected"):
        answer = jan.kind
    elif jan.kind == "nonparametric":
        answer = "nonparametric"
    elif truth_kinematic:
        answer = "spurious law"
    else:
        answer = (
            "correct law" if law_ok else "right form" if sym == "up_to_constants" else "wrong law"
        )
    out["answer"] = answer
    if truth_kinematic:
        out["rejection_credit"] = bool(jan.kind in ("kinematic", "rejected") and good_pred)
        out["rejection_penalty"] = bool(jan.kind == "dynamical")
    else:
        out["rejection_credit"] = None
        out["rejection_penalty"] = None
    return out


def vertex_metrics(sealed: Sealed, jan: JanOutputs, body_id: str) -> dict[str, Any] | None:
    """Polygon truths: distance from each true vertex to the recovered path, as a fraction
    of the shortest side."""
    verts = sealed.body_truth(body_id).get("vertices_3d")
    pred = jan.position("dense", body_id)
    if not verts or pred is None:
        return None
    v = np.asarray(verts)
    sides = np.linalg.norm(np.roll(v, -1, axis=0) - v, axis=1)
    d = np.array([float(np.min(np.linalg.norm(pred - p, axis=1))) for p in v])
    return {
        "vertex_error_max_au": float(d.max()),
        "vertex_error_frac": float(d.max() / sides.min()),
    }


def hidden_body_metrics(sealed: Sealed, jan: JanOutputs, body_id: str) -> dict[str, Any] | None:
    """Longitude error of the inferred perturber at the epoch of maximum perturbation."""
    hidden = sealed.hidden_ids
    pred = jan.position("dense", "hidden:perturber")
    if not hidden or pred is None:
        return None
    path = sealed.field_for(body_id, "path")
    if path is None:
        return None
    x = io.cols(path, ["x", "y", "z"]).to_numpy()
    a = io.cols(path, ["ax", "ay", "az"]).to_numpy()
    gm = float(sealed.body_truth(body_id).get("gm", 0.01720209895**2))
    pert = np.linalg.norm(a + gm * x / np.linalg.norm(x, axis=1, keepdims=True) ** 3, axis=1)
    t_star = float(path["t"].to_numpy()[int(np.argmax(pert))])
    dense_t = sealed.dense_t()
    k = int(np.argmin(np.abs(dense_t - t_star)))
    true_xyz = sealed.dense_positions(hidden[0])[k]
    lam_true = float(np.arctan2(true_xyz[1], true_xyz[0]))
    lam_pred = float(np.arctan2(pred[k, 1], pred[k, 0]))
    return {
        "hidden_id": hidden[0],
        "t_max_perturbation": t_star,
        "longitude_error_deg": abs(float(wrap_pi(lam_pred - lam_true))) * 180 / np.pi,
        "distance_true_au": float(np.linalg.norm(true_xyz)),
        "distance_pred_au": float(np.linalg.norm(pred[k])),
    }
