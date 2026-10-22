"""`classical.gauss_iod`: Gauss's method for an initial orbit from three sky directions.

Assumptions: the observer's positions are known (T1) and the body moves on a heliocentric
conic. Gauss's method fails when the three lines of sight and the observer positions are
nearly coplanar (D0 ≈ 0, e.g. an orbit in the ecliptic seen from an ecliptic Earth) or the
arc is very short or very long; the strategy therefore tries many triplets and keeps the
candidate that best explains *all* training observations.

Range ambiguity (SPEC §12): angles alone fix distance only through the dynamics; the
eighth-degree polynomial can have several positive roots and every root is tried.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from jan.base import BaseHypothesis, PerBodyHypothesis, StrategyConfig, register
from jan.fitting import ConicProblem
from jan.models.conic import ConicOrbit
from jan.models.conic_hypothesis import ConicHypothesis
from jan.models.observer import observer_function
from jt_core.conics import ConicElements, lagrange_fg, state_to_elements
from jt_core.schema import ObservationSet, PublicMeta

FloatArray = NDArray[np.float64]


def gauss_method(
    t: FloatArray, rho_hat: FloatArray, big_r: FloatArray, mu: float, n_refine: int = 50
) -> list[tuple[FloatArray, FloatArray, float]]:
    """Gauss's method (Curtis, Algorithms 5.5 + 5.6) for one triplet.

    Returns candidate (r2, v2, t2) for every admissible positive root, after iterative
    refinement with exact Lagrange coefficients.
    """
    t1, t2, t3 = (float(x) for x in t)
    tau1, tau3 = t1 - t2, t3 - t2
    tau = tau3 - tau1
    p1 = np.cross(rho_hat[1], rho_hat[2])
    p2 = np.cross(rho_hat[0], rho_hat[2])
    p3 = np.cross(rho_hat[0], rho_hat[1])
    d0 = float(rho_hat[0] @ p1)
    if abs(d0) < 1e-12:
        return []
    d = np.array([[big_r[i] @ p for p in (p1, p2, p3)] for i in range(3)])
    a_ = (-d[0, 1] * tau3 / tau + d[1, 1] + d[2, 1] * tau1 / tau) / d0
    b_ = (d[0, 1] * (tau3**2 - tau**2) * tau3 / tau + d[2, 1] * (tau**2 - tau1**2) * tau1 / tau) / (
        6.0 * d0
    )
    e_ = float(big_r[1] @ rho_hat[1])
    r2sq = float(big_r[1] @ big_r[1])
    a = -(a_**2 + 2.0 * a_ * e_ + r2sq)
    b = -2.0 * mu * b_ * (a_ + e_)
    c = -(mu**2) * b_**2
    roots = np.roots([1.0, 0.0, a, 0.0, 0.0, b, 0.0, 0.0, c])
    out: list[tuple[FloatArray, FloatArray, float]] = []
    for root in roots:
        if abs(root.imag) > 1e-9 * max(1.0, abs(root.real)) or root.real <= 0:
            continue
        r2n = float(root.real)
        rho2 = a_ + mu * b_ / r2n**3
        num1 = (
            6.0 * (d[2, 0] * tau1 / tau3 + d[1, 0] * tau / tau3) * r2n**3
            + mu * d[2, 0] * (tau**2 - tau1**2) * tau1 / tau3
        )
        rho1 = (num1 / (6.0 * r2n**3 + mu * (tau**2 - tau3**2)) - d[0, 0]) / d0
        num3 = (
            6.0 * (d[0, 2] * tau3 / tau1 - d[1, 2] * tau / tau1) * r2n**3
            + mu * d[0, 2] * (tau**2 - tau3**2) * tau3 / tau1
        )
        rho3 = (num3 / (6.0 * r2n**3 + mu * (tau**2 - tau1**2)) - d[2, 2]) / d0
        if min(rho1, rho2, rho3) <= 0:
            continue
        r1 = big_r[0] + rho1 * rho_hat[0]
        r2 = big_r[1] + rho2 * rho_hat[1]
        r3 = big_r[2] + rho3 * rho_hat[2]
        f1 = 1.0 - 0.5 * mu * tau1**2 / r2n**3
        f3 = 1.0 - 0.5 * mu * tau3**2 / r2n**3
        g1 = tau1 - mu * tau1**3 / (6.0 * r2n**3)
        g3 = tau3 - mu * tau3**3 / (6.0 * r2n**3)
        v2 = (-f3 * r1 + f1 * r3) / (f1 * g3 - f3 * g1)
        try:
            for _ in range(n_refine):
                f1, g1 = lagrange_fg(r2, v2, t1 - t2, mu)
                f3, g3 = lagrange_fg(r2, v2, t3 - t2, mu)
                den = f1 * g3 - f3 * g1
                c1, c3 = g3 / den, -g1 / den
                rho1n = (-d[0, 0] + d[1, 0] / c1 - c3 / c1 * d[2, 0]) / d0
                rho2n = (-c1 * d[0, 1] + d[1, 1] - c3 * d[2, 1]) / d0
                rho3n = (-c1 / c3 * d[0, 2] + d[1, 2] / c3 - d[2, 2]) / d0
                done = max(abs(rho1n - rho1), abs(rho2n - rho2), abs(rho3n - rho3)) < 1e-12 * rho2
                rho1, rho2, rho3 = rho1n, rho2n, rho3n
                r1 = big_r[0] + rho1 * rho_hat[0]
                r2 = big_r[1] + rho2 * rho_hat[1]
                r3 = big_r[2] + rho3 * rho_hat[2]
                v2 = (-f3 * r1 + f1 * r3) / den
                if done:
                    break
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            continue
        if np.all(np.isfinite(r2)) and np.all(np.isfinite(v2)) and min(rho1, rho2, rho3) > 0:
            out.append((r2, v2, t2))
    return out


def triplets(
    t: FloatArray, n: int, spacing_days: float, rng: np.random.Generator
) -> list[tuple[int, int, int]]:
    """Index triplets (i, j, k) with roughly `spacing_days` between consecutive members."""
    out: list[tuple[int, int, int]] = []
    if len(t) < 3:
        return out
    for _ in range(n * 4):
        i = int(rng.integers(0, len(t)))
        j = int(np.searchsorted(t, t[i] + spacing_days))
        k = int(np.searchsorted(t, t[i] + 2 * spacing_days))
        if k < len(t) and i < j < k:
            out.append((i, j, k))
        if len(out) >= n:
            break
    return out


def gauss_candidates(
    obs: ObservationSet,
    r_obs: FloatArray,
    mu: float,
    n_triplets: int,
    spacings: tuple[float, ...],
    rng: np.random.Generator,
) -> list[ConicElements]:
    t = obs.t
    u = obs.unit_vectors()
    cands: list[ConicElements] = []
    for s in spacings:
        for i, j, k in triplets(t, n_triplets, s, rng):
            idx = [i, j, k]
            with np.errstate(all="ignore"):  # degenerate triplets are expected and skipped
                sols = gauss_method(t[idx], u[idx], r_obs[idx], mu)
            for r2, v2, t2 in sols:
                try:
                    cands.append(state_to_elements(r2, v2, t2, mu))
                except ValueError:
                    continue
    return [c for c in cands if np.isfinite(c.as_array()).all() and 0.01 < c.q < 1e4 and c.e < 20]


class GaussIODConfig(StrategyConfig):
    n_triplets: int = Field(12, description="Triplets tried per spacing.")
    spacings_days: tuple[float, ...] = (15.0, 30.0, 60.0)


@register("classical.gauss_iod")
class GaussIOD:
    """Initial orbit from three observations; the best of many triplets is returned."""

    name: ClassVar[str]
    supported_tiers: ClassVar[set[str]] = {"T1"}
    Config: ClassVar[type[StrategyConfig]] = GaussIODConfig

    def fit(
        self,
        obs: ObservationSet,
        meta: PublicMeta,
        config: StrategyConfig,
        rng_seed: int,
        workdir: Path,
    ) -> BaseHypothesis:
        assert isinstance(config, GaussIODConfig)
        observer = observer_function(meta)
        rng = np.random.default_rng(rng_seed)
        parts: dict[str, BaseHypothesis] = {}
        for bid in obs.body_ids:
            ob = obs.for_body(bid)
            r_obs = observer(ob.t)
            t0 = float(np.median(ob.t))
            fam = ConicOrbit(t0=t0, gm=meta.gm_sun)
            cands = gauss_candidates(
                ob, r_obs, meta.gm_sun, config.n_triplets, config.spacings_days, rng
            )
            if not cands:
                raise RuntimeError(f"Gauss's method found no admissible orbit for {bid!r}")
            prob = ConicProblem(fam, ob.t, sky=True, u_obs=ob.unit_vectors(), r_obs=r_obs)
            thetas = np.array([fam.theta_from_elements(c) for c in cands])
            costs = prob.costs(thetas)
            best = int(np.argmin(costs))
            rms = float(np.sqrt(costs[best] / (2 * len(ob))))
            parts[bid] = ConicHypothesis(
                fam,
                thetas[best],
                bid,
                observer,
                info={
                    "n_candidates": len(cands),
                    "rms_rad": rms,
                    "notes": f"best of {len(cands)} Gauss candidates; training RMS {rms:.3g} rad",
                },
            )
        return PerBodyHypothesis(parts)
