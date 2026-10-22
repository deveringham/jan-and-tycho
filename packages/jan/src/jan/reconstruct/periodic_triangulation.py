"""Kepler's trick: periodic triangulation of a trajectory from lines of sight (SPEC §6.4).

A body with period P is at the same heliocentric point at t, t+P, t+2P, …, while the
observer has moved; the lines of sight at those epochs intersect at that point. This needs
no dynamics, only periodicity, so it works for any periodic path (physical or not).

Under T2 the observer is unknown, but the Sun stream gives the direction s_k from the
observer to the Sun. With the target fixed at M over a period group,

    M + ρ_k s_k − μ_k u_k = 0          (observer E_k = −ρ_k s_k, M − E_k = μ_k u_k)

is linear and homogeneous in (M, ρ, μ): its null vector gives the observer positions and M
up to one scale per group — Kepler's reconstruction of the Earth's orbit from Mars.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jt_core.schema import ObservationSet

FloatArray = NDArray[np.float64]


class DirectionInterpolator:
    """Unit-vector track interpolated at arbitrary times from nearby observations.

    A local quadratic in time is fitted to each component using observations within
    `window` days; returns None where the data do not bracket the requested time.
    """

    def __init__(self, t: FloatArray, u: FloatArray, window: float, min_points: int = 3) -> None:
        order = np.argsort(t)
        self.t = np.asarray(t, dtype=np.float64)[order]
        self.u = np.asarray(u, dtype=np.float64)[order]
        self.window = window
        self.min_points = min_points

    @classmethod
    def from_obs(cls, ob: ObservationSet, window: float | None = None) -> DirectionInterpolator:
        dt = float(np.median(np.diff(np.sort(ob.t)))) if len(ob) > 1 else 1.0
        return cls(ob.t, ob.unit_vectors(), window or 2.5 * dt)

    def __call__(self, tq: float) -> FloatArray | None:
        lo = np.searchsorted(self.t, tq - self.window)
        hi = np.searchsorted(self.t, tq + self.window)
        if hi - lo < self.min_points:
            return None
        ts = self.t[lo:hi] - tq
        if ts.min() > 0 or ts.max() < 0:  # extrapolation, not interpolation
            return None
        deg = min(2, hi - lo - 1)
        coef = np.polynomial.polynomial.polyfit(ts, self.u[lo:hi], deg)
        v = coef[0]
        return v / np.linalg.norm(v)


def intersect_lines(
    origins: FloatArray,
    dirs: FloatArray,
    sigma: float = 1.0,
) -> tuple[FloatArray, FloatArray, float]:
    """Least-squares intersection of lines R_k + s u_k, weighted by angular error σ.

    Returns (point, covariance, RMS perpendicular miss / (σ·range)).
    """
    x = None
    w = np.ones(len(origins))
    for _ in range(3):  # reweight by range: angular error ⇒ miss distance ∝ range
        a = np.zeros((3, 3))
        b = np.zeros(3)
        for r, u, wk in zip(origins, dirs, w, strict=True):
            p = np.eye(3) - np.outer(u, u)
            a += wk * p
            b += wk * p @ r
        x = np.linalg.lstsq(a, b, rcond=None)[0]
        rng = np.maximum(np.linalg.norm(x - origins, axis=1), 1e-6)
        w = 1.0 / (sigma * rng) ** 2
    assert x is not None
    a = sum(wk * (np.eye(3) - np.outer(u, u)) for u, wk in zip(dirs, w, strict=True))
    cov = np.linalg.pinv(a)
    d = x - origins
    perp = d - np.sum(d * dirs, axis=1, keepdims=True) * dirs
    chi = float(np.sqrt(np.mean(np.sum(perp**2, axis=1) * w)))
    return x, cov, chi


@dataclass
class TriPoint:
    """One triangulated point: the body's position at phase `tau` (mod P)."""

    tau: float
    x: FloatArray
    cov: FloatArray
    chi: float
    times: FloatArray
    origins: FloatArray
    dirs: FloatArray
    extra: dict[str, Any] = field(default_factory=dict[str, Any])

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau": self.tau,
            "x": self.x.tolist(),
            "sd": np.sqrt(np.clip(np.diag(self.cov), 0, None)).tolist(),
            "chi": self.chi,
            "times": self.times.tolist(),
            "origins": self.origins.tolist(),
            "dirs": self.dirs.tolist(),
            **self.extra,
        }


def _min_angle_ok(dirs: FloatArray, min_angle: float) -> bool:
    c = np.clip(dirs @ dirs.T, -1, 1)
    return bool(np.arccos(c).max() >= min_angle)


def triangulate_known_observer(
    interp: DirectionInterpolator,
    observer: Callable[[FloatArray], FloatArray],
    period: float,
    t_range: tuple[float, float],
    n_anchor: int = 72,
    sigma: float = 1.0,
    min_lines: int = 2,
    min_angle: float = np.radians(2.0),
) -> list[TriPoint]:
    """Triangulate the body over one period with the observer's positions known (T1)."""
    t0, t1 = t_range
    out: list[TriPoint] = []
    for tau in t0 + np.linspace(0, period, n_anchor, endpoint=False):
        times = np.arange(tau, t1 + 1e-9, period)
        dirs, used = [], []
        for tk in times:
            u = interp(float(tk))
            if u is not None:
                dirs.append(u)
                used.append(tk)
        if len(used) < min_lines:
            continue
        d = np.array(dirs)
        if not _min_angle_ok(d, min_angle):
            continue
        tt = np.array(used)
        r = observer(tt)
        x, cov, chi = intersect_lines(r, d, sigma)
        out.append(TriPoint(float(tau), x, cov, chi, tt, r, d))
    return out


def consistency(points: list[TriPoint]) -> float:
    """Mean normalized miss (lower = lines meet better); inf when nothing triangulates."""
    if len(points) < 3:
        return np.inf
    return float(np.median([p.chi for p in points]))


def refine_period(
    make: Callable[[float], list[TriPoint]], p0: float, rel: float = 0.01, n: int = 41
) -> tuple[float, list[TriPoint]]:
    """Pick the period whose lines of sight meet best near p0 (coarse grid, then finer)."""
    best_p, best_pts, best_c = p0, make(p0), np.inf
    best_c = consistency(best_pts)
    for width in (rel, rel / 10):
        for p in best_p * (1 + np.linspace(-width, width, n)):
            pts = make(float(p))
            c = consistency(pts)
            if c < best_c:
                best_p, best_pts, best_c = float(p), pts, c
    return best_p, best_pts


def triangulate_unknown_observer(
    interp: DirectionInterpolator,
    sun_interp: DirectionInterpolator,
    period: float,
    t_range: tuple[float, float],
    n_anchor: int = 72,
    min_lines: int = 3,
) -> list[TriPoint]:
    """Kepler's reconstruction (T2): per period group, solve for the target point M and the
    observer positions E_k = −ρ_k s_k up to a common scale (fixed later by the caller).

    Each TriPoint carries `x` = M and `extra["observer"]` = the E_k, in group units.
    """
    t0, t1 = t_range
    out: list[TriPoint] = []
    for tau in t0 + np.linspace(0, period, n_anchor, endpoint=False):
        times = np.arange(tau, t1 + 1e-9, period)
        us, ss, used = [], [], []
        for tk in times:
            u, s = interp(float(tk)), sun_interp(float(tk))
            if u is not None and s is not None:
                us.append(u)
                ss.append(s)
                used.append(tk)
        k = len(used)
        if k < min_lines:
            continue
        # Unknowns z = (M[3], ρ[k], μ[k]); rows: M + ρ_k s_k − μ_k u_k = 0.
        a = np.zeros((3 * k, 3 + 2 * k))
        for j, (u, s) in enumerate(zip(us, ss, strict=True)):
            a[3 * j : 3 * j + 3, :3] = np.eye(3)
            a[3 * j : 3 * j + 3, 3 + j] = s
            a[3 * j : 3 * j + 3, 3 + k + j] = -u
        _, sv, vt = np.linalg.svd(a)
        z = vt[-1]
        rho = z[3 : 3 + k]
        if np.mean(rho) < 0:
            z = -z
            rho = -rho
        if np.any(rho <= 0) or np.any(z[3 + k :] <= 0):
            continue
        scale = float(np.mean(rho))  # group units: mean observer distance 1
        m = z[:3] / scale
        e_pos = -(rho / scale)[:, None] * np.array(ss)
        chi = float(sv[-1] / max(sv[-2], 1e-300))  # null-space separation (small = consistent)
        out.append(
            TriPoint(
                float(tau),
                m,
                np.eye(3) * np.nan,
                chi,
                np.array(used),
                e_pos,
                np.array(us),
                extra={"observer": e_pos.tolist()},
            )
        )
    return out
