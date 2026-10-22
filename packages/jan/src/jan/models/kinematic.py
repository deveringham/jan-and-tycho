"""`PeriodicKinematic`: a closed 3D path as a Fourier series with a shared period.

Non-physical but periodic paths (rectangles, figure-eights) and anything a conic cannot
fit: positions only, no force law.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jan.base import BaseHypothesis
from jt_core.schema import HypothesisKind, PositionPrediction, SkyPrediction

FloatArray = NDArray[np.float64]


def _design(t: FloatArray, period: float, order: int) -> FloatArray:
    ph = 2 * np.pi * np.outer(np.asarray(t, dtype=np.float64), np.arange(1, order + 1)) / period
    return np.hstack([np.ones((len(ph), 1)), np.cos(ph), np.sin(ph)])


def fit_periodic(
    t: FloatArray, x: FloatArray, period: float, order: int, weights: FloatArray | None = None
) -> tuple[FloatArray, float]:
    """Least-squares Fourier coefficients (2·order+1, 3) and the per-axis RMS residual."""
    order = max(1, min(order, (len(t) - 1) // 2 - 1))
    a = _design(t, period, order)
    w = np.ones(len(t)) if weights is None else weights
    coef, *_ = np.linalg.lstsq(a * w[:, None], x * w[:, None], rcond=None)
    rms = float(np.sqrt(np.mean((a @ coef - x) ** 2)))
    return coef, rms


class PeriodicLinearPath:
    """A closed path through phase-ordered points, linearly interpolated (keeps corners)."""

    def __init__(self, taus: FloatArray, points: FloatArray, period: float) -> None:
        phase = np.mod(np.asarray(taus) - taus[0], period)
        order = np.argsort(phase)
        self.phase = phase[order]
        self.points = np.asarray(points)[order]
        self.period = period
        self.t_ref = float(taus[0])

    def __call__(self, t: FloatArray) -> FloatArray:
        ph = np.mod(np.atleast_1d(np.asarray(t, dtype=np.float64)) - self.t_ref, self.period)
        xp = np.concatenate([self.phase, [self.phase[0] + self.period]])
        out = np.empty((len(ph), 3))
        for k in range(3):
            fp = np.concatenate([self.points[:, k], [self.points[0, k]]])
            out[:, k] = np.interp(ph, xp, fp, period=self.period)
        return out


class PolygonPath:
    """A closed polygon traversed periodically: arc length s(phase) = L·phase/P plus a
    low-order Fourier correction (constant speed is the zeroth-order case), fitted to the
    triangulated points projected onto the polygon. Noise is averaged along each edge
    instead of being joined point to point."""

    def __init__(
        self,
        taus: FloatArray,
        points: FloatArray,
        vertices: FloatArray,
        period: float,
        order: int = 4,
    ) -> None:
        self.v = np.asarray(vertices)
        self.seg = np.roll(self.v, -1, axis=0) - self.v
        lens = np.linalg.norm(self.seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(lens)])
        self.length = float(self.cum[-1])
        self.period = period
        self.t_ref = float(taus[0])
        s = self.project(np.asarray(points))
        ph = np.mod(np.asarray(taus) - self.t_ref, period)
        lin = self.length * ph / period
        off = np.mod(s - lin + 0.5 * self.length, self.length) - 0.5 * self.length
        # Re-centre on the circular mean so the residual offsets do not wrap.
        c = self.length / (2 * np.pi) * np.angle(np.mean(np.exp(2j * np.pi * off / self.length)))
        off = np.mod(off - c + 0.5 * self.length, self.length) - 0.5 * self.length + c
        self.order = max(0, min(order, (len(ph) - 2) // 4))
        a = _design(ph, period, self.order) if self.order else np.ones((len(ph), 1))
        self.coef, *_ = np.linalg.lstsq(a, off, rcond=None)

    def project(self, x: FloatArray) -> FloatArray:
        """Arc length of each point's closest point on the polygon."""
        best_d = np.full(len(x), np.inf)
        best_s = np.zeros(len(x))
        for k, (a, ab) in enumerate(zip(self.v, self.seg, strict=True)):
            u = np.clip(((x - a) @ ab) / max(float(ab @ ab), 1e-30), 0, 1)
            d = np.linalg.norm(x - (a + u[:, None] * ab), axis=1)
            better = d < best_d
            best_d[better] = d[better]
            best_s[better] = self.cum[k] + u[better] * (self.cum[k + 1] - self.cum[k])
        return best_s

    def __call__(self, t: FloatArray) -> FloatArray:
        ph = np.mod(np.atleast_1d(np.asarray(t, dtype=np.float64)) - self.t_ref, self.period)
        a = _design(ph, self.period, self.order) if self.order else np.ones((len(ph), 1))
        s = np.mod(self.length * ph / self.period + a @ self.coef, self.length)
        k = np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.v) - 1)
        frac = (s - self.cum[k]) / np.maximum(self.cum[k + 1] - self.cum[k], 1e-30)
        return self.v[k] + frac[:, None] * self.seg[k]


class FourierPath:
    """The least-squares Fourier path as a callable (smooth closed curves)."""

    def __init__(self, coef: FloatArray, period: float) -> None:
        self.coef, self.period = np.asarray(coef), period
        self.order = (len(self.coef) - 1) // 2

    def __call__(self, t: FloatArray) -> FloatArray:
        return _design(np.atleast_1d(t), self.period, self.order) @ self.coef


KinematicPath = PeriodicLinearPath | PolygonPath | FourierPath


class PeriodicKinematicHypothesis(BaseHypothesis):
    kind: HypothesisKind = "kinematic"

    def __init__(
        self,
        body_id: str,
        period: float,
        coef: FloatArray,
        observer: Callable[[FloatArray], FloatArray] | None,
        info: dict[str, Any],
        path: KinematicPath | None = None,
    ) -> None:
        self.path = path
        self.body_id = body_id
        self.period = period
        self.coef = np.asarray(coef)
        self.order = (len(self.coef) - 1) // 2
        self.observer = observer
        self.info = info

    def _pos(self, t: FloatArray) -> FloatArray:
        if self.path is not None:
            return self.path(t)
        return _design(np.atleast_1d(t), self.period, self.order) @ self.coef

    def heliocentric_position(self, t: FloatArray, body_id: str) -> PositionPrediction:
        return PositionPrediction(mean=self._pos(t))

    def predict_sky(self, t: FloatArray, body_id: str) -> SkyPrediction:
        if self.observer is None:
            raise ValueError("sky predictions need an observer")
        d = self._pos(t) - self.observer(np.atleast_1d(t))
        return SkyPrediction(mean=d / np.linalg.norm(d, axis=1, keepdims=True))

    def notes(self) -> str:
        return str(self.info.get("notes", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "period": self.period,
            "coef": self.coef.tolist(),
            "info": self.info,
            "notes": self.notes(),
        }
