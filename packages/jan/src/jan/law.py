"""Hypotheses defined by an acceleration law a(x, v, t) plus an initial state (Phase 4).

Shared by the discovery (SINDy, PySR) and physics-informed ML strategies: whatever the law
(symbolic or a neural network), predictions come from integrating it, and the field itself
is exposed for structural scoring.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from jan.base import BaseHypothesis
from jt_core.angles import tangent_basis
from jt_core.integrate import propagate
from jt_core.schema import (
    HypothesisKind,
    LawEstimate,
    ObservationSet,
    PositionPrediction,
    SkyPrediction,
    TraceFrame,
)

FloatArray = NDArray[np.float64]
Accel = Callable[[Any, Any, Any, Any], Any]
FIT_RTOL, FIT_ATOL = 1e-9, 1e-12


class LawHypothesis(BaseHypothesis):
    kind: HypothesisKind = "dynamical"

    def __init__(
        self,
        body_id: str,
        accel: Accel,
        params: Any,
        x0: FloatArray,
        v0: FloatArray,
        t0: float,
        observer: Callable[[FloatArray], FloatArray] | None,
        law: LawEstimate | None,
        info: dict[str, Any],
        frames: list[TraceFrame] | None = None,
        frame_hypothesis: Callable[[TraceFrame], BaseHypothesis | None] | None = None,
    ) -> None:
        self.body_id = body_id
        self.accel = accel
        self.params = params
        self.x0 = np.asarray(x0, dtype=np.float64).reshape(1, 3)
        self.v0 = np.asarray(v0, dtype=np.float64).reshape(1, 3)
        self.t0 = t0
        self.observer = observer
        self._law = law
        self.info = info
        self._frames = frames
        self._frame_hypothesis = frame_hypothesis

    def _pos(self, t: FloatArray) -> FloatArray:
        x, _ = propagate(
            self.accel,
            jnp.asarray(self.x0),
            jnp.asarray(self.v0),
            self.t0,
            jnp.asarray(np.atleast_1d(t)),
            self.params,
            FIT_RTOL,
            FIT_ATOL,
        )
        return np.asarray(x)[:, 0]

    def heliocentric_position(self, t: FloatArray, body_id: str) -> PositionPrediction:
        return PositionPrediction(mean=self._pos(t))

    def predict_sky(self, t: FloatArray, body_id: str) -> SkyPrediction:
        if self.observer is None:
            raise ValueError("sky predictions need an observer")
        d = self._pos(t) - self.observer(np.atleast_1d(t))
        return SkyPrediction(mean=d / np.linalg.norm(d, axis=1, keepdims=True))

    def acceleration(self, x: FloatArray, v: FloatArray, t: FloatArray, body_id: str) -> FloatArray:
        acc = self.accel
        f = jax.vmap(lambda xi, vi, ti: acc(xi[None], vi[None], ti, self.params)[0])
        return np.asarray(f(jnp.asarray(x), jnp.asarray(np.nan_to_num(v)), jnp.asarray(t)))

    def law(self) -> LawEstimate | None:
        return self._law

    def observation_sigma(self, body_id: str) -> float | None:
        v = self.info.get("sigma_rad")
        return None if v is None else float(v)

    def trace(self) -> list[TraceFrame] | None:
        return self._frames

    def at_frame(self, frame: TraceFrame) -> BaseHypothesis | None:
        return None if self._frame_hypothesis is None else self._frame_hypothesis(frame)

    def notes(self) -> str:
        return str(self.info.get("notes", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "x0": self.x0[0].tolist(),
            "v0": self.v0[0].tolist(),
            "t0": self.t0,
            "law": self._law.to_dict() if self._law else None,
            "info": self.info,
            "notes": self.notes(),
        }


def fit_initial_state(accel: Accel, params: Any, ob: ObservationSet, t0: float,
                      x0: FloatArray, v0: FloatArray,
                      observer: Callable[[FloatArray], FloatArray] | None,
                      max_nfev: int = 50) -> tuple[FloatArray, FloatArray, float]:  # fmt: skip
    """Least-squares initial state (at t0) for a fixed law, against positions (T0) or sky
    directions (T1). Returns (x0, v0, per-axis RMS residual: AU or radians)."""
    t = jnp.asarray(ob.t)
    if ob.is_sky:
        assert observer is not None
        u = ob.unit_vectors()
        east, north = tangent_basis(u)
        r_obs, uj, ej, nj = (jnp.asarray(a) for a in (observer(ob.t), u, east, north))

        def res_j(s: jax.Array) -> jax.Array:
            x, _ = propagate(
                accel,
                s[None, :3],
                s[None, 3:],
                t0,
                t,
                params,
                FIT_RTOL,
                FIT_ATOL,
                differentiable=True,
            )
            d = x[:, 0] - r_obs
            d = d / jnp.linalg.norm(d, axis=1, keepdims=True)
            k = jnp.sum(d * uj, 1)
            return jnp.stack([jnp.sum(d * ej, 1) / k, jnp.sum(d * nj, 1) / k], 1).ravel()

    else:
        pos = jnp.asarray(ob.positions())

        def res_j(s: jax.Array) -> jax.Array:
            x, _ = propagate(
                accel,
                s[None, :3],
                s[None, 3:],
                t0,
                t,
                params,
                FIT_RTOL,
                FIT_ATOL,
                differentiable=True,
            )
            return (x[:, 0] - pos).ravel()

    rf, jf = jax.jit(res_j), jax.jit(jax.jacfwd(res_j))
    s0 = np.concatenate([np.ravel(x0), np.ravel(v0)])
    res = least_squares(lambda s: np.asarray(rf(jnp.asarray(s))), s0,
                        jac=lambda s: np.asarray(jf(jnp.asarray(s))),  # pyright: ignore[reportArgumentType]
                        x_scale="jac", max_nfev=max_nfev)  # fmt: skip
    r = np.asarray(rf(jnp.asarray(res.x)))
    return res.x[:3], res.x[3:], float(np.sqrt(np.mean(r**2)))
