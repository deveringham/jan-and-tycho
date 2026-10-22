"""A fitted conic: predictions, elements and law with linearized (Gaussian) uncertainty."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from jan.base import BaseHypothesis
from jan.models.conic import CircularOrbit, OrbitFamily, canonical_elements, sky_directions
from jt_core.angles import tangent_basis
from jt_core.conics import bucket, pad_rows
from jt_core.schema import (
    ELEMENT_NAMES,
    ElementsEstimate,
    HypothesisKind,
    LawEstimate,
    ModelScore,
    PositionPrediction,
    SkyPrediction,
    TraceFrame,
)

FloatArray = NDArray[np.float64]

NEWTON_LAW = [
    "-GM*x/(x**2 + y**2 + z**2)**(3/2)",
    "-GM*y/(x**2 + y**2 + z**2)**(3/2)",
    "-GM*z/(x**2 + y**2 + z**2)**(3/2)",
]


@jax.jit
def _sky(family: OrbitFamily, theta: jax.Array, t: jax.Array, r_obs: jax.Array) -> jax.Array:
    return sky_directions(family, theta, t, r_obs)


@jax.jit
def _sky_jac(family: OrbitFamily, theta: jax.Array, t: jax.Array, r_obs: jax.Array) -> jax.Array:
    return jax.jacfwd(lambda th: sky_directions(family, th, t, r_obs))(theta)


@jax.jit
def _sky_batch(family: OrbitFamily, thetas: jax.Array, t: jax.Array, r_obs: jax.Array) -> jax.Array:
    return jax.vmap(lambda th: sky_directions(family, th, t, r_obs))(thetas)


@jax.jit
def _pos(family: OrbitFamily, theta: jax.Array, t: jax.Array) -> jax.Array:
    return family.position(theta, t)


@jax.jit
def _pos_batch(family: OrbitFamily, thetas: jax.Array, t: jax.Array) -> jax.Array:
    return jax.vmap(lambda th: family.position(th, t))(thetas)


@jax.jit
def _el_jac(family: OrbitFamily, theta: jax.Array) -> jax.Array:
    return jax.jacfwd(family.elements)(theta)


def _psd_sqrt(cov: FloatArray) -> FloatArray:
    w, v = np.linalg.eigh(0.5 * (cov + cov.T))
    return v * np.sqrt(np.clip(w, 0.0, None))


class ConicHypothesis(BaseHypothesis):
    kind: HypothesisKind = "dynamical"

    def __init__(
        self,
        family: OrbitFamily,
        theta: FloatArray,
        body_id: str,
        observer: Callable[[FloatArray], FloatArray] | None,
        cov: FloatArray | None = None,
        n_samples: int = 0,
        seed: int = 0,
        info: dict[str, Any] | None = None,
        frames: list[TraceFrame] | None = None,
    ) -> None:
        self.family = family
        self.theta = np.asarray(theta, dtype=np.float64)
        self.body_id = body_id
        self.observer = observer
        self.cov = cov
        self.info = info or {}
        self._frames = frames
        self._samples: FloatArray | None = None
        if cov is not None and n_samples > 0:
            g = np.random.default_rng(seed)
            z = g.standard_normal((n_samples, self.theta.size))
            self._samples = self.theta + z @ _psd_sqrt(cov).T

    @property
    def family_name(self) -> str:
        return self.family.name

    def _check(self, body_id: str) -> None:
        if body_id != self.body_id:
            raise KeyError(f"hypothesis covers body {self.body_id!r}, not {body_id!r}")

    # ------------------------------------------------------------------ predictions

    def predict_sky(self, t: FloatArray, body_id: str) -> SkyPrediction:
        self._check(body_id)
        if self.observer is None:
            raise ValueError("sky predictions need a known observer")
        n = np.size(t)
        tp = pad_rows(np.atleast_1d(np.asarray(t, dtype=np.float64)), bucket(n))
        tj = jnp.asarray(tp)
        r_obs = jnp.asarray(self.observer(tp))
        th = jnp.asarray(self.theta)
        mean = np.asarray(_sky(self.family, th, tj, r_obs))[:n]
        cov2 = samples = None
        if self.cov is not None:
            jac = np.asarray(_sky_jac(self.family, th, tj, r_obs))[:n]  # (N, 3, P)
            east, north = tangent_basis(mean)
            jt = np.stack(
                [np.einsum("nk,nkp->np", east, jac), np.einsum("nk,nkp->np", north, jac)], 1
            )
            cov2 = np.einsum("nap,pq,nbq->nab", jt, self.cov, jt)
        if self._samples is not None:
            samples = np.asarray(_sky_batch(self.family, jnp.asarray(self._samples), tj, r_obs))[
                :, :n
            ]
        return SkyPrediction(mean=mean, samples=samples, cov=cov2)

    def heliocentric_position(self, t: FloatArray, body_id: str) -> PositionPrediction:
        self._check(body_id)
        n = np.size(t)
        tj = jnp.asarray(pad_rows(np.atleast_1d(np.asarray(t, dtype=np.float64)), bucket(n)))
        mean = np.asarray(_pos(self.family, jnp.asarray(self.theta), tj))[:n]
        samples = None
        if self._samples is not None:
            samples = np.asarray(_pos_batch(self.family, jnp.asarray(self._samples), tj))[:, :n]
        return PositionPrediction(mean=mean, samples=samples)

    # ------------------------------------------------------------------ physics

    def element_values(self) -> FloatArray:
        return np.asarray(self.family.elements(jnp.asarray(self.theta)))

    def elements(self, body_id: str) -> ElementsEstimate:
        self._check(body_id)
        el = canonical_elements(self.element_values())
        point = el.to_dict()
        if isinstance(self.family, CircularOrbit):
            point["e"] = 0.0
        est = ElementsEstimate(point=point)
        if self.cov is not None:
            jac = np.asarray(_el_jac(self.family, jnp.asarray(self.theta)))
            est.cov = (jac @ self.cov @ jac.T).tolist()
            est.cov_names = list(ELEMENT_NAMES)
        return est

    def acceleration(self, x: FloatArray, v: FloatArray, t: FloatArray, body_id: str) -> FloatArray:
        r = np.linalg.norm(x, axis=1, keepdims=True)
        return -self.family.gm * x / r**3

    def law(self) -> LawEstimate:
        return LawEstimate(
            expression=list(NEWTON_LAW), constants={"GM": self.family.gm}, radial="-GM/r**2"
        )

    def model_comparison(self) -> list[ModelScore] | None:
        bic = self.info.get("bic")
        if bic is None:
            return None
        return [ModelScore(model_name=self.family_name, information_criterion=float(bic))]

    # ------------------------------------------------------------------ traces

    def trace(self) -> list[TraceFrame] | None:
        return self._frames

    def at_frame(self, frame: TraceFrame) -> ConicHypothesis:
        return ConicHypothesis(
            self.family, np.asarray(frame.state["theta"]), self.body_id, self.observer
        )

    def observation_sigma(self, body_id: str) -> float | None:
        v = self.info.get("sigma_rad")
        return None if v is None else float(v)

    # ------------------------------------------------------------------ archive

    def notes(self) -> str:
        return str(self.info.get("notes", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "family": self.family_name,
            "body_id": self.body_id,
            "t0": self.family.t0,
            "gm": self.family.gm,
            "param_names": list(self.family.param_names),
            "theta": self.theta.tolist(),
            "theta_cov": None if self.cov is None else self.cov.tolist(),
            "elements": self.elements(self.body_id).to_dict(),
            "law": self.law().to_dict(),
            "info": self.info,
            "notes": self.notes(),
        }
