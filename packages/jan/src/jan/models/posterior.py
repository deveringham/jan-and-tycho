"""A hypothesis represented by posterior samples over a Bayesian family."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from jan.base import BaseHypothesis
from jan.models.conic_hypothesis import NEWTON_LAW
from jan.models.families import BayesFamily, ObserverFamily
from jt_core.angles import tangent_basis, wrap_pi
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


@jax.jit
def _sky_batch(fam: BayesFamily, phis: jax.Array, t: jax.Array, r_obs: jax.Array) -> jax.Array:
    return jax.vmap(lambda p: fam.sky(p, t, r_obs))(phis)


@jax.jit
def _sky_batch_free(
    fam: BayesFamily, obs_fam: ObserverFamily, phis: jax.Array, phis_o: jax.Array, t: jax.Array
) -> jax.Array:
    """Sky directions when every draw carries its own observer orbit."""
    return jax.vmap(lambda p, po: fam.sky(p, t, obs_fam.position(po, t)))(phis, phis_o)


@jax.jit
def _pos_batch(fam: BayesFamily, phis: jax.Array, t: jax.Array) -> jax.Array:
    return jax.vmap(lambda p: fam.position(p, t))(phis)


@jax.jit
def _el_batch(fam: BayesFamily, phis: jax.Array) -> jax.Array:
    return jax.vmap(fam.elements)(phis)


def _pad_batch(x: FloatArray) -> tuple[FloatArray, int]:
    n = len(x)
    return pad_rows(x, bucket(n, 4)), n


def unwrap_elements(el: FloatArray, ref: FloatArray, gm: float) -> FloatArray:
    """Make element samples continuous around a reference: angles near the reference's,
    and (for ellipses) t_p within half a period of the reference t_p."""
    out = el.copy()
    for j in (3, 4):  # Ω, ω
        out[:, j] = ref[j] + wrap_pi(el[:, j] - ref[j])
    ell = el[:, 1] < 1
    if ell.any():
        a = el[ell, 0] / (1 - el[ell, 1])
        period = 2 * np.pi * np.sqrt(a**3 / gm)
        out[ell, 5] = ref[5] + np.mod(el[ell, 5] - ref[5] + period / 2, period) - period / 2
    return out


class PosteriorHypothesis(BaseHypothesis):
    """Predictions average over posterior draws; elements report medians and samples."""

    def __init__(
        self,
        family: BayesFamily,
        x_map: FloatArray,
        samples: FloatArray,
        body_id: str,
        observer: Callable[[FloatArray], FloatArray] | None,
        sigma_free: bool,
        sigma_known: float | None = None,
        n_predictive: int = 100,
        seed: int = 0,
        info: dict[str, Any] | None = None,
        scores: list[ModelScore] | None = None,
        frames: list[TraceFrame] | None = None,
        observer_family: ObserverFamily | None = None,
    ) -> None:
        self.family = family
        self.x_map = np.asarray(x_map, dtype=np.float64)
        self.samples = np.atleast_2d(np.asarray(samples, dtype=np.float64))
        self.body_id = body_id
        self.observer = observer
        self.sigma_free = sigma_free
        self.sigma_known = sigma_known
        self.info = info or {}
        self.scores = scores
        self.frames = frames
        self.kind: HypothesisKind = "dynamical" if family.kind == "dynamical" else "kinematic"
        self.observer_family = observer_family
        p = family.n_params
        g = np.random.default_rng(seed)
        idx = g.choice(len(self.samples), size=min(n_predictive, len(self.samples)), replace=False)
        self.pred_phi = self.samples[np.sort(idx), :p]
        no = 4 if observer_family is not None else 0
        self.pred_phi_obs = self.samples[np.sort(idx), p : p + no]

    def _check(self, body_id: str) -> None:
        if body_id != self.body_id:
            raise KeyError(f"hypothesis covers body {self.body_id!r}, not {body_id!r}")

    # ------------------------------------------------------------------ predictions

    def predict_sky(self, t: FloatArray, body_id: str) -> SkyPrediction:
        self._check(body_id)
        n = np.size(t)
        tp = pad_rows(np.atleast_1d(np.asarray(t, dtype=np.float64)), bucket(n))
        phis, m = _pad_batch(self.pred_phi)
        if self.observer_family is not None:
            phis_o, _ = _pad_batch(self.pred_phi_obs)
            u = np.asarray(
                _sky_batch_free(
                    self.family,
                    self.observer_family,
                    jnp.asarray(phis),
                    jnp.asarray(phis_o),
                    jnp.asarray(tp),
                )
            )
        else:
            r_obs = self.observer(tp) if self.observer is not None else np.zeros((len(tp), 3))
            u = np.asarray(
                _sky_batch(self.family, jnp.asarray(phis), jnp.asarray(tp), jnp.asarray(r_obs))
            )
        u = u[:m, :n]
        mean = u.mean(axis=0)
        mean /= np.linalg.norm(mean, axis=1, keepdims=True)
        east, north = tangent_basis(mean)
        d = u - mean[None]
        de = np.einsum("snk,nk->sn", d, east)
        dn = np.einsum("snk,nk->sn", d, north)
        cov = np.stack(
            [
                np.stack([np.mean(de * de, 0), np.mean(de * dn, 0)], -1),
                np.stack([np.mean(de * dn, 0), np.mean(dn * dn, 0)], -1),
            ],
            -2,
        )
        return SkyPrediction(mean=mean, samples=u, cov=cov)

    def heliocentric_position(self, t: FloatArray, body_id: str) -> PositionPrediction | None:
        self._check(body_id)
        if self.family.geocentric:
            return None  # scale is unobservable for a geocentric kinematic model
        n = np.size(t)
        tp = pad_rows(np.atleast_1d(np.asarray(t, dtype=np.float64)), bucket(n))
        phis, m = _pad_batch(self.pred_phi)
        r = np.asarray(_pos_batch(self.family, jnp.asarray(phis), jnp.asarray(tp)))[:m, :n]
        return PositionPrediction(mean=r.mean(axis=0), samples=r)

    # ------------------------------------------------------------------ physics

    def element_samples(self) -> FloatArray | None:
        if not self.family.conic:
            return None
        p = self.family.n_params
        phis, m = _pad_batch(self.samples[:, :p])
        el = np.asarray(_el_batch(self.family, jnp.asarray(phis)))[:m]
        ref = np.asarray(self.family.elements(jnp.asarray(self.x_map[:p])))
        return unwrap_elements(el, ref, self.family.gm)

    def elements(self, body_id: str) -> ElementsEstimate | None:
        self._check(body_id)
        el = self.element_samples()
        if el is None:
            return None
        med = np.median(el, axis=0)
        point = {k: float(v) for k, v in zip(ELEMENT_NAMES, med, strict=True)}
        point["Omega"] = float(np.mod(point["Omega"], 2 * np.pi))
        point["omega"] = float(np.mod(point["omega"], 2 * np.pi))
        cov = np.cov(el.T) if len(el) > 1 else np.zeros((6, 6))
        return ElementsEstimate(
            point=point,
            cov=cov.tolist(),
            cov_names=list(ELEMENT_NAMES),
            samples={k: el[:, j].tolist() for j, k in enumerate(ELEMENT_NAMES)},
        )

    def acceleration(
        self, x: FloatArray, v: FloatArray, t: FloatArray, body_id: str
    ) -> FloatArray | None:
        if self.family.kind != "dynamical":
            return None
        fn = getattr(self.family, "acceleration", None)
        if fn is not None:  # families with their own force law (central forces, perturbers)
            return np.asarray(
                fn(jnp.asarray(self.x_map[: self.family.n_params]), jnp.asarray(x), jnp.asarray(t))
            )
        r = np.linalg.norm(x, axis=1, keepdims=True)
        return -self.family.gm * x / r**3

    def law(self) -> LawEstimate | None:
        if self.family.kind != "dynamical":
            return None
        law_strings = getattr(self.family, "law_strings", None)
        if law_strings is not None:  # the family's own fitted law (at the MAP)
            vec, consts, radial = law_strings(self.x_map[: self.family.n_params])
            return LawEstimate(expression=vec, constants=consts, radial=radial)
        return LawEstimate(expression=list(NEWTON_LAW), constants={"GM": self.family.gm})

    def model_comparison(self) -> list[ModelScore] | None:
        return self.scores

    def observer_position(self, t: FloatArray) -> PositionPrediction | None:
        if self.observer_family is None:
            return None
        n = np.size(t)
        tp = pad_rows(np.atleast_1d(np.asarray(t, dtype=np.float64)), bucket(n))
        phis, m = _pad_batch(self.pred_phi_obs)
        r = np.asarray(_pos_batch(self.observer_family, jnp.asarray(phis), jnp.asarray(tp)))[:m, :n]
        return PositionPrediction(mean=r.mean(axis=0), samples=r)

    def observer_elements(self) -> ElementsEstimate | None:
        if self.observer_family is None:
            return None
        p = self.family.n_params
        phis, m = _pad_batch(self.samples[:, p : p + 4])
        el = np.asarray(_el_batch(self.observer_family, jnp.asarray(phis)))[:m]
        ref = np.asarray(self.observer_family.elements(jnp.asarray(self.x_map[p : p + 4])))
        el = unwrap_elements(el, ref, self.observer_family.gm)
        point = {k: float(v) for k, v in zip(ELEMENT_NAMES, np.median(el, axis=0), strict=True)}
        point["Omega"] = float(np.mod(point["Omega"], 2 * np.pi))
        point["omega"] = float(np.mod(point["omega"], 2 * np.pi))
        return ElementsEstimate(
            point=point,
            cov=np.cov(el.T).tolist() if len(el) > 1 else None,
            cov_names=list(ELEMENT_NAMES),
            samples={k: el[:, j].tolist() for j, k in enumerate(ELEMENT_NAMES)},
        )

    def hidden_bodies(self, t: FloatArray) -> dict[str, PositionPrediction] | None:
        fn = getattr(self.family, "perturber", None)
        if fn is None:
            return None
        tj = jnp.asarray(np.atleast_1d(t), dtype=jnp.float64)
        p = self.family.n_params
        mean = np.asarray(fn(jnp.asarray(self.x_map[:p]), tj))
        smp = np.stack([np.asarray(fn(jnp.asarray(x), tj)) for x in self.pred_phi[:32]])
        return {"perturber": PositionPrediction(mean=mean, samples=smp)}

    def observation_sigma(self, body_id: str) -> float | None:
        if self.sigma_free:
            return float(np.exp(np.median(self.samples[:, -1])))
        return self.sigma_known

    # ------------------------------------------------------------------ traces

    def trace(self) -> list[TraceFrame] | None:
        return self.frames

    def at_frame(self, frame: TraceFrame) -> PosteriorHypothesis:
        x = np.asarray(frame.state["x"])
        smp = np.asarray(frame.state.get("samples", [frame.state["x"]]))
        return PosteriorHypothesis(
            self.family,
            x,
            smp,
            self.body_id,
            self.observer,
            self.sigma_free,
            self.sigma_known,
            n_predictive=len(smp),
            observer_family=self.observer_family,
        )

    # ------------------------------------------------------------------ archive

    def notes(self) -> str:
        return str(self.info.get("notes", ""))

    def to_dict(self) -> dict[str, Any]:
        q = np.quantile(self.samples, [0.025, 0.5, 0.975], axis=0)
        names = list(self.family.param_names) + (["log_sigma"] if self.sigma_free else [])
        return {
            "kind": self.kind,
            "family": self.family.name,
            "body_id": self.body_id,
            "t0": self.family.t0,
            "param_names": names,
            "x_map": self.x_map.tolist(),
            "posterior_quantiles": {
                "q025": q[0].tolist(),
                "q50": q[1].tolist(),
                "q975": q[2].tolist(),
            },
            "n_samples": len(self.samples),
            "model_comparison": [s.to_dict() for s in self.scores] if self.scores else None,
            "info": self.info,
            "notes": self.notes(),
        }
