"""`gp.sky`: Gaussian process on the unit-vector components of the sky track (jan[gp]).

Each component of u(t) is modelled as a constant plus a sum of K quasi-periodic terms
(ExpSineSquared × ExpSquared) with white noise; hyperparameters are shared across the
three components and fitted by maximizing the summed marginal likelihood, starting from
the Lomb–Scargle periods. Predictions are renormalized onto the sphere, and the
component variances are projected onto the tangent plane.

Nonparametric: excellent interpolation with honest error bars that grow away from the
data, but the quasi-periodic envelope decays, so extrapolation degrades with time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from scipy.optimize import minimize
from tinygp import GaussianProcess, kernels

from jan.base import BaseHypothesis, PerBodyHypothesis, StrategyConfig, make_config, register
from jan.strategies.ptolemy.fourier import PtolemyFourier
from jt_core.angles import tangent_basis
from jt_core.constants import ARCSEC
from jt_core.schema import HypothesisKind, ObservationSet, PublicMeta, SkyPrediction

FloatArray = NDArray[np.float64]


class GpConfig(StrategyConfig):
    n_periods: int = Field(
        2, ge=1, le=3, description="Quasi-periodic terms (from the periodogram)."
    )
    fit_periods: bool = Field(
        False, description="Free periods drift badly; default keeps the periodogram ones."
    )
    envelope_spans: float = Field(2.0, description="Initial envelope length in training spans.")
    max_iter: int = 300
    n_samples: int = 64


def _kernel(p: dict[str, jax.Array], k: int) -> kernels.Kernel:
    out = None
    for j in range(k):
        term = (
            jnp.exp(2 * p["log_amp"][j])
            * kernels.ExpSineSquared(
                scale=jnp.exp(p["log_period"][j]), gamma=jnp.exp(p["log_gamma"][j])
            )
            * kernels.ExpSquared(scale=jnp.exp(p["log_ell"][j]))
        )
        out = term if out is None else out + term
    assert out is not None
    return out


def _gp(p: dict[str, jax.Array], t: jax.Array, k: int) -> GaussianProcess:
    return GaussianProcess(_kernel(p, k), t, diag=jnp.exp(2 * p["log_noise"]))


class GpHypothesis(BaseHypothesis):
    kind: HypothesisKind = "nonparametric"

    def __init__(
        self,
        body_id: str,
        t: FloatArray,
        y: FloatArray,
        mean: FloatArray,
        params: dict[str, FloatArray],
        k: int,
        n_samples: int,
        seed: int,
        info: dict[str, Any],
    ) -> None:
        self.body_id = body_id
        self.t = t
        self.y = y
        self.mean0 = mean
        self.params = {n: jnp.asarray(v) for n, v in params.items()}
        self.k = k
        self.n_samples = n_samples
        self.seed = seed
        self.info = info

    def predict_sky(self, t: FloatArray, body_id: str) -> SkyPrediction:
        if body_id != self.body_id:
            raise KeyError(body_id)
        gp = _gp(self.params, jnp.asarray(self.t), self.k)
        tq = jnp.asarray(t, dtype=jnp.float64)
        mus, vars_ = [], []
        for c in range(3):
            _, cond = gp.condition(jnp.asarray(self.y[:, c] - self.mean0[c]), tq)
            mus.append(np.asarray(cond.loc) + self.mean0[c])
            vars_.append(np.asarray(cond.variance))
        m = np.stack(mus, 1)
        var = np.stack(vars_, 1)
        mean = m / np.linalg.norm(m, axis=1, keepdims=True)
        east, north = tangent_basis(mean)
        norm2 = np.sum(m * m, axis=1)
        cee = np.sum(east**2 * var, 1) / norm2
        cnn = np.sum(north**2 * var, 1) / norm2
        cen = np.sum(east * north * var, 1) / norm2
        cov = np.stack([np.stack([cee, cen], -1), np.stack([cen, cnn], -1)], -2)
        g = np.random.default_rng(self.seed)
        z = g.standard_normal((self.n_samples, len(mean), 3)) * np.sqrt(np.clip(var, 0, None))[None]
        s = m[None] + z
        s /= np.linalg.norm(s, axis=2, keepdims=True)
        return SkyPrediction(mean=mean, samples=s, cov=cov)

    def observation_sigma(self, body_id: str) -> float | None:
        return float(np.exp(self.params["log_noise"]))

    def notes(self) -> str:
        per = ", ".join(f"{np.exp(v):.1f} d" for v in np.asarray(self.params["log_period"]))
        return (
            f"GP with {self.k} quasi-periodic term(s), periods [{per}], "
            f"noise {np.exp(float(self.params['log_noise'])) / ARCSEC:.1f}″."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "body_id": self.body_id,
            "params": {k: np.asarray(v).tolist() for k, v in self.params.items()},
            "info": self.info,
            "notes": self.notes(),
        }


@register("gp.sky")
class GpSky:
    """Gaussian process on unit-vector components with quasi-periodic kernels."""

    name: ClassVar[str]
    supported_tiers: ClassVar[set[str]] = {"T1", "T2", "T3"}
    Config: ClassVar[type[StrategyConfig]] = GpConfig

    def fit(
        self,
        obs: ObservationSet,
        meta: PublicMeta,
        config: StrategyConfig,
        rng_seed: int,
        workdir: Path,
    ) -> BaseHypothesis:
        assert isinstance(config, GpConfig)
        if not obs.is_sky:
            raise ValueError("gp.sky needs sky observations")
        return PerBodyHypothesis(
            {b: self._fit_body(obs.for_body(b), b, config, rng_seed) for b in obs.body_ids}
        )

    def _fit_body(self, ob: ObservationSet, bid: str, cfg: GpConfig, seed: int) -> GpHypothesis:
        t = ob.t
        y = ob.unit_vectors()
        k = cfg.n_periods
        four = PtolemyFourier()
        fh = four._fit_body(ob, bid, make_config(four, {"n_frequencies": k, "n_harmonics": 2}))  # pyright: ignore[reportPrivateUsage]
        periods = np.sort(1.0 / np.asarray(fh.base))[::-1][:k]
        span = float(np.ptp(t))
        mean = y.mean(axis=0)
        amp = float(np.std(y - mean))
        p0 = {
            "log_amp": np.full(k, np.log(amp)),
            "log_period": np.log(periods),
            "log_gamma": np.zeros(k),
            "log_ell": np.full(k, np.log(cfg.envelope_spans * span)),
            "log_noise": np.array(np.log(60 * ARCSEC)),
        }
        names = list(p0)
        sizes = [np.size(p0[n]) for n in names]
        tj = jnp.asarray(t)
        yj = jnp.asarray(y - mean)

        def unpack(v: jax.Array) -> dict[str, jax.Array]:
            out, i = {}, 0
            for n, sz in zip(names, sizes, strict=True):
                out[n] = v[i : i + sz] if p0[n].ndim else v[i]
                i += sz
            if not cfg.fit_periods:
                out["log_period"] = jnp.asarray(p0["log_period"])
            return out

        @jax.jit
        def loss(v: jax.Array) -> jax.Array:
            gp = _gp(unpack(v), tj, k)
            return -jnp.sum(jnp.stack([gp.log_probability(yj[:, c]) for c in range(3)]))

        vg = jax.jit(jax.value_and_grad(loss))
        v0 = np.concatenate([np.atleast_1d(p0[n]) for n in names])

        def fun(v: FloatArray) -> tuple[float, FloatArray]:
            val, grad = vg(jnp.asarray(v))
            return float(val), np.asarray(grad, dtype=np.float64)

        res = minimize(fun, v0, jac=True, method="L-BFGS-B", options={"maxiter": cfg.max_iter})
        params = {n: np.asarray(x) for n, x in unpack(jnp.asarray(res.x)).items()}
        info = {
            "neg_log_marginal": float(res.fun),
            "converged": bool(res.success),
            "initial_periods": periods.tolist(),
        }
        return GpHypothesis(bid, t, y, mean, params, k, cfg.n_samples, seed, info)
