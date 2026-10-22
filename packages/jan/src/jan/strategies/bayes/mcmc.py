"""`bayes.mcmc`: emcee affine-invariant ensemble sampling of one family's posterior.

Walkers start in a tight Gaussian ball around the MAP (Laplace covariance). The likelihood
is Gaussian or Student-t on tangent-plane residuals with an optional free noise scale.
Failure modes: multimodal posteriors (walkers never leave the MAP mode), and very
anisotropic posteriors at small walker counts (slow mixing).

`coords="state"` (ellipse only) samples the same posterior in the heliocentric state
(x, v) at the reference epoch instead of the family's φ, with the prior carried over by the
Jacobian |∂φ/∂(x, v)|. A short arc pins the position and sky motion and leaves mostly the
range rate free, so its posterior is a nearly straight ridge in (x, v) but a curved banana
in (a, e, ϖ, λ0), along which the stretch move barely mixes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal

import emcee
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from jan.base import BaseHypothesis, PerBodyHypothesis, StrategyConfig, register, subsample_frames
from jan.inference import SkyModel, fit_map
from jan.models.families import TWO_PI, EllipseFamily
from jan.models.posterior import PosteriorHypothesis
from jan.setup import BayesConfig, setup_body
from jt_core.schema import ObservationSet, PublicMeta, TraceFrame


class McmcConfig(BayesConfig):
    family: str = "conic-elliptic"
    n_walkers: int = Field(0, description="0 → 4 × dimension.")
    n_steps: int = 4000
    burn: int = 1000
    thin: int = 5
    trace_walkers: int = Field(8, description="Walkers stored per trace frame.")
    coords: Literal["phi", "state"] = Field(
        "phi", description="Sampling coordinates: the family's φ, or the state at t0."
    )


FloatArray = NDArray[np.float64]


class StateCoords:
    """Whitened state coordinates for an ellipse model: z with y = y_map + A z, where
    y = (x, v at t0, other dims unchanged) and A = (∂y/∂φ)·L maps the Laplace factor L
    (Σ_φ = L Lᵀ) into y. A short arc pins the sky direction to ~1e-6 rad, so the state's
    Laplace cloud is nearly collinear in raw (x, v); whitening keeps the walker ensemble
    well conditioned. The affine map has a constant Jacobian, so `log_prob_batch` (the
    posterior density in z) only needs |∂φ/∂y|."""

    def __init__(self, model: SkyModel, x_map: FloatArray, chol: FloatArray) -> None:
        fam = model.family
        if not isinstance(fam, EllipseFamily):
            raise ValueError(f"coords='state' needs the conic-elliptic family, not {fam.name}")
        self.model = model
        self.x_map = np.asarray(x_map)

        def to_y(x: jax.Array) -> jax.Array:
            return jnp.concatenate([fam.to_state(x[:6]), x[6:]])

        xm = jnp.asarray(x_map)
        y_map = to_y(xm)
        a = jax.jacfwd(to_y)(xm) @ jnp.asarray(chol)

        def to_x(z: jax.Array) -> jax.Array:
            y = y_map + a @ z
            return jnp.concatenate([fam.from_state(y[:6]), y[6:]])

        def log_jac(z: jax.Array) -> jax.Array:
            y = y_map + a @ z
            return jnp.linalg.slogdet(jax.jacfwd(fam.from_state)(y[:6]))[1]

        self._to_x = jax.jit(jax.vmap(to_x))
        self._log_jac = jax.jit(jax.vmap(log_jac))

    def to_x(self, zs: FloatArray) -> FloatArray:
        """φ samples, periodic dims taken within ±π of the MAP (no split at a wrap point)."""
        x = np.array(self._to_x(jnp.asarray(zs)))
        per, xm = self.model.periodic(), self.x_map
        x[..., per] = xm[per] + np.mod(x[..., per] - xm[per] + np.pi, TWO_PI) - np.pi
        return x

    def log_prob_batch(self, zs: FloatArray) -> FloatArray:
        lp = self.model.log_prob_batch(self.to_x(zs))
        lj = np.asarray(self._log_jac(jnp.asarray(zs)))
        return np.where(np.isfinite(lp) & np.isfinite(lj), lp + lj, -np.inf)


@register("bayes.mcmc")
class BayesMcmc:
    """emcee posterior over a chosen family, initialized at the MAP."""

    name: ClassVar[str]
    supported_tiers: ClassVar[set[str]] = {"T1", "T2", "T3"}
    Config: ClassVar[type[StrategyConfig]] = McmcConfig

    def fit(
        self,
        obs: ObservationSet,
        meta: PublicMeta,
        config: StrategyConfig,
        rng_seed: int,
        workdir: Path,
    ) -> BaseHypothesis:
        assert isinstance(config, McmcConfig)
        return PerBodyHypothesis(
            {
                b: self._fit_body(obs.for_body(b), b, meta, config, rng_seed, obs.sun())
                for b in obs.body_ids
            }
        )

    def _fit_body(
        self,
        ob: ObservationSet,
        bid: str,
        meta: PublicMeta,
        cfg: McmcConfig,
        seed: int,
        sun: ObservationSet | None = None,
    ) -> PosteriorHypothesis:
        setup = setup_body(ob, meta, sun)
        model = setup.model(cfg.family, cfg)
        mp = fit_map(model, setup.starts(model.family, seed))
        ndim = len(mp.x)
        nw = cfg.n_walkers or max(4 * ndim, 16)
        nw += nw % 2
        rng = np.random.default_rng(seed)
        chol = np.linalg.cholesky(mp.cov + 1e-18 * np.eye(ndim))
        p0 = mp.x + 0.1 * rng.standard_normal((nw, ndim)) @ chol.T
        lo, hi = model.lo_hi()
        per = model.periodic()
        p0 = np.where(per, p0, np.clip(p0, lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo)))
        log_prob: Callable[[FloatArray], FloatArray] = model.log_prob_batch
        to_x: Callable[[FloatArray], FloatArray] = np.asarray
        if cfg.coords == "state":
            sc = StateCoords(model, mp.x, chol)
            p0 = 0.1 * rng.standard_normal((nw, ndim))
            log_prob, to_x = sc.log_prob_batch, sc.to_x
        sampler = emcee.EnsembleSampler(nw, ndim, log_prob, vectorize=True)
        sampler.random_state = np.random.RandomState(int(rng.integers(2**31))).get_state()
        sampler.run_mcmc(p0, cfg.n_steps, progress=False)
        samples = to_x(np.asarray(sampler.get_chain(discard=cfg.burn, thin=cfg.thin, flat=True)))
        frames = None
        if cfg.record_trace:
            chain = np.asarray(sampler.get_chain())  # (steps, walkers, dim)
            chain = to_x(chain.reshape(-1, ndim)).reshape(chain.shape)
            frames = [
                TraceFrame(
                    label=f"step {k}",
                    iteration=k,
                    state={
                        "x": chain[k].mean(axis=0).tolist(),
                        "samples": chain[k, : cfg.trace_walkers].tolist(),
                    },
                )
                for k in range(0, cfg.n_steps, max(cfg.n_steps // 200, 1))
            ]
            frames = subsample_frames(frames, cfg.trace_max_frames)
        try:
            logging.getLogger("emcee").setLevel(logging.ERROR)
            tau = float(np.max(sampler.get_autocorr_time(discard=cfg.burn, quiet=True)))
        except (emcee.autocorr.AutocorrError, ValueError):
            tau = float("nan")
        acc = float(np.mean(sampler.acceptance_fraction))
        info = {
            "family": cfg.family,
            "coords": cfg.coords,
            "n_walkers": nw,
            "acceptance": acc,
            "autocorr_time": tau,
            "n_samples": len(samples),
            "notes": f"emcee on {cfg.family} ({cfg.coords} coordinates): {nw} walkers × "
            f"{cfg.n_steps} steps, "
            f"acceptance {acc:.2f}, max τ ≈ {tau:.0f} steps.",
        }
        return PosteriorHypothesis(
            model.family,
            mp.x,
            samples,
            bid,
            setup.observer,
            model.sigma_free,
            sigma_known=None if model.sigma_free else setup.data.median_sigma(),
            observer_family=model.observer,
            n_predictive=cfg.n_predictive_samples,
            seed=seed,
            info=info,
            frames=frames,
        )
