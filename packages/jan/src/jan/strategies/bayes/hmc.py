"""`bayes.hmc`: NumPyro NUTS with the differentiable JAX forward model.

Every prior dimension is a bounded uniform (NumPyro handles the transform); periodic
dimensions are re-centred on the MAP (uniform over one period around it), which is the
same prior on the circle but keeps the sampler away from the wrap-around point.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from pydantic import Field

from jan.base import BaseHypothesis, PerBodyHypothesis, StrategyConfig, register, subsample_frames
from jan.inference import SkyModel, fit_map
from jan.models.posterior import PosteriorHypothesis
from jan.setup import BayesConfig, setup_body
from jt_core.schema import ObservationSet, PublicMeta, TraceFrame


class HmcConfig(BayesConfig):
    family: str = "conic-elliptic"
    n_warmup: int = 500
    n_samples: int = 1000
    n_chains: int = 1
    dense_mass: bool = True
    target_accept: float = Field(0.85, gt=0, lt=1)
    trace_samples: int = Field(8, description="Recent draws stored per trace frame.")


def _bounds(model: SkyModel, x_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = model.lo_hi()
    per = model.periodic()
    w = hi - lo
    lo = np.where(per, x_map - w / 2, lo)
    hi = np.where(per, x_map + w / 2, hi)
    return lo, hi


@register("bayes.hmc")
class BayesHmc:
    """No-U-Turn sampling of one family's posterior, initialized at the MAP."""

    name: ClassVar[str]
    supported_tiers: ClassVar[set[str]] = {"T1", "T2", "T3"}
    Config: ClassVar[type[StrategyConfig]] = HmcConfig

    def fit(
        self,
        obs: ObservationSet,
        meta: PublicMeta,
        config: StrategyConfig,
        rng_seed: int,
        workdir: Path,
    ) -> BaseHypothesis:
        assert isinstance(config, HmcConfig)
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
        cfg: HmcConfig,
        seed: int,
        sun: ObservationSet | None = None,
    ) -> PosteriorHypothesis:
        setup = setup_body(ob, meta, sun)
        model = setup.model(cfg.family, cfg)
        mp = fit_map(model, setup.starts(model.family, seed))
        lo, hi = _bounds(model, mp.x)
        eps = 1e-6 * (hi - lo)
        x0 = np.clip(mp.x, lo + eps, hi - eps)
        names = [d.name for d in model.dims]
        lo_j, hi_j = jnp.asarray(lo), jnp.asarray(hi)

        def numpyro_model() -> None:
            x = jnp.stack(
                [numpyro.sample(n, dist.Uniform(lo_j[k], hi_j[k])) for k, n in enumerate(names)]
            )
            numpyro.factor("loglike", model.loglike_jax(x))

        kernel = NUTS(
            numpyro_model,
            init_strategy=init_to_value(
                values={n: float(v) for n, v in zip(names, x0, strict=True)}
            ),
            dense_mass=cfg.dense_mass,
            target_accept_prob=cfg.target_accept,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=cfg.n_warmup,
            num_samples=cfg.n_samples,
            num_chains=cfg.n_chains,
            chain_method="sequential",
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(seed % (2**31)))
        draws = mcmc.get_samples()
        samples = np.stack([np.asarray(draws[n]) for n in names], axis=1)
        extra = mcmc.get_extra_fields()
        div = int(np.sum(np.asarray(extra["diverging"]))) if "diverging" in extra else -1
        frames = None
        if cfg.record_trace:
            stride = max(len(samples) // 200, 1)
            frames = [
                TraceFrame(
                    label=f"draw {k}",
                    iteration=k,
                    state={
                        "x": samples[k].tolist(),
                        "samples": samples[max(0, k - cfg.trace_samples + 1) : k + 1].tolist(),
                    },
                )
                for k in range(0, len(samples), stride)
            ]
            frames = subsample_frames(frames, cfg.trace_max_frames)
        info = {
            "family": cfg.family,
            "n_samples": len(samples),
            "divergences": div,
            "notes": f"NUTS on {cfg.family}: {cfg.n_warmup} warmup + {cfg.n_samples} draws × "
            f"{cfg.n_chains} chain(s); {div} divergences.",
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
