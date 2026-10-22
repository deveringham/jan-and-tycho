"""`sbi.npe`: neural posterior estimation (SPEC §6.5, Phase 5).

Simulations come from **Jan's own families** (never from Tycho): parameters from the
family's prior, directions from the family's forward model at the published epochs as seen
from the known observer (T1), plus isotropic tangent-plane noise at σ (published, or a MAP
estimate otherwise). Each data set is summarized by its mean direction in fixed time bins.

* Round 1 is trained on the whole prior: an **amortized** posterior, a single forward
  pass for any data set on this observing schedule.
* Later rounds (truncated sequential NPE) retrain on simulations from the prior truncated to
  the previous round's high-probability box: a **per-scenario** posterior focused on these
  data. The truncated prior is still uniform, so no proposal correction is needed. Rounds
  stop once the box stops shrinking (Mars-like data at 60″: ~5 rounds from the whole prior
  down to 1e-5 in q).

Each round is a trace frame, and its element samples are kept for Rudolf's
amortized-vs-per-scenario figure. Failure modes: the prior is vast compared with the
posterior, so the amortized round resolves elements to percents at best; NPE posteriors
are only as good as the network's fit, so coverage must be checked, not assumed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from jan.base import BaseHypothesis, PerBodyHypothesis, StrategyConfig, register
from jan.inference import fit_map
from jan.models.families import BayesFamily
from jan.models.posterior import PosteriorHypothesis
from jan.setup import BayesConfig, setup_body
from jt_core.schema import ELEMENT_NAMES, ObservationSet, PublicMeta, TraceFrame

FloatArray = NDArray[np.float64]
TWO_PI = 2.0 * np.pi


class NpeConfig(BayesConfig):
    family: str = "conic-elliptic"
    rounds: int = Field(8, ge=1, description="Maximum rounds; 1 = amortized over the prior only.")
    converged_shrink: float = Field(
        0.8,
        description="Stop once a truncated round shrinks no box side below this fraction of "
        "its previous width (the posterior has stopped narrowing).",
    )
    n_sims: int = Field(20_000, description="Simulations per round.")
    n_bins: int = Field(96, description="Time bins of the summary statistic.")
    hidden_features: int = 64
    num_transforms: int = 5
    max_epochs: int = 300
    truncation: float = Field(
        1e-3, description="Tail mass outside the truncated box, per dimension and side."
    )
    widen: float = Field(0.5, description="Fractional widening of the truncated box.")
    n_posterior: int = 4000
    round_samples: int = Field(500, description="Element samples stored per round.")
    threads: int = Field(2, description="PyTorch threads (sweeps run cells in parallel).")


def _bins(t: FloatArray, n_bins: int) -> tuple[NDArray[np.int64], int]:
    """Bin index per observation (empty bins dropped) and the number of bins."""
    edges = np.linspace(t.min(), t.max() + 1e-9, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_bins - 1)
    used, inv = np.unique(idx, return_inverse=True)
    return inv.astype(np.int64), len(used)


def make_simulator(
    fam: BayesFamily, t: FloatArray, r_obs: FloatArray, sigma: float, n_bins: int
) -> tuple[Any, Any]:
    """(simulate(φ batch, key) → summaries, summarize(unit vectors) → summary)."""
    idx, nb = _bins(t, n_bins)
    tj, rj, ij = jnp.asarray(t), jnp.asarray(r_obs), jnp.asarray(idx)

    def summarize(u: jax.Array) -> jax.Array:
        s = jax.ops.segment_sum(u, ij, num_segments=nb)
        return (s / jnp.linalg.norm(s, axis=1, keepdims=True)).reshape(-1)

    def one(phi: jax.Array, key: jax.Array) -> jax.Array:
        u = fam.sky(phi, tj, rj)
        e = jnp.stack([-u[:, 1], u[:, 0], jnp.zeros_like(u[:, 0])], axis=1)
        e = e / jnp.maximum(jnp.linalg.norm(e, axis=1, keepdims=True), 1e-300)
        n = jnp.cross(u, e)
        g = sigma * jax.random.normal(key, (len(t), 2))
        un = u + g[:, :1] * e + g[:, 1:] * n
        return summarize(un / jnp.linalg.norm(un, axis=1, keepdims=True))

    batch = jax.jit(jax.vmap(one))

    def simulate(phis: FloatArray, key: jax.Array) -> FloatArray:
        keys = jax.random.split(key, len(phis))
        return np.asarray(batch(jnp.asarray(phis), keys))

    return simulate, jax.jit(summarize)


def _truncated_box(
    samples: FloatArray,
    lo: FloatArray,
    hi: FloatArray,
    per: NDArray[np.bool_],
    q: float,
    widen: float,
) -> tuple[FloatArray, FloatArray]:
    """Per-dimension box holding the samples' central 1 − 2q mass, widened; periodic
    dimensions are centred on their circular mean and capped at one period."""
    new_lo, new_hi = lo.copy(), hi.copy()
    for k in range(samples.shape[1]):
        x = samples[:, k]
        if per[k]:
            c = float(np.angle(np.mean(np.exp(1j * x))))
            off = np.mod(x - c + np.pi, TWO_PI) - np.pi
            a, b = np.quantile(off, [q, 1 - q])
            w = (b - a) * widen
            a, b = max(a - w, -np.pi), min(b + w, np.pi)
            new_lo[k], new_hi[k] = c + a, c + b
        else:
            a, b = np.quantile(x, [q, 1 - q])
            w = (b - a) * widen
            new_lo[k], new_hi[k] = max(a - w, lo[k]), min(b + w, hi[k])
    return new_lo, new_hi


@register("sbi.npe")
class SbiNpe:
    """Neural posterior estimation on Jan's own family: amortized round + truncated rounds."""

    name: ClassVar[str]
    supported_tiers: ClassVar[set[str]] = {"T1"}
    Config: ClassVar[type[StrategyConfig]] = NpeConfig

    def fit(
        self,
        obs: ObservationSet,
        meta: PublicMeta,
        config: StrategyConfig,
        rng_seed: int,
        workdir: Path,
    ) -> BaseHypothesis:
        assert isinstance(config, NpeConfig)
        return PerBodyHypothesis(
            {b: self._fit_body(obs.for_body(b), b, meta, config, rng_seed) for b in obs.body_ids}
        )

    def _fit_body(
        self, ob: ObservationSet, bid: str, meta: PublicMeta, cfg: NpeConfig, seed: int
    ) -> PosteriorHypothesis:
        import torch
        from sbi.inference import NPE, DirectPosterior
        from sbi.neural_nets import posterior_nn
        from sbi.utils import BoxUniform

        t_start = time.perf_counter()
        torch.set_num_threads(cfg.threads)
        torch.manual_seed(seed)
        setup = setup_body(ob, meta)
        model = setup.model(cfg.family, cfg)
        fam = model.family
        sigma = setup.data.median_sigma()
        sigma_note = "published"
        if sigma is None:
            mp = fit_map(model, setup.starts(fam, seed))
            sigma = float(np.exp(mp.x[-1]))
            sigma_note = "MAP estimate"
        t = ob.t
        simulate, summarize = make_simulator(fam, t, setup.observer(t), sigma, cfg.n_bins)
        x_obs = np.asarray(summarize(jnp.asarray(ob.unit_vectors())))
        lo, hi = fam.lo_hi()
        per = fam.periodic_mask()
        rng = np.random.default_rng(seed)
        key = jax.random.PRNGKey(seed % (2**31))
        box_lo, box_hi = lo.copy(), hi.copy()
        frames: list[TraceFrame] = []
        rounds: list[dict[str, Any]] = []
        samples = np.zeros((0, fam.n_params))
        x_best = np.zeros(fam.n_params)
        for r in range(cfg.rounds):
            theta = box_lo + (box_hi - box_lo) * rng.uniform(size=(cfg.n_sims, fam.n_params))
            key, sub = jax.random.split(key)
            xs = simulate(theta, sub)
            ok = np.all(np.isfinite(xs), axis=1)
            prior = BoxUniform(torch.as_tensor(box_lo), torch.as_tensor(box_hi))
            builder = posterior_nn(
                "maf", hidden_features=cfg.hidden_features, num_transforms=cfg.num_transforms
            )
            inference = NPE(prior=prior, density_estimator=builder, show_progress_bars=False)
            inference.append_simulations(
                torch.as_tensor(theta[ok], dtype=torch.float32),
                torch.as_tensor(xs[ok], dtype=torch.float32),
            )
            estimator = inference.train(max_num_epochs=cfg.max_epochs)
            posterior = inference.build_posterior(estimator)
            assert isinstance(posterior, DirectPosterior)
            xo = torch.as_tensor(x_obs, dtype=torch.float32)
            smp = posterior.sample((cfg.n_posterior,), x=xo, show_progress_bars=False)
            logp = posterior.log_prob(smp, x=xo)
            samples = smp.numpy().astype(np.float64)
            x_best = samples[int(torch.argmax(logp))]
            label = "amortized (prior)" if r == 0 else f"truncated round {r + 1}"
            keep = rng.choice(len(samples), min(cfg.round_samples, len(samples)), replace=False)
            frames.append(
                TraceFrame(label, r, {"x": x_best.tolist(), "samples": samples[keep].tolist()})
            )
            round_h = PosteriorHypothesis(fam, x_best, samples[keep], bid, setup.observer, False)
            el = round_h.element_samples()
            rounds.append(
                {
                    "round": r + 1,
                    "label": label,
                    "n_sims": int(ok.sum()),
                    "box_width": (box_hi - box_lo).tolist(),
                    "samples": None
                    if el is None
                    else {n: el[:, k].tolist() for k, n in enumerate(ELEMENT_NAMES)},
                }
            )
            new_lo, new_hi = _truncated_box(samples, lo, hi, per, cfg.truncation, cfg.widen)
            shrink = (new_hi - new_lo) / np.maximum(box_hi - box_lo, 1e-300)
            box_lo, box_hi = new_lo, new_hi
            if r >= 1 and float(np.min(shrink)) > cfg.converged_shrink:
                break
        info = {
            "family": cfg.family,
            "sigma_arcsec": sigma * 206264.80624709636,
            "sigma_source": sigma_note,
            "rounds": rounds,
            "n_bins": int(_bins(t, cfg.n_bins)[1]),
            "fit_seconds": time.perf_counter() - t_start,
            "notes": f"NPE on {cfg.family}: {len(rounds)} round(s) × {cfg.n_sims} simulations "
            f"from Jan's family (round 1 amortized over the prior, later rounds truncated); "
            f"σ {sigma_note}.",
        }
        return PosteriorHypothesis(
            fam,
            x_best,
            samples,
            bid,
            setup.observer,
            False,
            sigma_known=sigma,
            n_predictive=cfg.n_predictive_samples,
            seed=seed,
            info=info,
            frames=frames if cfg.record_trace else None,
        )
