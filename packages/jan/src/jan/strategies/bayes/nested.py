"""`bayes.nested`: dynesty nested sampling over several families → evidences.

For each candidate family the MAP is found (multi-start least squares) and nested sampling
runs inside a box of ±`box_sigmas` Laplace standard deviations around it. Because every
family's prior is uniform in its physical coordinates (SPEC §6.3), the evidence is

    ln Z = ln Z_box + ln V_box + ln π,     π = 1 / Π (prior widths),

exact whenever the posterior mass lies inside the box (checked: if posterior samples crowd
a box edge that is not a prior edge, the box is doubled and the run repeated). Families
whose best fit trails the leader by more than `laplace_margin` nats get a Laplace
evidence instead: they are excluded either way, and nested sampling there is costly.
Limitation: modes far from the MAP (e.g. period aliases) are not explored.

Reports log-evidence and posterior model probabilities (equal model priors); predictions,
elements and samples come from the most probable family.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, ClassVar

import dynesty
import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from jan.base import BaseHypothesis, PerBodyHypothesis, StrategyConfig, register
from jan.inference import MapResult, SkyModel, fit_map
from jan.models.families import FAMILIES
from jan.models.posterior import PosteriorHypothesis
from jan.setup import BayesConfig, box_around, setup_body
from jt_core.schema import ModelScore, ObservationSet, PublicMeta

FloatArray = NDArray[np.float64]

DEFAULT_FAMILIES = ["circle", "equant-circle", "conic-elliptic", "conic-hyperbolic", "epicycles"]


class NestedConfig(BayesConfig):
    families: list[str] = Field(default_factory=lambda: list(DEFAULT_FAMILIES))
    nlive: int = 300
    box_sigmas: float = 8.0
    dlogz: float = 0.1
    maxcall: int = 400_000
    sample: str = Field("rwalk", description="dynesty sampling method.")
    laplace_margin: float = Field(
        200.0,
        description="Families whose MAP log-likelihood trails the best by more than "
        "this use a Laplace evidence instead of nested sampling.",
    )


def laplace_log_evidence(model: SkyModel, mp: MapResult) -> float:
    """ln Z ≈ ln L(x̂) + ln π + (d/2) ln 2π + ½ ln det Σ (uniform prior, Gaussian posterior)."""
    sign, logdet = np.linalg.slogdet(mp.cov)
    if sign <= 0:
        logdet = float(np.sum(np.log(np.clip(np.diag(mp.cov), 1e-300, None))))
    d = len(mp.x)
    return float(
        mp.loglike + model.log_prior_density() + 0.5 * d * np.log(2 * np.pi) + 0.5 * logdet
    )


def run_box(model: SkyModel, x: FloatArray, cov: FloatArray, cfg: NestedConfig, seed: int) -> dict:
    """Nested sampling inside a Laplace box; returns lnZ, error, samples and diagnostics."""
    n_sigma = cfg.box_sigmas
    lo, hi = model.lo_hi()
    for attempt in range(3):
        blo, bhi = box_around(x, cov, model, n_sigma)
        width = bhi - blo
        log_v = float(np.sum(np.log(width)))

        def prior_transform(
            u: FloatArray, blo: FloatArray = blo, width: FloatArray = width
        ) -> FloatArray:
            return blo + u * width

        def loglike(v: FloatArray) -> float:
            ll = model.loglike(model.wrap(v))
            return ll if np.isfinite(ll) else -1e300

        # dynesty's sampler factory is not typed for these keyword arguments.
        options: dict[str, Any] = {
            "nlive": cfg.nlive,
            "bound": "multi",
            "sample": cfg.sample,
            "rstate": np.random.default_rng(seed + attempt),
        }
        sampler: Any = dynesty.NestedSampler(loglike, prior_transform, len(x), **options)
        sampler.run_nested(dlogz=cfg.dlogz, maxcall=cfg.maxcall, print_progress=False)
        res: Any = sampler.results
        samples = res.samples_equal()
        # Edge check: mass near a box edge that is not also a prior edge → enlarge.
        per = model.periodic()
        frac = (samples - blo) / width
        inner_lo = blo > lo + 1e-12 * (hi - lo)
        inner_hi = bhi < hi - 1e-12 * (hi - lo)
        crowd = ((np.mean(frac < 0.02, 0) > 0.01) & (inner_lo | per)) | (
            (np.mean(frac > 0.98, 0) > 0.01) & (inner_hi | per)
        )
        if not crowd.any() or attempt == 2:
            return {
                "lnZ": float(res.logz[-1] + log_v + model.log_prior_density()),
                "lnZ_err": float(res.logzerr[-1]),
                "samples": samples,
                "ncall": int(np.sum(res.ncall)),
                "box_sigmas": n_sigma,
                "edge_warning": bool(crowd.any()),
            }
        n_sigma *= 2
    raise AssertionError("unreachable")


@register("bayes.nested")
class BayesNested:
    """Evidence for several model families (circle, equant, ellipse, hyperbola, epicycles)."""

    name: ClassVar[str]
    supported_tiers: ClassVar[set[str]] = {"T1", "T2", "T3"}
    Config: ClassVar[type[StrategyConfig]] = NestedConfig

    def fit(
        self,
        obs: ObservationSet,
        meta: PublicMeta,
        config: StrategyConfig,
        rng_seed: int,
        workdir: Path,
    ) -> BaseHypothesis:
        assert isinstance(config, NestedConfig)
        unknown = set(config.families) - set(FAMILIES)
        if unknown:
            raise ValueError(f"unknown families {sorted(unknown)}")
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
        cfg: NestedConfig,
        seed: int,
        sun: ObservationSet | None = None,
    ) -> PosteriorHypothesis:
        setup = setup_body(ob, meta, sun)
        results: dict[str, dict] = {}
        maps: dict[str, tuple[SkyModel, MapResult, float]] = {}
        for name in cfg.families:
            t_start = time.perf_counter()
            model = setup.model(name, cfg)
            starts = setup.starts(model.family, seed)
            if not starts:
                results[name] = {"lnZ": -np.inf, "error": "no starting point"}
                continue
            try:
                maps[name] = (model, fit_map(model, starts), time.perf_counter() - t_start)
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                results[name] = {"lnZ": -np.inf, "error": str(exc)}
        best_ll = max((mp.loglike for _, mp, _ in maps.values()), default=-np.inf)
        for name, (model, mp, t_map) in maps.items():
            t_start = time.perf_counter()
            if mp.loglike < best_ll - cfg.laplace_margin:
                # Hopeless family: a Laplace evidence is accurate enough to rule it out.
                r = {
                    "lnZ": laplace_log_evidence(model, mp),
                    "lnZ_err": None,
                    "method": "laplace",
                    "samples": mp.x[None, :],
                }
            else:
                try:
                    r = run_box(model, mp.x, mp.cov, cfg, seed)
                    r["method"] = "nested"
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                    results[name] = {"lnZ": -np.inf, "error": str(exc)}
                    continue
            r.update(
                {
                    "x_map": mp.x,
                    "loglike_map": mp.loglike,
                    "model": model,
                    "seconds": t_map + time.perf_counter() - t_start,
                }
            )
            results[name] = r
        lnz = np.array([results[n]["lnZ"] for n in cfg.families])
        finite = np.isfinite(lnz)
        if not finite.any():
            raise RuntimeError("no family could be fitted")
        prob = np.zeros_like(lnz)
        prob[finite] = np.exp(lnz[finite] - lnz[finite].max())
        prob /= prob.sum()
        scores = [
            ModelScore(
                model_name=n,
                log_evidence=float(z) if np.isfinite(z) else None,
                posterior_prob=float(p),
            )
            for n, z, p in zip(cfg.families, lnz, prob, strict=True)
        ]
        best = cfg.families[int(np.argmax(np.where(finite, lnz, -np.inf)))]
        rb = results[best]
        model: SkyModel = rb["model"]
        summary = {
            n: {k: v for k, v in r.items() if k not in ("samples", "model", "x_map")}
            for n, r in results.items()
        }
        ranking = ", ".join(
            f"{s.model_name} p={s.posterior_prob:.3g}"
            for s in sorted(scores, key=lambda s: -(s.posterior_prob or 0))
        )
        info = {
            "families": summary,
            "best_family": best,
            "notes": f"nested sampling per family (Laplace box); posterior: {ranking}.",
        }
        return PosteriorHypothesis(
            model.family,
            rb["x_map"],
            rb["samples"],
            bid,
            setup.observer,
            model.sigma_free,
            sigma_known=None if model.sigma_free else setup.data.median_sigma(),
            observer_family=model.observer,
            n_predictive=cfg.n_predictive_samples,
            seed=seed,
            info=info,
            scores=scores,
        )
