"""`classical.lsq`: batch nonlinear least squares (differential correction).

Fits a `ConicOrbit` (or `CircularOrbit`) to all training observations of each body with
scipy's trust-region least squares and an exact JAX Jacobian. Initialization is the union
of Gauss IOD candidates (T1) or finite-difference states (T0) and a circular multi-start
grid; the best few starts are refined and the lowest cost wins.

Assumptions: two-body heliocentric motion with known GM; observer known (T1) or
positions observed directly (T0). Failure modes: non-Keplerian truths (biased fits with
structured residuals), heavy-tailed noise with `loss: linear`, and short arcs where the
period–eccentricity degeneracy makes the covariance (linearized) overconfident.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from scipy.optimize import least_squares

from jan.base import (
    BaseHypothesis,
    PerBodyHypothesis,
    StrategyConfig,
    register,
    subsample_frames,
)
from jan.fitting import (
    ConicProblem,
    circular_grid_starts,
    finite_difference_state,
    period_grid_starts,
    sky_periods,
)
from jan.inference import SkyModel, fit_map
from jan.models.conic import OrbitFamily, make_family
from jan.models.conic_hypothesis import ConicHypothesis
from jan.models.observer import observer_function
from jan.models.posterior import PosteriorHypothesis
from jan.setup import observer_known, setup_body
from jan.strategies.classical.gauss_iod import gauss_candidates
from jt_core.conics import ConicElements, state_to_elements
from jt_core.constants import ARCSEC
from jt_core.schema import ObservationSet, PublicMeta, TraceFrame

FloatArray = NDArray[np.float64]


_CONIC = {"ellipse": "conic-elliptic", "circle": "circle", "conic": "conic-hyperbolic"}


class LsqConfig(StrategyConfig):
    family: Literal[
        "ellipse",
        "circle",
        "conic",
        "central-power-law",
        "central-hooke",
        "central-yukawa",
        "central-newton-plus-inverse-cube",
        "perturbed-conic",
    ] = Field(
        "ellipse",
        description="ellipse (non-singular, e < 1), circle, conic (any e), or a non-Keplerian "
        "family (numerically integrated; fitted by MAP + Laplace).",
    )
    loss: Literal["linear", "huber", "soft_l1", "cauchy"] = "linear"
    f_scale: float = Field(2.0, description="Robust-loss scale in units of the assumed σ.")
    assumed_sigma_arcsec: float = Field(
        60.0, description="Residual scale when Tycho does not publish σ (sky tiers)."
    )
    assumed_sigma_au: float = Field(1e-3, description="Residual scale for T0 positions.")
    n_triplets: int = 12
    spacings_days: tuple[float, ...] = (15.0, 30.0, 60.0)
    n_refine: int = Field(5, description="Number of best starts refined to convergence.")
    max_nfev: int = 300
    n_predictive_samples: int = Field(64, description="Posterior-approximation draws.")


@register("classical.lsq")
class ClassicalLsq:
    """Differential correction of a conic orbit with a Jacobian-based covariance."""

    name: ClassVar[str]
    supported_tiers: ClassVar[set[str]] = {"T0", "T1", "T2", "T3"}
    Config: ClassVar[type[StrategyConfig]] = LsqConfig

    def fit(
        self,
        obs: ObservationSet,
        meta: PublicMeta,
        config: StrategyConfig,
        rng_seed: int,
        workdir: Path,
    ) -> BaseHypothesis:
        assert isinstance(config, LsqConfig)
        if meta.tier not in self.supported_tiers:
            raise ValueError(f"classical.lsq does not support tier {meta.tier}")
        if obs.is_sky and (not observer_known(meta) or config.family not in _CONIC):
            parts: dict[str, BaseHypothesis] = {
                bid: self._fit_body_free(obs.for_body(bid), bid, meta, config, rng_seed, obs.sun())
                for bid in obs.body_ids
            }
        else:
            parts = {
                bid: self._fit_body(obs.for_body(bid), bid, meta, config, rng_seed)
                for bid in obs.body_ids
            }
        return PerBodyHypothesis(parts)

    def _fit_body_free(
        self,
        ob: ObservationSet,
        bid: str,
        meta: PublicMeta,
        cfg: LsqConfig,
        seed: int,
        sun: ObservationSet | None,
    ) -> BaseHypothesis:
        """T2/T3: fit the body and the observer's orbit jointly (with the Sun stream when
        published); uncertainty from the Laplace (Jacobian) covariance, drawn as samples."""
        t_start = time.perf_counter()
        setup = setup_body(ob, meta, sun)
        family = _CONIC.get(cfg.family, cfg.family)
        likelihood = "gaussian" if cfg.loss == "linear" else "student_t"
        model = SkyModel(setup.family(family), setup.data, likelihood, 1.0, setup.observer_family)
        mp = fit_map(model, setup.starts(model.family, seed))
        g = np.random.default_rng(seed)
        w, v = np.linalg.eigh(mp.cov)
        draws = (
            mp.x
            + g.standard_normal((max(cfg.n_predictive_samples, 2), len(mp.x)))
            @ (v * np.sqrt(np.clip(w, 0, None))).T
        )
        rms = mp.rms_rad
        rb, _ = model.residuals(mp.x)
        info = {
            "rms_per_axis": rms,
            "sigma_rad": 1.4826 * float(np.median(np.abs(rb))) if model.sigma_free else None,
            "fit_seconds": time.perf_counter() - t_start,
            "notes": (
                f"{family} fit"
                + (
                    ""
                    if model.observer is None
                    else " jointly with the observer's orbit "
                    f"({'with' if sun is not None else 'without'} the Sun stream)"
                )
                + f", loss={cfg.loss}; training RMS {rms / ARCSEC:.1f}″ per axis."
            ),
        }
        return PosteriorHypothesis(
            model.family,
            mp.x,
            draws,
            bid,
            setup.observer,
            model.sigma_free,
            sigma_known=None if model.sigma_free else setup.data.median_sigma(),
            n_predictive=cfg.n_predictive_samples,
            seed=seed,
            info=info,
            observer_family=model.observer,
        )

    def _fit_body(
        self, ob: ObservationSet, bid: str, meta: PublicMeta, cfg: LsqConfig, seed: int
    ) -> ConicHypothesis:
        t_start = time.perf_counter()
        gm = meta.gm_sun
        t0 = float(np.median(ob.t))
        fam: OrbitFamily = make_family(cfg.family, t0, gm)
        sky = ob.is_sky
        observer = observer_function(meta) if sky else None
        n = len(ob)
        if sky:
            assert observer is not None
            r_obs = observer(ob.t)
            sig = ob.sigmas()
            sigma_known = sig is not None
            weights = (
                1.0 / sig
                if sig is not None
                else np.full((n, 2), 1.0 / (cfg.assumed_sigma_arcsec * ARCSEC))
            )
            prob = ConicProblem(
                fam, ob.t, sky=True, u_obs=ob.unit_vectors(), r_obs=r_obs, weights=weights
            )
        else:
            r_obs = None
            sigma_known = False
            weights = np.full((n, 3), 1.0 / cfg.assumed_sigma_au)
            prob = ConicProblem(fam, ob.t, sky=False, pos_obs=ob.positions(), weights=weights)

        # ---- starts
        rng = np.random.default_rng(seed)
        starts: list[ConicElements] = []
        if sky:
            assert r_obs is not None
            starts += gauss_candidates(ob, r_obs, gm, cfg.n_triplets, cfg.spacings_days, rng)
        else:
            for k in np.linspace(1, n - 2, 7).astype(int):
                r, v, tk = finite_difference_state(ob, int(k))
                try:
                    starts.append(state_to_elements(r, v, tk, gm))
                except ValueError:
                    pass
        starts += circular_grid_starts(prob, gm, t0)
        if sky:
            starts += period_grid_starts(prob, gm, t0, sky_periods(ob))
        starts = [s for s in starts if fam.admits(s)]
        thetas = np.array([fam.theta_from_elements(s) for s in starts])
        costs = prob.costs(thetas)
        order = np.argsort(costs)[: cfg.n_refine]

        # ---- refine
        best = None
        frames: list[TraceFrame] = []
        for rank, j in enumerate(order):
            local: list[TraceFrame] = []

            def jac(th: FloatArray, _local: list[TraceFrame] = local) -> FloatArray:
                if cfg.record_trace:
                    _local.append(
                        TraceFrame(
                            label=f"iteration {len(_local)}",
                            iteration=len(_local),
                            state={"theta": th.tolist()},
                        )
                    )
                return prob.jacobian(th)

            try:
                res = least_squares(
                    prob.residuals,
                    thetas[j],
                    jac=jac,  # pyright: ignore[reportArgumentType]
                    method="trf",
                    loss=cfg.loss,
                    f_scale=cfg.f_scale,
                    max_nfev=cfg.max_nfev,
                    x_scale="jac",
                )
            except (ValueError, FloatingPointError):
                continue
            if not np.isfinite(res.cost):
                continue
            if best is None or res.cost < best.cost:
                best = res
                frames = [
                    TraceFrame(f.label, f.iteration, {**f.state, "start_rank": rank}) for f in local
                ]
        if best is None:
            raise RuntimeError(f"least squares failed for body {bid!r}")

        # ---- covariance from the Jacobian
        theta = best.x
        r = prob.residuals(theta)
        jm = prob.jacobian(theta)
        m = r.size
        p = theta.size
        chi2 = float(r @ r)
        red_chi2 = chi2 / max(m - p, 1)
        try:
            cov = np.linalg.pinv(jm.T @ jm, rcond=1e-13)
            if not sigma_known:
                cov = cov * red_chi2
        except np.linalg.LinAlgError:
            cov = None
        k = prob.n_residuals_per_obs
        scale = (
            (cfg.assumed_sigma_arcsec * ARCSEC)
            if (sky and not sigma_known)
            else (1.0 if sky else cfg.assumed_sigma_au)
        )
        raw = r / (weights.ravel() if sigma_known else 1.0 / scale)  # radians (or AU) per axis
        rms = float(np.sqrt(np.mean(raw**2)))
        robust_sigma = 1.4826 * float(np.median(np.abs(raw)))  # MAD: immune to outliers
        if sky and not sigma_known:
            rms_txt = f"{rms / ARCSEC:.1f}″ per axis"
        elif sky:
            rms_txt = f"{rms / ARCSEC:.1f}″ per axis"
        else:
            rms_txt = f"{rms:.2e} AU per axis"
        bic = chi2 / (red_chi2 if not sigma_known else 1.0) + p * np.log(m)
        info = {
            "cost": float(best.cost),
            "chi2": chi2,
            "reduced_chi2": red_chi2,
            "n_obs": n,
            "n_residuals": m,
            "n_starts": len(starts),
            "nfev": int(best.nfev),
            "status": int(best.status),
            "rms_per_axis": rms,
            "sigma_rad": robust_sigma if sky else None,
            "bic": float(bic),
            "fit_seconds": time.perf_counter() - t_start,
            "notes": (
                f"{cfg.family} fit, loss={cfg.loss}, {len(starts)} starts, best refined in "
                f"{best.nfev} evaluations; training residual RMS {rms_txt}, "
                f"reduced χ² {red_chi2:.2f} (residuals per obs: {k})."
            ),
        }
        if cfg.record_trace:
            frames = subsample_frames(frames, cfg.trace_max_frames)
        return ConicHypothesis(
            fam,
            theta,
            bid,
            observer,
            cov=cov,
            n_samples=cfg.n_predictive_samples,
            seed=seed,
            info=info,
            frames=frames if cfg.record_trace else None,
        )
