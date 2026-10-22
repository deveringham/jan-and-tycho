"""`ptolemy.fourier`: truncated Fourier series of the sky track ("epicycles").

Like Ptolemy's deferent + epicycles, the *unnormalized* geocentric vector d(t) is a sum
of uniformly rotating vectors: harmonics k · f_j (k = 1..n_harmonics) of K base
frequencies — for a planet, d = r_planet − r_earth is exactly a sum of two periodic
vectors, so no cross terms are needed. Only the direction u = d/|d| is observed, and the
condition u × d = 0 is linear in the coefficients, so they are the smallest-eigenvalue
solution of a homogeneous least-squares problem (overall scale is unobservable, which is
exactly the range ambiguity). A few reweighting passes turn |u × d| into angular error.
Base frequencies come from Lomb–Scargle periodograms of the u components (CLEAN-like,
with a small beam search over the top peaks scored by the fit), optionally polished by
variable projection.

This is the curve-fitting baseline: it interpolates very well but has no physics, so
errors in the estimated frequencies make the phase drift linearly after the training
window, and high orders overfit noise (SPEC §6.5). No elements, positions or law.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from scipy.optimize import least_squares
from scipy.signal import lombscargle

from jan.base import BaseHypothesis, PerBodyHypothesis, StrategyConfig, register
from jt_core.schema import HypothesisKind, ObservationSet, PublicMeta, SkyPrediction

FloatArray = NDArray[np.float64]


class FourierConfig(StrategyConfig):
    n_frequencies: int = Field(2, ge=1, le=4, description="Base frequencies K.")
    n_harmonics: int = Field(4, ge=1, description="Harmonics per base frequency.")
    ridge: float = Field(1e-8, ge=0, description="Relative ridge penalty on coefficients.")
    oversampling: int = Field(50, ge=1, description="Periodogram frequency-grid oversampling.")
    n_peaks: int = Field(4, ge=1, description="Periodogram peaks tried per frequency.")
    beam_width: int = Field(3, ge=1, description="Frequency sets kept per search step.")
    refine_frequencies: bool = Field(
        True, description="Polish base frequencies by variable-projection least squares."
    )
    min_period_days: float | None = Field(None, description="Default: 2 × median cadence.")
    max_period_days: float | None = Field(
        None, description="Default: half the training span (longer periods are unidentifiable)."
    )


def harmonic_frequencies(base: FloatArray | list[float], order: int) -> FloatArray:
    """k · f_j for every base frequency f_j and k = 1..order (duplicates removed)."""
    f = np.outer(np.arange(1, order + 1), np.asarray(base, dtype=np.float64)).T.ravel()
    keep: list[float] = []
    for x in f:
        if all(abs(x - y) > 1e-12 * max(x, y) for y in keep):
            keep.append(float(x))
    return np.array(keep)


def design(t: FloatArray, freqs: FloatArray) -> FloatArray:
    """[1, cos 2πf₁t, sin 2πf₁t, ...] for each frequency."""
    ph = 2 * np.pi * np.outer(t, freqs)
    return np.hstack([np.ones((t.size, 1)), np.cos(ph), np.sin(ph)])


def ridge_fit(x: FloatArray, y: FloatArray, ridge: float) -> FloatArray:
    scale = np.linalg.norm(x, axis=0)
    scale[scale == 0] = 1.0
    xs = x / scale
    a = xs.T @ xs + ridge * np.trace(xs.T @ xs) / xs.shape[1] * np.eye(xs.shape[1])
    coef = np.linalg.solve(a, xs.T @ y)
    return coef / scale[:, None]


def _cross_matrices(u: FloatArray) -> FloatArray:
    """[u]× for each row: (N, 3, 3)."""
    z = np.zeros(len(u))
    return np.stack(
        [
            np.stack([z, -u[:, 2], u[:, 1]], 1),
            np.stack([u[:, 2], z, -u[:, 0]], 1),
            np.stack([-u[:, 1], u[:, 0], z], 1),
        ],
        1,
    )


def epicycle_fit(
    x: FloatArray, u: FloatArray, ridge: float, n_reweight: int = 3
) -> tuple[FloatArray, FloatArray]:
    """Coefficients C (M, 3) with d = X C ∥ u, and the weighted residual vector.

    Minimizes Σ w_i² |u_i × d_i|² subject to Σ w_i (u_i · d_i) = N, with w_i = 1/|d_i| from
    the previous pass, so the objective approaches Σ sin²(angle error). The linear scale
    constraint rules out the spurious d ≈ 0 solutions a unit-norm constraint admits.
    """
    n, m = x.shape
    scale = np.linalg.norm(x, axis=0)
    scale[scale == 0] = 1.0
    xs = x / scale
    a = np.einsum("im,iab->iamb", xs, _cross_matrices(u)).reshape(3 * n, 3 * m)
    proj = np.einsum("im,ib->imb", xs, u).reshape(n, 3 * m)
    w = np.ones(n)
    c = np.zeros(3 * m)
    for _ in range(n_reweight + 1):
        aw = a * np.repeat(w, 3)[:, None]
        g = aw.T @ aw
        g += (ridge * np.trace(g) / g.shape[0] + 1e-300) * np.eye(g.shape[0])
        b = proj.T @ w
        gb = np.linalg.solve(g, b)
        c = gb * (n / (b @ gb))
        d = xs @ c.reshape(m, 3)
        w = 1.0 / np.maximum(np.linalg.norm(d, axis=1), 1e-12)
    resid = (a @ c) * np.repeat(w, 3)
    return c.reshape(m, 3) / scale[:, None], resid


class FourierHypothesis(BaseHypothesis):
    kind: HypothesisKind = "nonparametric"

    def __init__(
        self,
        body_id: str,
        t_ref: float,
        freqs: FloatArray,
        coef: FloatArray,
        base: FloatArray,
        info: dict[str, Any],
    ) -> None:
        self.body_id = body_id
        self.t_ref = t_ref
        self.freqs = freqs
        self.coef = coef
        self.base = base
        self.info = info

    def predict_sky(self, t: FloatArray, body_id: str) -> SkyPrediction:
        if body_id != self.body_id:
            raise KeyError(body_id)
        u = design(np.asarray(t) - self.t_ref, self.freqs) @ self.coef
        return SkyPrediction(mean=u / np.linalg.norm(u, axis=1, keepdims=True))

    def notes(self) -> str:
        periods = ", ".join(f"{1 / f:.1f} d" for f in self.base)
        return (
            f"{len(self.freqs)} epicycle frequencies (harmonics of base periods [{periods}]); "
            f"training RMS {self.info['train_rms_arcsec']:.1f}″."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "body_id": self.body_id,
            "t_ref": self.t_ref,
            "base_periods_days": (1 / self.base).tolist(),
            "frequencies": self.freqs.tolist(),
            "coef": self.coef.tolist(),
            "info": self.info,
            "notes": self.notes(),
        }


@register("ptolemy.fourier")
class PtolemyFourier:
    """Epicycle-like Fourier fit of the sky track with Lomb–Scargle base periods."""

    name: ClassVar[str]
    supported_tiers: ClassVar[set[str]] = {"T1", "T2", "T3"}
    Config: ClassVar[type[StrategyConfig]] = FourierConfig

    def fit(
        self,
        obs: ObservationSet,
        meta: PublicMeta,
        config: StrategyConfig,
        rng_seed: int,
        workdir: Path,
    ) -> BaseHypothesis:
        assert isinstance(config, FourierConfig)
        if not obs.is_sky:
            raise ValueError("ptolemy.fourier needs sky observations")
        return PerBodyHypothesis(
            {b: self._fit_body(obs.for_body(b), b, config) for b in obs.body_ids}
        )

    def _fit_body(self, ob: ObservationSet, bid: str, cfg: FourierConfig) -> FourierHypothesis:
        t = ob.t
        u = ob.unit_vectors()
        t_ref = float(np.mean(t))
        tc = t - t_ref
        span = float(t.max() - t.min())
        dt = float(np.median(np.diff(np.sort(t))))
        f_min = 1.0 / (cfg.max_period_days or span / 2.0)
        f_max = 1.0 / (cfg.min_period_days or 2.0 * dt)
        df = 1.0 / (span * cfg.oversampling)
        grid = np.arange(f_min, f_max, df)

        base = self._search(tc, u, grid, span, cfg)

        if cfg.refine_frequencies:
            base, _ = self._refine(tc, u, base, cfg)
        freqs = self._freqs(base, cfg.n_harmonics)
        x = design(tc, freqs)
        coef, _ = epicycle_fit(x, u, cfg.ridge)
        pred = x @ coef
        pred /= np.linalg.norm(pred, axis=1, keepdims=True)
        sep = np.arctan2(np.linalg.norm(np.cross(pred, u), axis=1), np.sum(pred * u, axis=1))
        info = {
            "n_terms": int(x.shape[1]),
            "n_coefficients": int(coef.size),
            "n_obs": len(t),
            "train_rms_arcsec": float(np.sqrt(np.mean(sep**2)) * 206264.80624709636),
        }
        return FourierHypothesis(bid, t_ref, freqs, coef, np.array(base), info)

    def _peaks(
        self,
        tc: FloatArray,
        resid: FloatArray,
        grid: FloatArray,
        span: float,
        exclude: list[float],
        cfg: FourierConfig,
    ) -> list[float]:
        """Top `cfg.n_peaks` distinct Lomb–Scargle peaks of the x/y residual components."""
        power = np.sum(
            [
                lombscargle(tc, resid[:, c] - resid[:, c].mean(), 2 * np.pi * grid, normalize=True)
                for c in (0, 1)
            ],
            axis=0,
        )
        if exclude:  # harmonics (already explained) and subharmonics (redundant) of chosen ones
            subs = [f / k for f in exclude for k in range(2, cfg.n_harmonics + 1)]
            for f in [*harmonic_frequencies(exclude, cfg.n_harmonics), *subs]:
                power[np.abs(grid - f) < 0.75 / span] = 0.0
        peaks: list[float] = []
        for j in np.argsort(power)[::-1]:
            if power[j] <= 0:
                break
            if all(abs(grid[j] - p) > 1.0 / span for p in peaks):
                peaks.append(float(grid[j]))
            if len(peaks) >= cfg.n_peaks:
                break
        return peaks

    def _search(
        self, tc: FloatArray, u: FloatArray, grid: FloatArray, span: float, cfg: FourierConfig
    ) -> list[float]:
        self._f_min = float(grid[0])
        """Beam search over periodogram peaks, scored by the epicycle-fit residual."""

        beams: list[tuple[float, list[float]]] = [(np.inf, [])]
        for _ in range(cfg.n_frequencies):
            cands: list[tuple[float, list[float]]] = []
            for _, base in beams:
                if base:
                    x = design(tc, self._freqs(base, cfg.n_harmonics))
                    resid = u - x @ ridge_fit(x, u, cfg.ridge)
                else:
                    resid = u
                for f in self._peaks(tc, resid, grid, span, base, cfg):
                    nb, c = self._refine(tc, u, [*base, f], cfg)
                    cands.append((c, nb))
            beams = sorted(cands, key=lambda c: c[0])[: cfg.beam_width] or beams
        return beams[0][1]

    def _refine(
        self,
        tc: FloatArray,
        u: FloatArray,
        base: list[float],
        cfg: FourierConfig,
        polish: bool = True,
    ) -> tuple[list[float], float]:
        """Improve base frequencies within ±1 periodogram resolution element: a coarse grid,
        then (if `polish`) variable-projection least squares. Returns (base, cost)."""
        f0 = np.array(base)
        bound = 1.0 / (f0 * np.ptp(tc))
        lower = np.maximum(-bound, getattr(self, "_f_min", 0.0) / f0 - 1.0)

        def resid(rel: FloatArray) -> FloatArray:
            x = design(tc, self._freqs(list(f0 * (1 + rel)), cfg.n_harmonics))
            return epicycle_fit(x, u, cfg.ridge, n_reweight=1)[1]

        def cost(rel: FloatArray) -> float:
            r = resid(rel)
            return float(r @ r) if np.all(np.isfinite(r)) else np.inf

        offsets = np.linspace(-1.0, 1.0, 9 if len(f0) <= 2 else 5)
        axes = [np.clip(offsets * b, lo, b) for b, lo in zip(bound, lower, strict=True)]
        grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(f0), -1).T
        costs = np.array([cost(g) for g in grid])
        best, best_cost = grid[int(np.argmin(costs))], float(costs.min())
        if polish:
            res = least_squares(
                resid, best, bounds=(lower - 1e-12, bound), x_scale=bound / 8, max_nfev=200
            )
            if np.isfinite(res.cost) and 2 * res.cost <= best_cost:
                best, best_cost = res.x, 2 * float(res.cost)
        return list(f0 * (1 + best)), best_cost

    @staticmethod
    def _freqs(base: list[float], order: int) -> FloatArray:
        return harmonic_frequencies(base, order)
