"""`discovery.sindy`: weak-form sparse regression of the force law (SPEC §6.5, Phase 4).

Stage 1 — positions: T0 uses the observed heliocentric positions; T1 reconstructs them
from a flexible central-force fit (power law with free exponent), which biases the library
towards central laws (documented).

Stage 2 — sparse regression of x'' = Θ(x) ξ without differencing noisy data twice: either
fit-based derivatives (a local polynomial per window, the default) or the weak form (for
compactly supported test functions φ_j, integrating by parts twice gives
∫ φ_j'' x dt = ∫ φ_j Θ(x) dt · ξ; each window is detrended linearly first, which is exact
because ∫ φ'' · (linear) = 0). The library is radial, a = Σ_p c_p r^p r̂ for p ∈ {−5, …, 2},
with coefficients shared by the three components (isotropy). Over the narrow range of r an
orbit visits, the powers of r are nearly collinear, which defeats thresholded least squares
(STLSQ); sparsity therefore comes from best-subset selection (≤ `max_terms` terms, the
sparsest within a tolerance of the best fit). The STLSQ path is still recorded as a trace
for the law-discovery animation.

Stage 3 — predictions: the discovered law is integrated from a fitted initial state.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, ClassVar, Literal

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from jan.base import BaseHypothesis, PerBodyHypothesis, StrategyConfig, make_config, register
from jan.law import LawHypothesis, fit_initial_state
from jan.models.observer import observer_function
from jt_core.schema import LawEstimate, ObservationSet, PublicMeta, TraceFrame

FloatArray = NDArray[np.float64]
POWERS = (-5, -4, -3, -2, -1, 0, 1, 2)


class SindyConfig(StrategyConfig):
    powers: tuple[int, ...] = Field(POWERS, description="Library: c_p r^p r̂ terms.")
    threshold: float = Field(0.05, description="STLSQ threshold (recorded path only).")
    max_terms: int = Field(3, description="Largest model considered by best-subset selection.")
    tolerance: float = Field(
        0.5, description="Pick the sparsest model with RSS ≤ (1 + tolerance) × the best RSS."
    )
    derivatives: Literal["local_poly", "weak"] = Field(
        "local_poly", description="Fit-based (local polynomial) or weak-form derivatives."
    )
    window_points: int = Field(11, description="Observations per window.")
    poly_degree: int = 4
    bump_power: int = 4
    max_gap_factor: float = Field(3.0, description="Split segments at gaps > factor × cadence.")


def _bump(t: FloatArray, a: float, b: float, p: int) -> tuple[FloatArray, FloatArray]:
    """φ = ((t−a)(b−t))^p and φ'' on [a, b], zero outside (smooth to order p−1)."""
    u = (t - a) * (b - t)
    inside = (t > a) & (t < b)
    up = (a + b) - 2 * t  # u'
    upp = -2.0
    phi = np.where(inside, u**p, 0.0)
    phi2 = np.where(inside, p * (p - 1) * u ** (p - 2) * up**2 + p * u ** (p - 1) * upp, 0.0)
    return phi, phi2


def _trapz(y: FloatArray, t: FloatArray) -> FloatArray:
    dt = np.diff(t)
    return np.sum(0.5 * (y[1:] + y[:-1]) * dt.reshape((-1,) + (1,) * (y.ndim - 1)), axis=0)


def weak_system(
    t: FloatArray,
    x: FloatArray,
    powers: tuple[int, ...],
    window: int,
    p: int,
    max_gap_factor: float,
) -> tuple[FloatArray, FloatArray]:
    """Rows (A ξ = b) of the weak-form regression, stacked over windows and components."""
    order = np.argsort(t)
    t, x = t[order], x[order]
    dt_med = float(np.median(np.diff(t)))
    cuts = np.where(np.diff(t) > max_gap_factor * dt_med)[0] + 1
    rows_a, rows_b = [], []
    r = np.linalg.norm(x, axis=1)
    feats = np.stack([x * (r ** (k - 1))[:, None] for k in powers], axis=-1)  # (N, 3, K)
    for seg in np.split(np.arange(len(t)), cuts):
        if len(seg) < window:
            continue
        for start in range(0, len(seg) - window + 1, max(window // 3, 1)):
            idx = seg[start : start + window]
            ts = t[idx]
            phi, phi2 = _bump(ts, float(ts[0]), float(ts[-1]), p)
            scale = float(np.max(np.abs(phi))) or 1.0
            # ∫φ''·(linear) = 0 exactly; removing the local linear trend first keeps the
            # quadrature error from being dominated by the large constant part of x.
            lin = np.polynomial.polynomial.polyfit(ts - ts.mean(), x[idx], 1)
            xd = x[idx] - np.polynomial.polynomial.polyval(ts - ts.mean(), lin).T
            lhs = _trapz(phi2[:, None] * xd, ts) / scale  # (3,)
            rhs = _trapz(phi[:, None, None] * feats[idx], ts) / scale  # (3, K)
            rows_b.append(lhs)
            rows_a.append(rhs)
    if not rows_a:
        raise RuntimeError("no window long enough for the weak form")
    return np.concatenate(rows_a, axis=0), np.concatenate(rows_b, axis=0)


def local_poly_system(
    t: FloatArray,
    x: FloatArray,
    powers: tuple[int, ...],
    window: int,
    degree: int,
    max_gap_factor: float,
) -> tuple[FloatArray, FloatArray]:
    """Rows (A ξ = b) from fit-based derivatives: a local polynomial per window of
    consecutive observations gives x and x'' at the window centre."""
    order = np.argsort(t)
    t, x = t[order], x[order]
    dt_med = float(np.median(np.diff(t)))
    cuts = np.where(np.diff(t) > max_gap_factor * dt_med)[0] + 1
    rows_a, rows_b = [], []
    for seg in np.split(np.arange(len(t)), cuts):
        for start in range(0, len(seg) - window + 1):
            idx = seg[start : start + window]
            tc = float(np.mean(t[idx]))
            c = np.polynomial.polynomial.polyfit(t[idx] - tc, x[idx], degree)  # (deg+1, 3)
            xc, acc = c[0], 2.0 * c[2]
            r = float(np.linalg.norm(xc))
            rows_a.append(np.stack([xc * r ** (k - 1) for k in powers], axis=-1))  # (3, K)
            rows_b.append(acc)
    if not rows_a:
        raise RuntimeError("no window long enough for local derivatives")
    return np.concatenate(rows_a, axis=0), np.concatenate(rows_b, axis=0)


def stlsq(
    a: FloatArray,
    b: FloatArray,
    threshold: float,
    max_iter: int = 20,
) -> tuple[FloatArray, list[FloatArray]]:
    """Sequentially thresholded least squares on normalized columns; returns ξ and path."""
    scale = np.linalg.norm(a, axis=0)
    scale[scale == 0] = 1.0
    an = a / scale
    xi = np.linalg.lstsq(an, b, rcond=None)[0]
    path = [xi / scale]
    active = np.ones(a.shape[1], dtype=bool)
    for _ in range(max_iter):
        mag = np.abs(xi) / max(float(np.max(np.abs(xi))), 1e-300)
        new = active & (mag >= threshold)
        if not new.any():
            break
        xi = np.zeros_like(xi)
        xi[new] = np.linalg.lstsq(an[:, new], b, rcond=None)[0]
        path.append(xi / scale)
        if np.array_equal(new, active):
            break
        active = new
    return xi / scale, path


def best_subset(
    a: FloatArray,
    b: FloatArray,
    max_terms: int,
    tolerance: float,
) -> tuple[FloatArray, list[dict[str, Any]]]:
    """Sparsest subset (≤ max_terms) whose residual is within (1 + tolerance) of the best
    achievable with ≤ max_terms terms. Robust where library columns are nearly collinear
    (over a narrow range of r, every power of r looks like every other)."""
    from itertools import combinations

    n_terms = a.shape[1]
    fits: list[dict[str, Any]] = []
    for k in range(1, max_terms + 1):
        for subset in combinations(range(n_terms), k):
            cols = list(subset)
            xi, *_ = np.linalg.lstsq(a[:, cols], b, rcond=None)
            rss = float(np.sum((a[:, cols] @ xi - b) ** 2))
            fits.append({"terms": cols, "coef": xi, "rss": rss, "k": k})
    best_rss = min(f["rss"] for f in fits)
    for k in range(1, max_terms + 1):
        ok = [f for f in fits if f["k"] == k and f["rss"] <= (1 + tolerance) * best_rss]
        if ok:
            pick = min(ok, key=lambda f: f["rss"])
            coef = np.zeros(n_terms)
            coef[pick["terms"]] = pick["coef"]
            return coef, fits
    raise AssertionError("unreachable")


def radial_law(
    coef: FloatArray, powers: tuple[int, ...]
) -> tuple[list[str], dict[str, float], str]:
    terms = [(c, k) for c, k in zip(coef, powers, strict=True) if c != 0]
    consts = {f"c{k if k >= 0 else 'm' + str(-k)}": float(c) for c, k in terms}
    radial = " + ".join(f"c{k if k >= 0 else 'm' + str(-k)}*r**({k})" for _, k in terms) or "0"
    r = "sqrt(x**2 + y**2 + z**2)"
    vec = [f"({radial.replace('r**', f'({r})**')})*{c}/({r})" for c in ("x", "y", "z")]
    return vec, consts, radial


def _accel(coef: Any, powers: tuple[int, ...]) -> Any:
    pw = jnp.asarray(powers, dtype=jnp.float64)
    c = jnp.asarray(coef)

    def accel(x: Any, v: Any, t: Any, args: Any) -> Any:
        r = jnp.linalg.norm(x, axis=-1, keepdims=True)
        f = jnp.sum(c * r**pw, axis=-1, keepdims=True)
        return f * x / r

    return accel


DISCOVERY_POSITION_FAMILIES = ("central-power-law", "central-hooke", "ellipse")


def discovery_positions(
    ob: ObservationSet, meta: PublicMeta, seed: int
) -> tuple[FloatArray, FloatArray, str]:
    """(t, positions, provenance) for the regression stage."""
    if not ob.is_sky:
        return ob.t, ob.positions(), "observed positions (T0)"
    from jan.base import get_strategy

    lsq = get_strategy("classical.lsq")
    bid = str(ob.table["body_id"].iloc[0])
    obs_all = ObservationSet(ob.table)
    u = ob.unit_vectors()
    best: tuple[float, FloatArray, str] | None = None
    # Flexible central families (a power law cannot bend into Hooke's centred ellipse
    # within its exponent range; the ellipse is a fallback that always integrates):
    # keep whichever matches the sky best.
    for fam in DISCOVERY_POSITION_FAMILIES:
        h = lsq.fit(obs_all, meta, make_config(lsq, {"family": fam}), seed, Path("."))
        pred = h.predict_sky(ob.t, bid).mean
        rms = float(np.sqrt(np.mean(np.sum((pred - u) ** 2, axis=1))))
        pp = h.heliocentric_position(ob.t, bid)
        if pp is None or not np.isfinite(rms) or not np.all(np.isfinite(pp.mean)):
            continue  # the integration failed for this family's fit
        if best is None or rms < best[0]:
            best = (rms, pp.mean, fam)
    if best is None:
        raise RuntimeError("no central-family fit gave finite positions")
    return (
        ob.t,
        best[1],
        f"positions from the better of {'/'.join(DISCOVERY_POSITION_FAMILIES)} fits "
        f"({best[2]}; T1, biased to central laws)",
    )


@register("discovery.sindy")
class DiscoverySindy:
    """Weak-form SINDy on a radial library (shared coefficients), STLSQ sparsity."""

    name: ClassVar[str]
    supported_tiers: ClassVar[set[str]] = {"T0", "T1"}
    Config: ClassVar[type[StrategyConfig]] = SindyConfig

    def fit(
        self,
        obs: ObservationSet,
        meta: PublicMeta,
        config: StrategyConfig,
        rng_seed: int,
        workdir: Path,
    ) -> BaseHypothesis:
        assert isinstance(config, SindyConfig)
        return PerBodyHypothesis(
            {b: self._fit_body(obs.for_body(b), b, meta, config, rng_seed) for b in obs.body_ids}
        )

    def _fit_body(
        self, ob: ObservationSet, bid: str, meta: PublicMeta, cfg: SindyConfig, seed: int
    ) -> LawHypothesis:
        t_start = time.perf_counter()
        t, x, provenance = discovery_positions(ob, meta, seed)
        if cfg.derivatives == "weak":
            a, b = weak_system(
                t, x, cfg.powers, cfg.window_points, cfg.bump_power, cfg.max_gap_factor
            )
        else:
            a, b = local_poly_system(
                t, x, cfg.powers, cfg.window_points, cfg.poly_degree, cfg.max_gap_factor
            )
        _, path = stlsq(a, b, cfg.threshold)
        coef, subsets = best_subset(a, b, cfg.max_terms, cfg.tolerance)
        vec, consts, radial = radial_law(coef, cfg.powers)
        observer = observer_function(meta) if ob.is_sky else None
        accel = _accel(coef, cfg.powers)
        t0 = float(np.median(t))
        k = int(np.argmin(np.abs(t - t0)))
        k = min(max(k, 1), len(t) - 2)
        v0 = (x[k + 1] - x[k - 1]) / (t[k + 1] - t[k - 1])
        t0 = float(t[k])
        x0, v0, rms = fit_initial_state(accel, None, ob, t0, x[k], v0, observer)
        frames = None
        if cfg.record_trace:
            frames = [
                TraceFrame(f"STLSQ pass {j}", j, {"coef": c.tolist()}) for j, c in enumerate(path)
            ]
        powers = cfg.powers
        info: dict[str, Any] = {
            "positions": provenance,
            "n_windows": int(len(b) // 3),
            "subsets": [
                {"terms": [cfg.powers[i] for i in f["terms"]], "rss": f["rss"]} for f in subsets
            ],
            "coefficients": dict(zip([str(p) for p in powers], coef.tolist(), strict=True)),
            "sigma_rad": rms if ob.is_sky else None,
            "fit_seconds": time.perf_counter() - t_start,
            "notes": f"weak-form SINDy on {provenance}: f(r) = {radial}",
        }

        def frame_h(fr: TraceFrame) -> LawHypothesis:
            c = np.asarray(fr.state["coef"])
            v_, c_, r_ = radial_law(c, powers)
            return LawHypothesis(
                bid, _accel(c, powers), None, x0, v0, t0, observer, LawEstimate(v_, c_, r_), {}
            )

        law = LawEstimate(
            expression=vec,
            constants=consts,
            radial=radial,
            extra={"path": [p.tolist() for p in path], "powers": list(powers)},
        )
        return LawHypothesis(bid, accel, None, x0, v0, t0, observer, law, info, frames, frame_h)
