"""Residuals, Jacobians and initializers for fitting conic families to observations.

Sky residuals are tangent-plane offsets of the model direction in the (east, north)
basis at each *observed* direction (fixed by the data, so cheap and differentiable),
never raw longitude differences (SPEC §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from jan.models.conic import OrbitFamily, sky_directions
from jt_core.angles import tangent_basis
from jt_core.conics import ConicElements, bucket, pad_rows
from jt_core.schema import ObservationSet

FloatArray = NDArray[np.float64]


def _sky_res(
    fam: OrbitFamily,
    theta: jax.Array,
    t: jax.Array,
    r_obs: jax.Array,
    u_obs: jax.Array,
    east: jax.Array,
    north: jax.Array,
    w: jax.Array,
) -> jax.Array:
    u = sky_directions(fam, theta, t, r_obs)
    d = jnp.sum(u * u_obs, axis=1)
    out = jnp.stack([jnp.sum(u * east, 1) / d, jnp.sum(u * north, 1) / d], axis=1)
    return (out * w).ravel()


def _pos_res(
    fam: OrbitFamily, theta: jax.Array, t: jax.Array, pos: jax.Array, w: jax.Array
) -> jax.Array:
    return ((fam.position(theta, t) - pos) * w).ravel()


_sky_res_j = jax.jit(_sky_res)
_sky_jac_j = jax.jit(jax.jacfwd(_sky_res, argnums=1))
_sky_cost_j = jax.jit(
    jax.vmap(lambda f, th, *a: jnp.sum(_sky_res(f, th, *a) ** 2), in_axes=(None, 0) + (None,) * 6),
)
_pos_res_j = jax.jit(_pos_res)
_pos_jac_j = jax.jit(jax.jacfwd(_pos_res, argnums=1))
_pos_cost_j = jax.jit(
    jax.vmap(lambda f, th, *a: jnp.sum(_pos_res(f, th, *a) ** 2), in_axes=(None, 0) + (None,) * 3),
)


@dataclass
class ConicProblem:
    """Weighted residuals r(θ) for one body under a conic family.

    Sky data: 2 residuals per observation (radians / σ). T0 data: 3 per observation (AU / σ).
    Inputs are padded to a power-of-two length with zero weights so the jitted kernels are
    reused across datasets; padding is sliced off before returning.
    """

    family: OrbitFamily
    t: FloatArray
    sky: bool
    u_obs: FloatArray | None = None
    r_obs: FloatArray | None = None
    """Observer positions at `t` (sky) or None (T0)."""
    pos_obs: FloatArray | None = None
    weights: FloatArray | None = None
    """Per-residual 1/σ, shape (N, 2) or (N, 3). None → unit weights."""
    _args: tuple[jax.Array, ...] = field(init=False, repr=False)
    _n_res: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = len(self.t)
        k = 2 if self.sky else 3
        nb = bucket(n)
        self._n_res = n * k
        w = np.ones((n, k)) if self.weights is None else np.asarray(self.weights)
        w = np.concatenate([w, np.zeros((nb - n, k))])
        t = jnp.asarray(pad_rows(self.t, nb))
        if self.sky:
            assert self.u_obs is not None and self.r_obs is not None
            east, north = tangent_basis(self.u_obs)
            self._args = tuple(
                jnp.asarray(pad_rows(a, nb)) for a in (self.r_obs, self.u_obs, east, north)
            )
            self._args = (t, *self._args, jnp.asarray(w))
        else:
            assert self.pos_obs is not None
            self._args = (t, jnp.asarray(pad_rows(self.pos_obs, nb)), jnp.asarray(w))

    @property
    def n_residuals_per_obs(self) -> int:
        return 2 if self.sky else 3

    def residuals(self, theta: FloatArray) -> FloatArray:
        f = _sky_res_j if self.sky else _pos_res_j
        return np.asarray(f(self.family, jnp.asarray(theta), *self._args))[: self._n_res]

    def jacobian(self, theta: FloatArray) -> FloatArray:
        f = _sky_jac_j if self.sky else _pos_jac_j
        return np.asarray(f(self.family, jnp.asarray(theta), *self._args))[: self._n_res]

    def costs(self, thetas: FloatArray) -> FloatArray:
        """Sum of squared residuals for a batch of parameter vectors (M, P)."""
        f = _sky_cost_j if self.sky else _pos_cost_j
        m = len(thetas)
        padded = pad_rows(np.asarray(thetas, dtype=np.float64), bucket(m))
        c = np.asarray(f(self.family, jnp.asarray(padded), *self._args))[:m]
        return np.where(np.isfinite(c), c, np.inf)


def circular_grid_starts(
    problem: ConicProblem,
    gm: float,
    t0: float,
    n_radius: int = 48,
    n_phase: int = 72,
    r_min: float = 0.2,
    r_max: float = 60.0,
    inclinations: tuple[float, ...] = (0.0,),
    n_keep: int = 6,
) -> list[ConicElements]:
    """Multi-start grid of circular orbits (radius × phase), best `n_keep` by cost.

    Robust where Gauss's method degenerates (coplanar geometry) and as a global safety net.
    """
    radii = np.geomspace(r_min, r_max, n_radius)
    phases = np.linspace(0, 2 * np.pi, n_phase, endpoint=False)
    cands: list[ConicElements] = []
    for inc in inclinations:
        for Om in [0.0] if inc == 0.0 else list(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
            for a in radii:
                for ph in phases:
                    n = np.sqrt(gm / a**3)
                    el = ConicElements(a, 0.0, inc, Om, 0.0, t0 - ph / n)
                    cands.append(el)
    fam = problem.family
    full = [fam.theta_from_elements(c) for c in cands]
    costs = problem.costs(np.asarray(full))
    order = np.argsort(costs)
    # Keep distinct radii among the best so starts are diverse.
    out: list[ConicElements] = []
    seen: list[float] = []
    for j in order:
        a = cands[j].q
        if all(abs(np.log(a / s)) > 0.05 for s in seen):
            out.append(cands[j])
            seen.append(a)
        if len(out) >= n_keep:
            break
    return out


def sky_periods(obs: ObservationSet, n: int = 2) -> list[float]:
    """Candidate orbital periods (days) from the sky track's epicycle periodogram.

    For a planet seen from a moving observer the strongest periods are the planet's and the
    observer's; both are returned (the fit decides). Robust to heavy noise, unlike Gauss.
    """
    from jan.base import make_config
    from jan.strategies.ptolemy.fourier import PtolemyFourier

    strat = PtolemyFourier()
    cfg = make_config(strat, {"n_frequencies": n, "n_harmonics": 2})
    h = strat._fit_body(obs, "x", cfg)  # pyright: ignore[reportPrivateUsage]
    return [float(1.0 / f) for f in h.base]


def period_grid_starts(
    problem: ConicProblem,
    gm: float,
    t0: float,
    periods: list[float],
    n_phase: int = 180,
    n_keep: int = 4,
) -> list[ConicElements]:
    """Circular starts at the periodogram periods (a from Kepler's third law) × a fine
    phase grid: the period fixes a precisely enough that the phase never drifts."""
    cands: list[ConicElements] = []
    for period in periods:
        a = (gm * (period / (2 * np.pi)) ** 2) ** (1 / 3)
        n = 2 * np.pi / period
        for ph in np.linspace(0, 2 * np.pi, n_phase, endpoint=False):
            for inc, Om in (
                (1e-3, 0.0),
                (0.05, 0.0),
                (0.05, np.pi / 2),
                (0.05, np.pi),
                (0.05, 1.5 * np.pi),
            ):
                cands.append(ConicElements(a, 0.0, inc, Om, 0.0, t0 - ph / n))
    fam = problem.family
    costs = problem.costs(np.asarray([fam.theta_from_elements(c) for c in cands]))
    return [cands[j] for j in np.argsort(costs)[:n_keep]]


def finite_difference_state(obs: ObservationSet, k: int) -> tuple[FloatArray, FloatArray, float]:
    """T0 initializer: position at observation k and velocity from its neighbours."""
    t = obs.t
    p = obs.positions()
    k = int(np.clip(k, 1, len(t) - 2))
    v = (p[k + 1] - p[k - 1]) / (t[k + 1] - t[k - 1])
    return p[k], v, float(t[k])
