"""Conic orbit families: differentiable forward models in JAX (SPEC §6.3).

Three parametrizations of the same two-body physics:

* `ConicOrbit` — raw elements θ = (log q, e, i, Ω, ω, t_p), valid for every e (ellipses,
  parabolas, hyperbolas). Singular at e = 0 (ω, t_p) and i = 0 (Ω).
* `EllipticOrbit` — non-singular elements for e < 1, used by default for fitting:

      θ = (log a, h, k, p, r, λ0)
      (e sin ϖ, e cos ϖ) = (h, k) · tanh(ρ)/ρ,  ρ = √(h² + k²)    → e = tanh ρ < 1
      p = sin(i/2) sin Ω, r = sin(i/2) cos Ω                        (ϖ = Ω + ω)
      λ0 = ϖ + n (t0 − t_p)   mean longitude at the reference epoch t0

  A 2π wrap of ϖ shifts t_p by exactly one period, so the map is consistent everywhere.
* `CircularOrbit` — e ≡ 0: θ = (log a, p, r, λ0).

Exactly e = 0 or i = 0 has zero gradient through the (e, ω)/(i, Ω) chain, so
`theta_from_elements` nudges such starts by 1e-7.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from jt_core.angles import wrap_2pi
from jt_core.conics import ConicElements, conic_position

FloatArray = NDArray[np.float64]
_TINY = 1e-300
_NUDGE = 1e-7


def canonical_elements(x: FloatArray) -> ConicElements:
    """Elements array → `ConicElements` with Ω and ω wrapped into [0, 2π)."""
    q, e, i, Om, om, tp = (float(v) for v in x)
    return ConicElements(q, e, i, float(wrap_2pi(Om)), float(wrap_2pi(om)), tp)


@dataclass(frozen=True)
class OrbitFamily:
    """Base: a map θ → conic elements (q, e, i, Ω, ω, t_p), plus the inverse for starts."""

    t0: float
    gm: float
    name: str = "base"
    n_params: int = 6
    param_names: tuple[str, ...] = ()

    def elements(self, theta: jax.Array) -> jax.Array:
        raise NotImplementedError

    def theta_from_elements(self, el: ConicElements) -> FloatArray:
        raise NotImplementedError

    def admits(self, el: ConicElements) -> bool:
        """Whether `el` can be represented (e.g. e < 1 for ellipses)."""
        return True

    def position(self, theta: jax.Array, t: jax.Array) -> jax.Array:
        return conic_position(self.elements(theta), t, self.gm)


@dataclass(frozen=True)
class ConicOrbit(OrbitFamily):
    name: str = "conic"
    n_params: int = 6
    param_names: tuple[str, ...] = ("log_q", "e", "i", "Omega", "omega", "t_p")

    def elements(self, theta: jax.Array) -> jax.Array:
        return jnp.stack(
            [jnp.exp(theta[0]), jnp.abs(theta[1]), theta[2], theta[3], theta[4], theta[5]]
        )

    def theta_from_elements(self, el: ConicElements) -> FloatArray:
        return np.array(
            [np.log(el.q), max(el.e, _NUDGE), max(el.i, _NUDGE), el.Omega, el.omega, el.t_p]
        )


def _elliptic_elements(
    log_a: jax.Array,
    h: jax.Array,
    k: jax.Array,
    p: jax.Array,
    r: jax.Array,
    lam0: jax.Array,
    t0: float,
    gm: float,
) -> jax.Array:
    a = jnp.exp(log_a)
    rho2 = h * h + k * k
    rho = jnp.sqrt(rho2 + _TINY)
    e = jnp.tanh(rho)
    varpi = jnp.arctan2(h, jnp.where(rho2 == 0, 1.0, k))
    s2 = p * p + r * r
    s = jnp.sqrt(s2 + _TINY)
    inc = 2.0 * jnp.arcsin(jnp.clip(s, 0.0, 1.0))
    Omega = jnp.arctan2(p, jnp.where(s2 == 0, 1.0, r))
    n = jnp.sqrt(gm / a**3)
    t_p = t0 - (lam0 - varpi) / n
    return jnp.stack([a * (1.0 - e), e, inc, Omega, varpi - Omega, t_p])


def _elliptic_theta(el: ConicElements, t0: float, gm: float) -> FloatArray:
    if el.e >= 1.0:
        raise ValueError("elliptic parametrization requires e < 1")
    a = el.q / (1.0 - el.e)
    varpi = el.Omega + el.omega
    e = max(el.e, _NUDGE)
    rho = np.arctanh(e)
    s = np.sin(max(el.i, _NUDGE) / 2)
    n = np.sqrt(gm / a**3)
    return np.array(
        [
            np.log(a),
            rho * np.sin(varpi),
            rho * np.cos(varpi),
            s * np.sin(el.Omega),
            s * np.cos(el.Omega),
            varpi + n * (t0 - el.t_p),
        ]
    )


@dataclass(frozen=True)
class EllipticOrbit(OrbitFamily):
    name: str = "ellipse"
    n_params: int = 6
    param_names: tuple[str, ...] = ("log_a", "h", "k", "p", "r", "lambda0")

    def elements(self, theta: jax.Array) -> jax.Array:
        return _elliptic_elements(*(theta[j] for j in range(6)), t0=self.t0, gm=self.gm)

    def theta_from_elements(self, el: ConicElements) -> FloatArray:
        return _elliptic_theta(el, self.t0, self.gm)

    def admits(self, el: ConicElements) -> bool:
        return el.e < 0.999


@dataclass(frozen=True)
class CircularOrbit(OrbitFamily):
    name: str = "circle"
    n_params: int = 4
    param_names: tuple[str, ...] = ("log_a", "p", "r", "lambda0")

    def elements(self, theta: jax.Array) -> jax.Array:
        z = jnp.zeros((), dtype=theta.dtype)
        x = _elliptic_elements(theta[0], z, z, theta[1], theta[2], theta[3], self.t0, self.gm)
        return x.at[1].set(0.0)

    def theta_from_elements(self, el: ConicElements) -> FloatArray:
        a = el.q / (1.0 - el.e) if el.e < 1 else el.q
        full = _elliptic_theta(
            ConicElements(a, 0.0, el.i, el.Omega, el.omega, el.t_p), self.t0, self.gm
        )
        return full[[0, 3, 4, 5]]

    def admits(self, el: ConicElements) -> bool:
        return el.e < 0.999


for _cls in (OrbitFamily, ConicOrbit, EllipticOrbit, CircularOrbit):
    # t0 and gm are traced leaves, so new data never forces a recompile.
    jax.tree_util.register_dataclass(
        _cls, data_fields=["t0", "gm"], meta_fields=["name", "n_params", "param_names"]
    )

FAMILIES: dict[str, type[OrbitFamily]] = {
    "conic": ConicOrbit,
    "ellipse": EllipticOrbit,
    "circle": CircularOrbit,
}


def make_family(name: str, t0: float, gm: float) -> OrbitFamily:
    return FAMILIES[name](t0=t0, gm=gm)


def sky_directions(
    family: OrbitFamily, theta: jax.Array, t: jax.Array, r_obs: jax.Array
) -> jax.Array:
    """Geocentric unit vectors (N, 3)."""
    d = family.position(theta, t) - r_obs
    return d / jnp.linalg.norm(d, axis=1, keepdims=True)
