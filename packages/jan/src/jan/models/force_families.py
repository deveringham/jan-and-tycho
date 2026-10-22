"""Non-Keplerian dynamical families, integrated numerically (SPEC §6.3, Phase 4).

All start from osculating elliptic elements at the reference epoch t0 (the same physical
coordinates as `conic-elliptic`: log a, e, ϖ, cos i, Ω, λ0) and add law parameters:

* `central-power-law`  f = −GM·μ · r^(−n)                   (+ n, log μ)
* `central-hooke`      f = −k r                              (+ log k)
* `central-yukawa`     f = −GM/r² · (1 + r/λ) e^(−r/λ)       (+ log λ)
* `central-newton-plus-inverse-cube`  f = −GM/r² · (1 + ε/r) (+ ε)
* `perturbed-conic`    Newton + one unseen perturber on a circular orbit in the ecliptic
                       (+ log m [M☉], log a_p, λ_p at t0) — Le Verrier's model.

Positions come from jt_core.integrate (diffrax, Dopri8), so every family is differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from jan.models.families import (
    _COS_I,
    _LAMBDA0,
    _LOG_A,
    _OMEGA,
    _VARPI,
    FAMILIES,
    BayesFamily,
    Dim,
    EllipseFamily,
)
from jt_core.conics import ConicElements, conic_state
from jt_core.integrate import propagate

FloatArray = NDArray[np.float64]
TWO_PI = 2.0 * np.pi
RTOL = 1e-10
ATOL = 1e-12


@dataclass(frozen=True)
class _OdeFamily(BayesFamily):
    """Base: initial state from osculating ellipse elements at t0, then integrate."""

    conic: ClassVar[bool] = False
    n_law: ClassVar[int] = 0

    def orbit_dims(self) -> tuple[Dim, ...]:
        return (_LOG_A, Dim("e", 0.0, 0.95), _VARPI, _COS_I, _OMEGA, _LAMBDA0)

    def law_dims(self) -> tuple[Dim, ...]:
        raise NotImplementedError

    def dims(self) -> tuple[Dim, ...]:
        return (*self.orbit_dims(), *self.law_dims())

    def law_defaults(self, el: ConicElements) -> list[float]:
        raise NotImplementedError

    def osculating(self, phi: jax.Array) -> jax.Array:
        """Osculating elements (q, e, i, Ω, ω, t_p) of the initial state at t0."""
        return EllipseFamily(t0=self.t0, gm=self.gm).elements(phi[:6])

    def elements(self, phi: jax.Array) -> jax.Array:
        return self.osculating(phi)

    def accel_fn(self, phi: jax.Array) -> Any:
        raise NotImplementedError

    def position(self, phi: jax.Array, t: jax.Array) -> jax.Array:
        r0, v0 = conic_state(self.osculating(phi), jnp.atleast_1d(self.t0), self.gm)
        x, _ = propagate(
            self.accel_fn(phi), r0, v0, self.t0, t, None, RTOL, ATOL, differentiable=True
        )
        return x[:, 0]

    def acceleration(self, phi: jax.Array, x: jax.Array, t: jax.Array) -> jax.Array:
        acc = self.accel_fn(phi)
        return jax.vmap(lambda xi, ti: acc(xi[None], xi[None] * 0, ti, None)[0])(x, t)

    def from_elements(self, el: ConicElements) -> FloatArray | None:
        base = EllipseFamily(t0=self.t0, gm=self.gm).from_elements(el)
        if base is None:
            return None
        return self.wrap(np.concatenate([base, self.law_defaults(el)]))

    def law_strings(self, phi: FloatArray) -> tuple[list[str], dict[str, float], str]:
        """(vector expression in x, y, z, constants, radial f(r) string)."""
        raise NotImplementedError


def _radial(f: Any) -> Any:
    def accel(x: jax.Array, v: jax.Array, t: jax.Array, args: Any) -> jax.Array:
        r = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return f(r) * x / r

    return accel


def _vector(radial: str) -> list[str]:
    """Components of f(r) r̂ in x, y, z from a radial expression in `r` (sympy)."""
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    x, y, z = sp.symbols("x y z")
    r = sp.sqrt(x**2 + y**2 + z**2)
    f = parse_expr(radial, local_dict={"r": sp.Symbol("r"), "lam": sp.Symbol("lam")}).subs(
        sp.Symbol("r"), r
    )
    return [str(f * c / r) for c in (x, y, z)]


@dataclass(frozen=True)
class PowerLawFamily(_OdeFamily):
    name: ClassVar[str] = "central-power-law"
    n_law: ClassVar[int] = 2

    def law_dims(self) -> tuple[Dim, ...]:
        return (Dim("n", 1.0, 3.5), Dim("log_mu", -1.0, 1.0))

    def law_defaults(self, el: ConicElements) -> list[float]:
        return [2.0, 0.0]

    def accel_fn(self, phi: jax.Array) -> Any:
        n, mu = phi[6], jnp.exp(phi[7])
        return _radial(lambda r: -self.gm * mu * r ** (-n))

    def law_strings(self, phi: FloatArray) -> tuple[list[str], dict[str, float], str]:
        c = float(self.gm * np.exp(phi[7]))
        return _vector("-C*r**(-n)"), {"C": c, "n": float(phi[6])}, "-C*r**(-n)"


@dataclass(frozen=True)
class HookeFamily(_OdeFamily):
    name: ClassVar[str] = "central-hooke"
    n_law: ClassVar[int] = 1

    def law_dims(self) -> tuple[Dim, ...]:
        return (Dim("log_k", float(np.log(1e-7)), float(np.log(1e-2))),)

    def law_defaults(self, el: ConicElements) -> list[float]:
        a = el.q / (1 - el.e)
        return [float(np.log(self.gm / a**3))]  # same period as the Kepler orbit

    def accel_fn(self, phi: jax.Array) -> Any:
        k = jnp.exp(phi[6])
        return _radial(lambda r: -k * r)

    def law_strings(self, phi: FloatArray) -> tuple[list[str], dict[str, float], str]:
        return _vector("-k*r"), {"k": float(np.exp(phi[6]))}, "-k*r"


@dataclass(frozen=True)
class YukawaFamily(_OdeFamily):
    name: ClassVar[str] = "central-yukawa"
    n_law: ClassVar[int] = 1

    def law_dims(self) -> tuple[Dim, ...]:
        return (Dim("log_lambda", float(np.log(0.1)), float(np.log(1000.0))),)

    def law_defaults(self, el: ConicElements) -> list[float]:
        return [float(np.log(100.0))]

    def accel_fn(self, phi: jax.Array) -> Any:
        lam = jnp.exp(phi[6])
        return _radial(lambda r: -self.gm / r**2 * (1 + r / lam) * jnp.exp(-r / lam))

    def law_strings(self, phi: FloatArray) -> tuple[list[str], dict[str, float], str]:
        rad = "-GM/r**2*(1 + r/lam)*exp(-r/lam)"
        return _vector(rad), {"GM": self.gm, "lam": float(np.exp(phi[6]))}, rad


@dataclass(frozen=True)
class NewtonEpsFamily(_OdeFamily):
    name: ClassVar[str] = "central-newton-plus-inverse-cube"
    n_law: ClassVar[int] = 1

    def law_dims(self) -> tuple[Dim, ...]:
        return (Dim("epsilon", -0.01, 0.01),)

    def law_defaults(self, el: ConicElements) -> list[float]:
        return [0.0]

    def accel_fn(self, phi: jax.Array) -> Any:
        eps = phi[6]
        return _radial(lambda r: -self.gm / r**2 * (1 + eps / r))

    def law_strings(self, phi: FloatArray) -> tuple[list[str], dict[str, float], str]:
        rad = "-GM/r**2*(1 + eps/r)"
        return _vector(rad), {"GM": self.gm, "eps": float(phi[6])}, rad


@dataclass(frozen=True)
class PerturbedConicFamily(_OdeFamily):
    """Newton + one unseen perturber on a circular, ecliptic orbit (restricted 3-body)."""

    name: ClassVar[str] = "perturbed-conic"
    n_law: ClassVar[int] = 3

    def law_dims(self) -> tuple[Dim, ...]:
        return (
            Dim("log_m_pert", float(np.log(1e-6)), float(np.log(1e-2))),
            Dim("log_a_pert", float(np.log(1.0)), float(np.log(200.0))),
            Dim("lambda_pert", 0.0, TWO_PI, periodic=True),
        )

    def law_defaults(self, el: ConicElements) -> list[float]:
        a = el.q / (1 - el.e)
        return [float(np.log(1e-5)), float(np.log(1.6 * a)), 0.0]

    def grid_starts(self, base: FloatArray) -> list[FloatArray]:
        """Global search over the perturber: radius × longitude (mass fixed small)."""
        a = float(np.exp(base[0]))
        out = []
        for f in (1.3, 1.6, 2.0, 2.6):
            for lam in np.linspace(0, TWO_PI, 12, endpoint=False):
                out.append(
                    self.wrap(np.concatenate([base[:6], [np.log(1e-4), np.log(f * a), lam]]))
                )
        return out

    def perturber(self, phi: jax.Array, t: jax.Array) -> jax.Array:
        a_p = jnp.exp(phi[7])
        n_p = jnp.sqrt(self.gm / a_p**3)
        lam = phi[8] + n_p * (t - self.t0)
        return a_p * jnp.stack([jnp.cos(lam), jnp.sin(lam), jnp.zeros_like(lam)], axis=-1)

    def accel_fn(self, phi: jax.Array) -> Any:
        m = jnp.exp(phi[6])

        def accel(x: jax.Array, v: jax.Array, t: jax.Array, args: Any) -> jax.Array:
            xp = self.perturber(phi, t)
            r = jnp.linalg.norm(x, axis=-1, keepdims=True)
            d = xp - x
            dn = jnp.linalg.norm(d, axis=-1, keepdims=True)
            rp = jnp.linalg.norm(xp)
            return -self.gm * x / r**3 + self.gm * m * (d / dn**3 - xp / rp**3)

        return accel

    def law_strings(self, phi: FloatArray) -> tuple[list[str], dict[str, float], str]:
        consts = {"GM": self.gm, "m_pert": float(np.exp(phi[6])), "a_pert": float(np.exp(phi[7]))}
        return ["-GM*x/(x**2 + y**2 + z**2)**(3/2) + perturbation"] * 3, consts, "-GM/r**2"


FORCE_FAMILIES: dict[str, type[BayesFamily]] = {
    f.name: f
    for f in (PowerLawFamily, HookeFamily, YukawaFamily, NewtonEpsFamily, PerturbedConicFamily)
}
for _cls in FORCE_FAMILIES.values():
    jax.tree_util.register_dataclass(_cls, data_fields=["t0", "gm", "t_span"], meta_fields=[])
FAMILIES.update(FORCE_FAMILIES)
