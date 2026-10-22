"""Hypothesis families for Bayesian inference and model selection (SPEC §6.3).

Each family is parametrized by *physical* coordinates φ with an explicit, independent
prior per dimension: uniform on [lo, hi) in the coordinate as written (so `log_a` is
log-uniform in a, `cos_i` is isotropic in orientation) and `periodic` dimensions wrap.
That gives every family the same unit-cube transform for nested sampling and the same
bounded/periodic handling for MCMC and HMC, so evidences are comparable across families.

Families
--------
* `circle` — heliocentric circular orbit, Kepler's third law (φ: log a, cos i, Ω, λ0).
* `conic-elliptic` — Keplerian ellipse (φ: log a, e, ϖ, cos i, Ω, λ0).
* `conic-hyperbolic` — Keplerian hyperbola (φ: log q, log(e − 1), ϖ, cos i, Ω, t_p).
* `equant-circle` — Kepler's "vicarious hypothesis": a circle whose centre is offset from
  the Sun by e_c·a, with uniform angular motion about an equant point e_q·a from the Sun on
  the apse line. Its period follows the third law from a, like the conics, so model
  selection against the ellipse tests the *shape*: a free period would cost the equant
  ~10–20 nats of Occam penalty whatever the data say. Kinematic.
* `epicycles` — Ptolemaic, geocentric: an eccentric deferent with an equant plus one
  epicycle in a tilted plane; overall scale is unobservable (deferent radius ≡ 1). Kinematic.

λ0 is the mean longitude (ϖ + mean anomaly) at the reference epoch t0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from jt_core.conics import ConicElements, conic_position, conic_state

FloatArray = NDArray[np.float64]
TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class Dim:
    name: str
    lo: float
    hi: float
    periodic: bool = False

    @property
    def width(self) -> float:
        return self.hi - self.lo


@dataclass(frozen=True)
class BayesFamily:
    """Base class. `t0` is the reference epoch, `t_span` the training span (days)."""

    t0: float
    gm: float
    t_span: float = 1.0
    name: ClassVar[str] = "base"
    kind: ClassVar[str] = "dynamical"
    geocentric: ClassVar[bool] = False
    conic: ClassVar[bool] = True

    def dims(self) -> tuple[Dim, ...]:
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return len(self.dims())

    @property
    def param_names(self) -> list[str]:
        return [d.name for d in self.dims()]

    # ---- forward model (JAX)
    def position(self, phi: jax.Array, t: jax.Array) -> jax.Array:
        """Heliocentric (or geocentric, arbitrary scale) positions (N, 3)."""
        return conic_position(self.elements(phi), t, self.gm)

    def elements(self, phi: jax.Array) -> jax.Array:
        raise NotImplementedError

    def sky(self, phi: jax.Array, t: jax.Array, r_obs: jax.Array) -> jax.Array:
        d = self.position(phi, t)
        if not self.geocentric:
            d = d - r_obs
        return d / jnp.linalg.norm(d, axis=1, keepdims=True)

    # ---- initialization (numpy)
    def from_elements(self, el: ConicElements) -> FloatArray | None:
        """φ approximating a conic (for MAP starts); None if not representable."""
        return None

    # ---- prior helpers (numpy)
    def lo_hi(self) -> tuple[FloatArray, FloatArray]:
        d = self.dims()
        return np.array([x.lo for x in d]), np.array([x.hi for x in d])

    def periodic_mask(self) -> NDArray[np.bool_]:
        return np.array([x.periodic for x in self.dims()])

    def log_prior_density(self) -> float:
        """Constant log density of the (box-uniform) prior inside its support."""
        return -float(sum(np.log(x.width) for x in self.dims()))

    def wrap(self, phi: FloatArray) -> FloatArray:
        """Wrap periodic coordinates into [lo, hi)."""
        lo, hi = self.lo_hi()
        per = self.periodic_mask()
        out = np.array(phi, dtype=np.float64, copy=True)
        out[..., per] = lo[per] + np.mod(out[..., per] - lo[per], (hi - lo)[per])
        return out

    def in_support(self, phi: FloatArray) -> NDArray[np.bool_]:
        lo, hi = self.lo_hi()
        per = self.periodic_mask()
        ok = (phi >= lo) & (phi <= hi)
        ok[..., per] = True
        return np.all(ok, axis=-1)


def _mean_motion(gm: jax.Array | float, a: jax.Array) -> jax.Array:
    return jnp.sqrt(gm / a**3)


_LOG_A = Dim("log_a", float(np.log(0.2)), float(np.log(2000.0)))
_COS_I = Dim("cos_i", -1.0, 1.0)
_OMEGA = Dim("Omega", 0.0, TWO_PI, periodic=True)
_VARPI = Dim("varpi", 0.0, TWO_PI, periodic=True)
_LAMBDA0 = Dim("lambda0", 0.0, TWO_PI, periodic=True)


def _elliptic(
    a: jax.Array,
    e: jax.Array,
    varpi: jax.Array,
    cos_i: jax.Array,
    Omega: jax.Array,
    lam0: jax.Array,
    t0: jax.Array | float,
    gm: jax.Array | float,
) -> jax.Array:
    n = _mean_motion(gm, a)
    inc = jnp.arccos(jnp.clip(cos_i, -1.0, 1.0))
    return jnp.stack([a * (1 - e), e, inc, Omega, varpi - Omega, t0 - (lam0 - varpi) / n])


@dataclass(frozen=True)
class CircleFamily(BayesFamily):
    name: ClassVar[str] = "circle"

    def dims(self) -> tuple[Dim, ...]:
        return (_LOG_A, _COS_I, _OMEGA, _LAMBDA0)

    def elements(self, phi: jax.Array) -> jax.Array:
        a = jnp.exp(phi[0])
        z = jnp.zeros_like(a)
        return _elliptic(a, z, phi[2], phi[1], phi[2], phi[3], self.t0, self.gm)

    def from_elements(self, el: ConicElements) -> FloatArray | None:
        if el.e >= 1:
            return None
        a = el.q / (1 - el.e)
        n = np.sqrt(self.gm / a**3)
        lam0 = el.Omega + el.omega + n * (self.t0 - el.t_p)
        return self.wrap(np.array([np.log(a), np.cos(el.i), el.Omega, lam0]))


@dataclass(frozen=True)
class EllipseFamily(BayesFamily):
    name: ClassVar[str] = "conic-elliptic"

    def dims(self) -> tuple[Dim, ...]:
        return (_LOG_A, Dim("e", 0.0, 0.95), _VARPI, _COS_I, _OMEGA, _LAMBDA0)

    def elements(self, phi: jax.Array) -> jax.Array:
        return _elliptic(jnp.exp(phi[0]), phi[1], phi[2], phi[3], phi[4], phi[5], self.t0, self.gm)

    def from_elements(self, el: ConicElements) -> FloatArray | None:
        if el.e >= 0.95:
            return None
        a = el.q / (1 - el.e)
        n = np.sqrt(self.gm / a**3)
        varpi = el.Omega + el.omega
        lam0 = varpi + n * (self.t0 - el.t_p)
        return self.wrap(np.array([np.log(a), el.e, varpi, np.cos(el.i), el.Omega, lam0]))

    def to_state(self, phi: jax.Array) -> jax.Array:
        """Heliocentric state (x, v) at t0, shape (6,)."""
        r, v = conic_state(self.elements(phi), jnp.atleast_1d(self.t0), self.gm)
        return jnp.concatenate([r[0], v[0]])

    def from_state(self, y: jax.Array) -> jax.Array:
        """φ from the state at t0 (inverse of `to_state`; NaN for unbound states). Singular
        at e = 0 and i = 0, where ϖ and Ω are undefined."""
        r, v = y[:3], y[3:]
        rn = jnp.linalg.norm(r)
        h = jnp.cross(r, v)
        hn = jnp.linalg.norm(h)
        e_vec = ((v @ v - self.gm / rn) * r - (r @ v) * v) / self.gm
        a = 1.0 / (2.0 / rn - (v @ v) / self.gm)
        a = jnp.where(a > 0, a, jnp.nan)
        Omega = jnp.arctan2(h[0], -h[1])
        node = jnp.stack([jnp.cos(Omega), jnp.sin(Omega), 0.0])
        omega = jnp.arctan2(e_vec @ jnp.cross(h / hn, node), e_vec @ node)
        e = jnp.linalg.norm(e_vec)
        ecc_anom = jnp.arctan2((r @ v) / jnp.sqrt(self.gm * a), 1.0 - rn / a)
        varpi = Omega + omega
        lam0 = varpi + ecc_anom - e * jnp.sin(ecc_anom)
        return jnp.stack([jnp.log(a), e, varpi, h[2] / hn, Omega, lam0])


@dataclass(frozen=True)
class HyperbolaFamily(BayesFamily):
    name: ClassVar[str] = "conic-hyperbolic"

    def dims(self) -> tuple[Dim, ...]:
        return (
            Dim("log_q", float(np.log(0.05)), float(np.log(200.0))),
            Dim("log_e_minus_1", float(np.log(1e-4)), float(np.log(20.0))),
            _VARPI,
            _COS_I,
            _OMEGA,
            Dim("t_p", self.t0 - 2 * self.t_span, self.t0 + 2 * self.t_span),
        )

    def elements(self, phi: jax.Array) -> jax.Array:
        inc = jnp.arccos(jnp.clip(phi[3], -1.0, 1.0))
        return jnp.stack(
            [jnp.exp(phi[0]), 1 + jnp.exp(phi[1]), inc, phi[4], phi[2] - phi[4], phi[5]]
        )

    def from_elements(self, el: ConicElements) -> FloatArray | None:
        e = max(el.e, 1.0 + 1e-3)
        return self.wrap(
            np.array(
                [np.log(el.q), np.log(e - 1), el.Omega + el.omega, np.cos(el.i), el.Omega, el.t_p]
            )
        )


def equant_point(
    angle: jax.Array, radius: jax.Array | float, e_c: jax.Array, e_q: jax.Array, apse: jax.Array
) -> jax.Array:
    """In-plane point on a circle (centre at −e_c·radius·p̂) seen at `angle` (measured from
    the +x axis) from the equant point −e_q·radius·p̂, p̂ = (cos apse, sin apse). (N, 2)."""
    p_hat = jnp.stack([jnp.cos(apse), jnp.sin(apse)])
    c = -e_c * radius * p_hat
    eq = -e_q * radius * p_hat
    u = jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1)
    d = eq - c
    ud = u @ d
    rho = -ud + jnp.sqrt(jnp.maximum(ud**2 - d @ d + radius**2, 0.0))
    return eq + rho[:, None] * u


def _plane_to_3d(xy: jax.Array, cos_i: jax.Array, Omega: jax.Array) -> jax.Array:
    """Rotate in-plane points (x along the ascending node) by Rz(Ω) Rx(i)."""
    si = jnp.sqrt(jnp.clip(1 - cos_i**2, 0.0, 1.0))
    cO, sO = jnp.cos(Omega), jnp.sin(Omega)
    x, y = xy[:, 0], xy[:, 1]
    return jnp.stack([cO * x - sO * cos_i * y, sO * x + cO * cos_i * y, si * y], axis=1)


@dataclass(frozen=True)
class EquantFamily(BayesFamily):
    name: ClassVar[str] = "equant-circle"
    kind: ClassVar[str] = "kinematic"
    conic: ClassVar[bool] = False

    def dims(self) -> tuple[Dim, ...]:
        return (
            _LOG_A,
            Dim("e_c", 0.0, 0.5),
            Dim("e_q", 0.0, 0.5),
            _VARPI,
            _COS_I,
            _OMEGA,
            _LAMBDA0,
        )

    def position(self, phi: jax.Array, t: jax.Array) -> jax.Array:
        a = jnp.exp(phi[0])
        varpi, cos_i, Omega, lam0 = phi[3], phi[4], phi[5], phi[6]
        omega = varpi - Omega  # apse direction measured in the orbit plane from the node
        ang = omega + (lam0 - varpi) + _mean_motion(self.gm, a) * (t - self.t0)
        xy = equant_point(ang, a, phi[1], phi[2], omega)
        return _plane_to_3d(xy, cos_i, Omega)

    def from_elements(self, el: ConicElements) -> FloatArray | None:
        if el.e >= 0.5:
            return None
        a = el.q / (1 - el.e)
        varpi = el.Omega + el.omega
        lam0 = varpi + np.sqrt(self.gm / a**3) * (self.t0 - el.t_p)
        x = [np.log(a), el.e, min(2 * el.e, 0.49), varpi, np.cos(el.i), el.Omega, lam0]
        return self.wrap(np.array(x))


@dataclass(frozen=True)
class EpicycleFamily(BayesFamily):
    name: ClassVar[str] = "epicycles"
    kind: ClassVar[str] = "kinematic"
    geocentric: ClassVar[bool] = True
    conic: ClassVar[bool] = False

    def dims(self) -> tuple[Dim, ...]:
        return (
            Dim("f_def", 1e-4, 0.02),  # deferent frequency (cycles/day), prograde
            Dim("f_epi", -0.02, 0.02),  # epicycle frequency, either sense
            Dim("ratio", 0.0, 1.5),  # epicycle radius / deferent radius
            Dim("e_c", 0.0, 0.5),
            Dim("e_q", 0.0, 0.5),
            Dim("apse", 0.0, TWO_PI, periodic=True),
            Dim("phase_def", 0.0, TWO_PI, periodic=True),
            Dim("phase_epi", 0.0, TWO_PI, periodic=True),
            _COS_I,
            _OMEGA,
        )

    def position(self, phi: jax.Array, t: jax.Array) -> jax.Array:
        dt = t - self.t0
        ang_d = phi[6] + TWO_PI * phi[0] * dt
        ang_e = phi[7] + TWO_PI * phi[1] * dt
        xy = equant_point(ang_d, 1.0, phi[3], phi[4], phi[5])
        xy = xy + phi[2] * jnp.stack([jnp.cos(ang_e), jnp.sin(ang_e)], axis=-1)
        return _plane_to_3d(xy, phi[8], phi[9])


@dataclass(frozen=True)
class ObserverFamily(BayesFamily):
    """The observer's own orbit when it is not given (tiers T2/T3): a Keplerian ellipse in
    the ecliptic plane (i ≡ 0 by the frame's definition). Its scale follows from its period
    through GM, so angles alone fix it (φ: log a, e, ϖ, λ0)."""

    name: ClassVar[str] = "observer-ellipse"

    def dims(self) -> tuple[Dim, ...]:
        return (
            Dim("obs_log_a", float(np.log(0.3)), float(np.log(3.0))),
            Dim("obs_e", 0.0, 0.3),
            Dim("obs_varpi", 0.0, TWO_PI, periodic=True),
            Dim("obs_lambda0", 0.0, TWO_PI, periodic=True),
        )

    def elements(self, phi: jax.Array) -> jax.Array:
        one, zero = jnp.ones(()), jnp.zeros(())
        return _elliptic(jnp.exp(phi[0]), phi[1], phi[2], one, zero, phi[3], self.t0, self.gm)

    def from_elements(self, el: ConicElements) -> FloatArray | None:
        if el.e >= 0.3:
            return None
        a = el.q / (1 - el.e)
        varpi = el.Omega + el.omega
        lam0 = varpi + np.sqrt(self.gm / a**3) * (self.t0 - el.t_p)
        return self.wrap(np.array([np.log(a), el.e, varpi, lam0]))


FAMILIES: dict[str, type[BayesFamily]] = {
    f.name: f for f in (CircleFamily, EllipseFamily, HyperbolaFamily, EquantFamily, EpicycleFamily)
}

for _cls in (BayesFamily, ObserverFamily, *FAMILIES.values()):
    jax.tree_util.register_dataclass(_cls, data_fields=["t0", "gm", "t_span"], meta_fields=[])


def make_bayes_family(name: str, t0: float, gm: float, t_span: float) -> BayesFamily:
    if name not in FAMILIES:
        raise KeyError(f"unknown family {name!r}; known: {sorted(FAMILIES)}")
    return FAMILIES[name](t0=t0, gm=gm, t_span=t_span)


# Non-Keplerian (integrated) families register themselves into FAMILIES on import.
from jan.models import force_families as _force_families  # noqa: E402

__all_families__ = sorted(FAMILIES)
_ = _force_families
