# pyright: strict, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportMissingTypeStubs=false
"""Two-body conic propagation with universal variables, and elements ↔ state.

Elements are `(q, e, i, Ω, ω, t_p)` (SPEC §4) so ellipses, parabolas and hyperbolas share
one code path: the body is placed at perihelion at `t_p` and propagated by `t − t_p` with
the universal Kepler equation, measured from perihelion where the radial velocity is zero:

    √μ Δt = e χ³ S(αχ²) + q χ,       r = q + e χ² C(αχ²),       α = (1 − e) / q.

`F(χ) = e χ³ S + q χ` is odd, strictly increasing (F′ = r ≥ q > 0) and convex for χ > 0
within half a revolution, so a bracketed Newton iteration from a cubic (Barker-like)
initial guess converges robustly for any e, including e = 1 and e → 1 (SPEC §12).

The propagator is written in JAX and is differentiable: the root solve runs under
`stop_gradient` and one final Newton step restores exact first derivatives via the
implicit function theorem. Elliptic Δt is first reduced to (−P/2, P/2].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from numpy.typing import ArrayLike, NDArray

from jt_core.angles import wrap_2pi
from jt_core.constants import GM_SUN

FloatArray = NDArray[np.float64]
Array = Any  # jax.Array; kept loose so numpy inputs type-check under strict pyright

_SERIES_Z = 1.0
_N_SERIES = 10  # |z| < 1: the next term is < 1/23! ≈ 4e-23
_MAX_ITER = 200


# ----------------------------------------------------------------------------- elements


@dataclass(frozen=True)
class ConicElements:
    """Conic elements; angles in radians, q in AU, t_p in days."""

    q: float
    e: float
    i: float
    Omega: float
    omega: float
    t_p: float

    def as_array(self) -> FloatArray:
        return np.array([self.q, self.e, self.i, self.Omega, self.omega, self.t_p])

    @classmethod
    def from_array(cls, x: ArrayLike) -> ConicElements:
        a = np.asarray(x, dtype=np.float64)
        return cls(*(float(v) for v in a[:6]))

    def to_dict(self) -> dict[str, float]:
        return {
            "q": self.q,
            "e": self.e,
            "i": self.i,
            "Omega": self.Omega,
            "omega": self.omega,
            "t_p": self.t_p,
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> ConicElements:
        return cls(d["q"], d["e"], d["i"], d["Omega"], d["omega"], d["t_p"])

    @property
    def a(self) -> float:
        """Semi-major axis (negative for hyperbolas, ±inf for parabolas)."""
        if self.e == 1.0:
            return float("inf")
        return self.q / (1.0 - self.e)

    def period(self, mu: float = GM_SUN) -> float:
        """Orbital period in days (inf for e ≥ 1)."""
        if self.e >= 1.0:
            return float("inf")
        return 2.0 * np.pi * float(np.sqrt(self.a**3 / mu))

    def mean_motion(self, mu: float = GM_SUN) -> float:
        a = abs(self.a)
        return float(np.sqrt(mu / a**3)) if np.isfinite(a) else 0.0

    def to_keplerian(self, epoch: float = 0.0, mu: float = GM_SUN) -> dict[str, float]:
        """`(a, e, i, Ω, ω, M0)` with M0 the mean anomaly at `epoch` (e < 1 only)."""
        if self.e >= 1.0:
            raise ValueError("Keplerian (a, ..., M0) elements require e < 1")
        m0 = wrap_2pi(self.mean_motion(mu) * (epoch - self.t_p))
        return {
            "a": self.a,
            "e": self.e,
            "i": self.i,
            "Omega": self.Omega,
            "omega": self.omega,
            "M0": float(m0),
        }

    @classmethod
    def from_keplerian(
        cls,
        a: float,
        e: float,
        i: float,
        Omega: float,
        omega: float,
        M0: float,
        epoch: float = 0.0,
        mu: float = GM_SUN,
    ) -> ConicElements:
        if not 0.0 <= e < 1.0 or a <= 0.0:
            raise ValueError("from_keplerian requires 0 <= e < 1 and a > 0")
        n = float(np.sqrt(mu / a**3))
        return cls(a * (1.0 - e), e, i, Omega, omega, epoch - M0 / n)


# ----------------------------------------------------------------------------- Stumpff


def stumpff(z: Array) -> tuple[Array, Array]:
    """Stumpff functions C(z), S(z), stable for all real z (series near 0)."""
    z = jnp.asarray(z, dtype=jnp.float64)
    small = jnp.abs(z) < _SERIES_Z
    # Series: C = Σ (−z)^k / (2k+2)!, S = Σ (−z)^k / (2k+3)!  (Horner form).
    zs = jnp.where(small, z, 0.0)
    c_ser = jnp.zeros_like(zs)
    s_ser = jnp.zeros_like(zs)
    for k in range(_N_SERIES - 1, -1, -1):
        c_ser = 1.0 / _fact(2 * k + 2) - zs * c_ser
        s_ser = 1.0 / _fact(2 * k + 3) - zs * s_ser
    pos = z > 0
    zp = jnp.where(pos & ~small, z, 1.0)
    sp = jnp.sqrt(zp)
    c_pos = 2.0 * jnp.sin(0.5 * sp) ** 2 / zp
    s_pos = (sp - jnp.sin(sp)) / (zp * sp)
    zn = jnp.where(~pos & ~small, -z, 1.0)
    sn = jnp.sqrt(zn)
    c_neg = 2.0 * jnp.sinh(0.5 * sn) ** 2 / zn
    s_neg = (jnp.sinh(sn) - sn) / (zn * sn)
    c = jnp.where(small, c_ser, jnp.where(pos, c_pos, c_neg))
    s = jnp.where(small, s_ser, jnp.where(pos, s_pos, s_neg))
    return c, s


def stumpff_np(z: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Numpy twin of `stumpff` for non-differentiable host-side code."""
    za = np.asarray(z, dtype=np.float64)
    small = np.abs(za) < _SERIES_Z
    zs = np.where(small, za, 0.0)
    c_ser = np.zeros_like(zs)
    s_ser = np.zeros_like(zs)
    for k in range(_N_SERIES - 1, -1, -1):
        c_ser = 1.0 / _fact(2 * k + 2) - zs * c_ser
        s_ser = 1.0 / _fact(2 * k + 3) - zs * s_ser
    pos = za > 0
    zp = np.where(pos & ~small, za, 1.0)
    sp = np.sqrt(zp)
    zn = np.where(~pos & ~small, -za, 1.0)
    sn = np.sqrt(zn)
    c = np.where(
        small,
        c_ser,
        np.where(pos, 2.0 * np.sin(0.5 * sp) ** 2 / zp, 2.0 * np.sinh(0.5 * sn) ** 2 / zn),
    )
    s = np.where(
        small, s_ser, np.where(pos, (sp - np.sin(sp)) / (zp * sp), (np.sinh(sn) - sn) / (zn * sn))
    )
    return c, s


def _fact(n: int) -> float:
    return float(np.prod(np.arange(1, n + 1, dtype=np.float64))) if n > 0 else 1.0


_INV_FACT = [1.0 / _fact(n) for n in range(2 * _N_SERIES + 4)]


# ----------------------------------------------------------------------------- solver


def _kepler_f(chi: Array, q: Array, e: Array, alpha: Array) -> tuple[Array, Array]:
    """F(χ) = e χ³ S(αχ²) + q χ and F′(χ) = r(χ)."""
    z = alpha * chi * chi
    c, s = stumpff(z)
    f = e * chi**3 * s + q * chi
    r = q + e * chi * chi * c
    return f, r


def _initial_guess(target: Array, q: Array, e: Array) -> Array:
    """Root of the cubic (e/6)χ³ + qχ = target (exact for parabolas), χ ≥ 0."""
    es = jnp.maximum(e, 1e-12)
    p = 6.0 * q / es
    rr = 6.0 * target / es
    return 2.0 * jnp.sqrt(p / 3.0) * jnp.sinh(jnp.arcsinh(1.5 * rr / p * jnp.sqrt(3.0 / p)) / 3.0)


def _solve_chi_abs(target: Array, q: Array, e: Array, alpha: Array) -> Array:
    """Solve F(χ) = target ≥ 0 for χ ≥ 0 by bracketed Newton (no gradients)."""
    lo = jnp.zeros_like(target)
    hi = target / q  # F(χ) ≥ qχ, so F(hi) ≥ target
    x0 = jnp.clip(_initial_guess(target, q, e), lo, hi)
    done0 = target == 0.0

    def cond(state: tuple[Array, ...]) -> Array:
        _, _, _, it, done = state
        return (it < _MAX_ITER) & ~jnp.all(done)

    def body(state: tuple[Array, ...]) -> tuple[Array, ...]:
        x, lo, hi, it, done = state
        f, df = _kepler_f(x, q, e, alpha)
        g = f - target
        lo = jnp.where(g < 0, x, lo)
        hi = jnp.where(g > 0, x, hi)
        step = g / df
        tol = 4.0 * jnp.finfo(jnp.float64).eps * jnp.maximum(jnp.abs(x), 1e-300)
        # Converged when the Newton step is at rounding level. Test this before any
        # bisection: near the root, rounding can push the iterate onto a bracket end.
        conv = (g == 0) | (jnp.abs(step) <= tol) | (hi - lo <= tol)
        xn = x - step
        bad = ~conv & ((xn <= lo) | (xn >= hi))
        xn = jnp.where(bad, 0.5 * (lo + hi), xn)
        x = jnp.where(done, x, xn)
        return x, lo, hi, it + 1, done | conv

    x, *_ = lax.while_loop(cond, body, (x0, lo, hi, jnp.array(0), done0))
    return x


def _orientation(i: Array, Omega: Array, omega: Array) -> tuple[Array, Array]:
    """Perifocal unit vectors P̂ (to perihelion) and Q̂ in the ecliptic frame."""
    ci, si = jnp.cos(i), jnp.sin(i)
    cO, sO = jnp.cos(Omega), jnp.sin(Omega)
    co, so = jnp.cos(omega), jnp.sin(omega)
    p_hat = jnp.stack([cO * co - sO * so * ci, sO * co + cO * so * ci, so * si])
    q_hat = jnp.stack([-cO * so - sO * co * ci, -sO * so + cO * co * ci, co * si])
    return p_hat, q_hat


def conic_state(params: Array, t: Array, mu: float = GM_SUN) -> tuple[Array, Array]:
    """Heliocentric position and velocity at times t.

    Args:
        params: array `(q, e, i, Ω, ω, t_p)`.
        t: times (N,) in days.
        mu: gravitational parameter (AU³/day²).

    Returns:
        (r, v), each (N, 3), in AU and AU/day.
    """
    params = jnp.asarray(params, dtype=jnp.float64)
    q, e, i, Omega, omega, t_p = (params[k] for k in range(6))
    t = jnp.atleast_1d(jnp.asarray(t, dtype=jnp.float64))
    sqmu = jnp.sqrt(mu)
    alpha = (1.0 - e) / q

    # Reduce elliptic time offsets into one revolution around perihelion.
    dt = t - t_p
    is_ell = e < 1.0
    a_ell = jnp.where(is_ell, q / jnp.where(is_ell, 1.0 - e, 1.0), 1.0)
    period = 2.0 * jnp.pi * jnp.sqrt(a_ell**3 / mu)
    n_rev = lax.stop_gradient(jnp.where(is_ell, jnp.round(dt / period), 0.0))
    dt = dt - n_rev * period

    # Root solve without gradients, then one differentiable Newton step.
    q_sg, e_sg, a_sg, dt_sg = (lax.stop_gradient(v) for v in (q, e, alpha, dt))
    chi_abs = _solve_chi_abs(sqmu * jnp.abs(dt_sg), q_sg, e_sg, a_sg)
    chi0 = jnp.sign(dt_sg) * chi_abs
    f0, r0 = _kepler_f(chi0, q, e, alpha)
    chi = chi0 - (f0 - sqmu * dt) / r0

    z = alpha * chi * chi
    c, s = stumpff(z)
    r = q + e * chi * chi * c
    p = q * (1.0 + e)
    x_pf = q - chi * chi * c
    y_pf = chi * jnp.sqrt(p) * (1.0 - z * s)
    v_peri = jnp.sqrt(mu * (1.0 + e) / q)
    vx_pf = sqmu / r * chi * (z * s - 1.0)
    vy_pf = (1.0 - chi * chi * c / r) * v_peri

    p_hat, q_hat = _orientation(i, Omega, omega)
    pos = x_pf[:, None] * p_hat[None, :] + y_pf[:, None] * q_hat[None, :]
    vel = vx_pf[:, None] * p_hat[None, :] + vy_pf[:, None] * q_hat[None, :]
    return pos, vel


def conic_position(params: Array, t: Array, mu: float = GM_SUN) -> Array:
    """Heliocentric positions (N, 3) at times t. See `conic_state`."""
    return conic_state(params, t, mu)[0]


_conic_state_jit = jax.jit(conic_state, static_argnames=("mu",))


def bucket(n: int, minimum: int = 8) -> int:
    """Next power of two ≥ n: padding jitted inputs to few shapes avoids recompiles."""
    return max(minimum, 1 << max(n - 1, 0).bit_length())


def pad_rows(x: ArrayLike, n: int) -> FloatArray:
    """Pad along axis 0 to length n by repeating the first row."""
    a = np.asarray(x)
    if a.shape[0] >= n:
        return a
    fill = np.repeat(a[:1] if a.shape[0] else np.zeros((1, *a.shape[1:])), n - a.shape[0], axis=0)
    return np.concatenate([a, fill], axis=0)


def elements_to_state(
    el: ConicElements, t: ArrayLike, mu: float = GM_SUN
) -> tuple[FloatArray, FloatArray]:
    """Numpy convenience wrapper around `conic_state`."""
    ta = np.atleast_1d(np.asarray(t, dtype=np.float64))
    n = ta.size
    tp = pad_rows(ta, bucket(n))
    r, v = _conic_state_jit(jnp.asarray(el.as_array()), jnp.asarray(tp), mu)
    return np.asarray(r)[:n], np.asarray(v)[:n]


def _stumpff_scalar(z: float) -> tuple[float, float]:
    """Scalar Stumpff C, S (pure `math`, for tight host-side loops)."""
    if abs(z) < _SERIES_Z:
        c = s = 0.0
        for k in range(_N_SERIES - 1, -1, -1):
            c = _INV_FACT[2 * k + 2] - z * c
            s = _INV_FACT[2 * k + 3] - z * s
        return c, s
    if z > 0:
        sz = math.sqrt(z)
        return 2.0 * math.sin(0.5 * sz) ** 2 / z, (sz - math.sin(sz)) / (z * sz)
    sz = math.sqrt(-z)
    return 2.0 * math.sinh(0.5 * sz) ** 2 / -z, (math.sinh(sz) - sz) / (-z * sz)


def lagrange_fg(r0: ArrayLike, v0: ArrayLike, dt: float, mu: float = GM_SUN) -> tuple[float, float]:
    """Lagrange coefficients f, g with r(t0 + dt) = f r0 + g v0 (numpy, universal variables).

    Newton iteration on the universal Kepler equation from a general state; meant for short
    arcs (e.g. Gauss's method), where it converges in a handful of steps.
    """
    rx, ry, rz = (float(x) for x in np.asarray(r0, dtype=np.float64))
    vx, vy, vz = (float(x) for x in np.asarray(v0, dtype=np.float64))
    rn = math.sqrt(rx * rx + ry * ry + rz * rz)
    sqmu = math.sqrt(mu)
    vr0 = (rx * vx + ry * vy + rz * vz) / rn
    alpha = 2.0 / rn - (vx * vx + vy * vy + vz * vz) / mu
    chi = sqmu * dt / rn
    for _ in range(100):
        z = alpha * chi * chi
        c, s = _stumpff_scalar(z)
        f_val = (
            (rn * vr0 / sqmu) * chi * chi * c + (1 - alpha * rn) * chi**3 * s + rn * chi - sqmu * dt
        )
        df = (rn * vr0 / sqmu) * chi * (1 - z * s) + (1 - alpha * rn) * chi * chi * c + rn
        step = f_val / df
        chi -= step
        if abs(step) <= 1e-14 * max(1.0, abs(chi)):
            break
    z = alpha * chi * chi
    c, s = _stumpff_scalar(z)
    return 1.0 - chi * chi / rn * c, dt - chi**3 * s / sqmu


# ----------------------------------------------------------------------------- state → elements

_EPS_E = 1e-12
_EPS_I = 1e-12


def time_since_perihelion(nu: ArrayLike, q: float, e: float, mu: float = GM_SUN) -> FloatArray:
    """Δt = t − t_p for true anomaly ν (nearest perihelion for ellipses)."""
    nu_a = np.asarray(nu, dtype=np.float64)
    if e < 1.0:
        a = q / (1.0 - e)
        ecc_anom = 2.0 * np.arctan2(
            np.sqrt(1.0 - e) * np.sin(nu_a / 2), np.sqrt(1.0 + e) * np.cos(nu_a / 2)
        )
        chi = np.sqrt(a) * ecc_anom
    elif e > 1.0:
        hyp_anom = 2.0 * np.arctanh(np.sqrt((e - 1.0) / (e + 1.0)) * np.tan(nu_a / 2))
        chi = np.sqrt(q / (e - 1.0)) * hyp_anom
    else:
        chi = np.sqrt(2.0 * q) * np.tan(nu_a / 2)
    alpha = (1.0 - e) / q
    _, s = stumpff_np(alpha * chi * chi)
    return (e * chi**3 * s + q * chi) / np.sqrt(mu)


def state_to_elements(r: ArrayLike, v: ArrayLike, t: float, mu: float = GM_SUN) -> ConicElements:
    """Osculating conic elements from a single state (r, v) at time t.

    Degenerate cases (SPEC §12): for i ≈ 0 (or π) Ω is set to 0 and the node line is +x;
    for e ≈ 0 ω is set to 0 and "perihelion" is the node line, so t_p is the time of
    nodal passage. Elements stay self-consistent so the state round-trips.
    """
    rv = np.asarray(r, dtype=np.float64).reshape(3)
    vv = np.asarray(v, dtype=np.float64).reshape(3)
    rn = float(np.linalg.norm(rv))
    h = np.cross(rv, vv)
    hn = float(np.linalg.norm(h))
    if hn == 0.0:
        raise ValueError("rectilinear orbit (zero angular momentum) is not supported")
    h_hat = h / hn
    e_vec = ((vv @ vv - mu / rn) * rv - (rv @ vv) * vv) / mu
    e = float(np.linalg.norm(e_vec))
    inc = float(np.arctan2(np.hypot(h[0], h[1]), h[2]))

    node = np.cross(np.array([0.0, 0.0, 1.0]), h)
    node_n = float(np.linalg.norm(node))
    if node_n > _EPS_I * hn:
        node_hat = node / node_n
        Omega = float(wrap_2pi(np.arctan2(node_hat[1], node_hat[0])))
    else:
        node_hat = np.array([1.0, 0.0, 0.0])
        Omega = 0.0
    in_plane = np.cross(h_hat, node_hat)

    if e > _EPS_E:
        peri_hat = e_vec / e
        omega = float(wrap_2pi(np.arctan2(e_vec @ in_plane, e_vec @ node_hat)))
    else:
        e = 0.0
        peri_hat = node_hat
        omega = 0.0
    nu = float(np.arctan2(rv @ np.cross(h_hat, peri_hat), rv @ peri_hat))
    q = hn * hn / (mu * (1.0 + e))
    t_p = t - float(time_since_perihelion(nu, q, e, mu))
    return ConicElements(q, e, inc, Omega, omega, t_p)


def specific_energy(r: ArrayLike, v: ArrayLike, mu: float = GM_SUN) -> FloatArray:
    ra = np.asarray(r, dtype=np.float64)
    va = np.asarray(v, dtype=np.float64)
    return 0.5 * np.sum(va * va, axis=-1) - mu / np.linalg.norm(ra, axis=-1)


def angular_momentum(r: ArrayLike, v: ArrayLike) -> FloatArray:
    return np.cross(np.asarray(r, dtype=np.float64), np.asarray(v, dtype=np.float64))
