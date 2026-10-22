"""Conic propagation and elements ↔ state (SPEC §10 Phase 1 acceptance, §11, §12)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jt_core.conics import (
    ConicElements,
    angular_momentum,
    conic_position,
    elements_to_state,
    specific_energy,
    state_to_elements,
    stumpff,
)
from jt_core.constants import GM_SUN

ECCENTRICITIES = [0.0, 0.5, 0.99, 1.0, 1.5, 5.0]


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)))


def test_stumpff_continuity() -> None:
    z = np.array([-1.0000001, -0.9999999, -1e-8, 0.0, 1e-8, 0.9999999, 1.0000001, 30.0, -30.0])
    c, s = (np.asarray(x) for x in stumpff(jnp.asarray(z)))
    assert np.isclose(c[3], 0.5) and np.isclose(s[3], 1 / 6)
    assert np.isclose(c[0], c[1], rtol=1e-6) and np.isclose(s[5], s[6], rtol=1e-6)
    sz = np.sqrt(30.0)
    assert np.isclose(c[7], (1 - np.cos(sz)) / 30) and np.isclose(s[8], (np.sinh(sz) - sz) / sz**3)


@pytest.mark.parametrize("e", ECCENTRICITIES)
@pytest.mark.parametrize("inc", [0.0, 0.4, np.pi])
def test_state_elements_state_roundtrip(e: float, inc: float) -> None:
    el = ConicElements(1.3, e, inc, 0.7, 2.1, 15.0)
    ts = np.array([15.0, 60.0, -250.0, 900.0, 4000.0])
    r, v = elements_to_state(el, ts)
    for k, t in enumerate(ts):
        el2 = state_to_elements(r[k], v[k], float(t))
        r2, v2 = elements_to_state(el2, [t])
        assert _rel(r2, r[k : k + 1]) < 1e-10
        assert _rel(v2, v[k : k + 1]) < 1e-10
        assert np.isclose(el2.q, el.q, rtol=1e-10)
        assert np.isclose(el2.e, e, atol=1e-10)


def test_elements_recovered_generic() -> None:
    el = ConicElements(0.9, 0.3, 0.5, 1.0, 4.0, 33.0)
    r, v = elements_to_state(el, [100.0])
    el2 = state_to_elements(r[0], v[0], 100.0)
    assert np.allclose(el2.as_array(), el.as_array(), rtol=1e-10, atol=1e-9)


@pytest.mark.parametrize("e", ECCENTRICITIES)
def test_conservation_over_100_periods(e: float) -> None:
    el = ConicElements(1.0, e, 0.2, 0.3, 0.4, 0.0)
    span = 100 * el.period() if e < 1 else 100 * 365.25
    t = np.linspace(-span / 2, span / 2, 4001)
    r, v = elements_to_state(el, t)
    en = specific_energy(r, v)
    h = angular_momentum(r, v)
    scale_e = GM_SUN / el.q  # energy scale, since E = 0 for parabolas
    assert np.max(np.abs(en - en[0])) / scale_e < 1e-10
    assert np.max(np.linalg.norm(h - h[0], axis=1)) / np.linalg.norm(h[0]) < 1e-10


def test_perihelion_geometry() -> None:
    el = ConicElements(0.7, 0.6, 0.0, 0.0, 0.0, 5.0)
    r, v = elements_to_state(el, [5.0])
    assert np.allclose(r[0], [0.7, 0, 0], atol=1e-14)
    assert np.isclose(v[0, 1], np.sqrt(GM_SUN * 1.6 / 0.7))


def test_period_closes_orbit() -> None:
    el = ConicElements(1.2, 0.4, 0.3, 1.0, 2.0, 0.0)
    p = el.period()
    r, _ = elements_to_state(el, [3.0, 3.0 + p, 3.0 + 7 * p])
    assert _rel(r[1:], r[:1].repeat(2, 0)) < 1e-12


@pytest.mark.parametrize("e", [1 - 1e-9, 1 - 1e-6, 1 + 1e-6, 1 + 1e-9])
def test_near_parabolic_continuity(e: float) -> None:
    t = np.array([-400.0, -20.0, 0.5, 30.0, 400.0])
    rp, _ = elements_to_state(ConicElements(1.0, 1.0, 0.1, 0.2, 0.3, 0.0), t)
    re, _ = elements_to_state(ConicElements(1.0, e, 0.1, 0.2, 0.3, 0.0), t)
    # Positions differ smoothly in e (scale ~ |e − 1| · Δt-dependent factor).
    assert _rel(re, rp) < 1e-2 * max(abs(e - 1) * 1e6, 1e-3)


def test_gradient_matches_finite_difference() -> None:
    p = jnp.array([1.3, 0.2, 0.1, 0.5, 1.0, 20.0])
    t = jnp.array([0.0, 150.0, 900.0])
    jac = np.asarray(jax.jacfwd(conic_position)(p, t))
    h = 1e-6
    for k in range(6):
        dp = jnp.zeros(6).at[k].set(h)
        fd = (np.asarray(conic_position(p + dp, t)) - np.asarray(conic_position(p - dp, t))) / (
            2 * h
        )
        assert np.allclose(jac[..., k], fd, rtol=1e-5, atol=1e-8), k


@given(
    st.floats(0.1, 30.0),
    st.floats(0.0, 3.0),
    st.floats(0.0, np.pi),
    st.floats(0.0, 2 * np.pi),
    st.floats(0.0, 2 * np.pi),
    st.floats(-2000.0, 2000.0),
)
@settings(max_examples=60, deadline=None)
def test_roundtrip_property(q: float, e: float, i: float, Om: float, om: float, t: float) -> None:
    el = ConicElements(q, e, i, Om, om, 0.0)
    r, v = elements_to_state(el, [t])
    if np.linalg.norm(r[0]) > 1e4:  # far hyperbolic branch: skip numerically extreme cases
        return
    el2 = state_to_elements(r[0], v[0], t)
    r2, v2 = elements_to_state(el2, [t])
    assert _rel(r2, r) < 1e-9
    assert _rel(v2, v) < 1e-9


def test_keplerian_conversion() -> None:
    el = ConicElements(1.381, 0.0934, 0.03, 0.87, 5.0, 120.0)
    k = el.to_keplerian(epoch=0.0)
    back = ConicElements.from_keplerian(**k, epoch=0.0)
    assert np.allclose(back.as_array()[:5], el.as_array()[:5], rtol=1e-12)
    dtp = (back.t_p - el.t_p) / el.period()
    assert np.isclose(dtp, round(dtp), atol=1e-12)  # t_p is defined modulo the period
    assert np.isclose(k["a"], 1.381 / (1 - 0.0934))


def test_lagrange_fg_matches_propagation() -> None:
    from jt_core.conics import lagrange_fg

    el = ConicElements(1.2, 0.3, 0.2, 0.5, 1.0, 10.0)
    r, v = elements_to_state(el, [100.0, 160.0, 40.0])
    for k, dt in ((1, 60.0), (2, -60.0)):
        f, g = lagrange_fg(r[0], v[0], dt)
        assert np.allclose(f * r[0] + g * v[0], r[k], rtol=1e-11, atol=1e-13)
