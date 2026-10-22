"""Unit and property tests for jt_core frames, angles, units and rng (SPEC §11)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from jt_core import rng
from jt_core.angles import (
    angle_diff,
    great_circle_separation,
    tangent_basis,
    tangent_residuals,
    wrap_2pi,
    wrap_pi,
)
from jt_core.constants import ARCSEC, GM_SUN, K_GAUSS
from jt_core.frames import lonlat_to_unit, orbit_orientation, unit_to_lonlat
from jt_core.units import arcsec_to_rad, deg_to_rad, rad_to_arcsec, rad_to_deg

finite = st.floats(-1e4, 1e4, allow_nan=False)
lons = st.floats(0.0, 2 * np.pi, exclude_max=True)
lats = st.floats(-np.pi / 2 + 1e-9, np.pi / 2 - 1e-9)


def test_constants() -> None:
    assert GM_SUN == K_GAUSS**2
    assert np.isclose(ARCSEC * 206264.80624709636, 1.0)


def test_units_roundtrip() -> None:
    x = np.array([0.0, 1.0, -3.5, 1e5])
    assert np.allclose(rad_to_deg(deg_to_rad(x)), x)
    assert np.allclose(rad_to_arcsec(arcsec_to_rad(x)), x)


@given(finite)
def test_wrap_ranges(x: float) -> None:
    w2 = float(wrap_2pi(x))
    wp = float(wrap_pi(x))
    assert 0.0 <= w2 < 2 * np.pi
    assert -np.pi <= wp < np.pi
    assert np.isclose(np.cos(w2), np.cos(x), atol=1e-9)
    assert np.isclose(np.sin(wp), np.sin(x), atol=1e-9)


def test_angle_diff_wraparound() -> None:
    assert np.isclose(float(angle_diff(0.01, 2 * np.pi - 0.01)), 0.02)


@given(lons, lats)
def test_lonlat_roundtrip(lon: float, lat: float) -> None:
    u = lonlat_to_unit(lon, lat)
    assert np.isclose(np.linalg.norm(u), 1.0)
    lon2, lat2 = unit_to_lonlat(u)
    assert abs(float(angle_diff(lon2, lon))) < 1e-9
    assert abs(float(lat2) - lat) < 1e-12


@given(lons, lats, st.floats(1e-9, 1.0))
def test_great_circle_small_and_large(lon: float, lat: float, d: float) -> None:
    u = lonlat_to_unit(lon, lat)
    east, _ = tangent_basis(u)
    v = np.cos(d) * u + np.sin(d) * east
    assert np.isclose(float(great_circle_separation(u, v)), d, rtol=1e-9, atol=1e-15)


def test_great_circle_antipodal() -> None:
    u = np.array([1.0, 0.0, 0.0])
    assert np.isclose(float(great_circle_separation(u, -u)), np.pi)


@given(lons, lats)
@settings(max_examples=50)
def test_tangent_basis_orthonormal(lon: float, lat: float) -> None:
    u = lonlat_to_unit(lon, lat)
    e, n = tangent_basis(u)
    m = np.stack([u, e, n])
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)
    # East is the direction of increasing longitude.
    du = lonlat_to_unit(lon + 1e-6, lat) - u
    assert float(du @ e) > 0


def test_tangent_basis_pole() -> None:
    e, n = tangent_basis(np.array([[0.0, 0.0, 1.0]]))
    assert np.allclose(np.linalg.norm(e), 1.0) and np.allclose(np.linalg.norm(n), 1.0)


def test_tangent_residuals() -> None:
    u = lonlat_to_unit(1.0, 0.2)
    e, n = tangent_basis(u)
    eps = 3e-5
    v = u + eps * e - 2 * eps * n
    v /= np.linalg.norm(v)
    res = tangent_residuals(v, u)
    assert np.allclose(res, [eps, -2 * eps], rtol=1e-6)


def test_orbit_orientation_is_rotation() -> None:
    r = orbit_orientation(0.4, 1.1, 2.3)
    assert np.allclose(r @ r.T, np.eye(3))
    assert np.isclose(np.linalg.det(r), 1.0)


def test_rng_streams_are_named_and_deterministic() -> None:
    a1 = rng.generator(7, "noise").normal(size=5)
    a2 = rng.generator(7, "noise").normal(size=5)
    b = rng.generator(7, "schedule").normal(size=5)
    c = rng.generator(8, "noise").normal(size=5)
    assert np.array_equal(a1, a2)
    assert not np.allclose(a1, b)
    assert not np.allclose(a1, c)
    assert rng.derive_int(1, "x") == rng.derive_int(1, "x") != rng.derive_int(1, "y")
    assert rng.derive_int(1, "a", "b") != rng.derive_int(1, "b", "a")


def test_jax_x64_enabled() -> None:
    import jax.numpy as jnp

    import jt_core  # noqa: F401

    assert jnp.asarray(1.0).dtype == jnp.float64
