"""Composable noise models (SPEC §5.7), applied in order to unit vectors / (λ, β) / times.

Each model draws from its own named RNG sub-stream, so adding or reordering components
does not reshuffle the others' realizations. Stochastic directional errors are applied in
the tangent plane with an exact great-circle step; biases, drifts and quantization act on
the (λ, β) coordinates as an instrument would; `clock` errors only affect the *published*
times (the body is observed at the true time).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jt_core import rng as jrng
from jt_core.angles import tangent_basis, wrap_2pi
from jt_core.constants import ARCSEC, YEAR_DAYS
from jt_core.frames import lonlat_to_unit, unit_to_lonlat
from tycho.scenario import (
    Ar1NoiseCfg,
    BiasNoiseCfg,
    ClockNoiseCfg,
    DriftNoiseCfg,
    GaussianNoiseCfg,
    HeteroscedasticNoiseCfg,
    NoiseCfg,
    OutlierNoiseCfg,
    PositionGaussianNoiseCfg,
    QuantizeNoiseCfg,
)

FloatArray = NDArray[np.float64]


@dataclass
class SkyNoiseResult:
    lon: FloatArray
    lat: FloatArray
    t_pub: FloatArray
    """Published times (true times plus any clock error)."""
    outlier: NDArray[np.bool_]
    sigma: FloatArray
    """Per-observation 1σ (radians, per axis) of the stochastic, non-outlier noise."""


def _exp_map(u: FloatArray, de: FloatArray, dn: FloatArray) -> FloatArray:
    """Move u by tangent offset (de, dn) along a great circle (exact angular distance)."""
    east, north = tangent_basis(u)
    rho = np.hypot(de, dn)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(rho > 0, np.sin(rho) / np.where(rho > 0, rho, 1.0), 1.0)
    v = np.cos(rho)[:, None] * u + k[:, None] * (de[:, None] * east + dn[:, None] * north)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _shift_lonlat(u: FloatArray, dlon: FloatArray, dlat: FloatArray) -> FloatArray:
    lon, lat = unit_to_lonlat(u)
    lat = np.clip(lat + dlat, -np.pi / 2, np.pi / 2)
    return lonlat_to_unit(lon + dlon, lat)


def _ar1(t: FloatArray, sigma: float, rho: float, g: np.random.Generator) -> FloatArray:
    """Stationary AR(1) sampled at irregular times: corr(Δt) = rho ** Δt."""
    x = np.empty(t.size)
    if t.size == 0:
        return x
    order = np.argsort(t, kind="stable")
    ts = t[order]
    xs = np.empty(t.size)
    xs[0] = g.normal(0, sigma)
    for k in range(1, t.size):
        a = rho ** max(ts[k] - ts[k - 1], 0.0)
        xs[k] = a * xs[k - 1] + np.sqrt(max(1 - a * a, 0.0)) * g.normal(0, sigma)
    x[order] = xs
    return x


def hetero_sigma(
    m: HeteroscedasticNoiseCfg, magnitude: FloatArray | None, elongation: FloatArray | None, n: int
) -> FloatArray:
    s = np.full(n, m.sigma_arcsec * ARCSEC)
    if magnitude is not None and m.per_magnitude:
        mag = np.nan_to_num(magnitude, nan=m.mag_ref)
        s = s * (1 + m.per_magnitude * np.maximum(mag - m.mag_ref, 0.0))
    if elongation is not None and m.low_elongation:
        el = np.nan_to_num(elongation, nan=np.pi / 2)
        s = s * (1 + m.low_elongation * np.maximum(np.pi / 2 - el, 0.0) / (np.pi / 2))
    return s


def apply_sky_noise(
    u_true: FloatArray,
    t: FloatArray,
    models: list[NoiseCfg],
    seed: int,
    stream: str,
    magnitude: FloatArray | None = None,
    elongation: FloatArray | None = None,
) -> SkyNoiseResult:
    """Apply sky noise to true directions observed at true times `t`.

    `stream` distinguishes e.g. bodies; `magnitude` and `elongation` (radians) feed the
    heteroscedastic model.
    """
    u = u_true.copy()
    n = u.shape[0]
    t_pub = np.asarray(t, dtype=np.float64).copy()
    outlier = np.zeros(n, dtype=bool)
    var = np.zeros(n)
    for k, m in enumerate(models):
        g = jrng.generator(seed, "noise", stream, f"{k}:{m.type}")
        if isinstance(m, GaussianNoiseCfg):
            s = m.sigma_arcsec * ARCSEC
            u = _exp_map(u, g.normal(0, s, n), g.normal(0, s, n))
            var += s**2
        elif isinstance(m, HeteroscedasticNoiseCfg):
            s = hetero_sigma(m, magnitude, elongation, n)
            u = _exp_map(u, g.normal(0, 1, n) * s, g.normal(0, 1, n) * s)
            var += s**2
        elif isinstance(m, Ar1NoiseCfg):
            s = m.sigma_arcsec * ARCSEC
            u = _exp_map(u, _ar1(t, s, m.rho_per_day, g), _ar1(t, s, m.rho_per_day, g))
            var += s**2
        elif isinstance(m, OutlierNoiseCfg):
            hit = g.uniform(size=n) < m.fraction
            s = m.scale_arcsec * ARCSEC
            de = np.where(hit, g.normal(0, s, n), 0.0)
            dn = np.where(hit, g.normal(0, s, n), 0.0)
            u = _exp_map(u, de, dn)
            outlier |= hit
        elif isinstance(m, BiasNoiseCfg):
            u = _shift_lonlat(
                u, np.full(n, m.lon_arcsec * ARCSEC), np.full(n, m.lat_arcsec * ARCSEC)
            )
        elif isinstance(m, DriftNoiseCfg):
            yrs = np.asarray(t) / YEAR_DAYS
            u = _shift_lonlat(
                u, m.lon_arcsec_per_year * ARCSEC * yrs, m.lat_arcsec_per_year * ARCSEC * yrs
            )
        elif isinstance(m, QuantizeNoiseCfg):
            step = m.step_arcsec * ARCSEC
            lon, lat = unit_to_lonlat(u)
            u = lonlat_to_unit(np.round(lon / step) * step, np.round(lat / step) * step)
            var += step**2 / 12.0
        elif isinstance(m, ClockNoiseCfg):
            t_pub = t_pub + m.offset_days + g.normal(0, m.sigma_days, n)
        else:
            raise TypeError(f"noise model {m.type!r} does not apply to sky observations")
    lon, lat = unit_to_lonlat(u)
    return SkyNoiseResult(wrap_2pi(lon), lat, t_pub, outlier, np.sqrt(var))


def apply_position_noise(
    r_true: FloatArray, models: list[NoiseCfg], seed: int, stream: str
) -> FloatArray:
    r = r_true.copy()
    for k, m in enumerate(models):
        if not isinstance(m, PositionGaussianNoiseCfg):
            raise TypeError(f"noise model {m.type!r} does not apply to T0 positions")
        g = jrng.generator(seed, "noise", stream, f"{k}:{m.type}")
        r = r + g.normal(0.0, m.sigma_au, r.shape)
    return r


def nominal_sigma(models: list[NoiseCfg]) -> float:
    """Typical per-axis 1σ (radians) of the stochastic, non-outlier sky noise."""
    var = 0.0
    for m in models:
        if isinstance(m, GaussianNoiseCfg | Ar1NoiseCfg | HeteroscedasticNoiseCfg):
            var += (m.sigma_arcsec * ARCSEC) ** 2
        elif isinstance(m, QuantizeNoiseCfg):
            var += (m.step_arcsec * ARCSEC) ** 2 / 12.0
    return float(np.sqrt(var))
