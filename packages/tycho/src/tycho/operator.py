"""Observation operator (SPEC §5.4): geometry → observed quantities.

Δ (geocentric distance), r (heliocentric distance), solar elongation and magnitude are
computed for visibility but never published.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jt_core.angles import great_circle_separation
from jt_core.frames import unit_to_lonlat
from tycho.scenario import PhotometryCfg

FloatArray = NDArray[np.float64]


@dataclass
class Geometry:
    u: FloatArray
    """(N, 3) geocentric unit vectors."""
    lon: FloatArray
    lat: FloatArray
    delta: FloatArray
    """Geocentric distance (AU)."""
    r: FloatArray
    """Heliocentric distance (AU)."""
    elongation: FloatArray
    """Sun–observer–body angle (rad); NaN for an observer at the Sun."""
    magnitude: FloatArray
    """Apparent magnitude; NaN without photometry."""


def observe(
    r_body: FloatArray, r_obs: FloatArray, photometry: PhotometryCfg | None = None
) -> Geometry:
    d = r_body - r_obs
    delta = np.linalg.norm(d, axis=1)
    u = d / delta[:, None]
    lon, lat = unit_to_lonlat(u)
    r = np.linalg.norm(r_body, axis=1)
    obs_dist = np.linalg.norm(r_obs, axis=1)
    to_sun = np.where(
        obs_dist[:, None] > 0, -r_obs / np.where(obs_dist > 0, obs_dist, 1)[:, None], 0
    )
    elong = np.where(obs_dist > 0, great_circle_separation(to_sun, u), np.nan)
    if photometry is None:
        mag = np.full_like(r, np.nan)
    elif photometry.kappa is None:
        mag = photometry.H + 5.0 * np.log10(r * delta)
    else:
        mag = photometry.H + 5.0 * np.log10(delta) + 2.5 * photometry.kappa * np.log10(r)
    return Geometry(u, lon, lat, delta, r, elong, mag)
