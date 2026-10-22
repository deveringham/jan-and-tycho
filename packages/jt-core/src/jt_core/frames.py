# pyright: strict, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportMissingTypeStubs=false
"""Heliocentric ecliptic frame: Cartesian ↔ (λ, β), rotations, orbit orientation.

+x points to the reference direction ("vernal equinox"), +z to the ecliptic north pole.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from jt_core.angles import wrap_2pi

FloatArray = NDArray[np.float64]


def lonlat_to_unit(lon: ArrayLike, lat: ArrayLike) -> FloatArray:
    """Ecliptic longitude/latitude (radians) → unit vectors, shape (..., 3)."""
    lo = np.asarray(lon, dtype=np.float64)
    la = np.asarray(lat, dtype=np.float64)
    cb = np.cos(la)
    return np.stack([cb * np.cos(lo), cb * np.sin(lo), np.sin(la)], axis=-1)


def unit_to_lonlat(u: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Vectors (any length) → (λ ∈ [0, 2π), β ∈ [-π/2, π/2])."""
    ua = np.asarray(u, dtype=np.float64)
    x, y, z = ua[..., 0], ua[..., 1], ua[..., 2]
    lon = wrap_2pi(np.arctan2(y, x))
    lat = np.arctan2(z, np.hypot(x, y))
    return lon, lat


def normalize(v: ArrayLike) -> FloatArray:
    """Normalize vectors along the last axis."""
    va = np.asarray(v, dtype=np.float64)
    return va / np.linalg.norm(va, axis=-1, keepdims=True)


def rot_x(angle: float) -> FloatArray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_z(angle: float) -> FloatArray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def orbit_orientation(i: float, Omega: float, omega: float) -> FloatArray:
    """Perifocal → ecliptic rotation R = Rz(Ω) · Rx(i) · Rz(ω).

    Columns are the perihelion direction P̂, the in-plane normal Q̂ and the orbit pole Ŵ.
    """
    return rot_z(Omega) @ rot_x(i) @ rot_z(omega)
