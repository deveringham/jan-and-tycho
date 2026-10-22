# pyright: strict, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportMissingTypeStubs=false
"""Angle wrapping and great-circle geometry.

Always compare directions with these helpers, never with raw longitude differences
(SPEC §4, §12: wraparound bugs).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

TWO_PI = 2.0 * np.pi


def wrap_2pi(x: ArrayLike) -> FloatArray:
    """Wrap angles into [0, 2π)."""
    y = np.mod(np.asarray(x, dtype=np.float64), TWO_PI)
    # np.mod can return exactly 2π for tiny negative inputs.
    return np.where(y >= TWO_PI, 0.0, y)


def wrap_pi(x: ArrayLike) -> FloatArray:
    """Wrap angles into [-π, π)."""
    return wrap_2pi(np.asarray(x, dtype=np.float64) + np.pi) - np.pi


def angle_diff(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """Signed difference a − b wrapped into [-π, π)."""
    return wrap_pi(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))


def great_circle_separation(u: ArrayLike, v: ArrayLike) -> FloatArray:
    """Angle (radians) between unit vectors along the last axis.

    Uses atan2(|u×v|, u·v), which is accurate for both tiny and near-antipodal
    separations (unlike arccos of the dot product).
    """
    ua = np.asarray(u, dtype=np.float64)
    va = np.asarray(v, dtype=np.float64)
    cross = np.linalg.norm(np.cross(ua, va), axis=-1)
    dot = np.sum(ua * va, axis=-1)
    return np.arctan2(cross, dot)


def tangent_basis(u: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Local (east, north) unit vectors in the tangent plane at direction(s) u.

    "East" is the direction of increasing ecliptic longitude, "north" of increasing
    latitude. At the poles, east is taken along +y (an arbitrary but continuous choice).
    """
    ua = np.asarray(u, dtype=np.float64)
    z = np.zeros_like(ua)
    z[..., 2] = 1.0
    east = np.cross(z, ua)
    norm = np.linalg.norm(east, axis=-1, keepdims=True)
    polar = norm[..., 0] < 1e-15
    east = np.where(norm > 1e-15, east / np.where(norm > 0, norm, 1.0), 0.0)
    if np.any(polar):
        east[polar] = np.array([0.0, 1.0, 0.0])
    north = np.cross(ua, east)
    return east, north


def tangent_residuals(u_obs: ArrayLike, u_model: ArrayLike) -> FloatArray:
    """Residuals (east, north) in radians of observed vs model directions.

    Projects u_obs onto the tangent plane at u_model with a gnomonic projection, so the
    result is exact for the direction and accurate to second order in magnitude for
    small separations. Shape (..., 2).
    """
    uo = np.asarray(u_obs, dtype=np.float64)
    um = np.asarray(u_model, dtype=np.float64)
    east, north = tangent_basis(um)
    d = np.sum(uo * um, axis=-1)
    return np.stack([np.sum(uo * east, axis=-1) / d, np.sum(uo * north, axis=-1) / d], axis=-1)
