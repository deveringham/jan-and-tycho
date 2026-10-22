# pyright: strict, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportMissingTypeStubs=false
"""Angle unit helpers. Internally everything is radians (SPEC §4)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from jt_core.constants import ARCSEC, DEG

FloatArray = NDArray[np.float64]


def deg_to_rad(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64) * DEG


def rad_to_deg(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64) / DEG


def arcsec_to_rad(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64) * ARCSEC


def rad_to_arcsec(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64) / ARCSEC
