# pyright: strict, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportMissingTypeStubs=false
"""Observational realism (SPEC Phase 5): light-time, stellar aberration, equatorial frame.

Geometric direction → apparent direction, in this order:

1. Light-time: the body is seen where it was when the light left it, at t − τ with
   τ = |r_body(t − τ) − r_obs(t)| / c (fixed-point iteration).
2. Aberration: the observer's velocity tilts the apparent direction towards its motion,
   u' ∝ u + v_obs / c (classical, first order in v/c; the neglected O(v²/c²) term is
   ~2 mas for the Earth). `unaberrate` is the exact inverse of `aberrate`.
3. Frame: ecliptic (λ, β) or equatorial (α, δ) obtained by a rotation about +x by the
   obliquity ε.

The same functions serve Tycho (making the data), Jan (reducing it) and Rudolf (sealed
apparent directions), so all three agree on the conventions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field

from jt_core.constants import AU_KM, DAY_SECONDS
from jt_core.frames import normalize, rot_x

FloatArray = NDArray[np.float64]

C_AU_PER_DAY: float = 299_792.458 * DAY_SECONDS / AU_KM
"""Speed of light (AU/day)."""
OBLIQUITY_J2000_DEG: float = 23.439291111
"""Mean obliquity of the ecliptic at J2000 (IAU 1980: 23° 26′ 21.448″)."""


class Realism(BaseModel):
    """Which effects the published directions contain, and in which frame. Published to
    Jan as an observing convention (a catalogue states what it reports)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    light_time: bool = False
    aberration: bool = False
    frame: Literal["ecliptic", "equatorial"] = "ecliptic"
    obliquity_deg: float = Field(default=OBLIQUITY_J2000_DEG, ge=0.0, le=90.0)

    @property
    def any(self) -> bool:
        return self.light_time or self.aberration or self.frame != "ecliptic"

    @property
    def physical(self) -> bool:
        """Effects that depend on the geometry (not just a change of frame)."""
        return self.light_time or self.aberration


def ecliptic_to_frame(u: ArrayLike, realism: Realism) -> FloatArray:
    """Rotate ecliptic vectors (..., 3) into the published frame."""
    arr = np.asarray(u, dtype=np.float64)
    if realism.frame == "ecliptic":
        return arr
    return arr @ rot_x(np.radians(realism.obliquity_deg)).T


def frame_to_ecliptic(u: ArrayLike, realism: Realism) -> FloatArray:
    """Inverse of `ecliptic_to_frame`."""
    arr = np.asarray(u, dtype=np.float64)
    if realism.frame == "ecliptic":
        return arr
    return arr @ rot_x(np.radians(realism.obliquity_deg))


def aberrate(u: ArrayLike, v_obs: ArrayLike) -> FloatArray:
    """Apparent direction u' = (u + β)/|u + β|, β = v_obs / c. Shapes (..., 3)."""
    beta = np.asarray(v_obs, dtype=np.float64) / C_AU_PER_DAY
    return normalize(np.asarray(u, dtype=np.float64) + beta)


def unaberrate(u_app: ArrayLike, v_obs: ArrayLike) -> FloatArray:
    """Exact inverse of `aberrate`: the unit u with u + β ∥ u_app."""
    ua = np.asarray(u_app, dtype=np.float64)
    beta = np.asarray(v_obs, dtype=np.float64) / C_AU_PER_DAY
    ub = np.sum(ua * beta, axis=-1, keepdims=True)
    s = ub + np.sqrt(ub**2 - np.sum(beta**2, axis=-1, keepdims=True) + 1.0)
    return normalize(s * ua - beta)


def retarded_positions(
    position: Callable[[FloatArray], FloatArray],
    t: ArrayLike,
    r_obs: ArrayLike,
    iterations: int = 3,
) -> tuple[FloatArray, FloatArray]:
    """Body positions at the emission times t − τ and the light-times τ (days).

    Each iteration reduces the error by a factor ~ v/c, so three are exact to round-off
    for anything slower than light.
    """
    tt = np.asarray(t, dtype=np.float64)
    ro = np.asarray(r_obs, dtype=np.float64)
    tau = np.zeros_like(tt)
    r = position(tt)
    for _ in range(iterations):
        tau = np.linalg.norm(r - ro, axis=-1) / C_AU_PER_DAY
        r = position(tt - tau)
    return r, tau


def apparent_directions(
    r_body: ArrayLike, r_obs: ArrayLike, v_obs: ArrayLike, realism: Realism
) -> FloatArray:
    """Published unit vectors from (already retarded, if light-time applies) body
    positions, the observer's positions and velocities at reception."""
    u = normalize(np.asarray(r_body, dtype=np.float64) - np.asarray(r_obs, dtype=np.float64))
    if realism.aberration:
        u = aberrate(u, v_obs)
    return ecliptic_to_frame(u, realism)
