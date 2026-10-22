"""Ptolemaic truth: deferent (optionally eccentric with an equant) plus epicycles."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from jt_core.constants import DEG
from jt_core.frames import rot_x, rot_z
from tycho.scenario import EpicycleTruthCfg

FloatArray = NDArray[np.float64]


def equant_circle_point(
    angle: FloatArray, radius: float, center_offset: float, equant_offset: float, apse: float
) -> FloatArray:
    """In-plane point on a circle of `radius` whose centre sits `center_offset` from the
    origin (opposite the apse direction), moving so that it is seen at `angle` from an
    equant point `equant_offset` from the origin on the same line. Shape (N, 2)."""
    p_hat = np.array([np.cos(apse), np.sin(apse)])
    c = -center_offset * p_hat
    e = -equant_offset * p_hat
    u = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    d = e - c
    ud = u @ d
    rho = -ud + np.sqrt(ud**2 - d @ d + radius**2)
    return e + rho[:, None] * u


class EpicycleTruth:
    kind: Literal["dynamical", "kinematic"] = "kinematic"

    def __init__(self, cfg: EpicycleTruthCfg, center: Callable[[FloatArray], FloatArray]) -> None:
        self.cfg = cfg
        self.center = center
        self.rot = rot_z(cfg.Omega_deg * DEG) @ rot_x(cfg.i_deg * DEG)

    def _in_plane(self, t: FloatArray) -> FloatArray:
        t = np.asarray(t, dtype=np.float64)
        out = np.zeros((t.size, 2))
        for k, c in enumerate(self.cfg.circles):
            ang = 2 * np.pi * t / c.period_days + c.phase_deg * DEG
            if k == 0 and self.cfg.eccentric is not None:
                ec = self.cfg.eccentric
                out += equant_circle_point(
                    ang, c.radius_au, ec.center_offset_au, ec.equant_offset_au, ec.angle_deg * DEG
                )
            else:
                out += c.radius_au * np.stack([np.cos(ang), np.sin(ang)], axis=-1)
        return out

    def position(self, t: FloatArray) -> FloatArray:
        p = self._in_plane(t)
        p3 = np.concatenate([p, np.zeros((p.shape[0], 1))], axis=1) @ self.rot.T
        return self.center(np.asarray(t, dtype=np.float64)) + p3

    def velocity(self, t: FloatArray) -> FloatArray:
        h = 1e-3
        t = np.asarray(t, dtype=np.float64)
        return (self.position(t + h) - self.position(t - h)) / (2 * h)

    def describe(self) -> dict[str, Any]:
        return {
            "type": "epicycle",
            "kind": self.kind,
            "family": "epicycles",
            "params": self.cfg.model_dump(mode="json"),
            "elements": None,
            "law": None,
        }
