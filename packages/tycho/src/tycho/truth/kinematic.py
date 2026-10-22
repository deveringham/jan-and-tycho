"""Kinematic (non-physical) periodic truths: polygons and parametric curves. No force law."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from jt_core.constants import DEG
from jt_core.frames import orbit_orientation
from tycho.scenario import KinematicParametricCfg, KinematicPolygonCfg

FloatArray = NDArray[np.float64]


def _to_3d(xy: FloatArray, i: float, Omega: float, omega: float) -> FloatArray:
    rot = orbit_orientation(i * DEG, Omega * DEG, omega * DEG)
    return np.concatenate([xy, np.zeros((len(xy), 1))], axis=1) @ rot.T


class _Kinematic:
    kind: Literal["dynamical", "kinematic"] = "kinematic"

    def position(self, t: FloatArray) -> FloatArray:
        raise NotImplementedError

    def velocity(self, t: FloatArray) -> FloatArray:
        h = 1e-3
        t = np.asarray(t, dtype=np.float64)
        return (self.position(t + h) - self.position(t - h)) / (2 * h)


class KinematicPolygonTruth(_Kinematic):
    def __init__(self, cfg: KinematicPolygonCfg) -> None:
        self.cfg = cfg
        v = np.asarray(cfg.vertices_au, dtype=np.float64) + np.asarray(cfg.center_au)
        self.v = v
        nxt = np.roll(v, -1, axis=0)
        seg = np.linalg.norm(nxt - v, axis=1)
        if cfg.speed_profile == "constant_speed":
            w = seg
        else:  # equal areas swept about the Sun (origin) per unit time
            w = 0.5 * np.abs(v[:, 0] * nxt[:, 1] - v[:, 1] * nxt[:, 0])
        self.cum = np.concatenate([[0.0], np.cumsum(w)]) / np.sum(w)
        self.nxt = nxt

    def _in_plane(self, t: FloatArray) -> FloatArray:
        s = np.mod(np.asarray(t, dtype=np.float64) / self.cfg.period_days + self.cfg.phase, 1.0)
        k = np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.v) - 1)
        # Both profiles are linear within an edge: the area swept from the Sun along a
        # straight segment grows linearly with the fraction travelled.
        frac = (s - self.cum[k]) / np.maximum(self.cum[k + 1] - self.cum[k], 1e-300)
        return self.v[k] + frac[:, None] * (self.nxt[k] - self.v[k])

    def position(self, t: FloatArray) -> FloatArray:
        c = self.cfg
        return _to_3d(self._in_plane(np.atleast_1d(t)), c.i_deg, c.Omega_deg, c.omega_deg)

    def vertices_3d(self) -> FloatArray:
        c = self.cfg
        return _to_3d(self.v, c.i_deg, c.Omega_deg, c.omega_deg)

    def describe(self) -> dict[str, Any]:
        return {
            "type": "kinematic_polygon",
            "kind": self.kind,
            "family": "kinematic-polygon",
            "elements": None,
            "law": None,
            "params": self.cfg.model_dump(mode="json"),
            "vertices_3d": self.vertices_3d().tolist(),
        }


class KinematicParametricTruth(_Kinematic):
    def __init__(self, cfg: KinematicParametricCfg) -> None:
        self.cfg = cfg
        if cfg.curve == "custom":
            assert cfg.expressions is not None
            phi = sp.Symbol("phi")
            self._f = [sp.lambdify(phi, sp.sympify(e), "numpy") for e in cfg.expressions]

    def _curve(self, phi: FloatArray) -> FloatArray:
        c = self.cfg
        if c.curve == "lissajous":
            x = c.a_au * np.cos(c.p * phi + c.delta_deg * DEG)
            y = c.b_au * np.sin(c.q * phi)
            z = np.zeros_like(phi)
        elif c.curve == "figure_eight":  # lemniscate of Bernoulli
            d = 1 + np.sin(phi) ** 2
            x = c.a_au * np.cos(phi) / d
            y = c.a_au * np.sin(phi) * np.cos(phi) / d
            z = np.zeros_like(phi)
        else:
            x, y, z = (np.asarray(f(phi), dtype=np.float64) * np.ones_like(phi) for f in self._f)
        return np.stack([x + c.center_au[0], y + c.center_au[1], z], axis=1)

    def position(self, t: FloatArray) -> FloatArray:
        c = self.cfg
        phi = (
            2 * np.pi * np.atleast_1d(np.asarray(t, dtype=np.float64)) / c.period_days
            + c.phase_deg * DEG
        )
        p = self._curve(phi)
        rot = orbit_orientation(c.i_deg * DEG, c.Omega_deg * DEG, c.omega_deg * DEG)
        return p @ rot.T

    def describe(self) -> dict[str, Any]:
        return {
            "type": "kinematic_parametric",
            "kind": self.kind,
            "family": f"kinematic-{self.cfg.curve}",
            "elements": None,
            "law": None,
            "params": self.cfg.model_dump(mode="json"),
        }
