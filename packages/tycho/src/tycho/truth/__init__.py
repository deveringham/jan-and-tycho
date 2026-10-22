"""Truth models (SPEC §5.2)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from tycho.scenario import (
    CentralForceTruthCfg,
    CircularTruthCfg,
    EpicycleTruthCfg,
    KeplerianTruthCfg,
    KinematicParametricCfg,
    KinematicPolygonCfg,
    NBodyTruthCfg,
    TruthCfg,
)
from tycho.truth.central_force import CentralForceTruth
from tycho.truth.epicycle import EpicycleTruth
from tycho.truth.keplerian import KeplerianTruth
from tycho.truth.kinematic import KinematicParametricTruth, KinematicPolygonTruth
from tycho.truth.nbody import NBodyTruth

FloatArray = NDArray[np.float64]


class TruthModel(Protocol):
    kind: Literal["dynamical", "kinematic"]

    def position(self, t: FloatArray) -> FloatArray: ...

    def velocity(self, t: FloatArray) -> FloatArray: ...

    def describe(self) -> dict[str, Any]: ...


def build_truth(cfg: TruthCfg, observer: Callable[[FloatArray], FloatArray]) -> TruthModel:
    """Truth model for a body. `observer` gives positions for observer-centred truths."""
    if isinstance(cfg, KeplerianTruthCfg):
        return KeplerianTruth(cfg.elements.to_elements(), cfg.gm, family=None)
    if isinstance(cfg, CircularTruthCfg):
        return KeplerianTruth(cfg.to_elements(), cfg.gm, family="circle")
    if isinstance(cfg, CentralForceTruthCfg):
        return CentralForceTruth(cfg)
    if isinstance(cfg, NBodyTruthCfg):
        return NBodyTruth(cfg)
    if isinstance(cfg, KinematicPolygonCfg):
        return KinematicPolygonTruth(cfg)
    if isinstance(cfg, KinematicParametricCfg):
        return KinematicParametricTruth(cfg)
    assert isinstance(cfg, EpicycleTruthCfg)
    if cfg.center == "observer":
        return EpicycleTruth(cfg, observer)
    return EpicycleTruth(cfg, lambda t: np.zeros((np.size(t), 3)))


__all__ = [
    "CentralForceTruth",
    "EpicycleTruth",
    "KeplerianTruth",
    "KinematicParametricTruth",
    "KinematicPolygonTruth",
    "NBodyTruth",
    "TruthModel",
    "build_truth",
]
