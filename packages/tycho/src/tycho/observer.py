"""Observer ("Earth") models (SPEC §5.3)."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from tycho.scenario import (
    FixedObserverCfg,
    HeliocentricObserverCfg,
    KeplerianObserverCfg,
    ObserverCfg,
    TruthObserverCfg,
)
from tycho.truth.keplerian import KeplerianTruth

FloatArray = NDArray[np.float64]


class Observer(Protocol):
    kind: str

    def position(self, t: FloatArray) -> FloatArray: ...

    def velocity(self, t: FloatArray) -> FloatArray: ...

    def describe(self) -> dict[str, Any]: ...


class KeplerianObserver:
    kind = "keplerian"

    def __init__(self, cfg: KeplerianObserverCfg) -> None:
        self.truth = KeplerianTruth(cfg.elements.to_elements(), cfg.gm)

    def position(self, t: FloatArray) -> FloatArray:
        return self.truth.position(t)

    def velocity(self, t: FloatArray) -> FloatArray:
        return self.truth.velocity(t)

    def describe(self) -> dict[str, Any]:
        return {"type": self.kind, **self.truth.describe()}


class FixedObserver:
    kind = "fixed"

    def __init__(self, position: tuple[float, float, float]) -> None:
        self.r = np.asarray(position, dtype=np.float64)

    def position(self, t: FloatArray) -> FloatArray:
        return np.broadcast_to(self.r, (np.size(t), 3)).copy()

    def velocity(self, t: FloatArray) -> FloatArray:
        return np.zeros((np.size(t), 3))

    def describe(self) -> dict[str, Any]:
        return {"type": self.kind, "position_au": self.r.tolist()}


class TruthObserver:
    """Any truth model used as the observer's trajectory (non-Keplerian observer)."""

    kind = "truth"

    def __init__(self, cfg: TruthObserverCfg) -> None:
        from tycho.truth import build_truth

        self.model = build_truth(cfg.truth, lambda t: np.zeros((np.size(t), 3)))

    def position(self, t: FloatArray) -> FloatArray:
        return self.model.position(t)

    def velocity(self, t: FloatArray) -> FloatArray:
        return self.model.velocity(t)

    def describe(self) -> dict[str, Any]:
        return {"type": self.kind, **self.model.describe()}


def build_observer(cfg: ObserverCfg) -> Observer:
    if isinstance(cfg, KeplerianObserverCfg):
        return KeplerianObserver(cfg)
    if isinstance(cfg, FixedObserverCfg):
        return FixedObserver(cfg.position_au)
    if isinstance(cfg, TruthObserverCfg):
        return TruthObserver(cfg)
    assert isinstance(cfg, HeliocentricObserverCfg)
    obs = FixedObserver((0.0, 0.0, 0.0))
    obs.kind = "heliocentric"
    return obs
