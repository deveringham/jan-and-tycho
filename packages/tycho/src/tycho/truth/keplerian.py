"""Two-body conic truth (also used for `circular`)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from jt_core.conics import ConicElements, elements_to_state

FloatArray = NDArray[np.float64]

NEWTON_LAW = [
    "-GM*x/(x**2 + y**2 + z**2)**(3/2)",
    "-GM*y/(x**2 + y**2 + z**2)**(3/2)",
    "-GM*z/(x**2 + y**2 + z**2)**(3/2)",
]


def conic_family(e: float) -> str:
    if e == 0.0:
        return "circle"
    if e < 1.0:
        return "conic-elliptic"
    if e == 1.0:
        return "conic-parabolic"
    return "conic-hyperbolic"


class KeplerianTruth:
    kind: Literal["dynamical", "kinematic"] = "dynamical"

    def __init__(self, elements: ConicElements, gm: float, family: str | None = None) -> None:
        self.elements = elements
        self.gm = gm
        self.family = family or conic_family(elements.e)

    def state(self, t: FloatArray) -> tuple[FloatArray, FloatArray]:
        return elements_to_state(self.elements, np.asarray(t, dtype=np.float64), self.gm)

    def position(self, t: FloatArray) -> FloatArray:
        return self.state(t)[0]

    def velocity(self, t: FloatArray) -> FloatArray:
        return self.state(t)[1]

    def acceleration(self, x: FloatArray, t: FloatArray) -> FloatArray:
        """The law's acceleration at arbitrary points (AU/day²)."""
        r = np.linalg.norm(x, axis=1, keepdims=True)
        return -self.gm * x / r**3

    def describe(self) -> dict[str, Any]:
        el = self.elements
        return {
            "type": "keplerian",
            "kind": self.kind,
            "family": self.family,
            "elements": el.to_dict(),
            "a": el.a if np.isfinite(el.a) else None,
            "period": el.period(self.gm) if el.e < 1 else None,
            "gm": self.gm,
            "law": {"expression": NEWTON_LAW, "radial": "-GM/r**2", "constants": {"GM": self.gm}},
        }
