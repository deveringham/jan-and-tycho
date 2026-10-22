"""N-body truth: the Sun plus mutually attracting bodies, heliocentric with indirect terms.

r̈_i = −GM☉(1 + m_i) r_i / r_i³ + Σ_{j≠i} GM☉ m_j [(r_j − r_i)/|r_j − r_i|³ − r_j / r_j³]
"""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from jt_core.conics import elements_to_state
from jt_core.integrate import propagate
from tycho.scenario import NBodyTruthCfg

FloatArray = NDArray[np.float64]


def _accel(x: Any, v: Any, t: Any, args: Any) -> Any:
    gm, m = args
    r = jnp.linalg.norm(x, axis=1, keepdims=True)
    direct = -gm * (1 + m[:, None]) * x / r**3
    d = x[None, :, :] - x[:, None, :]  # d[i, j] = r_j − r_i
    dist = jnp.linalg.norm(d, axis=2, keepdims=True)
    dist = jnp.where(dist == 0, jnp.inf, dist)
    pair = jnp.sum(gm * m[None, :, None] * d / dist**3, axis=1)
    indirect = -jnp.sum(gm * m[:, None] * x / r**3, axis=0, keepdims=True)
    return direct + pair + indirect


class NBodyTruth:
    kind: Literal["dynamical", "kinematic"] = "dynamical"

    def __init__(self, cfg: NBodyTruthCfg) -> None:
        self.cfg = cfg
        self.ids = [m.id for m in cfg.system]
        self.index = self.ids.index(cfg.member)
        self.m = np.array([m.mass_msun for m in cfg.system])
        states = [
            elements_to_state(m.elements.to_elements(), [0.0], cfg.gm_sun * (1 + m.mass_msun))
            for m in cfg.system
        ]
        self.x0 = np.concatenate([s[0] for s in states])
        self.v0 = np.concatenate([s[1] for s in states])
        self._cache: dict[bytes, tuple[FloatArray, FloatArray]] = {}

    def states(self, t: FloatArray) -> tuple[FloatArray, FloatArray]:
        """All members' (positions, velocities), shape (N, n_bodies, 3)."""
        ta = np.atleast_1d(np.asarray(t, dtype=np.float64))
        key = ta.tobytes()
        if key not in self._cache:
            x, v = propagate(
                _accel,
                jnp.asarray(self.x0),
                jnp.asarray(self.v0),
                0.0,
                jnp.asarray(ta),
                (self.cfg.gm_sun, jnp.asarray(self.m)),
            )
            self._cache = {key: (np.asarray(x), np.asarray(v))}
        return self._cache[key]

    def position(self, t: FloatArray) -> FloatArray:
        return self.states(t)[0][:, self.index]

    def velocity(self, t: FloatArray) -> FloatArray:
        return self.states(t)[1][:, self.index]

    def acceleration(self, x: FloatArray, t: FloatArray) -> FloatArray:
        """Acceleration on this member at positions x and times t (others at their true
        positions): Sun + direct pulls of the other bodies + indirect (Sun's reflex) terms."""
        gm = self.cfg.gm_sun
        allx = self.states(np.asarray(t))[0]  # (N, n_bodies, 3)
        m_self = self.m[self.index]
        r = np.linalg.norm(x, axis=1, keepdims=True)
        a = -gm * (1 + m_self) * x / r**3
        for j, mj in enumerate(self.m):
            if j == self.index or mj == 0:
                continue
            xj = allx[:, j]
            d = xj - x
            a += (
                gm
                * mj
                * (
                    d / np.linalg.norm(d, axis=1, keepdims=True) ** 3
                    - xj / np.linalg.norm(xj, axis=1, keepdims=True) ** 3
                )
            )
        return a

    def describe(self) -> dict[str, Any]:
        me = self.cfg.system[self.index]
        others = [m for m in self.cfg.system if m.id != me.id]
        return {
            "type": "nbody",
            "kind": self.kind,
            "family": "perturbed-conic"
            if any(m.mass_msun > 0 for m in others)
            else "conic-elliptic",
            "elements": None,
            "initial_osculating_elements": me.elements.to_elements().to_dict(),
            "gm": self.cfg.gm_sun,
            "system": self.cfg.model_dump(mode="json")["system"],
            "member": me.id,
            "law": {
                "expression": ["-GM*x/(x**2 + y**2 + z**2)**(3/2) + perturbations"] * 3,
                "constants": {"GM": self.cfg.gm_sun},
                "perturbers": [{"id": m.id, "mass_msun": m.mass_msun} for m in others],
            },
        }
