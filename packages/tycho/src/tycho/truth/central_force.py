"""Central-force truths, integrated with jt_core.integrate (diffrax, Dopri8)."""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
import sympy as sp
from numpy.typing import NDArray
from sympy.parsing.sympy_parser import parse_expr

from jt_core.conics import elements_to_state
from jt_core.integrate import central_accel, propagate
from tycho.scenario import CentralForceTruthCfg

FloatArray = NDArray[np.float64]

_R = sp.Symbol("r", positive=True)


def radial_law(cfg: CentralForceTruthCfg) -> tuple[sp.Expr, dict[str, float]]:
    """f(r) as a sympy expression (symbol `r`, constants as symbols) and their values."""
    GM, n, eps, k, lam = sp.symbols("GM n epsilon k lam", positive=True)
    if cfg.law == "power_law":
        return -GM / _R**n, {"GM": cfg.gm, "n": cfg.exponent}
    if cfg.law == "newton_plus_inverse_cube":
        return -GM / _R**2 * (1 + eps / _R), {"GM": cfg.gm, "epsilon": cfg.epsilon_au}
    if cfg.law == "hooke":
        return -k * _R, {"k": cfg.k_per_day2 if cfg.k_per_day2 is not None else cfg.gm}
    if cfg.law == "yukawa":
        return -GM / _R**2 * (1 + _R / lam) * sp.exp(-_R / lam), {
            "GM": cfg.gm,
            "lam": cfg.yukawa_lambda_au,
        }
    assert cfg.expression is not None
    return parse_expr(cfg.expression, local_dict={"r": _R}), {}


def substitute(expr: sp.Expr, constants: dict[str, float]) -> sp.Expr:
    """Replace named constants (matched by symbol name, whatever their assumptions)."""
    names = {sym: getattr(sym, "name", "") for sym in expr.free_symbols}
    return expr.subs({sym: constants[n] for sym, n in names.items() if n in constants})


def vector_law(radial: sp.Expr) -> list[str]:
    """Components of a = f(r) r̂ in x, y, z (sympy strings)."""
    x, y, z = sp.symbols("x y z")
    r = sp.sqrt(x**2 + y**2 + z**2)
    f = radial.subs(_R, r)
    return [str(f * c / r) for c in (x, y, z)]


class CentralForceTruth:
    kind: Literal["dynamical", "kinematic"] = "dynamical"

    def __init__(self, cfg: CentralForceTruthCfg) -> None:
        self.cfg = cfg
        expr, consts = radial_law(cfg)
        self.radial_expr = expr
        self.constants = consts
        f = sp.lambdify(_R, substitute(expr, consts), modules="jax")
        self._accel = central_accel(lambda r, _: f(r))
        if cfg.elements is not None:
            r0, v0 = elements_to_state(cfg.elements.to_elements(), [0.0], cfg.gm)
            self.x0, self.v0 = r0, v0
        else:
            assert cfg.initial is not None
            self.x0 = np.array([cfg.initial.position_au])
            self.v0 = np.array([cfg.initial.velocity_au_per_day])
        self._cache: dict[bytes, tuple[FloatArray, FloatArray]] = {}

    def state(self, t: FloatArray) -> tuple[FloatArray, FloatArray]:
        ta = np.atleast_1d(np.asarray(t, dtype=np.float64))
        key = ta.tobytes()
        if key not in self._cache:
            x, v = propagate(
                self._accel, jnp.asarray(self.x0), jnp.asarray(self.v0), 0.0, jnp.asarray(ta)
            )
            self._cache = {key: (np.asarray(x)[:, 0], np.asarray(v)[:, 0])}
        return self._cache[key]

    def position(self, t: FloatArray) -> FloatArray:
        return self.state(t)[0]

    def velocity(self, t: FloatArray) -> FloatArray:
        return self.state(t)[1]

    def acceleration(self, x: FloatArray, t: FloatArray) -> FloatArray:
        r = np.linalg.norm(x, axis=1, keepdims=True)
        return self.radial(r) * x / r

    def radial(self, r: FloatArray) -> FloatArray:
        """f(r) (AU/day²) for plotting and structural scoring."""
        f = sp.lambdify(_R, substitute(self.radial_expr, self.constants), modules="numpy")
        return np.asarray(f(np.asarray(r, dtype=np.float64)), dtype=np.float64) * np.ones_like(r)

    def describe(self) -> dict[str, Any]:
        el = None
        if self.cfg.elements is not None:
            el = self.cfg.elements.to_elements().to_dict()
        return {
            "type": "central_force",
            "kind": self.kind,
            "family": f"central-{self.cfg.law.replace('_', '-')}",
            "elements": None,
            "initial_osculating_elements": el,
            "x0": self.x0[0].tolist(),
            "v0": self.v0[0].tolist(),
            "gm": self.cfg.gm,
            "law": {
                "expression": vector_law(substitute(self.radial_expr, self.constants)),
                "radial": str(self.radial_expr),
                "constants": self.constants,
                "symbolic": vector_law(self.radial_expr),
            },
        }
