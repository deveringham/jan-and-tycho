"""Central-force families from a radial law written as text (Phase 5, `llm.hypothesize`).

A proposal is f(r) in AU/day² as an expression in `r` (AU), `GM` (the Sun's, fixed) and
named parameters with bounds. The text comes from outside Jan (an LLM), so it is checked
token by token before sympy sees it: `parse_expr` evaluates Python, and only numbers,
arithmetic, a few functions and the declared names may reach it.
"""

from __future__ import annotations

import io
import re
import tokenize
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from numpy.typing import NDArray
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

from jan.models.families import Dim
from jan.models.force_families import _OdeFamily, _radial, _vector
from jt_core.conics import ConicElements

FloatArray = NDArray[np.float64]

FUNCTIONS = {"exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "tanh": sp.tanh, "pi": sp.pi}
_OPS = {"+", "-", "*", "/", "**", "^", "(", ")"}
_PARAM_NAME = re.compile(r"^[a-z][a-z0-9_]{0,15}$")
_RESERVED = {"r", "gm", *FUNCTIONS}


@dataclass(frozen=True)
class ParamSpec:
    name: str
    lower: float
    upper: float
    initial: float
    log_scale: bool = False

    def dim(self) -> Dim:
        if self.log_scale:
            return Dim(f"log_{self.name}", float(np.log(self.lower)), float(np.log(self.upper)))
        return Dim(self.name, self.lower, self.upper)

    def to_phi(self, value: float) -> float:
        return float(np.log(value)) if self.log_scale else float(value)


def symbol(name: str) -> sp.Symbol:
    """Parameters are prefixed so they never collide with sympy names downstream."""
    return sp.Symbol(f"p_{name}")


def check_params(params: list[ParamSpec]) -> None:
    names = [p.name for p in params]
    if len(set(names)) != len(names):
        raise ValueError("duplicate parameter names")
    for p in params:
        if not _PARAM_NAME.match(p.name) or p.name in _RESERVED:
            raise ValueError(f"invalid parameter name {p.name!r}")
        if not (np.isfinite(p.lower) and np.isfinite(p.upper) and p.lower < p.upper):
            raise ValueError(f"{p.name}: need finite lower < upper")
        if not p.lower <= p.initial <= p.upper:
            raise ValueError(f"{p.name}: initial value outside its bounds")
        if p.log_scale and p.lower <= 0:
            raise ValueError(f"{p.name}: log-scale bounds must be positive")


def parse_radial(text: str, params: list[ParamSpec]) -> sp.Expr:
    """Parse f(r) after whitelisting every token; raises ValueError otherwise."""
    if len(text) > 300:
        raise ValueError("expression too long")
    allowed = {"r", "GM", *FUNCTIONS, *(p.name for p in params)}
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise ValueError(f"cannot tokenize: {exc}") from exc
    for tok in tokens:
        if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER):
            continue
        if tok.type == tokenize.NUMBER:
            if not re.fullmatch(r"\d+(\.\d*)?([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?", tok.string):
                raise ValueError(f"unsupported number {tok.string!r}")
        elif tok.type == tokenize.NAME:
            if tok.string not in allowed:
                raise ValueError(f"unknown name {tok.string!r}")
        elif tok.type == tokenize.OP:
            if tok.string not in _OPS:
                raise ValueError(f"operator {tok.string!r} not allowed")
        else:
            raise ValueError(f"token {tok.string!r} not allowed")
    local: dict[str, Any] = {"r": sp.Symbol("r"), "GM": sp.Symbol("GM"), **FUNCTIONS}
    local.update({p.name: symbol(p.name) for p in params})
    expr = parse_expr(
        text,
        local_dict=local,
        transformations=(*standard_transformations, convert_xor),
        evaluate=True,
    )
    if not isinstance(expr, sp.Expr):
        raise ValueError("not an expression")
    extra = expr.free_symbols - {sp.Symbol("r"), sp.Symbol("GM"), *(symbol(p.name) for p in params)}
    if extra:
        raise ValueError(f"unexpected symbols {sorted(map(str, extra))}")
    return expr


@dataclass(frozen=True)
class RadialSpec:
    slug: str
    radial: str
    params: tuple[ParamSpec, ...]


def radial_family(spec: RadialSpec, gm: float) -> type[_OdeFamily]:
    """A new `_OdeFamily` subclass for f(r); validates the spec (ValueError if unusable)."""
    params = list(spec.params)
    check_params(params)
    expr = parse_radial(spec.radial, params)
    syms = [sp.Symbol("r"), sp.Symbol("GM"), *(symbol(p.name) for p in params)]
    fn = sp.lambdify(syms, expr, "jax")
    probe = np.asarray(
        [fn(jnp.asarray(r), gm, *(p.initial for p in params)) for r in (0.5, 1.0, 10.0)],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(probe)):
        raise ValueError("f(r) is not finite at the initial parameters")
    text = str(expr)
    dims = tuple(p.dim() for p in params)
    defaults = [p.to_phi(p.initial) for p in params]
    logs = tuple(p.log_scale for p in params)
    names = tuple(str(symbol(p.name)) for p in params)

    def values(phi: Any) -> list[Any]:
        return [jnp.exp(phi[6 + k]) if lg else phi[6 + k] for k, lg in enumerate(logs)]

    class _Radial(_OdeFamily):
        name: ClassVar[str] = f"llm:{spec.slug}"
        n_law: ClassVar[int] = len(params)
        grid_bases: ClassVar[int] = 8

        def law_dims(self) -> tuple[Dim, ...]:
            return dims

        def law_defaults(self, el: ConicElements) -> list[float]:
            return defaults

        def grid_starts(self, base: FloatArray) -> list[FloatArray]:
            """One parameter at a time across its bounds: the proposed `initial` alone can
            sit in the wrong basin (e.g. a Yukawa range far from the truth)."""
            out = []
            for k, d in enumerate(dims):
                for f in (0.1, 0.3, 0.5, 0.7, 0.9):
                    x = np.array(base, dtype=np.float64)
                    x[6 + k] = d.lo + f * d.width
                    out.append(x)
            return out

        def accel_fn(self, phi: jax.Array) -> Any:
            vals = values(phi)
            return _radial(lambda r: fn(r, self.gm, *vals))

        def law_strings(self, phi: FloatArray) -> tuple[list[str], dict[str, float], str]:
            consts = {"GM": self.gm}
            consts.update({n: float(v) for n, v in zip(names, values(phi), strict=True)})
            return _vector(text), consts, text

    cls = dataclass(frozen=True)(_Radial)
    jax.tree_util.register_dataclass(cls, data_fields=["t0", "gm", "t_span"], meta_fields=[])
    return cls
