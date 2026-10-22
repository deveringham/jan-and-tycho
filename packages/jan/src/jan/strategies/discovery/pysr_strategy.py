"""`discovery.pysr`: symbolic regression of the radial force law f(r) (SPEC §6.5, Phase 4).

Accelerations come from fit-based derivatives of the reconstructed positions (local
polynomials, as in `discovery.sindy`); their radial component f = a · r̂ is regressed on r
with PySR (genetic programming over expressions). The full Pareto front of complexity vs
loss is reported; the selection criterion is PySR's "best" (the largest drop in log-loss
per unit complexity among the accurate expressions). Needs `jan[discovery]` and a Julia
runtime (installed by PySR on first use); deterministic, single-threaded search.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import sympy as sp
from numpy.typing import NDArray
from pydantic import Field
from sympy.parsing.sympy_parser import parse_expr

from jan.base import BaseHypothesis, PerBodyHypothesis, StrategyConfig, register
from jan.law import LawHypothesis, fit_initial_state
from jan.models.observer import observer_function
from jan.strategies.discovery.sindy import discovery_positions, local_poly_system
from jt_core.schema import LawEstimate, ObservationSet, PublicMeta, TraceFrame

FloatArray = NDArray[np.float64]


class PysrConfig(StrategyConfig):
    niterations: int = 40
    populations: int = 15
    maxsize: int = 15
    binary_operators: list[str] = Field(default_factory=lambda: ["+", "-", "*", "/", "^"])
    unary_operators: list[str] = Field(default_factory=lambda: ["exp"])
    window_points: int = 11
    poly_degree: int = 4
    max_gap_factor: float = 3.0
    model_selection: Literal["best", "accuracy", "score"] = "best"


def radial_samples(t: FloatArray, x: FloatArray, cfg: PysrConfig) -> tuple[FloatArray, FloatArray]:
    """(r, f) pairs: f = a · r̂ from local-polynomial accelerations."""
    a, b = local_poly_system(t, x, (1,), cfg.window_points, cfg.poly_degree, cfg.max_gap_factor)
    xc = a[:, 0].reshape(-1, 3)  # feature for power 1 is x itself
    acc = b.reshape(-1, 3)
    r = np.linalg.norm(xc, axis=1)
    return r, np.sum(acc * xc, axis=1) / r


def _sympy_radial(expr: sp.Expr, scale: float) -> sp.Expr:
    """PySR's expression in x0 (= r), rescaled back to AU/day²."""
    return sp.sympify(expr).subs(sp.Symbol("x0"), sp.Symbol("r")) * scale


@register("discovery.pysr")
class DiscoveryPysr:
    """Symbolic regression (PySR) of the radial acceleration vs r, with a Pareto front."""

    name: ClassVar[str]
    supported_tiers: ClassVar[set[str]] = {"T0", "T1"}
    Config: ClassVar[type[StrategyConfig]] = PysrConfig

    def fit(
        self,
        obs: ObservationSet,
        meta: PublicMeta,
        config: StrategyConfig,
        rng_seed: int,
        workdir: Path,
    ) -> BaseHypothesis:
        assert isinstance(config, PysrConfig)
        return PerBodyHypothesis(
            {
                b: self._fit_body(obs.for_body(b), b, meta, config, rng_seed, workdir)
                for b in obs.body_ids
            }
        )

    def _fit_body(
        self,
        ob: ObservationSet,
        bid: str,
        meta: PublicMeta,
        cfg: PysrConfig,
        seed: int,
        workdir: Path,
    ) -> LawHypothesis:
        from pysr import PySRRegressor  # optional dependency (jan[discovery] + Julia)

        t_start = time.perf_counter()
        t, x, provenance = discovery_positions(ob, meta, seed)
        r, f = radial_samples(t, x, cfg)
        scale = float(np.median(np.abs(f)))
        model = PySRRegressor(
            niterations=cfg.niterations,
            populations=cfg.populations,
            maxsize=cfg.maxsize,
            binary_operators=cfg.binary_operators,
            unary_operators=cfg.unary_operators,
            model_selection=cfg.model_selection,
            random_state=seed % (2**31),
            deterministic=True,
            parallelism="serial",
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            tempdir=str(workdir),
        )
        model.fit(r[:, None], f / scale)
        eqs = model.equations_
        assert isinstance(eqs, pd.DataFrame)
        best = cast(dict[str, Any], cast(pd.Series, model.get_best()).to_dict())
        radial = _sympy_radial(best["sympy_format"], scale)
        r_sym = sp.Symbol("r")
        f_np = sp.lambdify(r_sym, radial, "jax")

        def accel(xx: Any, v: Any, tt: Any, args: Any) -> Any:
            rr = jnp.linalg.norm(xx, axis=-1, keepdims=True)
            return f_np(rr) * xx / rr

        pareto = [
            {
                "complexity": int(row["complexity"]),
                "loss": float(row["loss"]) * scale**2,
                "score": float(row.get("score", np.nan)),
                "expression": str(_sympy_radial(row["sympy_format"], scale)),
            }
            for row in cast(list[dict[str, Any]], eqs.to_dict("records"))
        ]
        observer = observer_function(meta) if ob.is_sky else None
        t0 = float(np.median(t))
        k = min(max(int(np.argmin(np.abs(t - t0))), 1), len(t) - 2)
        v0 = (x[k + 1] - x[k - 1]) / (t[k + 1] - t[k - 1])
        t0 = float(t[k])
        x0, v0, rms = fit_initial_state(accel, None, ob, t0, x[k], v0, observer)
        xs, ys, zs = sp.symbols("x y z")
        rr = sp.sqrt(xs**2 + ys**2 + zs**2)
        vec = [str(radial.subs(r_sym, rr) * c / rr) for c in (xs, ys, zs)]
        law = LawEstimate(
            expression=vec,
            constants={},
            radial=str(radial),
            extra={"pareto": pareto, "selected": str(radial), "criterion": cfg.model_selection},
        )
        frames = None
        if cfg.record_trace:
            frames = [
                TraceFrame(f"complexity {p['complexity']}", j, {"radial": p["expression"]})
                for j, p in enumerate(pareto)
            ]

        def frame_h(fr: TraceFrame) -> LawHypothesis:
            ex = parse_expr(fr.state["radial"], local_dict={"r": r_sym})
            g = sp.lambdify(r_sym, ex, "jax")

            def acc2(xx: Any, v: Any, tt: Any, args: Any) -> Any:
                q = jnp.linalg.norm(xx, axis=-1, keepdims=True)
                return g(q) * jnp.ones_like(q) * xx / q

            return LawHypothesis(
                bid, acc2, None, x0, v0, t0, observer, LawEstimate([str(ex)] * 3, {}, str(ex)), {}
            )

        info = {
            "positions": provenance,
            "n_samples": len(r),
            "selected": str(radial),
            "pareto": pareto,
            "sigma_rad": rms if ob.is_sky else None,
            "fit_seconds": time.perf_counter() - t_start,
            "notes": f"PySR on f(r) from {provenance}: f(r) = {radial} "
            f"({len(pareto)} Pareto entries)",
        }
        return LawHypothesis(bid, accel, None, x0, v0, t0, observer, law, info, frames, frame_h)
