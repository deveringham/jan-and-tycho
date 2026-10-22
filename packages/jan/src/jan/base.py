"""Strategy protocol, registry and hypothesis base classes (SPEC §6.1–6.2)."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from jt_core.schema import (
    ElementsEstimate,
    HypothesisKind,
    LawEstimate,
    ModelScore,
    ObservationSet,
    PositionPrediction,
    PublicMeta,
    SkyPrediction,
    TraceFrame,
)

FloatArray = NDArray[np.float64]


class StrategyConfig(BaseModel):
    """Base for all strategy configs (YAML-loadable)."""

    model_config = ConfigDict(extra="forbid")

    record_trace: bool = False
    trace_max_frames: int = 100
    realism: Literal["model", "ignore"] = Field(
        "model",
        description="Model the published light-time/aberration (see jan.realism), or ignore "
        "them to measure their bias. The frame is always converted.",
    )


class Strategy(Protocol):
    name: ClassVar[str]
    supported_tiers: ClassVar[set[str]]
    Config: ClassVar[type[StrategyConfig]]

    def fit(
        self,
        obs: ObservationSet,
        meta: PublicMeta,
        config: StrategyConfig,
        rng_seed: int,
        workdir: Path,
    ) -> BaseHypothesis:
        """Fit the data. Must not read files outside `workdir`."""
        ...


_REGISTRY: dict[str, type[Any]] = {}
_STRATEGY_MODULES = [
    "jan.strategies.classical",
    "jan.strategies.ptolemy",
    "jan.strategies.kepler",
    "jan.strategies.discovery",
    "jan.strategies.bayes",
    "jan.strategies.gp",
    "jan.strategies.piml",
    "jan.strategies.sbi",
    "jan.strategies.llm",
]


def register(name: str) -> Callable[[type[Any]], type[Any]]:
    """Class decorator: `@register("classical.lsq")`."""

    def deco(cls: type[Any]) -> type[Any]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"strategy {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


_OPTIONAL = {
    "jan.strategies.bayes": "jan[bayes]",
    "jan.strategies.gp": "jan[gp]",
    "jan.strategies.piml": "jan[piml]",
    "jan.strategies.sbi": "jan[sbi]",
    "jan.strategies.llm": "jan[llm,bayes]",
}
UNAVAILABLE: dict[str, str] = {}
"""Strategy packages skipped because their optional extra is not installed."""


def _load_all() -> None:
    for m in _STRATEGY_MODULES:
        try:
            importlib.import_module(m)
        except ImportError as exc:
            if m not in _OPTIONAL:
                raise
            UNAVAILABLE[m] = f"install {_OPTIONAL[m]} ({exc.name} missing)"


def list_strategies() -> list[str]:
    _load_all()
    return sorted(_REGISTRY)


def get_strategy(name: str) -> Strategy:
    _load_all()
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(_REGISTRY)}")
    from jan.realism import RealismAdapter  # imports jan.base

    # The adapter exposes the strategy's name, tiers and Config as instance attributes.
    return cast(Strategy, RealismAdapter(_REGISTRY[name]()))


def make_config(strategy: Strategy, data: dict[str, Any] | None = None) -> StrategyConfig:
    return strategy.Config.model_validate(data or {})


def subsample_frames(frames: list[TraceFrame], max_frames: int) -> list[TraceFrame]:
    """Evenly subsample a trace, always keeping the first and last frame."""
    if len(frames) <= max_frames:
        return frames
    idx = np.unique(np.linspace(0, len(frames) - 1, max_frames).round().astype(int))
    return [frames[i] for i in idx]


class BaseHypothesis:
    """Default implementations: every optional capability returns None."""

    kind: HypothesisKind = "nonparametric"

    def predict_sky(self, t: FloatArray, body_id: str) -> SkyPrediction:
        raise NotImplementedError

    def heliocentric_position(self, t: FloatArray, body_id: str) -> PositionPrediction | None:
        return None

    def elements(self, body_id: str) -> ElementsEstimate | None:
        return None

    def law(self) -> LawEstimate | None:
        return None

    def model_comparison(self) -> list[ModelScore] | None:
        return None

    def trace(self) -> list[TraceFrame] | None:
        return None

    def at_frame(self, frame: TraceFrame) -> BaseHypothesis | None:
        return None

    def observation_sigma(self, body_id: str) -> float | None:
        """Estimated per-axis observation noise σ (radians), for predictive densities."""
        return None

    def observer_position(self, t: FloatArray) -> PositionPrediction | None:
        """The observer's heliocentric positions as inferred (T2/T3), if the hypothesis
        estimates them."""
        return None

    def observer_elements(self) -> ElementsEstimate | None:
        """The observer's inferred orbital elements (T2/T3)."""
        return None

    def acceleration(
        self, x: FloatArray, v: FloatArray, t: FloatArray, body_id: str
    ) -> FloatArray | None:
        """The hypothesis's acceleration field (AU/day²) at arbitrary points, for dynamical
        and black-box models; None when it has no force law."""
        return None

    def hidden_bodies(self, t: FloatArray) -> dict[str, PositionPrediction] | None:
        """Positions of unseen bodies the hypothesis infers (e.g. a perturber)."""
        return None

    def notes(self) -> str:
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "notes": self.notes()}


class PerBodyHypothesis(BaseHypothesis):
    """Bundle independent single-body hypotheses into one (bodies fitted separately)."""

    def __init__(self, parts: dict[str, BaseHypothesis], notes: str = "") -> None:
        self.parts = parts
        kinds: set[HypothesisKind] = {p.kind for p in parts.values()}
        self.kind: HypothesisKind = kinds.pop() if len(kinds) == 1 else "nonparametric"
        self._notes = notes

    def _part(self, body_id: str) -> BaseHypothesis:
        if body_id not in self.parts:
            raise KeyError(f"no hypothesis for body {body_id!r}")
        return self.parts[body_id]

    def predict_sky(self, t: FloatArray, body_id: str) -> SkyPrediction:
        return self._part(body_id).predict_sky(t, body_id)

    def heliocentric_position(self, t: FloatArray, body_id: str) -> PositionPrediction | None:
        return self._part(body_id).heliocentric_position(t, body_id)

    def elements(self, body_id: str) -> ElementsEstimate | None:
        return self._part(body_id).elements(body_id)

    def law(self) -> LawEstimate | None:
        return next(iter(self.parts.values())).law() if len(self.parts) == 1 else None

    def model_comparison(self) -> list[ModelScore] | None:
        return next(iter(self.parts.values())).model_comparison() if len(self.parts) == 1 else None

    def trace(self) -> list[TraceFrame] | None:
        return next(iter(self.parts.values())).trace() if len(self.parts) == 1 else None

    def at_frame(self, frame: TraceFrame) -> BaseHypothesis | None:
        if len(self.parts) != 1:
            return None
        ((bid, part),) = self.parts.items()
        h = part.at_frame(frame)
        return None if h is None else PerBodyHypothesis({bid: h})

    def observation_sigma(self, body_id: str) -> float | None:
        return self._part(body_id).observation_sigma(body_id)

    def observer_position(self, t: FloatArray) -> PositionPrediction | None:
        # Bodies are fitted separately, each with its own observer estimate: report the first.
        return next(iter(self.parts.values())).observer_position(t)

    def observer_elements(self) -> ElementsEstimate | None:
        return next(iter(self.parts.values())).observer_elements()

    def acceleration(
        self, x: FloatArray, v: FloatArray, t: FloatArray, body_id: str
    ) -> FloatArray | None:
        return self._part(body_id).acceleration(x, v, t, body_id)

    def hidden_bodies(self, t: FloatArray) -> dict[str, PositionPrediction] | None:
        out: dict[str, PositionPrediction] = {}
        for p in self.parts.values():
            out.update(p.hidden_bodies(t) or {})
        return out or None

    def notes(self) -> str:
        parts = [f"[{b}] {p.notes()}" for b, p in self.parts.items() if p.notes()]
        return "\n".join(([self._notes] if self._notes else []) + parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "bodies": {b: p.to_dict() for b, p in self.parts.items()},
            "notes": self.notes(),
        }
