# pyright: strict, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportMissingTypeStubs=false
"""Data exchanged across the information boundary.

Everything here is truth-agnostic: Tycho produces `ObservationSet` + `PublicMeta`,
Jan returns objects satisfying `Hypothesis`, the harness serializes predictions on a
`QuerySheet`, and Rudolf scores the serialized files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from jt_core.frames import lonlat_to_unit
from jt_core.realism import Realism

FloatArray = NDArray[np.float64]

Tier = Literal["T0", "T1", "T2", "T3"]
HypothesisKind = Literal["dynamical", "kinematic", "nonparametric", "rejected"]

SKY_COLUMNS = ("obs_id", "body_id", "t", "lon", "lat")
POSITION_COLUMNS = ("obs_id", "body_id", "t", "x", "y", "z")
SIGMA_COLUMNS = ("sigma_lon", "sigma_lat")

SUN_ID = "sun"
"""Reserved body id of Tycho's Sun-direction stream (tier T2)."""

ELEMENT_NAMES = ("q", "e", "i", "Omega", "omega", "t_p")
"""Canonical conic elements, angles in radians (SPEC §4)."""


# --------------------------------------------------------------------------- public side


class ElementsModel(BaseModel):
    """Conic elements `(q, e, i, Ω, ω, t_p)`, angles in radians."""

    model_config = ConfigDict(frozen=True)

    q: float
    e: float
    i: float
    Omega: float
    omega: float
    t_p: float

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.q, self.e, self.i, self.Omega, self.omega, self.t_p)


class PublicMeta(BaseModel):
    """Everything Jan is told about a run besides the observations themselves.

    Deliberately excludes the scenario name and description, which could leak the truth.
    """

    model_config = ConfigDict(frozen=True)

    tier: Tier
    observed: Literal["sky", "position"] = Field(
        description="'sky' → lon/lat columns; 'position' → heliocentric x/y/z (T0)."
    )
    units: dict[str, str] = Field(
        default_factory=lambda: {"t": "day", "angle": "rad", "distance": "AU"}
    )
    gm_sun: float
    epoch_calendar: str | None = None
    t_range: tuple[float, float] = Field(description="Span of the public (training) window.")
    body_ids: list[str]
    observer_elements: ElementsModel | None = Field(
        default=None, description="Observer's orbit (T1 only)."
    )
    observer_kind: str | None = Field(
        default=None, description="Observer model type when known (T1 only)."
    )
    observer_position: tuple[float, float, float] | None = Field(
        default=None, description="Fixed observer location (T1 'fixed'/'heliocentric' only)."
    )
    report_uncertainties: bool = False
    sun_stream: bool = Field(
        default=False, description="T2: observations include body 'sun' (geocentric Sun direction)."
    )
    realism: Realism = Field(
        default_factory=Realism,
        description="Effects in the published directions (light-time, aberration) and their "
        "frame; with frame 'equatorial' the lon/lat columns hold right ascension/declination.",
    )


@dataclass(frozen=True)
class ObservationSet:
    """Tycho's public observations. One row per observation.

    Columns: obs_id, body_id, t, then lon, lat (radians) for sky tiers or x, y, z (AU) for
    T0, optionally sigma_lon, sigma_lat (radians, tangent-plane east/north).
    """

    table: pd.DataFrame

    def __post_init__(self) -> None:
        cols = set(self.table.columns)
        if not (set(SKY_COLUMNS) <= cols or set(POSITION_COLUMNS) <= cols):
            raise ValueError(f"ObservationSet table lacks required columns: {sorted(cols)}")

    @property
    def body_ids(self) -> list[str]:
        """Observed bodies (excluding the Sun stream)."""
        return sorted(b for b in self.table["body_id"].unique().tolist() if b != SUN_ID)

    def sun(self) -> ObservationSet | None:
        """The Sun-direction stream (T2), if published."""
        if SUN_ID not in set(self.table["body_id"].tolist()):
            return None
        return self.for_body(SUN_ID)

    @property
    def is_sky(self) -> bool:
        return "lon" in self.table.columns

    def for_body(self, body_id: str) -> ObservationSet:
        sub = cast(pd.DataFrame, self.table.loc[self.table["body_id"] == body_id])
        return ObservationSet(sub.reset_index(drop=True))

    def __len__(self) -> int:
        return len(self.table)

    @property
    def t(self) -> FloatArray:
        return self.table["t"].to_numpy(dtype=np.float64)

    def unit_vectors(self) -> FloatArray:
        """Observed directions as unit vectors (N, 3). Sky tiers only."""
        if not self.is_sky:
            raise ValueError("unit_vectors() requires sky observations")
        return lonlat_to_unit(
            self.table["lon"].to_numpy(dtype=np.float64),
            self.table["lat"].to_numpy(dtype=np.float64),
        )

    def positions(self) -> FloatArray:
        """Observed heliocentric positions (N, 3). T0 only."""
        if self.is_sky:
            raise ValueError("positions() requires T0 position observations")
        return self.table[["x", "y", "z"]].to_numpy(dtype=np.float64)

    def sigmas(self) -> FloatArray | None:
        """Per-observation (east, north) sigmas in radians, if published."""
        if not set(SIGMA_COLUMNS) <= set(self.table.columns):
            return None
        return self.table[list(SIGMA_COLUMNS)].to_numpy(dtype=np.float64)


@dataclass(frozen=True)
class QuerySheet:
    """Times at which the harness asks a hypothesis for predictions. No values, ever."""

    holdout: pd.DataFrame
    """Columns obs_id, body_id, t: the sealed holdout observation times."""
    dense_t: FloatArray
    """Uniform grid over the full span (training + holdout)."""
    body_ids: list[str]
    train: pd.DataFrame | None = None
    """Columns obs_id, body_id, t: the public training times (for fit residuals)."""


# --------------------------------------------------------------------------- Jan's outputs


@dataclass
class SkyPrediction:
    """Predicted geocentric directions."""

    mean: FloatArray
    """(N, 3) unit vectors."""
    samples: FloatArray | None = None
    """(S, N, 3) unit vectors drawn from the predictive distribution."""
    cov: FloatArray | None = None
    """(N, 2, 2) tangent-plane (east, north) covariance in rad²."""


@dataclass
class PositionPrediction:
    """Predicted heliocentric positions."""

    mean: FloatArray
    """(N, 3) AU."""
    samples: FloatArray | None = None
    """(S, N, 3) AU."""


@dataclass
class ElementsEstimate:
    """Point estimate (and optionally uncertainty) of conic elements for one body."""

    point: dict[str, float]
    """Keys from ELEMENT_NAMES, radians for angles."""
    cov: list[list[float]] | None = None
    """Covariance over `cov_names` (same order)."""
    cov_names: list[str] | None = None
    samples: dict[str, list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "cov": self.cov,
            "cov_names": self.cov_names,
            "samples": self.samples,
        }


@dataclass
class LawEstimate:
    """A recovered acceleration law: sympy string in x, y, z, vx, vy, vz, t."""

    expression: list[str]
    """Three component strings (ax, ay, az)."""
    constants: dict[str, float] = field(default_factory=dict[str, float])
    radial: str | None = None
    """For central laws: f(r) in `r`, with a = f(r) r̂ (same constants)."""
    extra: dict[str, Any] = field(default_factory=dict[str, Any])
    """Strategy-specific detail (e.g. a symbolic-regression Pareto front)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "constants": self.constants,
            "radial": self.radial,
            "extra": self.extra,
        }


@dataclass
class ModelScore:
    model_name: str
    log_evidence: float | None = None
    information_criterion: float | None = None
    posterior_prob: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "log_evidence": self.log_evidence,
            "information_criterion": self.information_criterion,
            "posterior_prob": self.posterior_prob,
        }


@dataclass
class TraceFrame:
    """A snapshot of a fit in progress, sufficient to rebuild a predicting hypothesis."""

    label: str
    iteration: int
    state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "iteration": self.iteration, "state": self.state}


@runtime_checkable
class Hypothesis(Protocol):
    """What every Jan strategy returns (SPEC §6.2)."""

    kind: HypothesisKind

    def predict_sky(self, t: FloatArray, body_id: str) -> SkyPrediction: ...

    def heliocentric_position(self, t: FloatArray, body_id: str) -> PositionPrediction | None: ...

    def elements(self, body_id: str) -> ElementsEstimate | None: ...

    def law(self) -> LawEstimate | None: ...

    def model_comparison(self) -> list[ModelScore] | None: ...

    def trace(self) -> list[TraceFrame] | None: ...

    def at_frame(self, frame: TraceFrame) -> Hypothesis | None:
        """A hypothesis that predicts from a trace frame (used by the harness)."""
        ...

    def notes(self) -> str: ...

    def to_dict(self) -> dict[str, Any]: ...
