"""Scenario configuration (SPEC §5.1): pydantic models loaded from YAML.

Every sub-component is a discriminated union keyed by `type`. Angles in configs carry an
explicit unit suffix (`_deg`, `_arcsec`); everything is converted to radians internally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from jt_core.conics import ConicElements
from jt_core.constants import DEG, GM_SUN
from jt_core.realism import Realism


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------------- elements


class ElementsCfg(_Cfg):
    """Conic elements with angles in degrees."""

    q: float = Field(gt=0, description="Perihelion distance (AU).")
    e: float = Field(ge=0)
    i_deg: float = 0.0
    Omega_deg: float = 0.0
    omega_deg: float = 0.0
    t_p: float = Field(default=0.0, description="Time of perihelion (days since epoch).")

    def to_elements(self) -> ConicElements:
        return ConicElements(
            self.q, self.e, self.i_deg * DEG, self.Omega_deg * DEG, self.omega_deg * DEG, self.t_p
        )


EARTH_LIKE = ElementsCfg(q=0.98329, e=0.0167, i_deg=0.0, Omega_deg=0.0, omega_deg=102.94, t_p=2.0)


# ----------------------------------------------------------------------------- truths


class KeplerianTruthCfg(_Cfg):
    type: Literal["keplerian"] = "keplerian"
    elements: ElementsCfg
    gm: float = GM_SUN


class CircularTruthCfg(_Cfg):
    """Circular orbit; `phase_deg` is the argument of latitude at t = 0."""

    type: Literal["circular"] = "circular"
    radius_au: float = Field(gt=0)
    i_deg: float = 0.0
    Omega_deg: float = 0.0
    phase_deg: float = 0.0
    gm: float = GM_SUN

    def to_elements(self) -> ConicElements:
        n = float(np.sqrt(self.gm / self.radius_au**3))
        return ConicElements(
            self.radius_au,
            0.0,
            self.i_deg * DEG,
            self.Omega_deg * DEG,
            0.0,
            -self.phase_deg * DEG / n,
        )


class EpicycleCircleCfg(_Cfg):
    radius_au: float = Field(gt=0)
    period_days: float = Field(description="Negative for retrograde rotation.")
    phase_deg: float = 0.0


class EccentricCfg(_Cfg):
    """Deferent with an eccentric centre and an equant (uniform-motion) point.

    Both points lie on the line at `angle_deg` from the centre of motion, at distances
    `center_offset_au` and `equant_offset_au` on the far side (Ptolemy: equant = 2 × centre).
    """

    center_offset_au: float = Field(0.0, ge=0)
    equant_offset_au: float = Field(0.0, ge=0)
    angle_deg: float = 0.0


class EpicycleTruthCfg(_Cfg):
    """Ptolemaic deferent + epicycles in a tilted plane (kinematic, no force law).

    `center: observer` makes the model geocentric (the classical form): the body sits at
    observer + Σ circles. `center: sun` makes it heliocentric.
    """

    type: Literal["epicycle"] = "epicycle"
    center: Literal["observer", "sun"] = "observer"
    i_deg: float = 0.0
    Omega_deg: float = 0.0
    circles: list[EpicycleCircleCfg] = Field(min_length=1)
    eccentric: EccentricCfg | None = None


class InitialStateCfg(_Cfg):
    """Heliocentric state at t = 0 (AU, AU/day)."""

    position_au: tuple[float, float, float]
    velocity_au_per_day: tuple[float, float, float]


class CentralForceTruthCfg(_Cfg):
    """r̈ = f(r) r̂, integrated numerically (Dormand–Prince 8, tight tolerances).

    Laws (f < 0 is attractive):
      power_law                 f = −GM / r^n                      (n = 2: Newton)
      newton_plus_inverse_cube  f = −GM / r² · (1 + ε / r)         (GR-like precession)
      hooke                     f = −k r                           (ellipse *centred* on the Sun)
      yukawa                    f = −GM / r² · (1 + r/λ) e^(−r/λ)
      custom                    f = sympy expression in `r` (AU/day²)
    The initial state is either osculating Keplerian `elements` (under GM) or `initial`.
    """

    type: Literal["central_force"] = "central_force"
    law: Literal["power_law", "newton_plus_inverse_cube", "hooke", "yukawa", "custom"]
    gm: float = GM_SUN
    exponent: float = 2.0
    epsilon_au: float = 0.0
    k_per_day2: float | None = Field(default=None, description="Hooke constant (default GM/AU³).")
    yukawa_lambda_au: float = 1.0
    expression: str | None = None
    elements: ElementsCfg | None = None
    initial: InitialStateCfg | None = None

    @model_validator(mode="after")
    def _check(self) -> CentralForceTruthCfg:
        if (self.elements is None) == (self.initial is None):
            raise ValueError("give exactly one of `elements` or `initial`")
        if self.law == "custom" and not self.expression:
            raise ValueError("law 'custom' needs `expression`")
        return self


class NBodyMemberCfg(_Cfg):
    id: str
    mass_msun: float = Field(ge=0, description="Mass in solar masses (0: test particle).")
    elements: ElementsCfg = Field(description="Osculating heliocentric elements at t = 0.")


class NBodyTruthCfg(_Cfg):
    """Sun + bodies with mutual gravity (heliocentric frame, indirect terms included).

    Every scenario body in the same system lists the same `system`; `member` picks which
    one this body is (a hidden perturber is just a member marked `hidden: true`).
    """

    type: Literal["nbody"] = "nbody"
    system: list[NBodyMemberCfg] = Field(min_length=1)
    member: str
    gm_sun: float = GM_SUN

    @model_validator(mode="after")
    def _check(self) -> NBodyTruthCfg:
        if self.member not in {m.id for m in self.system}:
            raise ValueError(f"member {self.member!r} not in system")
        return self


class KinematicPolygonCfg(_Cfg):
    """A closed polygon in a tilted plane, traversed periodically (no force law)."""

    type: Literal["kinematic_polygon"] = "kinematic_polygon"
    vertices_au: list[tuple[float, float]] = Field(min_length=3, description="In-plane, in order.")
    center_au: tuple[float, float] = (0.0, 0.0)
    i_deg: float = 0.0
    Omega_deg: float = 0.0
    omega_deg: float = Field(default=0.0, description="Rotation of the shape within its plane.")
    period_days: float = Field(gt=0)
    phase: float = Field(default=0.0, description="Fraction of the cycle completed at t = 0.")
    speed_profile: Literal["constant_speed", "constant_areal_velocity"] = "constant_speed"


class KinematicParametricCfg(_Cfg):
    """A periodic parametric curve r(φ), φ = 2π t / P + φ0, in a tilted plane.

    curves: lissajous (x = a cos(p φ + δ), y = b sin(q φ)), figure_eight (lemniscate of
    Bernoulli with half-width a), custom (sympy x, y, z in `phi`).
    """

    type: Literal["kinematic_parametric"] = "kinematic_parametric"
    curve: Literal["lissajous", "figure_eight", "custom"]
    a_au: float = 1.0
    b_au: float = 1.0
    p: int = 1
    q: int = 2
    delta_deg: float = 0.0
    expressions: tuple[str, str, str] | None = None
    center_au: tuple[float, float] = (0.0, 0.0)
    i_deg: float = 0.0
    Omega_deg: float = 0.0
    omega_deg: float = 0.0
    period_days: float = Field(gt=0)
    phase_deg: float = 0.0


TruthCfg = Annotated[
    KeplerianTruthCfg
    | CircularTruthCfg
    | EpicycleTruthCfg
    | CentralForceTruthCfg
    | NBodyTruthCfg
    | KinematicPolygonCfg
    | KinematicParametricCfg,
    Field(discriminator="type"),
]


class PhotometryCfg(_Cfg):
    H: float = Field(description="Absolute magnitude.")
    kappa: float | None = Field(
        default=None, description="Comet activity: m = H + 5 log Δ + 2.5 κ log r."
    )


class BodyCfg(_Cfg):
    id: str
    truth: TruthCfg
    photometry: PhotometryCfg | None = None
    hidden: bool = False

    @model_validator(mode="after")
    def _id_ok(self) -> BodyCfg:
        if self.id in {"sun", "observer"}:
            raise ValueError(f"body id {self.id!r} is reserved")
        return self


# ----------------------------------------------------------------------------- observers


class KeplerianObserverCfg(_Cfg):
    type: Literal["keplerian"] = "keplerian"
    elements: ElementsCfg = EARTH_LIKE
    gm: float = GM_SUN


class FixedObserverCfg(_Cfg):
    type: Literal["fixed"] = "fixed"
    position_au: tuple[float, float, float]


class HeliocentricObserverCfg(_Cfg):
    type: Literal["heliocentric"] = "heliocentric"


class TruthObserverCfg(_Cfg):
    """Any truth model as the observer's trajectory (e.g. an Earth under a non-Newtonian
    force law)."""

    type: Literal["truth"] = "truth"
    truth: TruthCfg


ObserverCfg = Annotated[
    KeplerianObserverCfg | FixedObserverCfg | HeliocentricObserverCfg | TruthObserverCfg,
    Field(discriminator="type"),
]


# ----------------------------------------------------------------------------- schedules


class VisibilityCfg(_Cfg):
    min_solar_elongation_deg: float | None = None
    max_magnitude: float | None = Field(default=None, description="Requires body photometry.")
    dropout_prob: float = Field(
        default=0.0, ge=0, lt=1, description="Weather: night lost entirely."
    )


class RegularScheduleCfg(_Cfg):
    type: Literal["regular"] = "regular"
    cadence_days: float = Field(gt=0)
    jitter_days: float = Field(default=0.0, ge=0)
    start_days: float = 0.0
    visibility: VisibilityCfg = Field(default_factory=VisibilityCfg)


class PoissonScheduleCfg(_Cfg):
    type: Literal["poisson"] = "poisson"
    rate_per_day: float = Field(gt=0)
    visibility: VisibilityCfg = Field(default_factory=VisibilityCfg)


class SeasonsScheduleCfg(_Cfg):
    """Regular cadence, but only inside observing windows that repeat every `cycle_days`
    (e.g. the months a target is well placed, or an observatory's working season)."""

    type: Literal["seasons"] = "seasons"
    cadence_days: float = Field(gt=0)
    jitter_days: float = Field(0.0, ge=0)
    cycle_days: float = Field(365.25, gt=0)
    windows: list[tuple[float, float]] = Field(
        description="(start, end) day-in-cycle intervals; end < start wraps around."
    )
    visibility: VisibilityCfg = Field(default_factory=VisibilityCfg)


class ExplicitScheduleCfg(_Cfg):
    type: Literal["explicit"] = "explicit"
    times: list[float]
    visibility: VisibilityCfg = Field(default_factory=VisibilityCfg)


ScheduleCfg = Annotated[
    RegularScheduleCfg | PoissonScheduleCfg | SeasonsScheduleCfg | ExplicitScheduleCfg,
    Field(discriminator="type"),
]


# ----------------------------------------------------------------------------- noise


class GaussianNoiseCfg(_Cfg):
    """Isotropic on the sphere: per-axis σ in the tangent plane."""

    type: Literal["gaussian"] = "gaussian"
    sigma_arcsec: float = Field(ge=0)


class BiasNoiseCfg(_Cfg):
    """Constant offset in the λ and β coordinates."""

    type: Literal["bias"] = "bias"
    lon_arcsec: float = 0.0
    lat_arcsec: float = 0.0


class QuantizeNoiseCfg(_Cfg):
    """Round λ and β to multiples of the instrument graduation."""

    type: Literal["quantize"] = "quantize"
    step_arcsec: float = Field(gt=0)


class DriftNoiseCfg(_Cfg):
    """Bias growing linearly with time (instrument drift), in λ and β coordinates."""

    type: Literal["drift"] = "drift"
    lon_arcsec_per_year: float = 0.0
    lat_arcsec_per_year: float = 0.0


class Ar1NoiseCfg(_Cfg):
    """Temporally correlated error: per-axis AR(1) with corr(Δt) = rho_per_day ** Δt."""

    type: Literal["ar1"] = "ar1"
    sigma_arcsec: float = Field(ge=0)
    rho_per_day: float = Field(ge=0, lt=1)


class HeteroscedasticNoiseCfg(_Cfg):
    """Gaussian noise whose σ grows for faint targets and low solar elongation (twilight,
    low altitude): σ = sigma_arcsec · (1 + per_magnitude · max(m − mag_ref, 0))
    · (1 + low_elongation · max(90° − elongation, 0) / 90°)."""

    type: Literal["heteroscedastic"] = "heteroscedastic"
    sigma_arcsec: float = Field(ge=0)
    per_magnitude: float = Field(default=0.0, ge=0)
    mag_ref: float = 0.0
    low_elongation: float = Field(default=0.0, ge=0)


class OutlierNoiseCfg(_Cfg):
    """Gross errors: a `fraction` of observations get an extra isotropic N(0, scale) error."""

    type: Literal["outliers"] = "outliers"
    fraction: float = Field(ge=0, le=1)
    scale_arcsec: float = Field(gt=0)


class ClockNoiseCfg(_Cfg):
    """Errors in the *published* times: constant offset plus Gaussian jitter (days)."""

    type: Literal["clock"] = "clock"
    sigma_days: float = Field(default=0.0, ge=0)
    offset_days: float = 0.0


class PositionGaussianNoiseCfg(_Cfg):
    """T0 only: isotropic 3D Gaussian noise on published heliocentric positions."""

    type: Literal["position_gaussian"] = "position_gaussian"
    sigma_au: float = Field(ge=0)


NoiseCfg = Annotated[
    GaussianNoiseCfg
    | BiasNoiseCfg
    | QuantizeNoiseCfg
    | DriftNoiseCfg
    | Ar1NoiseCfg
    | HeteroscedasticNoiseCfg
    | OutlierNoiseCfg
    | ClockNoiseCfg
    | PositionGaussianNoiseCfg,
    Field(discriminator="type"),
]


# ----------------------------------------------------------------------------- scenario


class Scenario(_Cfg):
    name: str
    description: str = ""
    duration_days: float = Field(gt=0)
    holdout_fraction: float = Field(0.25, gt=0, lt=1)
    knowledge_tier: Literal["T0", "T1", "T2", "T3"] = "T1"
    epoch_calendar: str | None = None
    observer: ObserverCfg = KeplerianObserverCfg()
    bodies: list[BodyCfg]
    schedule: ScheduleCfg
    noise: list[NoiseCfg] = Field(default_factory=list[NoiseCfg])
    report_uncertainties: bool = False
    realism: Realism = Field(default_factory=Realism)
    dense_points: int = Field(2001, ge=10, description="Sealed dense truth grid size.")
    definitions: dict[str, Any] = Field(
        default_factory=dict[str, Any],
        description="Free-form YAML anchors shared by several entries (e.g. an N-body system).",
    )

    @model_validator(mode="after")
    def _check(self) -> Scenario:
        ids = [b.id for b in self.bodies]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate body ids")
        if not [b for b in self.bodies if not b.hidden]:
            raise ValueError("at least one observed (non-hidden) body is required")
        sky_noise = [n for n in self.noise if n.type != "position_gaussian"]
        if self.knowledge_tier == "T0" and sky_noise:
            raise ValueError("T0 publishes positions: use position_gaussian noise only")
        if self.knowledge_tier == "T1" and isinstance(self.observer, TruthObserverCfg):
            raise ValueError(
                "T1 publishes the observer's orbital elements, which a 'truth' observer does "
                "not have: use T2/T3 (observer inferred) for a non-Keplerian observer"
            )
        if self.knowledge_tier == "T0" and self.realism.any:
            raise ValueError("realism effects apply to sky directions, not T0 positions")
        if self.knowledge_tier != "T0" and len(sky_noise) != len(self.noise):
            raise ValueError("position_gaussian noise is only valid for tier T0")
        return self

    @property
    def t_split(self) -> float:
        """Start of the sealed holdout window."""
        return self.duration_days * (1.0 - self.holdout_fraction)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


def load_scenario(path: str | Path) -> Scenario:
    return Scenario.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def apply_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply dotted-path overrides (`noise.0.sigma_arcsec: 60`) to a raw config dict."""
    import copy

    out = copy.deepcopy(data)
    for path, value in overrides.items():
        node: Any = out
        keys = path.split(".")
        for k in keys[:-1]:
            node = node[int(k)] if isinstance(node, list) else node.setdefault(k, {})
        last = keys[-1]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value
    return out
