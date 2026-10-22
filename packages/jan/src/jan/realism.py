"""Realism effects on Jan's side (SPEC Phase 5): every strategy is run through
`RealismAdapter`, so no strategy has to know about light-time, aberration or the frame.

* Frame: equatorial data are rotated to the ecliptic before fitting and predictions are
  rotated back (exact, always applied: it is a convention, not physics).
* Aberration: with the observer's orbit known (T1) the data are reduced exactly
  (`unaberrate`) before fitting, and predictions re-aberrated.
* Light-time (and aberration with an inferred observer): a two-pass differential
  correction, as astronomers reduce astrometry. Pass 1 fits the data as they are; its
  heliocentric positions give each observation's correction (geometric − apparent model
  direction, ~10″), which is applied to the data for pass 2. Predictions add the
  correction back from the final hypothesis. A hypothesis without heliocentric positions
  (a sky-plane model) has no light-time to model: it keeps pass 1, which describes the
  light-time-delayed path directly.

`realism: ignore` in a strategy's config skips the physical corrections (the frame is
still converted), to measure the bias they would otherwise leave. Published per-axis
sigmas are taken as isotropic (unchanged by the frame rotation).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jan.base import BaseHypothesis, StrategyConfig
from jan.models.observer import observer_function
from jt_core.angles import tangent_basis
from jt_core.frames import lonlat_to_unit, normalize, unit_to_lonlat
from jt_core.realism import (
    Realism,
    aberrate,
    ecliptic_to_frame,
    frame_to_ecliptic,
    retarded_positions,
    unaberrate,
)
from jt_core.schema import (
    SUN_ID,
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
PosFn = Callable[[FloatArray], FloatArray]
FD_STEP_DAYS = 0.01


def _velocity(pos: PosFn) -> PosFn:
    """Central-difference velocity of a position function (relative error ~1e-9 for the
    Earth at a 0.01-day step)."""
    return lambda t: (pos(t + FD_STEP_DAYS) - pos(t - FD_STEP_DAYS)) / (2 * FD_STEP_DAYS)


def _with_directions(obs: ObservationSet, u: FloatArray) -> ObservationSet:
    lon, lat = unit_to_lonlat(u)
    table = obs.table.copy()
    table["lon"], table["lat"] = lon, lat
    return ObservationSet(table)


def _units(obs: ObservationSet) -> FloatArray:
    return lonlat_to_unit(obs.table["lon"].to_numpy(), obs.table["lat"].to_numpy())


def _frame_jacobian(u_ecl: FloatArray, real: Realism) -> FloatArray:
    """(N, 2, 2) map of tangent-plane (east, north) offsets from the ecliptic frame to the
    published frame at each direction."""
    e1, n1 = tangent_basis(u_ecl)
    u2 = ecliptic_to_frame(u_ecl, real)
    e2, n2 = tangent_basis(u2)
    re1, rn1 = ecliptic_to_frame(e1, real), ecliptic_to_frame(n1, real)
    j = np.empty((len(u_ecl), 2, 2))
    j[:, 0, 0] = np.sum(e2 * re1, axis=1)
    j[:, 0, 1] = np.sum(e2 * rn1, axis=1)
    j[:, 1, 0] = np.sum(n2 * re1, axis=1)
    j[:, 1, 1] = np.sum(n2 * rn1, axis=1)
    return j


class ApparentHypothesis(BaseHypothesis):
    """Wraps a hypothesis fitted to geometric ecliptic directions so that its sky
    predictions are apparent and in the published frame. Everything else (positions,
    elements, laws, the observer) is geometric and passes through unchanged."""

    def __init__(
        self,
        inner: BaseHypothesis,
        real: Realism,
        observer: PosFn | None,
        light_time: bool,
        aberration: bool,
        notes: str,
    ) -> None:
        self.inner = inner
        self.real = real
        self.observer = observer
        self.light_time = light_time
        self.aberration = aberration
        self._notes = notes
        self.kind: HypothesisKind = inner.kind

    def __getattr__(self, name: str) -> Any:
        # Strategy-specific attributes (info, parts, ...) of the wrapped hypothesis.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def observer_fn(self) -> PosFn | None:
        if self.observer is not None:
            return self.observer
        if self.inner.observer_position(np.zeros(1)) is None:
            return None

        def fn(t: FloatArray) -> FloatArray:
            p = self.inner.observer_position(t)
            assert p is not None
            return p.mean

        return fn

    def offset(self, t: FloatArray, body_id: str, u_model: FloatArray) -> FloatArray | None:
        """Apparent − geometric model direction (N, 3) in the ecliptic frame, given the
        model's geometric directions; None when the hypothesis cannot supply what the
        effects need (observer positions, or body positions for light-time)."""
        obs = self.observer_fn()
        if obs is None:
            return None
        r_o = obs(t)
        g = a = u_model
        if self.light_time and body_id != SUN_ID:  # the Sun is at rest at the origin
            pos = self.inner.heliocentric_position(t, body_id)
            if pos is None:
                return None

            def body(tt: FloatArray) -> FloatArray:
                p = self.inner.heliocentric_position(tt, body_id)
                assert p is not None
                return p.mean

            r_ret, _ = retarded_positions(body, t, r_o)
            g, a = normalize(pos.mean - r_o), normalize(r_ret - r_o)
        if self.aberration:
            a = aberrate(a, _velocity(obs)(t))
        return a - g

    def predict_sky(self, t: FloatArray, body_id: str) -> SkyPrediction:
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        p = self.inner.predict_sky(t, body_id)
        mean, samples = p.mean, p.samples
        if self.light_time or self.aberration:
            off = self.offset(t, body_id, mean)
            if off is not None:
                mean = normalize(mean + off)
                samples = None if samples is None else normalize(samples + off)
        cov = p.cov
        if cov is not None and self.real.frame != "ecliptic":
            j = _frame_jacobian(mean, self.real)
            cov = j @ cov @ np.transpose(j, (0, 2, 1))
        mean = ecliptic_to_frame(mean, self.real)
        samples = None if samples is None else ecliptic_to_frame(samples, self.real)
        return SkyPrediction(mean=mean, samples=samples, cov=cov)

    # ---- pass-through
    def heliocentric_position(self, t: FloatArray, body_id: str) -> PositionPrediction | None:
        return self.inner.heliocentric_position(t, body_id)

    def elements(self, body_id: str) -> ElementsEstimate | None:
        return self.inner.elements(body_id)

    def law(self) -> LawEstimate | None:
        return self.inner.law()

    def model_comparison(self) -> list[ModelScore] | None:
        return self.inner.model_comparison()

    def trace(self) -> list[TraceFrame] | None:
        return self.inner.trace()

    def at_frame(self, frame: TraceFrame) -> BaseHypothesis | None:
        h = self.inner.at_frame(frame)
        if h is None:
            return None
        return ApparentHypothesis(
            h, self.real, self.observer, self.light_time, self.aberration, self._notes
        )

    def observation_sigma(self, body_id: str) -> float | None:
        return self.inner.observation_sigma(body_id)

    def observer_position(self, t: FloatArray) -> PositionPrediction | None:
        return self.inner.observer_position(t)

    def observer_elements(self) -> ElementsEstimate | None:
        return self.inner.observer_elements()

    def acceleration(
        self, x: FloatArray, v: FloatArray, t: FloatArray, body_id: str
    ) -> FloatArray | None:
        return self.inner.acceleration(x, v, t, body_id)

    def hidden_bodies(self, t: FloatArray) -> dict[str, PositionPrediction] | None:
        return self.inner.hidden_bodies(t)

    def notes(self) -> str:
        inner = self.inner.notes()
        return f"{inner} [{self._notes}]" if inner else f"[{self._notes}]"

    def to_dict(self) -> dict[str, Any]:
        return {**self.inner.to_dict(), "realism": self._notes, "notes": self.notes()}


class RealismAdapter:
    """A strategy run through the realism reductions of `meta.realism`."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name: str = inner.name
        self.supported_tiers: set[str] = inner.supported_tiers
        self.Config: type[StrategyConfig] = inner.Config

    def fit(
        self,
        obs: ObservationSet,
        meta: PublicMeta,
        config: StrategyConfig,
        rng_seed: int,
        workdir: Path,
    ) -> BaseHypothesis:
        real = meta.realism
        if not real.any or not obs.is_sky:
            return self.inner.fit(obs, meta, config, rng_seed, workdir)
        plain = meta.model_copy(update={"realism": Realism()})
        u = frame_to_ecliptic(_units(obs), real)
        obs_e = _with_directions(obs, u)
        ignore = config.realism == "ignore"
        light_time = real.light_time and not ignore
        aberration = real.aberration and not ignore
        try:
            known: PosFn | None = observer_function(meta)
        except ValueError:
            known = None
        frame_note = f"{real.frame} → ecliptic" if real.frame != "ecliptic" else "ecliptic"
        if not (light_time or aberration):
            h = self.inner.fit(obs_e, plain, config, rng_seed, workdir)
            what = "ignored" if real.physical else "none"
            return ApparentHypothesis(h, real, known, False, False, f"{frame_note}; effects {what}")
        exact_ab = aberration and known is not None
        if exact_ab:
            # Exact reduction with the known observer's velocity, no hypothesis needed.
            assert known is not None
            u = unaberrate(u, _velocity(known)(obs.table["t"].to_numpy()))
            obs_e = _with_directions(obs, u)
        notes = ["aberration reduced with the known observer"] if exact_ab else []
        if not light_time and exact_ab:
            h = self.inner.fit(obs_e, plain, config, rng_seed, workdir)
            return ApparentHypothesis(h, real, known, False, True, "; ".join([frame_note, *notes]))
        # Two-pass differential correction for what is left (light-time, and aberration with
        # an inferred observer).
        h1 = self.inner.fit(obs_e, plain, config, rng_seed, workdir)
        probe = ApparentHypothesis(h1, real, known, light_time, aberration and not exact_ab, "")
        offsets = np.zeros_like(u)
        t_all = obs_e.table["t"].to_numpy()
        for bid in obs_e.table["body_id"].unique().tolist():
            m = (obs_e.table["body_id"] == bid).to_numpy()
            if bid == SUN_ID:  # the Sun stream's model direction is −r_obs
                obs_fn = probe.observer_fn()
                u_model = None if obs_fn is None else normalize(-obs_fn(t_all[m]))
            else:
                u_model = h1.predict_sky(t_all[m], str(bid)).mean
            off = None if u_model is None else probe.offset(t_all[m], str(bid), u_model)
            if off is None:
                notes.append("no heliocentric model for light-time: fitted to its apparent path")
                return ApparentHypothesis(
                    h1, real, known, False, exact_ab, "; ".join([frame_note, *notes])
                )
            offsets[m] = off
        obs_c = _with_directions(obs, normalize(u - offsets))
        h2 = self.inner.fit(obs_c, plain, config, rng_seed, workdir)
        two_pass = [n for n, on in (("light-time", light_time), ("aberration", not exact_ab)) if on]
        notes.append(f"{' + '.join(two_pass)} by two-pass differential correction")
        return ApparentHypothesis(
            h2, real, known, light_time, aberration, "; ".join([frame_note, *notes])
        )
