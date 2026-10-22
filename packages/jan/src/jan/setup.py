"""Shared setup for strategies fitting Bayesian families: data, starts, configs (core: no
optional dependencies)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from jan.base import StrategyConfig
from jan.inference import (
    SkyData,
    SkyModel,
    conic_starts,
    epicycle_starts,
    fit_observer_from_sun,
)
from jan.models.families import BayesFamily, ObserverFamily, make_bayes_family
from jan.models.observer import observer_function
from jt_core.conics import ConicElements, elements_to_state
from jt_core.schema import ObservationSet, PublicMeta

FloatArray = NDArray[np.float64]


class BayesConfig(StrategyConfig):
    likelihood: Literal["gaussian", "student_t"] = "gaussian"
    nu: float = Field(4.0, gt=0, description="Student-t degrees of freedom.")
    n_predictive_samples: int = Field(100, description="Posterior draws used for predictions.")


@dataclass
class BodySetup:
    """One body's data and starting points.

    T1: `observer` is the published observer. T2/T3: it is a first estimate (from the Sun
    stream, or the sky track's yearly epicycle), and `observer_family`/`observer_start`
    make every model fit the observer's orbit jointly.
    """

    ob: ObservationSet
    observer: Callable[[FloatArray], FloatArray]
    data: SkyData
    t0: float
    span: float
    gm: float
    observer_family: ObserverFamily | None = None
    observer_start: FloatArray | None = None
    _conics: list[ConicElements] | None = None

    def conics(self, seed: int) -> list[ConicElements]:
        if self._conics is None:
            r_obs = self.observer(self.ob.t)
            self._conics = conic_starts(self.ob, r_obs, self.gm, self.t0, seed)
        return self._conics

    def family(self, name: str) -> BayesFamily:
        return make_bayes_family(name, self.t0, self.gm, self.span)

    def model(self, name: str, cfg: BayesConfig) -> SkyModel:
        return SkyModel(self.family(name), self.data, cfg.likelihood, cfg.nu, self.observer_family)

    def starts(self, fam: BayesFamily, seed: int) -> list[FloatArray]:
        if fam.name == "epicycles":
            out = epicycle_starts(self.ob, self.t0)
            # A heliocentric conic seen from a circular Earth is an epicycle model too.
            out = out + _copernican_epicycles(self, seed)
        else:
            out = [x for x in (fam.from_elements(c) for c in self.conics(seed)) if x is not None]
            grid = getattr(fam, "grid_starts", None)
            if grid is not None and out:
                n = getattr(fam, "grid_bases", 1)  # conic starts the grid is laid around
                out = out[:2] + [g for b in out[:n] for g in grid(b)] + out[2:]
        if self.observer_start is None:
            return out
        return [np.concatenate([x, self.observer_start]) for x in out]


def _copernican_epicycles(setup: BodySetup, seed: int) -> list[FloatArray]:
    """Epicycle starts equivalent to the best heliocentric circle + the Earth's orbit."""
    out: list[FloatArray] = []
    conics = [c for c in setup.conics(seed) if c.e < 1]
    if not conics:
        return out
    c = conics[0]
    a = c.q / (1 - c.e)
    r_e = setup.observer(np.array([setup.t0]))[0]
    f_planet = 1.0 / (2 * np.pi * np.sqrt(a**3 / setup.gm))
    f_earth = 1.0 / 365.25
    lam_earth = float(np.arctan2(r_e[1], r_e[0]))
    n = 2 * np.pi * f_planet
    lam_planet = c.Omega + c.omega + n * (setup.t0 - c.t_p)
    # Geocentric vector = r_planet − r_earth: deferent = planet circle (radius 1 after
    # scaling by a), epicycle = the Earth's circle reversed (phase + π), ratio 1/a.
    out.append(
        np.array(
            [
                f_planet,
                f_earth,
                min(1.0 / a, 1.49),
                0.0,
                0.0,
                0.0,
                lam_planet - c.Omega,
                lam_earth + np.pi - c.Omega,
                np.cos(c.i),
                c.Omega,
            ]
        )
    )
    return out


def observer_known(meta: PublicMeta) -> bool:
    return meta.observer_elements is not None or meta.observer_position is not None


def setup_body(
    ob: ObservationSet, meta: PublicMeta, sun: ObservationSet | None = None
) -> BodySetup:
    """Data + starts for one body. Under T2/T3 the observer's orbit becomes a free family,
    started from the Sun stream (T2) or the sky track's yearly epicycle (T3)."""
    if not ob.is_sky:
        raise ValueError("Bayesian strategies need sky observations")
    t0, span, gm = float(np.median(ob.t)), float(np.ptp(ob.t)), meta.gm_sun
    if observer_known(meta):
        observer = observer_function(meta)
        return BodySetup(ob, observer, SkyData.from_obs(ob, observer(ob.t)), t0, span, gm)
    obs_fam = ObserverFamily(t0=t0, gm=gm)
    phi_o = fit_observer_from_sun(sun, gm, t0) if sun is not None else observer_from_sky(ob, t0, gm)
    observer = observer_positions(obs_fam, phi_o)
    data = SkyData.from_obs(ob, None, sun)
    return BodySetup(ob, observer, data, t0, span, gm, obs_fam, phi_o)


def observer_positions(fam: ObserverFamily, phi: FloatArray) -> Callable[[FloatArray], FloatArray]:
    el = ConicElements.from_array(np.asarray(fam.elements(jnp.asarray(phi))))
    return lambda t: elements_to_state(el, np.atleast_1d(np.asarray(t, dtype=np.float64)), fam.gm)[
        0
    ]


def observer_from_sky(ob: ObservationSet, t0: float, gm: float) -> FloatArray:
    """T3 start: the planet's yearly epicycle is the observer's orbit reflected
    (Copernicus), so its phase gives the observer's longitude; start circular at 1 AU."""
    best = None
    for st in epicycle_starts(ob, t0):
        f_d, f_e, _, _, _, _, ph_d, ph_e, cos_i, Om = st
        if cos_i < 0:
            continue
        # The term closest to one cycle per year is the reflected observer orbit.
        f, ph = (
            (f_e, ph_e) if abs(abs(f_e) - 1 / 365.25) < abs(abs(f_d) - 1 / 365.25) else (f_d, ph_d)
        )
        best = float(Om + ph + np.pi) if f > 0 else float(Om - ph + np.pi)
        break
    lam0 = 0.0 if best is None else best
    a = (gm * (365.25 / (2 * np.pi)) ** 2) ** (1 / 3)
    return ObserverFamily(t0=t0, gm=gm).wrap(np.array([np.log(a), 0.0, 0.0, lam0]))


def box_around(
    x: FloatArray, cov: FloatArray, model: SkyModel, n_sigma: float
) -> tuple[FloatArray, FloatArray]:
    """Box of ±n_sigma Laplace standard deviations, clipped to the prior (not for periodic dims)."""
    lo, hi = model.lo_hi()
    per = model.periodic()
    x = np.where(per, x, np.clip(x, lo, hi))
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    width = hi - lo
    sd = np.where(np.isfinite(sd), sd, np.inf)  # degenerate Laplace → whole prior range
    half = np.minimum(n_sigma * sd, np.where(per, width / 2, width))
    half = np.maximum(half, 1e-9 * width)
    blo, bhi = x - half, x + half
    blo = np.where(per, blo, np.maximum(blo, lo))
    bhi = np.where(per, bhi, np.minimum(bhi, hi))
    return blo, bhi
