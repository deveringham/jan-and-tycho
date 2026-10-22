"""Observation times and visibility (SPEC §5.5)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from jt_core.constants import DEG
from tycho.operator import Geometry
from tycho.scenario import (
    ExplicitScheduleCfg,
    PoissonScheduleCfg,
    RegularScheduleCfg,
    ScheduleCfg,
    SeasonsScheduleCfg,
    VisibilityCfg,
)

FloatArray = NDArray[np.float64]

# Gap reasons recorded in the sealed schedule (first failing filter wins).
OK = "observed"
WEATHER = "weather"
ELONGATION = "elongation"
MAGNITUDE = "magnitude"
SEASON = "season"


def candidate_times(cfg: ScheduleCfg, duration: float, rng: np.random.Generator) -> FloatArray:
    """Candidate observing nights in [0, duration], sorted and unique."""
    if isinstance(cfg, RegularScheduleCfg | SeasonsScheduleCfg):
        start = cfg.start_days if isinstance(cfg, RegularScheduleCfg) else 0.0
        base = np.arange(start, duration + 1e-9, cfg.cadence_days)
        jitter = rng.uniform(-cfg.jitter_days, cfg.jitter_days, size=base.size)
        t = base + jitter
    elif isinstance(cfg, PoissonScheduleCfg):
        n_max = int(duration * cfg.rate_per_day * 2 + 50)
        t = np.cumsum(rng.exponential(1.0 / cfg.rate_per_day, size=n_max))
    else:
        assert isinstance(cfg, ExplicitScheduleCfg)
        t = np.asarray(cfg.times, dtype=np.float64)
    t = t[(t >= 0.0) & (t <= duration)]
    return np.unique(t)


def in_season(cfg: ScheduleCfg, t: FloatArray) -> NDArray[np.bool_]:
    """True where t falls inside an observing window (always true for other schedules)."""
    if not isinstance(cfg, SeasonsScheduleCfg):
        return np.ones(t.shape, dtype=bool)
    phase = np.mod(t, cfg.cycle_days)
    ok = np.zeros(t.shape, dtype=bool)
    for start, end in cfg.windows:
        if start <= end:
            ok |= (phase >= start) & (phase <= end)
        else:
            ok |= (phase >= start) | (phase <= end)
    return ok


def weather(vis: VisibilityCfg, n: int, rng: np.random.Generator) -> NDArray[np.bool_]:
    """True where the night is clear. Shared by all bodies on the same night."""
    return rng.uniform(size=n) >= vis.dropout_prob


def visibility_reasons(
    vis: VisibilityCfg,
    clear: NDArray[np.bool_],
    geom: Geometry,
    season: NDArray[np.bool_] | None = None,
) -> NDArray[np.str_]:
    reason = np.full(clear.shape, OK, dtype=object)
    if vis.max_magnitude is not None:
        reason[np.isfinite(geom.magnitude) & (geom.magnitude > vis.max_magnitude)] = MAGNITUDE
    if vis.min_solar_elongation_deg is not None:
        too_close = np.isfinite(geom.elongation) & (
            geom.elongation < vis.min_solar_elongation_deg * DEG
        )
        reason[too_close] = ELONGATION
    reason[~clear] = WEATHER
    if season is not None:
        reason[~season] = SEASON
    return reason.astype(str)
