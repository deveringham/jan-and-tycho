"""Phase 3: realistic noise, schedules, unknown observers, Kepler's triangulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import jan
from jan.base import make_config
from jan.reconstruct.periodic_triangulation import DirectionInterpolator, intersect_lines
from jt_core.angles import great_circle_separation
from jt_core.constants import ARCSEC
from jt_core.frames import lonlat_to_unit
from jt_core.schema import SUN_ID
from tycho import Scenario, generate
from tycho.noise import apply_sky_noise
from tycho.scenario import (
    Ar1NoiseCfg,
    ClockNoiseCfg,
    DriftNoiseCfg,
    HeteroscedasticNoiseCfg,
    OutlierNoiseCfg,
)

from .conftest import small_scenario_dict


def _u(n: int) -> np.ndarray:
    g = np.random.default_rng(0)
    return lonlat_to_unit(g.uniform(0, 2 * np.pi, n), g.uniform(-0.2, 0.2, n))


def _sep(res, u: np.ndarray) -> np.ndarray:
    return great_circle_separation(lonlat_to_unit(res.lon, res.lat), u) / ARCSEC


def test_outliers_fraction_and_flags() -> None:
    u = _u(4000)
    res = apply_sky_noise(
        u, np.arange(4000.0), [OutlierNoiseCfg(fraction=0.05, scale_arcsec=3600)], 1, "x"
    )
    assert abs(res.outlier.mean() - 0.05) < 0.015
    sep = _sep(res, u)
    assert np.median(sep[res.outlier]) > 1000 and np.all(sep[~res.outlier] < 1e-6)


def test_drift_grows_linearly() -> None:
    u = lonlat_to_unit(np.zeros(3), np.zeros(3))
    t = np.array([0.0, 365.25, 730.5])
    res = apply_sky_noise(u, t, [DriftNoiseCfg(lon_arcsec_per_year=10)], 0, "x")
    assert np.allclose(_sep(res, u), [0, 10, 20], atol=1e-6)


def test_ar1_is_correlated() -> None:
    n = 3000
    u = lonlat_to_unit(np.zeros(n), np.zeros(n))
    t = np.arange(n, dtype=float)
    res = apply_sky_noise(u, t, [Ar1NoiseCfg(sigma_arcsec=60, rho_per_day=0.9)], 2, "x")
    e = (res.lon + np.pi) % (2 * np.pi) - np.pi
    assert abs(np.corrcoef(e[:-1], e[1:])[0, 1] - 0.9) < 0.05
    assert abs(np.std(e) / ARCSEC / 60 - 1) < 0.2


def test_heteroscedastic_sigma_grows_with_magnitude() -> None:
    u = _u(2000)
    mag = np.where(np.arange(2000) < 1000, 0.0, 4.0)
    res = apply_sky_noise(
        u,
        np.zeros(2000),
        [HeteroscedasticNoiseCfg(sigma_arcsec=10, per_magnitude=0.5)],
        0,
        "x",
        magnitude=mag,
    )
    assert np.allclose(res.sigma[:1000] / ARCSEC, 10) and np.allclose(res.sigma[1000:] / ARCSEC, 30)
    sep = _sep(res, u)
    assert np.median(sep[1000:]) / np.median(sep[:1000]) == pytest.approx(3, rel=0.15)


def test_clock_changes_published_times_only() -> None:
    u = _u(10)
    t = np.arange(10.0)
    res = apply_sky_noise(u, t, [ClockNoiseCfg(offset_days=0.5)], 0, "x")
    assert np.allclose(res.t_pub, t + 0.5) and np.allclose(_sep(res, u), 0, atol=1e-6)


def test_seasons_schedule_and_reasons() -> None:
    d = small_scenario_dict()
    d["schedule"] = {
        "type": "seasons",
        "cadence_days": 5,
        "windows": [[300, 60]],
        "visibility": {"dropout_prob": 0.0},
    }
    run = generate(Scenario.model_validate(d), 0)
    phase = np.mod(run.observations.t, 365.25)
    assert np.all((phase >= 300) | (phase <= 60))
    assert "season" in set(run.schedule["reason"])


def test_t2_sun_stream_and_hidden_observer() -> None:
    run = generate(Scenario.model_validate(small_scenario_dict(knowledge_tier="T2")), 0)
    assert run.meta.observer_elements is None and run.meta.sun_stream
    sun = run.observations.sun()
    assert sun is not None and len(sun) > 50
    assert run.observations.body_ids == ["target"]  # the Sun stream is not a "body"
    t3 = generate(Scenario.model_validate(small_scenario_dict(knowledge_tier="T3")), 0)
    assert t3.observations.sun() is None and not t3.meta.sun_stream


def test_line_intersection_exact() -> None:
    target = np.array([1.0, 2.0, 0.1])
    origins = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1.0, 0.0]])
    dirs = (target - origins) / np.linalg.norm(target - origins, axis=1, keepdims=True)
    x, _, chi = intersect_lines(origins, dirs)
    assert np.allclose(x, target, atol=1e-12) and chi < 1e-9


def test_direction_interpolator() -> None:
    t = np.arange(0, 100, 10.0)
    u = lonlat_to_unit(0.01 * t, np.zeros_like(t))
    it = DirectionInterpolator(t, u, window=25)
    v = it(35.0)
    assert v is not None and np.allclose(v, lonlat_to_unit(0.35, 0.0), atol=1e-3)  # quadratic fit
    assert it(150.0) is None


@pytest.mark.parametrize("tier", ["T2", "T3"])
def test_lsq_recovers_observer(tier: str) -> None:
    run = generate(Scenario.model_validate(small_scenario_dict(knowledge_tier=tier)), 0)
    s = jan.get_strategy("classical.lsq")
    h = s.fit(run.observations, run.meta, make_config(s), 0, Path("."))
    oe = h.observer_elements()
    assert oe is not None
    assert abs(oe.point["e"] - 0.0167) < 0.005
    assert abs(h.elements("target").point["e"] - 0.0934) < 0.005  # type: ignore[union-attr]
    assert h.observer_position(np.array([10.0])) is not None


@pytest.mark.parametrize("tier", ["T1", "T2"])
def test_kepler_triangulate(tier: str) -> None:
    d = small_scenario_dict(knowledge_tier=tier, duration_days=4000)
    run = generate(Scenario.model_validate(d), 0)
    s = jan.get_strategy("kepler.triangulate")
    h = s.fit(run.observations, run.meta, make_config(s), 0, Path("."))
    part = h.parts["target"]  # type: ignore[attr-defined]
    assert abs(part.period / 687.0 - 1) < 0.01
    tri = part.info["triangulated_elements"]
    a = tri["q"] / (1 - tri["e"])
    assert abs(a / (1.381 / (1 - 0.0934)) - 1) < 0.01  # geometry alone: a within 1%
    assert h.kind == "dynamical"
    if tier == "T2":
        assert abs(part.info["triangulated_observer_elements"]["e"] - 0.0167) < 0.005
        assert h.observer_elements() is not None
    assert len(h.to_dict()["bodies"]["target"]["triangulation"]) > 20


def test_sun_stream_row_ids() -> None:
    run = generate(Scenario.model_validate(small_scenario_dict(knowledge_tier="T2")), 0)
    assert SUN_ID in set(run.holdout["body_id"])
