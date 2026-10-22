"""Jan: registry, Gauss IOD, least squares, Fourier (SPEC §6, Phase 1 acceptance)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import jan
from jan.base import make_config
from jan.models.conic import CircularOrbit, ConicOrbit, EllipticOrbit
from jan.strategies.classical.gauss_iod import gauss_method
from jt_core.angles import great_circle_separation
from jt_core.conics import ConicElements, elements_to_state
from jt_core.constants import ARCSEC, DEG, GM_SUN
from jt_core.frames import lonlat_to_unit
from tycho import Scenario, generate, load_scenario

from .conftest import SCENARIOS, small_scenario_dict

MARS = ConicElements(1.381, 0.0934, 1.85 * DEG, 49.6 * DEG, 286.5 * DEG, 120.0)
EARTH = ConicElements(0.98329, 0.0167, 0.0, 0.0, 102.94 * DEG, 2.0)


def test_registry() -> None:
    names = jan.list_strategies()
    assert {"classical.gauss_iod", "classical.lsq", "ptolemy.fourier"} <= set(names)
    s = jan.get_strategy("classical.lsq")
    assert "T1" in s.supported_tiers
    with pytest.raises(KeyError):
        jan.get_strategy("nope")


@pytest.mark.parametrize("family", [EllipticOrbit, ConicOrbit, CircularOrbit])
def test_family_roundtrip(family: type) -> None:
    import jax.numpy as jnp

    el = ConicElements(1.5, 0.0 if family is CircularOrbit else 0.2, 0.1, 1.0, 2.0, 50.0)
    fam = family(t0=300.0, gm=GM_SUN)
    th = fam.theta_from_elements(el)
    t = np.linspace(0, 2000, 7)
    r1, _ = elements_to_state(el, t)
    r2 = np.asarray(fam.position(jnp.asarray(th), jnp.asarray(t)))
    assert np.allclose(r1, r2, atol=1e-6)


def test_elliptic_family_wraps_consistently() -> None:
    """A 2π wrap of ϖ must shift t_p by exactly one period (regression)."""
    import jax.numpy as jnp

    fam = EllipticOrbit(t0=2700.0, gm=GM_SUN)
    th = fam.theta_from_elements(MARS)
    t = np.linspace(0, 7000, 9)
    r1, _ = elements_to_state(MARS, t)
    assert np.allclose(np.asarray(fam.position(jnp.asarray(th), jnp.asarray(t))), r1, atol=1e-9)


def test_gauss_method_noise_free() -> None:
    t = np.array([300.0, 330.0, 360.0])
    rm, vm = elements_to_state(MARS, t)
    re, _ = elements_to_state(EARTH, t)
    u = (rm - re) / np.linalg.norm(rm - re, axis=1, keepdims=True)
    sols = gauss_method(t, u, re, GM_SUN)
    best = min(sols, key=lambda s: np.linalg.norm(s[0] - rm[1]))
    assert np.linalg.norm(best[0] - rm[1]) < 1e-9
    assert np.linalg.norm(best[1] - vm[1]) < 1e-11


def _fit(run, name: str, cfg: dict | None = None):
    s = jan.get_strategy(name)
    return s.fit(run.observations, run.meta, make_config(s, cfg), 0, Path("."))


def _holdout_rms(run, h) -> float:
    hold = run.holdout
    u_true = lonlat_to_unit(hold["lon_true"].to_numpy(), hold["lat_true"].to_numpy())
    pred = h.predict_sky(hold["t"].to_numpy(), "target").mean
    return float(np.sqrt(np.mean((great_circle_separation(pred, u_true) / ARCSEC) ** 2)))


def test_lsq_mars_like_acceptance() -> None:
    """Phase 1: a within 0.5%, e within 0.005, angles within 0.5°, holdout RMS < 2σ."""
    run = generate(load_scenario(SCENARIOS / "mars_like.yaml"), 1)
    h = _fit(run, "classical.lsq")
    est = h.elements("target")
    assert est is not None
    el = est.point
    tr = run.truth["bodies"]["target"]["elements"]
    a_hat, a_true = el["q"] / (1 - el["e"]), tr["q"] / (1 - tr["e"])
    assert abs(a_hat / a_true - 1) < 0.005
    assert abs(el["e"] - tr["e"]) < 0.005
    for k in ("i", "Omega", "omega"):
        d = (el[k] - tr[k] + np.pi) % (2 * np.pi) - np.pi
        assert abs(d) < 0.5 * DEG, k
    assert _holdout_rms(run, h) < 2 * run.run_info["noise_sigma_arcsec"]
    assert est.cov is not None and est.cov_names is not None


def test_lsq_handles_coplanar_circle() -> None:
    run = generate(load_scenario(SCENARIOS / "circle_coplanar.yaml"), 0)
    h = _fit(run, "classical.lsq")
    est = h.elements("target")
    assert est is not None and abs(est.point["q"] / 1.52 - 1) < 1e-3
    assert _holdout_rms(run, h) < 10


def test_lsq_t0_positions() -> None:
    d = small_scenario_dict(
        knowledge_tier="T0", noise=[{"type": "position_gaussian", "sigma_au": 1e-4}]
    )
    run = generate(Scenario.model_validate(d), 0)
    h = _fit(run, "classical.lsq")
    est = h.elements("target")
    assert est is not None
    el = est.point
    assert abs(el["q"] / 1.381 - 1) < 1e-3 and abs(el["e"] - 0.0934) < 1e-3


def test_lsq_trace() -> None:
    run = generate(Scenario.model_validate(small_scenario_dict()), 0)
    h = _fit(run, "classical.lsq", {"record_trace": True, "trace_max_frames": 5})
    frames = h.trace()
    assert frames and len(frames) <= 5
    h0 = h.at_frame(frames[0])
    assert h0 is not None and h0.predict_sky(np.array([10.0]), "target").mean.shape == (1, 3)


def test_fourier_interpolates_and_has_no_physics() -> None:
    run = generate(Scenario.model_validate(small_scenario_dict()), 0)
    h = _fit(run, "ptolemy.fourier")
    assert h.kind == "nonparametric"
    assert h.heliocentric_position(np.array([1.0]), "target") is None
    assert h.elements("target") is None
    assert _holdout_rms(run, h) < 1000


@pytest.mark.slow
def test_fourier_extrapolates_worse_than_lsq() -> None:
    """Phase 1 acceptance, measured as the median over seeds 0–4 (see notebooks/phase1)."""
    sc = load_scenario(SCENARIOS / "mars_like.yaml")
    ratios = []
    for seed in range(5):
        run = generate(sc, seed)
        ratios.append(
            _holdout_rms(run, _fit(run, "ptolemy.fourier"))
            / _holdout_rms(run, _fit(run, "classical.lsq"))
        )
    assert np.median(ratios) >= 5.0, ratios
