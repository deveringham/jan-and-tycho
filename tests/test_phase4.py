"""Phase 4: alien physics, kinematic paths, law discovery and structural scoring."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from jan.reconstruct.polygon import fit_polygon
from jan.strategies.discovery.sindy import best_subset
from jt_core.conics import elements_to_state
from rudolf.metrics.structural import compare_radial
from tycho.scenario import (
    CentralForceTruthCfg,
    ElementsCfg,
    InitialStateCfg,
    KinematicPolygonCfg,
    NBodyMemberCfg,
    NBodyTruthCfg,
)
from tycho.truth.central_force import CentralForceTruth
from tycho.truth.kinematic import KinematicPolygonTruth
from tycho.truth.nbody import NBodyTruth

MARS = ElementsCfg(q=1.381, e=0.0934, i_deg=1.85, Omega_deg=49.6, omega_deg=286.5, t_p=120.0)
R = sp.Symbol("r", positive=True)


def test_power_law_two_is_kepler() -> None:
    truth = CentralForceTruth(CentralForceTruthCfg(law="power_law", exponent=2.0, elements=MARS))
    t = np.linspace(0, 2000, 50)
    kep, _ = elements_to_state(MARS.to_elements(), t, truth.cfg.gm)
    assert np.max(np.linalg.norm(truth.position(t) - kep, axis=1)) < 1e-8


def test_hooke_orbit_is_centred_ellipse() -> None:
    k = 1e-4
    cfg = CentralForceTruthCfg(
        law="hooke",
        k_per_day2=k,
        initial=InitialStateCfg(position_au=(1.0, 0.0, 0.0), velocity_au_per_day=(0.0, 0.005, 0.0)),
    )
    period = 2 * np.pi / np.sqrt(k)
    x = CentralForceTruth(cfg).position(np.array([0.0, period / 2, period]))
    assert np.allclose(x[1], -x[0], atol=1e-8) and np.allclose(x[2], x[0], atol=1e-8)


def test_massless_nbody_member_is_kepler() -> None:
    cfg = NBodyTruthCfg(
        system=[
            NBodyMemberCfg(id="a", mass_msun=0.0, elements=MARS),
            NBodyMemberCfg(
                id="b",
                mass_msun=0.0,
                elements=ElementsCfg(q=5.0, e=0.05, i_deg=1, Omega_deg=10, omega_deg=20, t_p=0),
            ),
        ],
        member="a",
    )
    t = np.linspace(0, 1500, 20)
    kep, _ = elements_to_state(MARS.to_elements(), t, cfg.gm_sun)
    assert np.max(np.linalg.norm(NBodyTruth(cfg).position(t) - kep, axis=1)) < 1e-7


def test_polygon_truth_visits_vertices_at_constant_speed() -> None:
    cfg = KinematicPolygonCfg(
        vertices_au=[(-1.0, -0.5), (1.0, -0.5), (1.0, 0.5), (-1.0, 0.5)],
        period_days=600,
        i_deg=10,
    )
    truth = KinematicPolygonTruth(cfg)
    # Constant speed: vertex k is reached after the cumulative perimeter fraction.
    x = truth.position(np.array([0.0, 200.0, 300.0, 500.0]))
    assert np.allclose(x, truth.vertices_3d(), atol=1e-12)
    speed = np.linalg.norm(truth.velocity(np.linspace(10, 590, 40)), axis=1)
    assert np.ptp(speed[speed > 0.5 * speed.max()]) / speed.max() < 1e-6


def _rectangle_points(n: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    truth = KinematicPolygonTruth(
        KinematicPolygonCfg(
            vertices_au=[(-1.5, -0.8), (1.5, -0.8), (1.5, 0.8), (-1.5, 0.8)],
            period_days=1.0,
            i_deg=20,
            Omega_deg=40,
        )
    )
    s = np.sort(np.random.default_rng(seed).uniform(0, 1, n))
    # No points near the first corner (as if it always fell in an observing gap).
    s = s[np.abs(np.mod(s + 0.5, 1.0) - 0.5) > 0.04]
    pts = truth.position(s) + np.random.default_rng(seed + 1).normal(0, noise, (len(s), 3))
    return pts, truth.vertices_3d()


@pytest.mark.parametrize("eps", [0.04, 0.08])
def test_polygon_fit_recovers_corner_in_gap(eps: float) -> None:
    pts, verts = _rectangle_points(150, 0.01, 0)
    diam = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    poly = fit_polygon(pts, eps * diam)
    assert poly is not None and len(poly.vertices) == 4
    err = [np.min(np.linalg.norm(poly.vertices - v, axis=1)) for v in verts]
    assert max(err) < 0.02 * 1.6  # within 2% of the short side


def test_polygon_fit_rejects_smooth_curve() -> None:
    phi = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    ellipse = np.stack([1.5 * np.cos(phi), np.sin(phi), 0 * phi], axis=1)
    poly = fit_polygon(ellipse, 0.02 * 1.5)
    # Either no small polygon, or one with many vertices (never a 4-gon).
    assert poly is None or len(poly.vertices) > 6


def test_best_subset_handles_collinear_library() -> None:
    r = np.linspace(1.3, 1.7, 400)
    powers = (-3, -2, -1, 0, 1)
    a = np.stack([r**p for p in powers], axis=1)
    b = -3e-4 * r**-2.0 + np.random.default_rng(0).normal(0, 1e-9, r.size)
    coef, _ = best_subset(a, b, 2, 0.5)
    assert np.flatnonzero(coef).tolist() == [1]
    assert coef[1] == pytest.approx(-3e-4, rel=1e-4)


def test_compare_radial_verdicts() -> None:
    gm, r = 2.96e-4, np.linspace(0.8, 2.0, 200)
    truth = -gm / R**2
    assert compare_radial(truth, -gm / R**2, r)[0] == "exact"
    assert compare_radial(truth, sp.Float(-1.05 * gm) / R**2, r)[0] == "up_to_constants"
    assert compare_radial(truth, sp.Float(-gm) * R, r)[0] == "mismatch"


# --- end to end (slow) ----------------------------------------------------------------


def _run(name: str, strategy: str, config: dict | None, tmp_path, seed: int = 0, **over):
    import yaml

    from jt_bench.harness import ensure_run, fit, score
    from tycho import Scenario
    from tycho.scenario import apply_overrides

    from .conftest import SCENARIOS

    data = yaml.safe_load((SCENARIOS / f"{name}.yaml").read_text(encoding="utf-8"))
    sc = Scenario.model_validate(apply_overrides(data, over))
    d = ensure_run(sc, seed, run_dir=tmp_path / name / str(seed))
    res = fit(d, strategy, config)
    return score(d, res.label, plots=False)["bodies"]["target"]


@pytest.mark.slow
def test_rectangle_vertices_and_rejection(tmp_path) -> None:
    s = _run("rectangle", "kepler.triangulate", None, tmp_path)
    assert s["vertices"]["vertex_error_frac"] < 0.02
    assert s["structural"]["answer"] == "kinematic" and s["structural"]["rejection_credit"]


@pytest.mark.slow
def test_power_law_exponent_t1(tmp_path) -> None:
    s = _run("precessing_power_law", "classical.lsq", {"family": "central-power-law"}, tmp_path)
    assert s["structural"]["symbolic"] in ("exact", "up_to_constants")
    assert s["structural"]["field_rel_error"] < 1e-3


@pytest.mark.slow
def test_hidden_perturber_longitude(tmp_path) -> None:
    s = _run("hidden_perturber", "classical.lsq", {"family": "perturbed-conic"}, tmp_path)
    assert s["hidden"]["longitude_error_deg"] < 10


@pytest.mark.slow
@pytest.mark.discovery
@pytest.mark.parametrize("strategy", ["discovery.sindy", "discovery.pysr"])
@pytest.mark.parametrize("scenario", ["mars_like", "hooke_trap"])
def test_discovery_recovers_law_t0(strategy: str, scenario: str, tmp_path) -> None:
    over = {"knowledge_tier": "T0", "noise": [{"type": "position_gaussian", "sigma_au": 1e-5}]}
    s = _run(scenario, strategy, None, tmp_path, **over)
    assert s["structural"]["symbolic"] in ("exact", "up_to_constants")
