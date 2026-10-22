"""Phase 5: realism effects (light-time, aberration, equatorial frame)."""

from __future__ import annotations

import numpy as np
import pytest

from jt_core.angles import great_circle_separation
from jt_core.conics import ConicElements, elements_to_state
from jt_core.constants import ARCSEC, GM_SUN
from jt_core.frames import lonlat_to_unit, normalize, unit_to_lonlat
from jt_core.realism import (
    C_AU_PER_DAY,
    OBLIQUITY_J2000_DEG,
    Realism,
    aberrate,
    ecliptic_to_frame,
    frame_to_ecliptic,
    retarded_positions,
    unaberrate,
)
from tycho import Scenario, generate

from .conftest import small_scenario_dict

EARTH = ConicElements(0.98329, 0.0167, 0.0, 0.0, np.radians(102.94), 2.0)


def test_light_takes_499_seconds_per_au() -> None:
    assert 86400 / C_AU_PER_DAY == pytest.approx(499.004784, rel=1e-8)


def test_equatorial_frame_round_trip_and_ecliptic_pole() -> None:
    real = Realism(frame="equatorial")
    pole = ecliptic_to_frame(np.array([0.0, 0.0, 1.0]), real)
    ra, dec = unit_to_lonlat(pole[None])
    assert np.degrees(ra[0]) == pytest.approx(270.0)  # 18h
    assert np.degrees(dec[0]) == pytest.approx(90.0 - OBLIQUITY_J2000_DEG)
    u = normalize(np.random.default_rng(0).normal(size=(50, 3)))
    assert np.allclose(frame_to_ecliptic(ecliptic_to_frame(u, real), real), u, atol=1e-15)


def test_annual_aberration_amplitude_and_inverse() -> None:
    t = np.linspace(0, 365, 40)
    _, v = elements_to_state(EARTH, t, GM_SUN)
    # A star at the ecliptic pole traces a circle of radius κ ≈ 20.5″.
    u = np.tile([0.0, 0.0, 1.0], (len(t), 1))
    sep = great_circle_separation(aberrate(u, v), u) / ARCSEC
    assert 20.1 < sep.min() and sep.max() < 21.0
    w = normalize(np.random.default_rng(1).normal(size=(len(t), 3)))
    assert np.allclose(unaberrate(aberrate(w, v), v), w, atol=1e-15)


def test_retarded_positions_solve_the_light_time_equation() -> None:
    def pos(tt: np.ndarray) -> np.ndarray:  # a fast body: 0.1 AU/day along +y at 3 AU
        return np.stack([np.full_like(tt, 3.0), 0.1 * tt, np.zeros_like(tt)], axis=1)

    t = np.array([0.0, 5.0])
    r, tau = retarded_positions(pos, t, np.zeros((2, 3)))
    assert np.allclose(np.linalg.norm(r, axis=1) / C_AU_PER_DAY, tau, rtol=1e-12)
    assert np.allclose(r, pos(t - tau), atol=1e-15)


def test_published_directions_include_the_effects() -> None:
    base = small_scenario_dict(noise=[{"type": "gaussian", "sigma_arcsec": 1e-6}])
    real = Realism(light_time=True, aberration=True, frame="equatorial")
    plain = generate(Scenario.model_validate(base), 0)
    eff = generate(Scenario.model_validate({**base, "realism": real.model_dump()}), 0)
    assert eff.meta.realism == real and plain.meta.realism == Realism()
    pub_p, pub_e = plain.observations.table, eff.observations.table
    u_p = lonlat_to_unit(pub_p["lon"].to_numpy(), pub_p["lat"].to_numpy())
    u_e = lonlat_to_unit(pub_e["lon"].to_numpy(), pub_e["lat"].to_numpy())
    shift = great_circle_separation(frame_to_ecliptic(u_e, real), u_p) / ARCSEC
    assert 5 < np.median(shift) < 60  # aberration ~20″ plus light-time ~10″
    traj = eff.trajectory
    assert {"ux", "uy", "uz"} <= set(traj.columns)


def test_realism_rejected_at_t0() -> None:
    d = small_scenario_dict(
        knowledge_tier="T0",
        noise=[{"type": "position_gaussian", "sigma_au": 1e-5}],
        realism={"aberration": True},
    )
    with pytest.raises(ValueError, match="realism"):
        Scenario.model_validate(d)


@pytest.mark.slow
@pytest.mark.parametrize("realism", ["model", "ignore"])
def test_realism_corrections_remove_the_bias(realism: str, tmp_path) -> None:
    from jt_bench.harness import ensure_run, fit, score
    from tycho import load_scenario

    from .conftest import SCENARIOS

    d = ensure_run(load_scenario(SCENARIOS / "realism_mars.yaml"), 0, run_dir=tmp_path)
    res = fit(d, "classical.lsq", {"realism": realism})
    rms = score(d, res.label, plots=False)["bodies"]["target"]["predictive"]["holdout_rms_arcsec"]
    assert rms < 0.5 if realism == "model" else rms > 5


def test_apparent_hypothesis_rotates_predictions_and_covariances() -> None:
    from jan.base import BaseHypothesis
    from jan.realism import ApparentHypothesis
    from jt_core.schema import SkyPrediction

    u = normalize(np.random.default_rng(2).normal(size=(5, 3)))
    cov = np.tile(np.diag([4.0, 1.0]), (5, 1, 1))

    class Fixed(BaseHypothesis):
        def predict_sky(self, t: np.ndarray, body_id: str) -> SkyPrediction:
            return SkyPrediction(mean=u, cov=cov)

    real = Realism(frame="equatorial")
    p = ApparentHypothesis(Fixed(), real, None, False, False, "").predict_sky(np.zeros(5), "b")
    assert np.allclose(p.mean, ecliptic_to_frame(u, real))
    assert p.cov is not None
    # A rotation of the tangent plane: trace and determinant are preserved.
    assert np.allclose(np.trace(p.cov, axis1=1, axis2=2), 5.0)
    assert np.allclose(np.linalg.det(p.cov), 4.0)


@pytest.mark.slow
@pytest.mark.sbi
def test_npe_sequential_rounds_cover_truth(tmp_path) -> None:
    from jt_bench.harness import ensure_run, fit, score
    from tycho import load_scenario

    from .conftest import SCENARIOS

    d = ensure_run(load_scenario(SCENARIOS / "mars_like.yaml"), 0, run_dir=tmp_path)
    res = fit(d, "sbi.npe", {"n_sims": 10000})
    par = score(d, res.label, plots=False)["bodies"]["target"]["parameters"]
    assert all(v for k, v in par["coverage"].items() if k.startswith("in95"))
