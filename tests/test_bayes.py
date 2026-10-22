"""Phase 2: Bayesian families, likelihood, evidences and samplers (SPEC §6.3, §6.5)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import jan
from jan.base import make_config
from jan.inference import SkyModel, fit_map
from jan.models.families import FAMILIES, EllipseFamily, EpicycleFamily, make_bayes_family
from jan.setup import setup_body
from jt_core.conics import ConicElements, elements_to_state
from jt_core.constants import ARCSEC, DEG, GM_SUN
from tycho import Scenario, generate, load_scenario
from tycho.scenario import EpicycleTruthCfg
from tycho.truth.epicycle import EpicycleTruth

from .conftest import SCENARIOS, small_scenario_dict

MARS = ConicElements(1.381, 0.0934, 1.85 * DEG, 49.6 * DEG, 286.5 * DEG, 120.0)


@pytest.mark.parametrize("name", ["circle", "conic-elliptic", "conic-hyperbolic"])
def test_conic_families_reproduce_conics(name: str) -> None:
    el = MARS if name != "circle" else ConicElements(1.5, 0.0, 0.1, 1.0, 0.0, 30.0)
    if name == "conic-hyperbolic":
        el = ConicElements(1.2, 1.3, 0.4, 1.0, 2.0, 500.0)
    fam = make_bayes_family(name, 1000.0, GM_SUN, 2000.0)
    phi = fam.from_elements(el)
    assert phi is not None and bool(fam.in_support(phi))
    t = np.linspace(0, 2000, 9)
    r_true, _ = elements_to_state(el, t)
    r = np.asarray(fam.position(jnp.asarray(phi), jnp.asarray(t)))
    assert np.allclose(r, r_true, atol=1e-8)


def test_ellipse_state_round_trip() -> None:
    fam = EllipseFamily(t0=1000.0, gm=GM_SUN, t_span=2000.0)
    phi = fam.from_elements(MARS)
    assert phi is not None
    y = fam.to_state(jnp.asarray(phi))
    r_true, v_true = elements_to_state(MARS, np.array([1000.0]))
    assert np.allclose(y, np.concatenate([r_true[0], v_true[0]]), atol=1e-12)
    d = np.asarray(fam.wrap(np.asarray(fam.from_state(y)))) - phi
    assert np.max(np.abs((d + np.pi) % (2 * np.pi) - np.pi)) < 1e-10


def test_equant_close_to_ellipse_at_low_e() -> None:
    el = ConicElements(1.0 * (1 - 0.02), 0.02, 0.0, 0.0, 1.0, 0.0)
    fam = make_bayes_family("equant-circle", 0.0, GM_SUN, 1000.0)
    phi = fam.from_elements(el)
    t = np.linspace(0, 1000, 50)
    r_ell, _ = elements_to_state(el, t)
    r_eq = np.asarray(fam.position(jnp.asarray(phi), jnp.asarray(t)))
    # Bisected equant matches the ellipse to O(e²) in position.
    assert np.max(np.linalg.norm(r_eq - r_ell, axis=1)) < 3 * 0.02**2


def test_epicycle_family_matches_tycho_truth() -> None:
    cfg = EpicycleTruthCfg.model_validate(
        {
            "type": "epicycle",
            "center": "sun",
            "i_deg": 2.0,
            "Omega_deg": 40.0,
            "circles": [
                {"radius_au": 1.5, "period_days": 700.0, "phase_deg": 30.0},
                {"radius_au": 0.6, "period_days": 300.0, "phase_deg": 100.0},
            ],
            "eccentric": {"center_offset_au": 0.1, "equant_offset_au": 0.2, "angle_deg": 70.0},
        }
    )
    truth = EpicycleTruth(cfg, lambda t: np.zeros((np.size(t), 3)))
    t0 = 500.0
    fam = EpicycleFamily(t0=t0, gm=GM_SUN)
    phi = np.array(
        [
            1 / 700,
            1 / 300,
            0.4,
            0.1 / 1.5,
            0.2 / 1.5,
            70 * DEG,
            30 * DEG + 2 * np.pi * t0 / 700,
            100 * DEG + 2 * np.pi * t0 / 300,
            np.cos(2 * DEG),
            40 * DEG,
        ]
    )
    t = np.linspace(0, 3000, 31)
    r_fam = np.asarray(fam.position(jnp.asarray(phi), jnp.asarray(t))) * 1.5
    assert np.allclose(r_fam, truth.position(t), atol=1e-12)


def test_prior_box_consistency() -> None:
    for name in FAMILIES:
        fam = make_bayes_family(name, 100.0, GM_SUN, 1000.0)
        lo, hi = fam.lo_hi()
        assert np.all(hi > lo) and len(fam.param_names) == fam.n_params
        assert np.isclose(fam.log_prior_density(), -np.sum(np.log(hi - lo)))


def _setup(seed: int = 0, **kw: object):
    run = generate(Scenario.model_validate(small_scenario_dict(**kw)), seed)
    return run, setup_body(run.observations.for_body("target"), run.meta)


def test_likelihood_gaussian_and_student() -> None:
    _, setup = _setup()
    fam = setup.family("conic-elliptic")
    phi = fam.from_elements(MARS)
    assert phi is not None
    for kind in ("gaussian", "student_t"):
        m = SkyModel(fam, setup.data, kind)  # type: ignore[arg-type]
        x = np.append(phi, np.log(30 * ARCSEC))
        ll0 = m.loglike(x)
        x_bad = x.copy()
        x_bad[5] += 1e-3  # shift the mean longitude
        assert np.isfinite(ll0) and m.loglike(x_bad) < ll0
        batch = m.loglike_batch(np.stack([x, x_bad]))
        assert np.allclose(batch, [ll0, m.loglike(x_bad)])
    m = SkyModel(fam, setup.data)
    n = setup.data.n
    r, _ = m.residuals(phi)
    s = 30 * ARCSEC
    expected = -0.5 * np.sum(r**2) / s**2 - 2 * n * np.log(s) - n * np.log(2 * np.pi)
    assert np.isclose(m.loglike(np.append(phi, np.log(s))), expected)


def test_known_sigma_uses_published_values() -> None:
    run, _ = _setup(report_uncertainties=True)
    setup = setup_body(run.observations.for_body("target"), run.meta)
    assert setup.data.sigma_known
    m = SkyModel(setup.family("conic-elliptic"), setup.data)
    assert not m.sigma_free and len(m.dims) == 6


def test_map_recovers_ellipse() -> None:
    _, setup = _setup()
    m = SkyModel(setup.family("conic-elliptic"), setup.data)
    mp = fit_map(m, setup.starts(m.family, 0))
    el = np.asarray(m.family.elements(jnp.asarray(mp.x[:6])))
    assert abs(el[0] / MARS.q - 1) < 2e-3 and abs(el[1] - MARS.e) < 2e-3
    assert abs(np.exp(mp.x[-1]) / ARCSEC - 30) < 5  # noise σ estimate


@pytest.mark.bayes
def test_box_evidence_matches_laplace() -> None:
    from jan.strategies.bayes.nested import NestedConfig, laplace_log_evidence, run_box

    _, setup = _setup()
    cfg = NestedConfig.model_validate({"nlive": 150})
    m = setup.model("conic-elliptic", cfg)
    mp = fit_map(m, setup.starts(m.family, 0))
    r = run_box(m, mp.x, mp.cov, cfg, 0)
    # A near-Gaussian posterior: nested-in-box and Laplace evidences agree.
    assert abs(r["lnZ"] - laplace_log_evidence(m, mp)) < 1.5
    assert not r["edge_warning"]


@pytest.mark.bayes
def test_nested_prefers_true_family() -> None:
    run, _ = _setup()
    s = jan.get_strategy("bayes.nested")
    cfg = make_config(s, {"families": ["circle", "conic-elliptic", "equant-circle"], "nlive": 100})
    h = s.fit(run.observations, run.meta, cfg, 0, Path("."))
    scores = {m.model_name: m.posterior_prob for m in h.model_comparison() or []}
    assert (scores["conic-elliptic"] or 0) > 0.95
    assert h.kind == "dynamical" and h.elements("target") is not None


@pytest.mark.bayes
def test_nested_prefers_epicycles_on_epicycle_truth() -> None:
    run = generate(load_scenario(SCENARIOS / "epicycle_truth.yaml"), 0)
    s = jan.get_strategy("bayes.nested")
    cfg = make_config(s, {"families": ["conic-elliptic", "epicycles"], "nlive": 100})
    h = s.fit(run.observations, run.meta, cfg, 0, Path("."))
    scores = {m.model_name: m.posterior_prob for m in h.model_comparison() or []}
    assert (scores["epicycles"] or 0) > 0.99
    assert h.kind == "kinematic" and h.elements("target") is None
    assert h.heliocentric_position(np.array([1.0]), "target") is None


@pytest.mark.bayes
@pytest.mark.parametrize(
    ("name", "cfg"),
    [
        ("bayes.mcmc", {"n_steps": 800, "burn": 300, "record_trace": True, "trace_max_frames": 10}),
        (
            "bayes.mcmc",
            {
                "coords": "state",
                "n_steps": 800,
                "burn": 300,
                "record_trace": True,
                "trace_max_frames": 10,
            },
        ),
        (
            "bayes.hmc",
            {"n_warmup": 150, "n_samples": 200, "record_trace": True, "trace_max_frames": 10},
        ),
    ],
)
def test_samplers_cover_truth(name: str, cfg: dict) -> None:
    run, _ = _setup(seed=1)
    s = jan.get_strategy(name)
    h = s.fit(run.observations, run.meta, make_config(s, cfg), 0, Path("."))
    est = h.elements("target")
    assert est is not None and est.samples is not None
    for k, v in MARS.to_dict().items():
        if k in ("Omega", "omega"):
            continue
        smp = np.asarray(est.samples[k])
        lo, hi = np.quantile(smp, [0.0005, 0.9995])
        if k == "t_p":  # modulo the period
            period = MARS.period()
            v = lo + np.mod(v - lo, period)
        assert lo <= v <= hi, (k, lo, v, hi)
    sp = h.predict_sky(np.array([10.0, 2000.0]), "target")
    assert sp.samples is not None and sp.cov is not None
    assert h.observation_sigma("target") is not None
    frames = h.trace()
    assert frames and len(frames) <= 10
    assert h.at_frame(frames[-1]) is not None


@pytest.mark.slow
def test_gp_sky_interpolates() -> None:
    run, _ = _setup()
    s = jan.get_strategy("gp.sky")
    h = s.fit(run.observations, run.meta, make_config(s, {"max_iter": 50}), 0, Path("."))
    t = run.observations.t[:20]
    sp = h.predict_sky(t, "target")
    assert sp.mean.shape == (20, 3) and sp.cov is not None and sp.samples is not None
    sep = np.arccos(np.clip(np.sum(sp.mean * run.observations.unit_vectors()[:20], 1), -1, 1))
    assert np.median(sep) / ARCSEC < 300
