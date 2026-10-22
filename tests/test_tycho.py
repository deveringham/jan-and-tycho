"""Tycho: scenarios, generation, noise, visibility, bundles (SPEC §5, §11)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from jt_core import io
from jt_core.constants import ARCSEC
from tycho import Scenario, generate, load_scenario, write_run
from tycho.scenario import apply_overrides

from .conftest import SCENARIOS, small_scenario_dict


@pytest.mark.parametrize("path", sorted(SCENARIOS.glob("*.yaml")), ids=lambda p: p.stem)
def test_library_scenarios_load(path: Path) -> None:
    sc = load_scenario(path)
    assert sc.name == path.stem


def test_overrides() -> None:
    d = apply_overrides(
        small_scenario_dict(), {"noise.0.sigma_arcsec": 99, "schedule.cadence_days": 3}
    )
    sc = Scenario.model_validate(d)
    assert sc.noise[0].sigma_arcsec == 99  # type: ignore[union-attr]
    assert sc.schedule.cadence_days == 3  # type: ignore[union-attr]


def test_invalid_configs_rejected() -> None:
    with pytest.raises(ValueError):
        Scenario.model_validate(small_scenario_dict(knowledge_tier="T0"))  # sky noise under T0
    with pytest.raises(ValueError):
        Scenario.model_validate(small_scenario_dict(unknown_field=1))


def _digest(d: Path) -> dict[str, str]:
    return {
        str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(d.rglob("*"))
        if p.is_file()
    }


def test_determinism_byte_equal(tmp_path: Path) -> None:
    sc = Scenario.model_validate(small_scenario_dict())
    write_run(generate(sc, 5), tmp_path / "a")
    write_run(generate(sc, 5), tmp_path / "b")
    assert _digest(tmp_path / "a") == _digest(tmp_path / "b")


def test_different_seeds_differ() -> None:
    sc = Scenario.model_validate(small_scenario_dict())
    a, b = generate(sc, 1), generate(sc, 2)
    assert not np.array_equal(
        a.noise_realization["de_arcsec"].to_numpy()[:20],
        b.noise_realization["de_arcsec"].to_numpy()[:20],
    )


def test_gaussian_noise_matches_sigma() -> None:
    """Great-circle residuals of isotropic gaussian noise are Rayleigh(σ) (KS test)."""
    sigma = 45.0
    d = small_scenario_dict(
        duration_days=20000, noise=[{"type": "gaussian", "sigma_arcsec": sigma}]
    )
    d["schedule"] = {"type": "regular", "cadence_days": 3}
    run = generate(Scenario.model_validate(d), 11)
    sep = run.noise_realization["sep_arcsec"].to_numpy()
    assert len(sep) > 3000
    ks = stats.kstest(sep, stats.rayleigh(scale=sigma).cdf)
    assert ks.pvalue > 0.01
    comp = run.noise_realization[["de_arcsec", "dn_arcsec"]].to_numpy().ravel()
    assert abs(np.std(comp) / sigma - 1) < 0.05


def test_quantize_noise() -> None:
    d = small_scenario_dict(noise=[{"type": "quantize", "step_arcsec": 60}])
    run = generate(Scenario.model_validate(d), 0)
    lon = run.observations.table["lon"].to_numpy() / (60 * ARCSEC)
    assert np.allclose(lon, np.round(lon), atol=1e-6)


def test_split_and_public_bundle() -> None:
    sc = Scenario.model_validate(small_scenario_dict())
    run = generate(sc, 0)
    pub = run.observations.table
    assert list(pub.columns) == ["obs_id", "body_id", "t", "lon", "lat"]
    assert pub["t"].max() < sc.t_split <= run.holdout["t"].min()
    assert set(pub["obs_id"]).isdisjoint(set(run.holdout["obs_id"]))
    meta = run.meta.model_dump()
    assert "name" not in meta and "description" not in meta  # no truth leakage via names
    assert meta["observer_elements"] is not None and meta["t_range"] == (0.0, sc.t_split)


def test_visibility_filters() -> None:
    d = small_scenario_dict()
    d["schedule"]["visibility"] = {
        "min_solar_elongation_deg": 40,
        "max_magnitude": 0.5,
        "dropout_prob": 0.3,
    }
    run = generate(Scenario.model_validate(d), 0)
    s = run.schedule
    assert {"weather", "elongation", "observed"} <= set(s["reason"])
    obs = s[s["reason"] == "observed"]
    assert (obs["elongation_deg"] >= 40).all() and (obs["magnitude"] <= 0.5).all()
    assert abs((s["reason"] == "weather").mean() - 0.3) < 0.08


def test_report_uncertainties_columns() -> None:
    run = generate(Scenario.model_validate(small_scenario_dict(report_uncertainties=True)), 0)
    assert np.allclose(run.observations.table["sigma_lon"], 30 * ARCSEC)


def test_t0_tier_publishes_positions() -> None:
    d = small_scenario_dict(
        knowledge_tier="T0", noise=[{"type": "position_gaussian", "sigma_au": 1e-4}]
    )
    run = generate(Scenario.model_validate(d), 0)
    assert list(run.observations.table.columns) == ["obs_id", "body_id", "t", "x", "y", "z"]
    assert run.meta.observer_elements is None and run.meta.observed == "position"


def test_written_bundle_layout(tmp_path: Path) -> None:
    write_run(generate(Scenario.model_validate(small_scenario_dict()), 0), tmp_path)
    for f in [
        "public/observations.parquet",
        "public/meta.json",
        "sealed/scenario.yaml",
        "sealed/truth.json",
        "sealed/trajectory.parquet",
        "sealed/holdout.parquet",
        "sealed/noise_realization.parquet",
    ]:
        assert (tmp_path / f).exists(), f
    obs, meta = io.read_public(tmp_path)
    assert meta.tier == "T1" and len(obs) > 50
    traj = pd.read_parquet(tmp_path / "sealed/trajectory.parquet")
    assert set(traj["body_id"]) == {"observer", "target"}
