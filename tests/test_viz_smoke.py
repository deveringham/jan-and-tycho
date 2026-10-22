"""Every Phase 1 figure and animation renders in --quick mode and lands in figures.json."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jt_core import io
from rudolf.data import JanOutputs, Sealed
from rudolf.report import bench_report, run_report
from rudolf.viz.manifest import FigureManifest
from rudolf.viz.run_plots import render_run

STATIC = [
    "sky_track",
    "residuals_vs_time",
    "orbit_topdown",
    "orbit_3d",
    "element_errors",
    "observation_coverage",
    "corner",
    "posterior_predictive",
    "model_probabilities",
    "noise_diagnostics",
    "triangulation_geometry",
]
ANIMATIONS = ["orbit_animation", "sky_animation"]
# Need a fit trace / predictive draws: produced by classical.lsq in the fixture.
LSQ_ANIMATIONS = ["fit_progress", "posterior_samples"]


@pytest.mark.parametrize("label", ["classical.lsq", "ptolemy.fourier"])
def test_per_run_figures_and_animations(tiny_run: Path, tmp_path: Path, label: str) -> None:
    out = tmp_path / label
    m = render_run(
        Sealed.load(tiny_run), JanOutputs.load(tiny_run, label), out, animate=True, quick=True
    )
    saved = FigureManifest.load(out)
    assert set(STATIC + ANIMATIONS) <= set(saved.names())
    if label == "classical.lsq":
        assert set(LSQ_ANIMATIONS) <= set(saved.names())
    for e in m.entries:
        for p in e.paths:
            f = out / p
            assert f.exists() and f.stat().st_size > 0, p
    if label == "ptolemy.fourier":  # missing layers are noted, not fatal
        assert "Omitted" in (saved.get("orbit_topdown").caption)  # type: ignore[union-attr]
        assert "no orbital elements" in saved.get("element_errors").caption  # type: ignore[union-attr]


def test_run_report(tiny_run: Path, tmp_path: Path) -> None:
    d = tmp_path / "run"
    shutil.copytree(tiny_run, d)
    from rudolf.score import score_run

    for label in ("classical.lsq", "ptolemy.fourier"):
        score_run(d, label, plots=True)
    p = run_report(d)
    html = p.read_text()
    assert "classical.lsq" in html and "ptolemy.fourier" in html and "<img" in html
    assert FigureManifest.load(d / "rudolf").get("strategies_residuals") is not None


def test_bench_report_from_results(tiny_run: Path, tmp_path: Path) -> None:
    from rudolf.score import flatten_score

    rows = []
    for label in ("classical.lsq", "ptolemy.fourier"):
        sc = io.read_json(tiny_run / "rudolf" / label / "score.json")
        rows.append({**flatten_score(sc), "status": "ok", "run_dir": "x"})
    res = tmp_path / "res"
    res.mkdir()
    io.write_table(pd.DataFrame(rows), res / "results.parquet")
    p = bench_report(res, requests=[{"figure": "leaderboard"}, {"figure": "extrapolation_curves"}])
    m = FigureManifest.load(res / "figures")
    assert {"leaderboard", "extrapolation_curves", "leaderboard_table"} <= set(m.names())
    assert "Leaderboard" in p.read_text()


def _synthetic_results(tiny_run: Path, tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    """A results table over a 2 × 2 Tycho grid × 2 seeds whose rows all point to tiny_run."""
    from rudolf.score import flatten_score

    res = tmp_path / "res"
    shutil.copytree(tiny_run, res / "cell")
    rng = np.random.default_rng(0)
    rows = []
    for label in ("classical.lsq", "ptolemy.fourier"):
        base = flatten_score(io.read_json(tiny_run / "rudolf" / label / "score.json"))
        for noise in (10, 100):
            for cad in (10, 30):
                for seed in (0, 1):
                    r = dict(base, seed=seed, status="ok", run_dir="cell")
                    r["noise.0.sigma_arcsec"] = noise
                    r["schedule.cadence_days"] = cad
                    r["jan.n_harmonics"] = cad // 10
                    r["holdout_rms_arcsec"] = noise * (1 + rng.random())
                    r["p_true_family"] = 1 / (1 + noise / 50)
                    r["p_conic-elliptic"] = r["p_true_family"]
                    r["true_family"] = "conic-elliptic"
                    rows.append(r)
    return pd.DataFrame(rows), res


def test_comparison_figures_and_cross_run_animations(tiny_run: Path, tmp_path: Path) -> None:
    from rudolf.viz.compare_plots import render_comparisons

    df, res = _synthetic_results(tiny_run, tmp_path)
    m = FigureManifest(res / "figures")
    requests = [
        {
            "figure": "error_vs_tycho_param",
            "x": "noise.0.sigma_arcsec",
            "by": "schedule.cadence_days",
        },
        {"figure": "error_vs_jan_param", "strategy": "ptolemy.fourier", "x": "n_harmonics"},
        {"figure": "heatmap_2d", "x": "noise.0.sigma_arcsec", "y": "schedule.cadence_days"},
        {"figure": "calibration"},
        {"figure": "model_selection_vs_noise", "x": "noise.0.sigma_arcsec"},
        {"figure": "accuracy_vs_runtime"},
        {
            "figure": "sweep_animation",
            "base_figure": "orbit_topdown",
            "x": "noise.0.sigma_arcsec",
            "quick": True,
        },
        {"figure": "strategy_race", "seed": 0, "quick": True},
    ]
    skipped = render_comparisons(df, res / "figures", requests, m, results_dir=res)
    m.save()
    assert skipped == []
    names = set(FigureManifest.load(res / "figures").names())
    for prefix in (
        "leaderboard",
        "extrapolation_curves",
        "error_vs_tycho_param",
        "error_vs_jan_param",
        "heatmap_2d",
        "calibration",
        "model_selection_vs_noise",
        "accuracy_vs_runtime",
        "sweep_animation",
        "strategy_race",
    ):
        assert any(n.startswith(prefix) for n in names), prefix
    for e in FigureManifest.load(res / "figures").entries:
        assert all((res / "figures" / p).stat().st_size > 0 for p in e.paths)


def test_triangulation_figures_t2(tmp_path: Path) -> None:
    from jt_bench.harness import ensure_run, fit, score
    from tycho import Scenario

    from .conftest import small_scenario_dict

    sc = Scenario.model_validate(small_scenario_dict(knowledge_tier="T2", duration_days=4000))
    d = ensure_run(sc, 0, run_dir=tmp_path / "t2")
    res = fit(d, "kepler.triangulate")
    score(d, res.label, plots=True, animate=True, quick=True)
    m = FigureManifest.load(d / "rudolf" / res.label)
    for name in ("triangulation_geometry", "triangulation_animation", "orbit_animation"):
        e = m.get(name)
        assert e is not None and (d / "rudolf" / res.label / e.paths[0]).stat().st_size > 0
    assert "reconstructed" in m.get("triangulation_geometry").caption  # type: ignore[union-attr]
    sc_json = io.read_json(d / "rudolf" / res.label / "score.json")
    assert sc_json["observer"]["errors"]["e"] is not None


@pytest.mark.discovery
def test_phase4_per_run_figures(tmp_path: Path) -> None:
    from jt_bench.harness import ensure_run, fit, score
    from tycho import Scenario

    from .conftest import small_scenario_dict

    d0 = small_scenario_dict()
    el = d0["bodies"][0]["truth"]["elements"]
    d0["bodies"][0]["truth"] = {
        "type": "central_force",
        "law": "power_law",
        "exponent": 2.05,
        "elements": el,
    }
    d = ensure_run(Scenario.model_validate(d0), 0, run_dir=tmp_path / "p4")
    res = fit(d, "discovery.sindy", {"record_trace": True})
    score(d, res.label, plots=True, animate=True, quick=True)
    m = FigureManifest.load(d / "rudolf" / res.label)
    for name in ("force_law", "acceleration_field", "conserved_quantities", "law_discovery"):
        e = m.get(name)
        assert e is not None, name
        assert (d / "rudolf" / res.label / e.paths[0]).stat().st_size > 0


def test_phase4_comparison_figures(tmp_path: Path) -> None:
    from rudolf.viz.compare_plots import render_comparisons

    rows = []
    for sc, kind, fam in [
        ("hooke_trap", "dynamical", "central-hooke"),
        ("hidden_perturber", "dynamical", "perturbed-conic"),
        ("rectangle", "kinematic", "kinematic-polygon"),
    ]:
        for lab, sym, answer in [
            ("classical.lsq", "exact" if sc == "hooke_trap" else None, "correct law"),
            ("piml.neural_ode", "black_box", "wrong law"),
            ("kepler.triangulate", None, "kinematic"),
        ]:
            if kind == "kinematic" and answer != "kinematic":
                answer, sym = "spurious law", None
            rows.append(
                {
                    "scenario": sc,
                    "seed": 0,
                    "strategy": lab,
                    "label": lab,
                    "status": "ok",
                    "truth_kind": kind,
                    "truth_family": fam,
                    "symbolic": sym,
                    "answer": answer,
                    "runtime_s": 1.0,
                    "holdout_rms_arcsec": 1.0,
                }
            )
    df = pd.DataFrame(rows)
    m = FigureManifest(tmp_path / "figures")
    reqs = [{"figure": "rejection_matrix"}, {"figure": "structural_success"}]
    assert render_comparisons(df, tmp_path / "figures", reqs, m, results_dir=tmp_path) == []
    m.save()
    names = set(FigureManifest.load(tmp_path / "figures").names())
    assert {"rejection_matrix", "structural_success"} <= names


@pytest.mark.sbi
def test_phase5_npe_figure(tmp_path: Path) -> None:
    from jt_bench.harness import ensure_run, fit, score
    from tycho import Scenario

    from .conftest import small_scenario_dict

    d = ensure_run(Scenario.model_validate(small_scenario_dict()), 0, run_dir=tmp_path / "p5")
    res = fit(d, "sbi.npe", {"rounds": 2, "n_sims": 1000, "max_epochs": 20})
    score(d, res.label, plots=True)
    e = FigureManifest.load(d / "rudolf" / res.label).get("npe_rounds")
    assert e is not None
    assert (d / "rudolf" / res.label / e.paths[0]).stat().st_size > 0


def test_phase5_realism_bias_figure(tmp_path: Path) -> None:
    from rudolf.viz.compare_plots import render_comparisons

    rows = [
        {
            "scenario": "realism_mars",
            "seed": 0,
            "strategy": "classical.lsq",
            "label": f"classical.lsq[realism={mode}]",
            "jan.realism": mode,
            "realism.light_time": lt,
            "realism.aberration": ab,
            "status": "ok",
            "holdout_rms_arcsec": 0.1 if mode == "model" else 0.1 + 10 * lt + 20 * ab,
        }
        for mode in ("model", "ignore")
        for lt in (False, True)
        for ab in (False, True)
    ]
    m = FigureManifest(tmp_path / "figures")
    reqs = [{"figure": "realism_bias"}]
    df = pd.DataFrame(rows)
    assert render_comparisons(df, tmp_path / "figures", reqs, m, results_dir=tmp_path) == []
    m.save()
    assert "realism_bias" in set(FigureManifest.load(tmp_path / "figures").names())


def test_results_explorer(tmp_path: Path) -> None:
    from rudolf.explorer import write_explorer

    df = pd.DataFrame(
        {
            "scenario": ["a", "a", "b"],
            "label": ["x", "y", "x"],
            "seed": [0, 0, 1],
            "holdout_rms_arcsec": [1.0, float("nan"), 3.0],
            "elements": [{"q": 1}, None, None],  # non-scalar columns are left out
        }
    )
    path = write_explorer(df, tmp_path / "explorer.html", name="t</title>")
    assert path is not None
    html = path.read_text(encoding="utf-8")
    data = html.split("const DATA = ", 1)[1].split(", COLS = ", 1)[0]
    assert '"holdout_rms_arcsec": null' in data and '"elements"' not in data
    assert "t&lt;/title&gt;" in html
