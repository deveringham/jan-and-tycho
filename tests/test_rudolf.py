"""Rudolf: metrics and scoring (SPEC §7)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jt_core import io
from jt_core.constants import DEG
from rudolf.metrics.parameters import element_errors


def test_element_errors_wrap_and_degeneracies() -> None:
    truth = {"q": 1.0, "e": 0.0, "i": 0.0, "Omega": 0.0, "omega": 0.0, "t_p": 0.0}
    est = {"q": 1.01, "e": 0.001, "i": 0.001, "Omega": 3.0, "omega": 1.0, "t_p": 40.0}
    err = element_errors(truth, est, 0.01720209895**2)
    assert err["Omega_deg"] is None and err["omega_deg"] is None and err["t_p_days"] is None
    assert err["varpi_deg"] is None
    assert err["lambda_deg"] is not None
    assert np.isclose(err["q_rel"], 0.01)


def test_element_errors_angles_and_period_wrap() -> None:
    gm = 0.01720209895**2
    truth = {"q": 1.381, "e": 0.0934, "i": 0.03, "Omega": 0.01, "omega": 6.27, "t_p": 120.0}
    period = 2 * np.pi * np.sqrt((1.381 / (1 - 0.0934)) ** 3 / gm)
    est = dict(truth, Omega=2 * np.pi - 0.01, t_p=120.0 + 3 * period + 1.0)
    err = element_errors(truth, est, gm)
    assert np.isclose(err["Omega_deg"], -0.02 / DEG)
    assert np.isclose(err["t_p_days"], 1.0)


def test_scores_written(tiny_run: Path) -> None:
    for label in ("classical.lsq", "ptolemy.fourier"):
        sc = io.read_json(tiny_run / "rudolf" / label / "score.json")
        p = sc["bodies"]["target"]["predictive"]
        assert p["holdout_rms_arcsec"] is not None
        assert len(p["extrapolation"]["rms"]) == 10
    lsq = io.read_json(tiny_run / "rudolf" / "classical.lsq" / "score.json")["bodies"]["target"]
    four = io.read_json(tiny_run / "rudolf" / "ptolemy.fourier" / "score.json")["bodies"]["target"]
    assert lsq["trajectory"]["traj_rms_holdout_au"] < 1e-3
    assert lsq["parameters"]["errors"]["a_rel"] is not None
    # Unsupported metrics are null, never zero.
    assert four["trajectory"]["traj_rms_holdout_au"] is None
    assert four["parameters"]["errors"] is None
    json.dumps(lsq)


def test_jan_outputs_have_no_truth(tiny_run: Path) -> None:
    preds = io.read_table(tiny_run / "jan" / "classical.lsq" / "predictions.parquet")
    assert not {"lon_true", "lat_true", "x_true"} & set(preds.columns)
    q = io.read_table(tiny_run / "jan" / "classical.lsq" / "query.parquet")
    assert set(q.columns) == {"query", "obs_id", "body_id", "t"}
    trace = io.read_table(tiny_run / "jan" / "classical.lsq" / "trace_predictions.parquet")
    assert trace["frame"].nunique() >= 2


def test_probabilistic_metrics(tiny_run: Path) -> None:
    sc = io.read_json(tiny_run / "rudolf" / "classical.lsq" / "score.json")["bodies"]["target"]
    p = sc["probabilistic"]
    assert p["nlpd"] is not None and np.isfinite(p["nlpd"])
    assert p["energy_score_arcsec"] is not None and p["energy_score_arcsec"] > 0
    assert 0 <= p["pred_cover68"] <= p["pred_cover95"] <= 1
    cov = sc["parameters"]["coverage"]
    assert any(k.startswith("in95_") for k in cov)
    assert sc["predictive"]["train_fit_rms_arcsec"] > 10  # ≈ noise σ (30″) for a good fit
    four = io.read_json(tiny_run / "rudolf" / "ptolemy.fourier" / "score.json")["bodies"]["target"]
    assert four["probabilistic"]["nlpd"] is None  # no noise estimate → not scorable


def test_selection_metric() -> None:
    from rudolf.metrics.selection import selection_metrics

    comp = [
        {"model_name": "circle", "posterior_prob": 0.1},
        {"model_name": "conic-elliptic", "posterior_prob": 0.9},
    ]
    s = selection_metrics({"family": "conic-elliptic"}, comp)
    assert s["p_true_family"] == 0.9 and s["selected_correct"] is True
    assert selection_metrics({"family": "x"}, None)["p_true_family"] is None
