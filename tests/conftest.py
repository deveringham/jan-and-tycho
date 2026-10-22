from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"


def small_scenario_dict(**overrides: Any) -> dict[str, Any]:
    """A small, fast Mars-like scenario for unit and smoke tests."""
    d: dict[str, Any] = {
        "name": "tiny_mars",
        "description": "fast fixture",
        "duration_days": 2400,
        "holdout_fraction": 0.25,
        "knowledge_tier": "T1",
        "bodies": [
            {
                "id": "target",
                "truth": {
                    "type": "keplerian",
                    "elements": {
                        "q": 1.381,
                        "e": 0.0934,
                        "i_deg": 1.85,
                        "Omega_deg": 49.6,
                        "omega_deg": 286.5,
                        "t_p": 120.0,
                    },
                },
                "photometry": {"H": -1.5},
            }
        ],
        "schedule": {
            "type": "regular",
            "cadence_days": 12,
            "jitter_days": 2,
            "visibility": {"min_solar_elongation_deg": 15, "dropout_prob": 0.2},
        },
        "noise": [{"type": "gaussian", "sigma_arcsec": 30}],
        "dense_points": 400,
    }
    d.update(overrides)
    return d


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A generated + fitted + scored run with both Phase 1 strategies (no animations)."""
    from jt_bench.harness import ensure_run, fit, score
    from tycho import Scenario

    sc = Scenario.model_validate(small_scenario_dict())
    d = ensure_run(sc, 3, run_dir=tmp_path_factory.mktemp("run") / "tiny_mars" / "3")
    for s in ("classical.lsq", "ptolemy.fourier"):
        cfg = {"record_trace": True, "trace_max_frames": 8} if s == "classical.lsq" else None
        res = fit(d, s, cfg, label=s)
        score(d, res.label, plots=False)
    return d
