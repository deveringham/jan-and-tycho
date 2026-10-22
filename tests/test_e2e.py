"""End-to-end: CLI and minimal sweep (SPEC §11)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from jt_bench.cli import app
from jt_bench.sweep import SweepConfig, run_sweep

from .conftest import ROOT, SCENARIOS


def test_cli_lists() -> None:
    r = CliRunner().invoke(app, ["list", "strategies"])
    assert r.exit_code == 0 and "classical.lsq" in r.stdout
    r = CliRunner().invoke(app, ["list", "scenarios", "--directory", str(SCENARIOS)])
    assert r.exit_code == 0 and "mars_like" in r.stdout


def test_jt_run_circle_under_60s(tmp_path: Path) -> None:
    t0 = time.perf_counter()
    r = CliRunner().invoke(
        app,
        [
            "run",
            str(SCENARIOS / "circle_coplanar.yaml"),
            "--strategy",
            "classical.lsq",
            "--out",
            str(tmp_path),
        ],
    )
    elapsed = time.perf_counter() - t0
    assert r.exit_code == 0, r.stdout
    assert elapsed < 60, elapsed
    run_dir = tmp_path / "circle_coplanar" / "1"
    assert (run_dir / "rudolf" / "classical.lsq" / "figures.json").exists()
    assert (run_dir / "rudolf" / "report.html").exists()


def test_minimal_sweep(tmp_path: Path) -> None:
    cfg = SweepConfig.model_validate(
        {
            "name": "t",
            "scenarios": ["scenarios/circle_coplanar.yaml"],
            "overrides": {"noise.0.sigma_arcsec": [5, 50]},
            "strategies": ["classical.lsq"],
            "seeds": {"start": 0, "count": 1},
            "viz": {"per_run": "none"},
        }
    )
    df = run_sweep(cfg, tmp_path / "res", workers=0, root=ROOT, log=lambda *_: None)
    assert len(df) == 2 and (df["status"] == "ok").all()
    stored = pd.read_parquet(tmp_path / "res" / "results.parquet")
    assert "noise.0.sigma_arcsec" in stored.columns
    lo, hi = stored.sort_values("noise.0.sigma_arcsec")["holdout_rms_arcsec"]
    assert lo < hi


def test_sweep_cache_hits(tmp_path: Path) -> None:
    cfg = SweepConfig.model_validate({
        "name": "c", "scenarios": ["scenarios/circle_coplanar.yaml"],
        "strategies": ["ptolemy.fourier"], "seeds": {"start": 0, "count": 1},
        "viz": {"per_run": "none"},
    })  # fmt: skip
    first = run_sweep(cfg, tmp_path / "res", workers=0, root=ROOT, log=lambda *_: None)
    second = run_sweep(cfg, tmp_path / "res", workers=0, root=ROOT, log=lambda *_: None)
    assert not bool(first["cached"].any()) and bool(second["cached"].all())
    assert np.allclose(first["holdout_rms_arcsec"], second["holdout_rms_arcsec"])


def _sleepy(x: float) -> dict:
    time.sleep(x)
    return {"status": "ok", "slept": x}


def test_run_pool_enforces_timeout() -> None:
    from jt_bench.sweep import run_pool

    out = run_pool(_sleepy, [0.0, 30.0], workers=2, timeout_s=8.0)
    assert out[0]["status"] == "ok"
    assert out[1]["status"] == "timeout"
