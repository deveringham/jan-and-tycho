"""Load a run's sealed bundle and one strategy's serialized outputs.

Rudolf never executes Jan code: it reads `predictions.parquet` / `hypothesis.json` written
by the harness and compares them with the sealed truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from jt_core import io
from jt_core.angles import great_circle_separation
from jt_core.constants import ARCSEC
from jt_core.frames import lonlat_to_unit

FloatArray = NDArray[np.float64]


@dataclass
class Sealed:
    run_dir: Path
    info: dict[str, Any]
    scenario: dict[str, Any]
    truth: dict[str, Any]
    trajectory: pd.DataFrame
    holdout: pd.DataFrame
    noise: pd.DataFrame
    schedule: pd.DataFrame
    public: pd.DataFrame

    @classmethod
    def load(cls, run_dir: str | Path) -> Sealed:
        d = Path(run_dir)
        s = d / io.SEALED_DIR
        sched = s / "schedule.parquet"
        return cls(
            run_dir=d,
            info=io.read_json(s / "run.json"),
            scenario=yaml.safe_load((s / "scenario.yaml").read_text(encoding="utf-8")),
            truth=io.read_json(s / "truth.json"),
            trajectory=io.read_table(s / "trajectory.parquet"),
            holdout=io.read_table(s / "holdout.parquet"),
            noise=io.read_table(s / "noise_realization.parquet"),
            schedule=io.read_table(sched) if sched.exists() else pd.DataFrame(),
            public=io.read_table(d / io.PUBLIC_DIR / io.OBSERVATIONS_FILE),
        )

    @property
    def is_sky(self) -> bool:
        return self.info["tier"] != "T0"

    @property
    def t_split(self) -> float:
        return float(self.info["t_split"])

    @property
    def duration(self) -> float:
        return float(self.info["duration_days"])

    @property
    def body_ids(self) -> list[str]:
        return list(self.info["body_ids"])

    @property
    def scenario_name(self) -> str:
        return str(self.info["scenario"])

    @property
    def seed(self) -> int:
        return int(self.info["seed"])

    @property
    def noise_sigma_arcsec(self) -> float | None:
        v = self.info.get("noise_sigma_arcsec")
        return None if v is None else float(v)

    def traj(self, body_id: str) -> pd.DataFrame:
        return io.rows(self.trajectory, self.trajectory["body_id"] == body_id)

    def dense_t(self) -> FloatArray:
        return self.traj("observer")["t"].to_numpy()

    def dense_positions(self, body_id: str) -> FloatArray:
        return self.traj(body_id)[["x", "y", "z"]].to_numpy()

    def dense_directions(self, body_id: str) -> FloatArray:
        """True directions on the dense grid, as published: apparent and in the published
        frame when the scenario has realism effects (sealed by Tycho), else geometric."""
        tr = self.traj(body_id)
        if "ux" in tr.columns and not bool(tr["ux"].isna().all()):
            return tr[["ux", "uy", "uz"]].to_numpy()
        d = self.dense_positions(body_id) - self.dense_positions("observer")
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def holdout_body(self, body_id: str) -> pd.DataFrame:
        return io.rows(self.holdout, self.holdout["body_id"] == body_id)

    def public_body(self, body_id: str) -> pd.DataFrame:
        return io.rows(self.public, self.public["body_id"] == body_id)

    def holdout_true_directions(self, body_id: str) -> FloatArray:
        h = self.holdout_body(body_id)
        return lonlat_to_unit(h["lon_true"].to_numpy(), h["lat_true"].to_numpy())

    def body_truth(self, body_id: str) -> dict[str, Any]:
        return self.truth["bodies"][body_id]

    @property
    def field_table(self) -> pd.DataFrame | None:
        p = self.run_dir / io.SEALED_DIR / io.FIELD_FILE
        if "_field" not in self.__dict__:
            self.__dict__["_field"] = io.read_table(p) if p.exists() else None
        return self.__dict__["_field"]

    def field_for(self, body_id: str, kind: str = "near") -> pd.DataFrame | None:
        f = self.field_table
        if f is None:
            return None
        sel = io.rows(f, (f["body_id"] == body_id) & (f["kind"] == kind))
        return None if sel.empty else sel

    @property
    def hidden_ids(self) -> list[str]:
        return [b for b, d in self.truth["bodies"].items() if d.get("hidden")]


@dataclass
class JanOutputs:
    label: str
    dir: Path
    hypothesis: dict[str, Any]
    predictions: pd.DataFrame
    samples: pd.DataFrame | None = None
    trace: pd.DataFrame | None = None
    trace_samples: pd.DataFrame | None = None
    field_df: pd.DataFrame | None = None
    trace_field_df: pd.DataFrame | None = None
    _cache: dict[str, Any] = field(default_factory=dict[str, Any], repr=False)

    @classmethod
    def load(cls, run_dir: str | Path, label: str) -> JanOutputs:
        d = Path(run_dir) / io.JAN_DIR / label
        samples = d / io.SAMPLES_FILE
        trace = d / io.TRACE_PREDICTIONS_FILE
        trace_s = d / io.TRACE_SAMPLES_FILE
        field = d / io.FIELD_FILE
        trace_f = d / io.TRACE_FIELD_FILE
        return cls(
            label=label,
            dir=d,
            hypothesis=io.read_json(d / io.HYPOTHESIS_FILE),
            predictions=io.read_table(d / io.PREDICTIONS_FILE),
            samples=io.read_table(samples) if samples.exists() else None,
            trace=io.read_table(trace) if trace.exists() else None,
            trace_samples=io.read_table(trace_s) if trace_s.exists() else None,
            field_df=io.read_table(field) if field.exists() else None,
            trace_field_df=io.read_table(trace_f) if trace_f.exists() else None,
        )

    @property
    def strategy(self) -> str:
        return str(self.hypothesis["strategy"])

    @property
    def kind(self) -> str:
        return str(self.hypothesis["kind"])

    def _sel(self, df: pd.DataFrame, query: str, body_id: str) -> pd.DataFrame:
        return io.rows(df, (df["query"] == query) & (df["body_id"] == body_id))

    def sky(self, query: str, body_id: str) -> FloatArray | None:
        p = self._sel(self.predictions, query, body_id)
        if "ux" not in p.columns or bool(p["ux"].isna().all()):
            return None
        return p[["ux", "uy", "uz"]].to_numpy()

    def sky_cov(self, query: str, body_id: str) -> FloatArray | None:
        p = self._sel(self.predictions, query, body_id)
        if "cov_ee" not in p.columns or bool(p["cov_ee"].isna().all()):
            return None
        c = p[["cov_ee", "cov_en", "cov_nn"]].to_numpy()
        return np.stack([np.stack([c[:, 0], c[:, 1]], -1), np.stack([c[:, 1], c[:, 2]], -1)], -2)

    def position(self, query: str, body_id: str) -> FloatArray | None:
        p = self._sel(self.predictions, query, body_id)
        if "px" not in p.columns or bool(p["px"].isna().all()):
            return None
        return p[["px", "py", "pz"]].to_numpy()

    def times(self, query: str, body_id: str) -> FloatArray:
        return self._sel(self.predictions, query, body_id)["t"].to_numpy()

    def sample_array(self, query: str, body_id: str, kind: str) -> FloatArray | None:
        """(S, N, 3) sky ('u') or position ('p') samples."""
        if self.samples is None:
            return None
        s = self._sel(self.samples, query, body_id)
        cols = ["ux", "uy", "uz"] if kind == "u" else ["px", "py", "pz"]
        if s.empty or cols[0] not in s.columns or bool(s[cols[0]].isna().all()):
            return None
        n_s = int(np.max(s["sample"].to_numpy())) + 1
        ordered = s.sort_values(by=["sample", "idx"])
        return io.cols(ordered, cols).to_numpy().reshape(n_s, -1, 3)

    def elements(self, body_id: str) -> dict[str, Any] | None:
        el = self.hypothesis.get("elements") or {}
        return el.get(body_id)

    def law(self) -> dict[str, Any] | None:
        return self.hypothesis.get("law")

    def field_for(self, body_id: str, kind: str = "near") -> pd.DataFrame | None:
        if self.field_df is None:
            return None
        fd = self.field_df
        f = io.rows(fd, (fd["body_id"] == body_id) & (fd["kind"] == kind))
        return None if f.empty else f

    def obs_sigma(self, body_id: str) -> float | None:
        v = (self.hypothesis.get("obs_sigma_rad") or {}).get(body_id)
        return None if v is None else float(v)

    def model_comparison(self) -> list[dict[str, Any]] | None:
        return self.hypothesis.get("model_comparison")

    def trace_frames(self) -> list[int]:
        if self.trace is None:
            return []
        return sorted(int(f) for f in self.trace["frame"].unique())

    def trace_array(
        self, frame: int, body_id: str, kind: str
    ) -> tuple[FloatArray, FloatArray] | None:
        """(t, values (N, 3)) of a trace frame: sky ('u') or position ('p') means."""
        if self.trace is None:
            return None
        tr = self.trace
        sel = io.rows(
            tr, (tr["frame"] == frame) & (tr["query"] == "dense") & (tr["body_id"] == body_id)
        )
        cols = ["ux", "uy", "uz"] if kind == "u" else ["px", "py", "pz"]
        if sel.empty or bool(sel[cols[0]].isna().all()):
            return None
        return sel["t"].to_numpy(), io.cols(sel, cols).to_numpy()

    def trace_sample_array(self, frame: int, body_id: str, kind: str) -> FloatArray | None:
        """(S, N, 3) draws of a trace frame on the trace grid."""
        if self.trace_samples is None:
            return None
        ts = self.trace_samples
        sel = io.rows(
            ts, (ts["frame"] == frame) & (ts["body_id"] == body_id) & (ts["query"] == "dense")
        )
        cols = ["ux", "uy", "uz"] if kind == "u" else ["px", "py", "pz"]
        if sel.empty or bool(sel[cols[0]].isna().all()):
            return None
        n_s = int(np.max(sel["sample"].to_numpy())) + 1
        return io.cols(sel.sort_values(by=["sample", "idx"]), cols).to_numpy().reshape(n_s, -1, 3)


def sep_arcsec(u: FloatArray, v: FloatArray) -> FloatArray:
    return great_circle_separation(u, v) / ARCSEC


def list_labels(run_dir: str | Path) -> list[str]:
    d = Path(run_dir) / io.JAN_DIR
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if (p / io.HYPOTHESIS_FILE).exists())


@dataclass
class RunView:
    """Sealed truth + one strategy's outputs, with derived comparison arrays."""

    sealed: Sealed
    jan: JanOutputs
    body_id: str

    @cached_property
    def dense_t(self) -> FloatArray:
        return self.sealed.dense_t()

    @cached_property
    def dense_true_u(self) -> FloatArray:
        return self.sealed.dense_directions(self.body_id)

    @cached_property
    def dense_pred_u(self) -> FloatArray | None:
        return self.jan.sky("dense", self.body_id)

    @cached_property
    def dense_err_arcsec(self) -> FloatArray | None:
        if self.dense_pred_u is None:
            return None
        return sep_arcsec(self.dense_pred_u, self.dense_true_u)

    @cached_property
    def dense_true_r(self) -> FloatArray:
        return self.sealed.dense_positions(self.body_id)

    @cached_property
    def dense_pred_r(self) -> FloatArray | None:
        return self.jan.position("dense", self.body_id)

    @cached_property
    def observer_r(self) -> FloatArray:
        return self.sealed.dense_positions("observer")
