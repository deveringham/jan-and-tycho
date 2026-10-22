# pyright: strict, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportMissingTypeStubs=false
"""Deterministic Parquet + JSON I/O, run-directory layout and prediction file format."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from jt_core.schema import ObservationSet, PublicMeta, QuerySheet

# ----------------------------------------------------------------------------- layout

PUBLIC_DIR = "public"
SEALED_DIR = "sealed"
JAN_DIR = "jan"
RUDOLF_DIR = "rudolf"

OBSERVATIONS_FILE = "observations.parquet"
META_FILE = "meta.json"
QUERY_FILE = "query.parquet"
PREDICTIONS_FILE = "predictions.parquet"
SAMPLES_FILE = "prediction_samples.parquet"
TRACE_PREDICTIONS_FILE = "trace_predictions.parquet"
TRACE_SAMPLES_FILE = "trace_samples.parquet"
FIELD_FILE = "field.parquet"
TRACE_FIELD_FILE = "trace_field.parquet"
HYPOTHESIS_FILE = "hypothesis.json"
SCORE_FILE = "score.json"
FIGURES_MANIFEST = "figures.json"


# ----------------------------------------------------------------------------- primitives


def rows(df: pd.DataFrame, mask: Any) -> pd.DataFrame:
    """Boolean-mask row selection as a fresh DataFrame indexed from 0.

    (pandas' stubs type `df[mask]` as `DataFrame | Series`; this keeps call sites typed.)
    """
    return cast(pd.DataFrame, df.loc[mask]).reset_index(drop=True)


def cols(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Column subset as a DataFrame."""
    return cast(pd.DataFrame, df[names])


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to Parquet byte-deterministically (no index, fixed settings)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False)
    pq.write_table(
        table,
        p,
        compression="zstd",
        use_dictionary=False,
        write_statistics=False,
        store_schema=True,
    )
    return p


def read_table(path: str | Path) -> pd.DataFrame:
    return pq.read_table(Path(path)).to_pandas()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(obj: Any, path: str | Path) -> Path:
    """Write JSON with sorted keys; non-finite floats become null."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_jsonable(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- bundles


def write_public(run_dir: str | Path, obs: ObservationSet, meta: PublicMeta) -> None:
    d = Path(run_dir) / PUBLIC_DIR
    write_table(obs.table, d / OBSERVATIONS_FILE)
    write_json(meta.model_dump(mode="json"), d / META_FILE)


def read_public(run_dir: str | Path) -> tuple[ObservationSet, PublicMeta]:
    d = Path(run_dir) / PUBLIC_DIR
    obs = ObservationSet(read_table(d / OBSERVATIONS_FILE))
    meta = PublicMeta.model_validate(read_json(d / META_FILE))
    return obs, meta


def write_query(path: str | Path, query: QuerySheet) -> None:
    keep = ["query", "obs_id", "body_id", "t"]
    parts = [cols(query.holdout.assign(query="holdout"), keep)]
    if query.train is not None:
        parts.append(cols(query.train.assign(query="train"), keep))
    for b in query.body_ids:
        parts.append(
            pd.DataFrame(
                {"query": "dense", "obs_id": -1, "body_id": b, "t": np.asarray(query.dense_t)}
            )
        )
    write_table(pd.concat(parts, ignore_index=True), path)


def read_query(path: str | Path) -> QuerySheet:
    df = read_table(path)
    hold = cols(rows(df, df["query"] == "holdout"), ["obs_id", "body_id", "t"])
    dense = rows(df, df["query"] == "dense")
    body_ids = sorted(df["body_id"].unique().tolist())
    first = body_ids[0] if body_ids else None
    dense_t = rows(dense, dense["body_id"] == first)["t"].to_numpy(dtype=np.float64)
    train = cols(rows(df, df["query"] == "train"), ["obs_id", "body_id", "t"])
    return QuerySheet(hold, dense_t, body_ids, train if len(train) else None)
