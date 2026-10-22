"""Scenario + seed → public bundle + sealed bundle (SPEC §5.8)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jt_core import io
from jt_core import rng as jrng
from jt_core.angles import great_circle_separation, tangent_residuals
from jt_core.constants import ARCSEC, GM_SUN
from jt_core.frames import lonlat_to_unit, unit_to_lonlat
from jt_core.realism import apparent_directions, retarded_positions
from jt_core.schema import SUN_ID, ElementsModel, ObservationSet, PublicMeta
from tycho.noise import SkyNoiseResult, apply_position_noise, apply_sky_noise, nominal_sigma
from tycho.observer import KeplerianObserver, build_observer
from tycho.operator import observe
from tycho.scenario import (
    FixedObserverCfg,
    HeliocentricObserverCfg,
    KeplerianObserverCfg,
    Scenario,
)
from tycho.schedule import OK, candidate_times, in_season, visibility_reasons, weather
from tycho.truth import build_truth

SCHEDULE_FILE = "schedule.parquet"
FIELD_FILE = "field.parquet"
RUN_FILE = "run.json"


@dataclass
class GeneratedRun:
    scenario: Scenario
    seed: int
    observations: ObservationSet
    meta: PublicMeta
    truth: dict[str, Any]
    trajectory: pd.DataFrame
    holdout: pd.DataFrame
    noise_realization: pd.DataFrame
    schedule: pd.DataFrame
    run_info: dict[str, Any]
    field: pd.DataFrame | None = None


def _public_meta(sc: Scenario, body_ids: list[str]) -> PublicMeta:
    obs_elements = None
    obs_kind = None
    obs_pos = None
    if sc.knowledge_tier == "T1":
        obs_kind = sc.observer.type
        if isinstance(sc.observer, FixedObserverCfg):
            obs_pos = sc.observer.position_au
        elif isinstance(sc.observer, HeliocentricObserverCfg):
            obs_pos = (0.0, 0.0, 0.0)
        elif isinstance(sc.observer, KeplerianObserverCfg):
            el = sc.observer.elements.to_elements()
            obs_elements = ElementsModel(**el.to_dict())
    return PublicMeta(
        tier=sc.knowledge_tier,
        observed="position" if sc.knowledge_tier == "T0" else "sky",
        gm_sun=GM_SUN,
        epoch_calendar=sc.epoch_calendar,
        t_range=(0.0, sc.t_split),
        body_ids=body_ids,
        observer_elements=obs_elements,
        observer_kind=obs_kind,
        observer_position=obs_pos,
        report_uncertainties=sc.report_uncertainties,
        sun_stream=sc.knowledge_tier == "T2",
        realism=sc.realism,
    )


def field_points(
    body_id: str, model: Any, t_dense: Any, seed: int, n_near: int = 300, n_grid: int = 25
) -> pd.DataFrame:
    """Sealed evaluation points for the acceleration field (structural scoring, field plots).

    path: the true trajectory (with velocities); near: points scattered ±20 % radially and a
    little out of plane around it; grid: an in-plane grid at the mid-epoch. True
    accelerations are attached where the truth has a force law (NaN for kinematic paths).
    """
    step = max(1, len(t_dense) // 500)
    tp = t_dense[::step]
    xp, vp = model.position(tp), model.velocity(tp)
    g = jrng.generator(seed, "field", body_id)
    k = g.integers(0, len(tp), n_near)
    r = np.linalg.norm(xp[k], axis=1, keepdims=True)
    xn = xp[k] * g.uniform(0.8, 1.2, (n_near, 1))
    xn[:, 2] += g.normal(0, 0.02, n_near) * r[:, 0]
    h = np.cross(xp, vp).mean(axis=0)
    nhat = h / np.linalg.norm(h)
    e1 = np.cross([0.0, 0.0, 1.0], nhat)
    e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-9 else np.array([1.0, 0.0, 0.0])
    e2 = np.cross(nhat, e1)
    rmax = 1.3 * float(np.max(np.linalg.norm(xp, axis=1)))
    rmin = 0.3 * float(np.min(np.linalg.norm(xp, axis=1)))
    u = np.linspace(-rmax, rmax, n_grid)
    gu, gv = np.meshgrid(u, u)
    xg = gu.ravel()[:, None] * e1 + gv.ravel()[:, None] * e2
    xg = xg[np.linalg.norm(xg, axis=1) > rmin]
    t_mid = float(np.median(tp))
    parts = [
        ("path", tp, xp, vp),
        ("near", tp[k], xn, np.full_like(xn, np.nan)),
        ("grid", np.full(len(xg), t_mid), xg, np.full_like(xg, np.nan)),
    ]
    rows = []
    for kind, t, x, v in parts:
        acc = (
            model.acceleration(x, t) if hasattr(model, "acceleration") else np.full_like(x, np.nan)
        )
        df = pd.DataFrame({"kind": kind, "body_id": body_id, "t": t})
        df[["x", "y", "z"]] = x
        df[["vx", "vy", "vz"]] = v
        df[["ax", "ay", "az"]] = acc
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    plane = np.concatenate([e1, e2])
    out.attrs["plane"] = plane.tolist()
    return out


def _record_sky(df: pd.DataFrame, u_true: Any, res: SkyNoiseResult) -> None:
    """Fill published (λ, β, t) and the sealed noise realization for one stream."""
    lon_t, lat_t = unit_to_lonlat(u_true)
    df["t"] = res.t_pub
    df["lon"], df["lat"] = res.lon, res.lat
    df["lon_true"], df["lat_true"] = lon_t, lat_t
    u_noisy = lonlat_to_unit(res.lon, res.lat)
    off = tangent_residuals(u_noisy, u_true) / ARCSEC
    df["de_arcsec"], df["dn_arcsec"] = off[:, 0], off[:, 1]
    df["sep_arcsec"] = great_circle_separation(u_noisy, u_true) / ARCSEC
    df["outlier"] = res.outlier
    df["sigma"] = res.sigma


def generate(sc: Scenario, seed: int) -> GeneratedRun:
    """Simulate a scenario. Same (scenario, seed) ⇒ bit-identical output."""
    observer = build_observer(sc.observer)
    truths = {b.id: build_truth(b.truth, observer.position) for b in sc.bodies}
    vis = sc.schedule.visibility
    t_nights = candidate_times(sc.schedule, sc.duration_days, jrng.generator(seed, "schedule"))
    clear = weather(vis, t_nights.size, jrng.generator(seed, "weather"))
    season = in_season(sc.schedule, t_nights)
    r_obs = observer.position(t_nights)
    v_obs = observer.velocity(t_nights)
    is_t0 = sc.knowledge_tier == "T0"
    real = sc.realism

    sched_rows: list[pd.DataFrame] = []
    obs_rows: list[pd.DataFrame] = []
    for body in sc.bodies:
        if body.hidden:
            continue
        r_body = truths[body.id].position(t_nights)
        if real.light_time:
            r_body, _ = retarded_positions(truths[body.id].position, t_nights, r_obs)
        geom = observe(r_body, r_obs, body.photometry)
        # Published directions: apparent (aberration) and in the published frame; noise is
        # applied to these. Visibility uses the geometry.
        u_pub = apparent_directions(r_body, r_obs, v_obs, real) if real.any else geom.u
        reason = visibility_reasons(vis, clear, geom, season)
        sched_rows.append(
            pd.DataFrame(
                {
                    "t": t_nights,
                    "body_id": body.id,
                    "reason": reason,
                    "elongation_deg": np.degrees(geom.elongation),
                    "magnitude": geom.magnitude,
                    "delta": geom.delta,
                    "r": geom.r,
                }
            )
        )
        keep = reason == OK
        t = t_nights[keep]
        df = pd.DataFrame({"body_id": body.id, "t": t, "t_true": t})
        if is_t0:
            r_true = r_body[keep]
            r_noisy = apply_position_noise(r_true, sc.noise, seed, body.id)
            df[["x", "y", "z"]] = r_noisy
            df[["x_true", "y_true", "z_true"]] = r_true
            err = r_noisy - r_true
            df["err_x"], df["err_y"], df["err_z"] = err[:, 0], err[:, 1], err[:, 2]
            df["sep_arcsec"] = np.nan
        else:
            res = apply_sky_noise(
                u_pub[keep],
                t,
                sc.noise,
                seed,
                body.id,
                magnitude=geom.magnitude[keep],
                elongation=geom.elongation[keep],
            )
            _record_sky(df, u_pub[keep], res)
        obs_rows.append(df)

    if sc.knowledge_tier == "T2":
        # Tycho's daily Sun observations: the geocentric Sun direction on every clear night.
        sun_ok = clear & season
        t_sun = t_nights[sun_ok]
        u_sun = -r_obs[sun_ok] / np.linalg.norm(r_obs[sun_ok], axis=1, keepdims=True)
        if real.any:  # the Sun sits at the origin: no light-time displacement
            u_sun = apparent_directions(0 * r_obs[sun_ok], r_obs[sun_ok], v_obs[sun_ok], real)
        res = apply_sky_noise(u_sun, t_sun, sc.noise, seed, SUN_ID)
        df = pd.DataFrame({"body_id": SUN_ID, "t": t_sun, "t_true": t_sun})
        _record_sky(df, u_sun, res)
        obs_rows.append(df)

    all_obs = pd.concat(obs_rows, ignore_index=True)
    all_obs = all_obs.sort_values(["t_true", "body_id"], kind="stable").reset_index(drop=True)
    all_obs.insert(0, "obs_id", np.arange(len(all_obs), dtype=np.int64))
    all_obs["split"] = np.where(all_obs["t_true"] < sc.t_split, "train", "holdout")

    pub_cols = ["obs_id", "body_id", "t", *(["x", "y", "z"] if is_t0 else ["lon", "lat"])]
    public = io.cols(io.rows(all_obs, all_obs["split"] == "train"), pub_cols)
    if sc.report_uncertainties and not is_t0:
        s = io.rows(all_obs, all_obs["split"] == "train")["sigma"].to_numpy()
        public["sigma_lon"] = s
        public["sigma_lat"] = s
    observed_ids = [b.id for b in sc.bodies if not b.hidden]
    meta = _public_meta(sc, observed_ids)

    hold_cols = (
        pub_cols
        + ["t_true"]
        + (["x_true", "y_true", "z_true"] if is_t0 else ["lon_true", "lat_true"])
    )
    holdout = io.cols(io.rows(all_obs, all_obs["split"] == "holdout"), hold_cols)
    noise_cols = ["obs_id", "body_id", "t", "t_true", "split", "sep_arcsec"] + (
        ["err_x", "err_y", "err_z"] if is_t0 else ["de_arcsec", "dn_arcsec", "outlier", "sigma"]
    )
    noise_real = io.cols(all_obs, noise_cols)

    schedule = pd.concat(sched_rows, ignore_index=True)
    schedule["split"] = np.where(schedule["t"] < sc.t_split, "train", "holdout")

    t_dense = np.linspace(0.0, sc.duration_days, sc.dense_points)
    r_obs_d, v_obs_d = observer.position(t_dense), observer.velocity(t_dense)
    traj_rows: list[pd.DataFrame] = []
    for bid, model in [("observer", observer), *truths.items()]:
        r = model.position(t_dense)
        v = model.velocity(t_dense)
        df = pd.DataFrame(
            {
                "body_id": bid,
                "t": t_dense,
                "x": r[:, 0],
                "y": r[:, 1],
                "z": r[:, 2],
                "vx": v[:, 0],
                "vy": v[:, 1],
                "vz": v[:, 2],
            }
        )
        if real.any and not is_t0 and bid != "observer":
            # Sealed apparent directions in the published frame (what Jan's dense sky
            # predictions are compared with).
            if real.light_time:
                r = retarded_positions(model.position, t_dense, r_obs_d)[0]
            u = apparent_directions(r, r_obs_d, v_obs_d, real)
            df["ux"], df["uy"], df["uz"] = u[:, 0], u[:, 1], u[:, 2]
        traj_rows.append(df)
    trajectory = pd.concat(traj_rows, ignore_index=True)
    field = pd.concat(
        [field_points(b.id, truths[b.id], t_dense, seed) for b in sc.bodies if not b.hidden],
        ignore_index=True,
    )

    truth: dict[str, Any] = {
        "bodies": {b.id: {**truths[b.id].describe(), "hidden": b.hidden} for b in sc.bodies},
        "observer": observer.describe(),
    }
    run_info = {
        "scenario": sc.name,
        "seed": seed,
        "tier": sc.knowledge_tier,
        "duration_days": sc.duration_days,
        "t_split": sc.t_split,
        "noise_sigma_arcsec": nominal_sigma(sc.noise) / ARCSEC if not is_t0 else None,
        "observer_is_keplerian": isinstance(observer, KeplerianObserver),
        "n_train": len(public),
        "n_holdout": len(holdout),
        "body_ids": observed_ids,
        "sun_stream": sc.knowledge_tier == "T2",
    }
    return GeneratedRun(
        sc,
        seed,
        ObservationSet(public),
        meta,
        truth,
        trajectory,
        holdout,
        noise_real,
        schedule,
        run_info,
        field,
    )


def write_run(run: GeneratedRun, run_dir: str | Path) -> Path:
    """Write public/ and sealed/ bundles into `run_dir`."""
    d = Path(run_dir)
    io.write_public(d, run.observations, run.meta)
    s = d / io.SEALED_DIR
    s.mkdir(parents=True, exist_ok=True)
    (s / "scenario.yaml").write_text(run.scenario.to_yaml(), encoding="utf-8")
    io.write_json(run.truth, s / "truth.json")
    io.write_json(run.run_info, s / RUN_FILE)
    io.write_table(run.trajectory, s / "trajectory.parquet")
    io.write_table(run.holdout, s / "holdout.parquet")
    io.write_table(run.noise_realization, s / "noise_realization.parquet")
    io.write_table(run.schedule, s / SCHEDULE_FILE)
    if run.field is not None:
        io.write_table(run.field, s / FIELD_FILE)
    return d


def default_run_dir(sc: Scenario, seed: int, out: str | Path = "runs") -> Path:
    return Path(out) / sc.name / str(seed)
