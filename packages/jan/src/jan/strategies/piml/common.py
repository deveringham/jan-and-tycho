"""Shared machinery for the physics-informed ML strategies (jan[piml]).

Each model is an acceleration law with neural parameters, a(x, v, t; θ), trained *end to
end through the observation operator*: the law is integrated (diffrax) from a fitted
initial state, and the loss compares predicted sky directions (T1) or positions (T0) with
the data. Training starts from a conic least-squares fit: the initial state comes from
it, and the networks are initialized so the law starts at (or near) Newton.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpy.typing import NDArray
from pydantic import Field

from jan.base import StrategyConfig, get_strategy, make_config, subsample_frames
from jan.models.observer import observer_function
from jt_core.angles import tangent_basis
from jt_core.constants import ARCSEC
from jt_core.integrate import propagate
from jt_core.schema import ObservationSet, PublicMeta, TraceFrame

FloatArray = NDArray[np.float64]
TRAIN_RTOL, TRAIN_ATOL = 1e-8, 1e-11


class PimlConfig(StrategyConfig):
    width: int = 32
    depth: int = 2
    steps: int = Field(300, description="End-to-end optimizer steps.")
    learning_rate: float = 3e-3
    pretrain_steps: int = Field(1500, description="Steps matching the initial law (no ODE).")
    weight_decay: float = 1e-6
    assumed_sigma_arcsec: float = 60.0
    assumed_sigma_au: float = 1e-4


@dataclass
class Start:
    """Initial state and scales from a conic least-squares fit."""

    t0: float
    x0: FloatArray
    v0: FloatArray
    a_scale: float
    """Typical heliocentric distance (AU)."""
    acc_scale: float
    """GM / a_scale² (AU/day²)."""
    gm: float
    sample_x: FloatArray
    """Points around the fitted orbit (for pretraining and distillation)."""


def conic_start(ob: ObservationSet, meta: PublicMeta, seed: int) -> Start:
    lsq = get_strategy("classical.lsq")
    bid = str(ob.table["body_id"].iloc[0])
    h = lsq.fit(ob, meta, make_config(lsq), seed, Path("."))
    t0 = float(np.median(ob.t))
    tt = np.array([t0 - 0.5, t0, t0 + 0.5])
    pp = h.heliocentric_position(tt, bid)
    assert pp is not None
    x = pp.mean
    t_path = np.linspace(ob.t.min(), ob.t.max(), 400)
    path = h.heliocentric_position(t_path, bid)
    assert path is not None
    g = np.random.default_rng(seed)
    k = g.integers(0, len(t_path), 1500)
    sample = path.mean[k] * g.uniform(0.85, 1.15, (1500, 1))
    a_scale = float(np.median(np.linalg.norm(path.mean, axis=1)))
    return Start(
        t0, x[1], (x[2] - x[0]) / 1.0, a_scale, meta.gm_sun / a_scale**2, meta.gm_sun, sample
    )


def newton(x: jax.Array, gm: float) -> jax.Array:
    r = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return -gm * x / r**3


STATE_STEP = 1e-3
"""Initial-state offsets are trained in units of 1e-3 × (orbit size, speed): Adam's
per-parameter step (≈ learning rate) must not throw the orbit away."""


def initial_state(theta: Any, start: Start) -> tuple[jax.Array, jax.Array]:
    vs = float(np.linalg.norm(start.v0))
    x0 = jnp.asarray(start.x0) + STATE_STEP * start.a_scale * theta["dx0"]
    v0 = jnp.asarray(start.v0) + STATE_STEP * vs * theta["dv0"]
    return x0, v0


def make_loss(
    ob: ObservationSet,
    meta: PublicMeta,
    cfg: PimlConfig,
    accel_of: Callable[[Any], Any],
    start: Start,
) -> Callable[[Any], jax.Array]:
    """Loss(θ) with θ = {"nn", "dx0", "dv0"}: mean squared normalized residual."""
    t0 = start.t0
    t = jnp.asarray(ob.t)
    if ob.is_sky:
        observer = observer_function(meta)
        u = ob.unit_vectors()
        east, north = tangent_basis(u)
        r_obs, uj, ej, nj = (jnp.asarray(a) for a in (observer(ob.t), u, east, north))
        s = cfg.assumed_sigma_arcsec * ARCSEC

        def resid(x: jax.Array) -> jax.Array:
            d = x - r_obs
            d = d / jnp.linalg.norm(d, axis=1, keepdims=True)
            k = jnp.sum(d * uj, 1)
            return jnp.stack([jnp.sum(d * ej, 1) / k, jnp.sum(d * nj, 1) / k], 1) / s

    else:
        pos = jnp.asarray(ob.positions())

        def resid(x: jax.Array) -> jax.Array:
            return (x - pos) / cfg.assumed_sigma_au

    def loss(theta: Any) -> jax.Array:
        acc = accel_of(theta["nn"])
        x0, v0 = initial_state(theta, start)
        x, _ = propagate(acc, x0[None], v0[None], t0, t, None, TRAIN_RTOL, TRAIN_ATOL)
        r = resid(x[:, 0])
        return jnp.mean(jnp.where(jnp.isfinite(r), r, 1e6) ** 2)

    return loss


def train(
    theta: Any,
    loss: Callable[[Any], jax.Array],
    cfg: PimlConfig,
    snapshot: Callable[[Any, int], TraceFrame] | None = None,
) -> tuple[Any, list[float], list[TraceFrame]]:
    """Adam on the trainable leaves of θ (equinox-filtered), keeping the best iterate."""
    params, static = eqx.partition(theta, eqx.is_inexact_array)
    opt = optax.adamw(cfg.learning_rate, weight_decay=cfg.weight_decay)
    state = opt.init(params)

    @eqx.filter_jit
    def step(p: Any, st: Any) -> tuple[Any, Any, jax.Array]:
        val, grads = jax.value_and_grad(lambda q: loss(eqx.combine(q, static)))(p)
        grads = jax.tree_util.tree_map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
        upd, st = opt.update(grads, st, p)
        return optax.apply_updates(p, upd), st, val

    history: list[float] = []
    frames: list[TraceFrame] = []
    best, best_val = params, np.inf
    for k in range(cfg.steps):
        params, state, val = step(params, state)
        v = float(val)
        history.append(v)
        if np.isfinite(v) and v < best_val:
            best, best_val = params, v
        if (
            snapshot is not None
            and cfg.record_trace
            and (k % max(cfg.steps // 50, 1) == 0 or k == cfg.steps - 1)
        ):
            frames.append(snapshot(eqx.combine(params, static), k))
    return eqx.combine(best, static), history, subsample_frames(frames, cfg.trace_max_frames)


def pretrain(
    nn: Any,
    target: Callable[[jax.Array], jax.Array],
    apply: Callable[[Any, jax.Array], jax.Array],
    xs: FloatArray,
    steps: int,
    lr: float = 3e-3,
) -> Any:
    """Fit `apply(nn, x)` to `target(x)` on sample points (behaviour cloning, no ODE)."""
    xj = jnp.asarray(xs)
    y = target(xj)
    scale = jnp.std(y) + 1e-30
    params, static = eqx.partition(nn, eqx.is_inexact_array)
    opt = optax.adam(lr)
    st = opt.init(params)

    @eqx.filter_jit
    def step(p: Any, s: Any) -> tuple[Any, Any]:
        g = jax.grad(lambda q: jnp.mean(((apply(eqx.combine(q, static), xj) - y) / scale) ** 2))(p)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s

    for _ in range(steps):
        params, st = step(params, st)
    return eqx.combine(params, static)


def timed() -> Callable[[], float]:
    t0 = time.perf_counter()
    return lambda: time.perf_counter() - t0
