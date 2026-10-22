# pyright: strict, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportMissingTypeStubs=false
"""Generic second-order ODE propagation (x'' = a(x, v, t; args)) with diffrax.

Truth-agnostic: Tycho uses it for central-force and N-body truths, Jan for central-force
families and learned force laws. Dormand–Prince 8(7) with a PID step-size controller and
tight default tolerances (SPEC §12: numerical drift must not masquerade as new physics).
Times on both sides of the epoch are handled by integrating forward and backward.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import diffrax
import jax.numpy as jnp

Array = Any
Accel = Callable[[Array, Array, Array, Any], Array]
"""a(x, v, t, args) with x, v of shape (n, 3) → (n, 3)."""

RTOL = 1e-12
ATOL = 1e-14
MAX_STEPS = 200_000
MAX_STEPS_AD = 16_384


def _solve(
    accel: Accel,
    y0: Array,
    t0: Array,
    ts: Array,
    args: Any,
    rtol: float,
    atol: float,
    differentiable: bool,
) -> Array:
    """States (len(ts), 2n, 3) at monotone ts, all on one side of t0."""
    n = y0.shape[0] // 2

    def rhs(t: Array, y: Array, a: Any) -> Array:
        x, v = y[:n], y[n:]
        return jnp.concatenate([v, accel(x, v, t, a)], axis=0)

    t1 = ts[-1]
    direction = jnp.sign(t1 - t0)
    # diffrax needs t1 != t0: pad the span by a tiny amount in the right direction.
    t1 = jnp.where(t1 == t0, t0 + jnp.where(direction == 0, 1e-9, direction * 1e-9), t1)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        diffrax.Dopri8(),
        t0=t0,
        t1=t1,
        dt0=None,
        y0=y0,
        args=args,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=MAX_STEPS_AD if differentiable else MAX_STEPS,
        # DirectAdjoint supports forward-mode (Jacobians for least squares) *and* reverse-mode
        # (HMC gradients); the default checkpointed adjoint is reverse-only.
        adjoint=diffrax.DirectAdjoint() if differentiable else diffrax.RecursiveCheckpointAdjoint(),
        throw=False,
    )
    assert sol.ys is not None
    # A failed solve (e.g. step budget exhausted for a wild trial law) returns NaN rather than
    # raising, so optimizers simply see an unacceptable step.
    ok = sol.result == diffrax.RESULTS.successful
    return jnp.where(ok, sol.ys, jnp.nan)


def propagate(
    accel: Accel,
    x0: Array,
    v0: Array,
    t0: float | Array,
    ts: Array,
    args: Any = None,
    rtol: float = RTOL,
    atol: float = ATOL,
    differentiable: bool = False,
) -> tuple[Array, Array]:
    """Positions and velocities (len(ts), n, 3) of n particles at times `ts` (any order).

    `x0`, `v0`: (n, 3) state at epoch t0. `differentiable=True` supports forward- and
    reverse-mode derivatives with respect to the inputs (for fitting).
    """
    ts = jnp.atleast_1d(jnp.asarray(ts, dtype=jnp.float64))
    x0 = jnp.atleast_2d(jnp.asarray(x0, dtype=jnp.float64))
    v0 = jnp.atleast_2d(jnp.asarray(v0, dtype=jnp.float64))
    t0 = jnp.asarray(t0, dtype=jnp.float64)
    y0 = jnp.concatenate([x0, v0], axis=0)
    n = x0.shape[0]
    order = jnp.argsort(ts)
    t_sorted = ts[order]
    fwd = t_sorted >= t0
    # Forward branch: times ≥ t0 in increasing order (earlier ones clamped to t0).
    tf = jnp.where(fwd, t_sorted, t0)
    yf = _solve(accel, y0, t0, tf, args, rtol, atol, differentiable)
    # Backward branch: times < t0 in decreasing order (later ones clamped to t0).
    tb = jnp.where(fwd, t0, t_sorted)[::-1]
    yb = _solve(accel, y0, t0, tb, args, rtol, atol, differentiable)[::-1]
    y_sorted = jnp.where(fwd[:, None, None], yf, yb)
    y = jnp.zeros_like(y_sorted).at[order].set(y_sorted)
    return y[:, :n], y[:, n:]


def central_accel(radial: Callable[[Array, Any], Array]) -> Accel:
    """Acceleration for a central force: a = f(r) r̂, f(r) < 0 attractive."""

    def accel(x: Array, v: Array, t: Array, args: Any) -> Array:
        r = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return radial(r, args) * x / r

    return accel
