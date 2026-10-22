"""`piml.neural_ode`, `piml.hnn` and `piml.ude` (SPEC §6.5, Phase 4).

* neural_ode — a = NN(x): a black-box acceleration field (pretrained to Newton along the
  conic fit, then trained end to end).
* hnn — a = −∇V(x) with V = NN(x): a learned potential, so the dynamics conserve
  energy by construction (pretrained to −GM/r's gradient, then trained end to end).
* ude — a = −GM x/r³ + NN(x, [t]): Newton plus a learned correction that starts at zero.
  Its correction is then distilled symbolically: the radial part is regressed on powers of
  r (best-subset, as in discovery.sindy), giving −GM/r² + δf(r). With `time_input` the
  correction can depend on time (e.g. an unseen perturber) and is then not distilled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from numpy.typing import NDArray
from pydantic import Field

from jan.base import BaseHypothesis, PerBodyHypothesis, StrategyConfig, register
from jan.law import LawHypothesis
from jan.models.observer import observer_function
from jan.strategies.discovery.sindy import POWERS, best_subset
from jan.strategies.piml.common import (
    PimlConfig,
    Start,
    conic_start,
    initial_state,
    make_loss,
    newton,
    pretrain,
    timed,
    train,
)
from jt_core.schema import LawEstimate, ObservationSet, PublicMeta, TraceFrame

FloatArray = NDArray[np.float64]


class UdeConfig(PimlConfig):
    time_input: bool = Field(False, description="Let the correction depend on time too.")
    correction_scale: float = Field(0.01, description="Correction output scale × Newton's.")
    distill_powers: tuple[int, ...] = POWERS
    distill_max_terms: int = 2


class _Base:
    name: ClassVar[str]
    supported_tiers: ClassVar[set[str]] = {"T0", "T1"}
    Config: ClassVar[type[StrategyConfig]] = PimlConfig
    kind: ClassVar[Literal["neural_ode", "hnn", "ude"]]

    def fit(
        self,
        obs: ObservationSet,
        meta: PublicMeta,
        config: StrategyConfig,
        rng_seed: int,
        workdir: Path,
    ) -> BaseHypothesis:
        assert isinstance(config, PimlConfig)
        return PerBodyHypothesis(
            {b: self._fit_body(obs.for_body(b), b, meta, config, rng_seed) for b in obs.body_ids}
        )

    # --- model definition -------------------------------------------------------------
    def build(self, cfg: PimlConfig, start: Start, key: jax.Array) -> tuple[Any, Any]:
        """(network, apply(nn, x, t) → accelerations (N, 3))."""
        raise NotImplementedError

    def initial_net(self, nn: Any, apply: Any, cfg: PimlConfig, start: Start) -> Any:
        return pretrain(
            nn,
            lambda x: newton(x, start.gm),
            lambda m, x: apply(m, x, jnp.zeros(len(x))),
            start.sample_x,
            cfg.pretrain_steps,
        )

    def law_estimate(
        self, nn: Any, apply: Any, cfg: PimlConfig, start: Start
    ) -> LawEstimate | None:
        return None  # black box: structural scoring uses the field only

    # --- fit ------------------------------------------------------------------------
    def _fit_body(
        self, ob: ObservationSet, bid: str, meta: PublicMeta, cfg: PimlConfig, seed: int
    ) -> LawHypothesis:
        elapsed = timed()
        start = conic_start(ob, meta, seed)
        nn, apply = self.build(cfg, start, jax.random.PRNGKey(seed % (2**31)))
        nn = self.initial_net(nn, apply, cfg, start)

        def accel_of(m: Any) -> Any:
            return lambda x, v, t, args: apply(m, x, jnp.broadcast_to(t, (x.shape[0],)))

        loss = make_loss(ob, meta, cfg, accel_of, start)
        flat0, unravel = ravel_pytree(eqx.filter(nn, eqx.is_inexact_array))
        static = eqx.filter(nn, eqx.is_inexact_array, inverse=True)

        def snapshot(theta: Any, k: int) -> TraceFrame:
            flat, _ = ravel_pytree(eqx.filter(theta["nn"], eqx.is_inexact_array))
            x0, v0 = initial_state(theta, start)
            state = {
                "nn": np.asarray(flat).tolist(),
                "x0": np.asarray(x0).tolist(),
                "v0": np.asarray(v0).tolist(),
            }
            return TraceFrame(f"step {k}", k, state)

        theta0 = {"nn": nn, "dx0": jnp.zeros(3), "dv0": jnp.zeros(3)}
        theta, history, frames = train(theta0, loss, cfg, snapshot)
        nn_fit = theta["nn"]
        observer = observer_function(meta) if ob.is_sky else None
        law = self.law_estimate(nn_fit, apply, cfg, start)

        def frame_h(fr: TraceFrame) -> LawHypothesis:
            m = eqx.combine(unravel(jnp.asarray(fr.state["nn"])), static)
            return LawHypothesis(
                bid,
                accel_of(m),
                None,
                np.asarray(fr.state["x0"]),
                np.asarray(fr.state["v0"]),
                start.t0,
                observer,
                None,
                {},
            )

        final_loss = float(min(history)) if history else float("nan")
        sigma = np.sqrt(final_loss) * (
            cfg.assumed_sigma_arcsec * 4.8481368e-6 if ob.is_sky else cfg.assumed_sigma_au
        )
        info = {
            "loss_history": history[:: max(len(history) // 100, 1)],
            "final_loss": final_loss,
            "n_parameters": int(flat0.size),
            "sigma_rad": float(sigma) if ob.is_sky else None,
            "fit_seconds": elapsed(),
            "notes": f"{self.name}: {cfg.steps} end-to-end steps, "
            f"final normalized MSE {final_loss:.3g}"
            + (f"; law {law.radial}" if law is not None and law.radial else ""),
        }
        x0, v0 = initial_state(theta, start)
        return LawHypothesis(
            bid,
            accel_of(nn_fit),
            None,
            np.asarray(x0),
            np.asarray(v0),
            start.t0,
            observer,
            law,
            info,
            frames,
            frame_h,
        )


def _mlp(
    in_size: int, out_size: int | Literal["scalar"], cfg: PimlConfig, key: jax.Array
) -> eqx.nn.MLP:
    return eqx.nn.MLP(in_size, out_size, cfg.width, cfg.depth, activation=jnp.tanh, key=key)


@register("piml.neural_ode")
class NeuralOde(_Base):
    """Black-box neural acceleration field a = NN(x), trained through the sky operator."""

    kind: ClassVar[Literal["neural_ode", "hnn", "ude"]] = "neural_ode"

    def build(self, cfg: PimlConfig, start: Start, key: jax.Array) -> tuple[Any, Any]:
        nn = _mlp(3, 3, cfg, key)

        def apply(m: Any, x: jax.Array, t: jax.Array) -> jax.Array:
            return start.acc_scale * jax.vmap(m)(x / start.a_scale)

        return nn, apply


@register("piml.hnn")
class HamiltonianNN(_Base):
    """Learned potential V(x): a = −∇V, energy conserved by construction."""

    kind: ClassVar[Literal["neural_ode", "hnn", "ude"]] = "hnn"

    def build(self, cfg: PimlConfig, start: Start, key: jax.Array) -> tuple[Any, Any]:
        nn = _mlp(3, "scalar", cfg, key)
        pot = start.gm / start.a_scale

        def apply(m: Any, x: jax.Array, t: jax.Array) -> jax.Array:
            grad_v = jax.vmap(jax.grad(lambda q: pot * m(q / start.a_scale)))(x)
            return -grad_v

        return nn, apply


@register("piml.ude")
class UniversalDE(_Base):
    """Newton + a neural correction (starts at zero), then symbolic distillation."""

    kind: ClassVar[Literal["neural_ode", "hnn", "ude"]] = "ude"
    Config: ClassVar[type[StrategyConfig]] = UdeConfig

    def build(self, cfg: PimlConfig, start: Start, key: jax.Array) -> tuple[Any, Any]:
        assert isinstance(cfg, UdeConfig)
        nn = _mlp(4 if cfg.time_input else 3, 3, cfg, key)
        last = nn.layers[-1]
        assert last.weight is not None and last.bias is not None
        nn = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias),
            nn,
            (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)),
        )  # type: ignore[union-attr]
        scale = cfg.correction_scale * start.acc_scale
        span = 365.25 * 10

        def apply(m: Any, x: jax.Array, t: jax.Array) -> jax.Array:
            inp = x / start.a_scale
            if cfg.time_input:
                inp = jnp.concatenate([inp, ((t - start.t0) / span)[:, None]], axis=1)
            return newton(x, start.gm) + scale * jax.vmap(m)(inp)

        return nn, apply

    def initial_net(self, nn: Any, apply: Any, cfg: PimlConfig, start: Start) -> Any:
        return nn  # already exactly Newton (zero correction)

    def law_estimate(
        self, nn: Any, apply: Any, cfg: PimlConfig, start: Start
    ) -> LawEstimate | None:
        assert isinstance(cfg, UdeConfig)
        if cfg.time_input:
            return None
        x = jnp.asarray(start.sample_x)
        corr = np.asarray(apply(nn, x, jnp.zeros(len(x))) - newton(x, start.gm))
        xn = np.asarray(x)
        r = np.linalg.norm(xn, axis=1)
        f = np.sum(corr * xn, axis=1) / r
        a = np.stack([r**p for p in cfg.distill_powers], axis=1)
        coef, _ = best_subset(a, f, cfg.distill_max_terms, 0.5)
        terms = [(c, p) for c, p in zip(coef, cfg.distill_powers, strict=True) if c != 0]
        consts = {
            "GM": start.gm,
            **{f"d{p if p >= 0 else 'm' + str(-p)}": float(c) for c, p in terms},
        }
        delta = " + ".join(f"d{p if p >= 0 else 'm' + str(-p)}*r**({p})" for _, p in terms)
        radial = "-GM/r**2" + (f" + {delta}" if delta else "")
        rr = "sqrt(x**2 + y**2 + z**2)"
        vec = [f"({radial.replace('r**', f'({rr})**')})*{c}/({rr})" for c in ("x", "y", "z")]
        resid = float(np.sqrt(np.mean((a @ coef - f) ** 2)) / (np.std(f) + 1e-300))
        return LawEstimate(
            expression=vec,
            constants=consts,
            radial=radial,
            extra={
                "distillation_relative_rms": resid,
                "correction_rms": float(np.sqrt(np.mean(f**2))),
            },
        )
