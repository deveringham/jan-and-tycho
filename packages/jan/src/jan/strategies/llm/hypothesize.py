"""`llm.hypothesize`: an LLM proposes model families, Jan fits and compares them (SPEC §6.5).

The loop, for up to `max_iterations` rounds:

1. Fit every candidate (MAP + Laplace evidence) on the training data.
2. Summarize the best fit numerically: residual RMS against σ, residuals binned by time
   and by heliocentric distance, a residual periodogram, and the apsidal precession
   between conic fits to the two halves of the arc.
3. Ask Claude (structured JSON output) for new candidates: registered Jan families, or new
   central radial laws f(r) with bounded parameters, or to stop.

The leading candidates are then compared with nested sampling (`bayes.nested`'s box
evidence), and predictions come from the most probable one.

**Reproducibility.** Every request is cached on disk, keyed by a hash of the exact request
body; the summaries are rounded so a re-run produces the same bytes. A cached run makes no
API call; `offline: true` turns a cache miss into an error. **Safety.** Proposed laws are
text from outside Jan: `jan.models.radial` whitelists every token before sympy parses it.
No scenario name or truth reaches the prompt (only `PublicMeta` and the observations).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, ClassVar, Literal

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from scipy.signal import lombscargle

from jan.base import BaseHypothesis, PerBodyHypothesis, StrategyConfig, register
from jan.inference import MapResult, SkyModel, fit_map
from jan.models.families import FAMILIES, BayesFamily, make_bayes_family
from jan.models.posterior import PosteriorHypothesis
from jan.models.radial import ParamSpec, RadialSpec, radial_family
from jan.setup import BayesConfig, BodySetup, setup_body
from jan.strategies.bayes.nested import NestedConfig, laplace_log_evidence, run_box
from jt_core.constants import ARCSEC
from jt_core.schema import ModelScore, ObservationSet, PublicMeta

FloatArray = NDArray[np.float64]

SYSTEM_PROMPT = """\
You assist an automated astronomer who infers how a body moves from angles-only sky \
observations (ecliptic longitude/latitude seen from an observer whose orbit is known). \
Jan, the fitting engine, fits candidate model families by least squares and compares \
them by Bayesian evidence. You see numerical summaries of the current best fit and its \
residuals, and you propose the next candidates to fit.

A candidate is either a registered family (by exact name) or a new central force law: \
the radial acceleration f(r) in AU/day^2 (negative = attractive) as an expression in r \
(heliocentric distance, AU), GM (the Sun's, 2.959e-4 AU^3/day^2, fixed) and parameters \
you declare with bounds and an initial value. Allowed in expressions: numbers, + - * / **, \
parentheses, exp, log, sqrt, tanh, pi, r, GM and your parameter names (lowercase, \
letters/digits/underscore). Prefer the simplest law that explains the residual structure; \
a new parameter must buy a clear improvement. Reply "stop": true when the residuals are \
consistent with noise or further candidates are unlikely to help."""

_PARAM_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "lower": {"type": "number"},
        "upper": {"type": "number"},
        "initial": {"type": "number"},
        "log_scale": {"type": "boolean"},
    },
    "required": ["name", "lower", "upper", "initial", "log_scale"],
    "additionalProperties": False,
}
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "stop": {"type": "boolean"},
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["registered", "central_radial"]},
                    "family": {"type": "string", "description": "registered: exact name"},
                    "slug": {"type": "string", "description": "central_radial: short id"},
                    "radial": {"type": "string", "description": "central_radial: f(r)"},
                    "params": {"type": "array", "items": _PARAM_SCHEMA},
                    "rationale": {"type": "string"},
                },
                "required": ["kind", "family", "slug", "radial", "params", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["analysis", "stop", "proposals"],
    "additionalProperties": False,
}

FAMILY_NOTES = {
    "conic-elliptic": "Keplerian ellipse under Newtonian gravity (6 elements)",
    "conic-hyperbolic": "Keplerian hyperbola (unbound)",
    "circle": "circular Keplerian orbit (4 parameters)",
    "central-power-law": "f = -C r^-n, free n and C (integrated)",
    "central-hooke": "f = -k r (integrated)",
    "central-yukawa": "f = -GM/r^2 (1 + r/lam) exp(-r/lam) (integrated)",
    "central-newton-plus-inverse-cube": "f = -GM/r^2 (1 + eps/r) (integrated)",
    "perturbed-conic": "Newton + one unseen planet on a circular ecliptic orbit "
    "(mass, radius, longitude; integrated)",
    "equant-circle": "Kepler's vicarious hypothesis: eccentric circle with an equant",
    "epicycles": "Ptolemaic geocentric deferent + epicycle",
}


class LlmConfig(BayesConfig):
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    max_tokens: int = 16000
    start_families: list[str] = Field(default_factory=lambda: ["conic-elliptic"])
    max_iterations: int = 3
    max_proposals: int = Field(3, description="Proposals fitted per iteration.")
    final_nested: int = Field(3, description="Leading candidates compared by nested sampling.")
    nlive: int = 200
    cache_dir: str = Field("runs/llm_cache", description="On-disk request/response cache.")
    offline: bool = Field(False, description="Fail on a cache miss instead of calling the API.")


# ----------------------------------------------------------------------------- the LLM call


def _client() -> Any:
    import anthropic  # optional dependency (jan[llm])

    return anthropic.Anthropic()


def request_key(request: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def cached_call(request: dict[str, Any], cfg: LlmConfig) -> dict[str, Any]:
    """The JSON object Claude returned for `request`, from the cache or the API."""
    path = Path(cfg.cache_dir) / f"{request_key(request)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["response"]
    if cfg.offline:
        raise RuntimeError(f"llm cache miss ({path.name}) and offline: true")
    msg = _client().beta.messages.create(**request)
    if msg.stop_reason == "refusal":
        raise RuntimeError("the model declined the request")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("response truncated at max_tokens")
    text = next(b.text for b in msg.content if b.type == "text")
    response = {
        "parsed": json.loads(text),
        "model": msg.model,
        "stop_reason": msg.stop_reason,
        "usage": {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"request": request, "response": response}, indent=1), "utf-8")
    tmp.replace(path)
    return response


def build_request(summary: dict[str, Any], history: list[dict[str, Any]], cfg: LlmConfig) -> dict:
    user = (
        "Data and current best fit:\n"
        + json.dumps(summary, indent=1, sort_keys=True)
        + "\n\nCandidates fitted so far (log-evidence, Laplace approximation):\n"
        + json.dumps(history, indent=1, sort_keys=True)
        + f"\n\nRegistered families: {json.dumps(FAMILY_NOTES, sort_keys=True)}"
        + f"\n\nPropose up to {cfg.max_proposals} new candidates (not already fitted), or stop."
    )
    return {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user}],
        "output_config": {
            "effort": cfg.effort,
            "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
        },
        # Server-side fallback if a safety classifier declines (routes by category).
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
    }


# ----------------------------------------------------------------------------- summaries


def _sig(x: float, n: int = 4) -> float | None:
    return float(f"{x:.{n}g}") if np.isfinite(x) else None


def _binned(key: FloatArray, res: FloatArray, n_bins: int) -> list[dict[str, Any]]:
    edges = np.quantile(key, np.linspace(0, 1, n_bins + 1))
    out = []
    for k in range(n_bins):
        m = (key >= edges[k]) & (key <= edges[k + 1])
        if m.sum() < 3:
            continue
        out.append(
            {
                "from": _sig(float(edges[k])),
                "to": _sig(float(edges[k + 1])),
                "mean_east_arcsec": _sig(float(np.mean(res[m, 0]))),
                "mean_north_arcsec": _sig(float(np.mean(res[m, 1]))),
                "rms_arcsec": _sig(float(np.sqrt(np.mean(res[m] ** 2)))),
            }
        )
    return out


def _precession(ob: ObservationSet, meta: PublicMeta, seed: int, cfg: LlmConfig) -> Any:
    """Δϖ between ellipse fits to the two halves of the arc, per orbit (deg)."""
    t = ob.t
    mid = float(np.median(t))
    fits = []
    for m in (t < mid, t >= mid):
        sub = ObservationSet(ob.table.loc[m].reset_index(drop=True))
        if len(sub.t) < 12:
            return None
        s = setup_body(sub, meta)
        model = s.model("conic-elliptic", cfg)
        starts = s.starts(model.family, seed)
        if not starts:
            return None
        fits.append((fit_map(model, starts).x, float(np.median(sub.t))))
    (x1, t1), (x2, t2) = fits
    a = float(np.exp(0.5 * (x1[0] + x2[0])))
    period = 2 * np.pi * np.sqrt(a**3 / meta.gm_sun)
    dvarpi = float(np.angle(np.exp(1j * (x2[2] - x1[2]))))
    return {
        "delta_varpi_deg_per_orbit": _sig(np.degrees(dvarpi) * period / (t2 - t1)),
        "period_days_first_half": _sig(2 * np.pi * np.sqrt(np.exp(3 * x1[0]) / meta.gm_sun)),
        "period_days_second_half": _sig(2 * np.pi * np.sqrt(np.exp(3 * x2[0]) / meta.gm_sun)),
    }


def summarize(cand: Candidate, setup: BodySetup, precession: Any) -> dict[str, Any]:
    model, mp = cand.model, cand.mp
    rb, _ = model.residuals(mp.x)
    res = rb / ARCSEC
    t = setup.data.t
    sigma = setup.data.median_sigma()
    sig_as = (sigma if sigma is not None else float(np.exp(mp.x[-1]))) / ARCSEC
    fam = model.family
    r = np.linalg.norm(
        np.asarray(fam.position(jnp.asarray(mp.x[: fam.n_params]), jnp.asarray(t))), axis=1
    )
    freqs = np.linspace(2 * np.pi / np.ptp(t), 2 * np.pi / (4 * np.median(np.diff(t))), 2000)
    power = lombscargle(t, res[:, 0] - res[:, 0].mean(), freqs, normalize=True)
    top = np.argsort(power)[::-1]
    peaks: list[dict[str, Any]] = []
    for k in top:
        p = 2 * np.pi / freqs[k]
        if all(abs(p / q["period_days"] - 1) > 0.1 for q in peaks):
            peaks.append({"period_days": _sig(float(p), 3), "power": _sig(float(power[k]), 2)})
        if len(peaks) == 3:
            break
    rms = float(np.sqrt(np.mean(res**2)))
    return {
        "n_observations": len(t),
        "span_days": _sig(float(np.ptp(t))),
        "noise_sigma_arcsec": _sig(sig_as),
        "noise_sigma_source": "published" if sigma is not None else "fitted",
        "best_family": cand.name,
        "best_fit_parameters": {
            n: _sig(float(v)) for n, v in zip(fam.param_names, mp.x, strict=False)
        },
        "residual_rms_arcsec": _sig(rms),
        "residual_rms_over_sigma": _sig(rms / sig_as),
        "residuals_by_time": _binned(t, res, 10),
        "residuals_by_heliocentric_distance_au": _binned(r, res, 8),
        "heliocentric_distance_range_au": [_sig(float(r.min())), _sig(float(r.max()))],
        "residual_periodogram_peaks": peaks,
        "two_half_ellipse_fits": precession,
    }


# ----------------------------------------------------------------------------- candidates


class Candidate:
    def __init__(
        self, name: str, model: SkyModel, mp: MapResult, proposal: dict[str, Any] | None
    ) -> None:
        self.name, self.model, self.mp, self.proposal = name, model, mp, proposal
        self.lnz = laplace_log_evidence(model, mp)
        self.method = "laplace"
        self.samples: FloatArray | None = None


def _family(proposal: dict[str, Any], setup: BodySetup) -> BayesFamily:
    if proposal["kind"] == "registered":
        name = proposal["family"]
        if name not in FAMILIES:
            raise ValueError(f"unknown family {name!r}")
        return make_bayes_family(name, setup.t0, setup.gm, setup.span)
    slug = re.sub(r"[^a-z0-9_-]", "-", proposal["slug"].lower())[:32] or "law"
    params = tuple(
        ParamSpec(p["name"], p["lower"], p["upper"], p["initial"], p["log_scale"])
        for p in proposal["params"]
    )
    cls = radial_family(RadialSpec(slug, proposal["radial"], params), setup.gm)
    return cls(t0=setup.t0, gm=setup.gm, t_span=setup.span)


def _fit(
    fam: BayesFamily, setup: BodySetup, cfg: LlmConfig, seed: int
) -> tuple[SkyModel, MapResult]:
    model = SkyModel(fam, setup.data, cfg.likelihood, cfg.nu, setup.observer_family)
    starts = setup.starts(fam, seed)
    if not starts:
        raise ValueError("no starting point")
    return model, fit_map(model, starts)


@register("llm.hypothesize")
class LlmHypothesize:
    """Propose (Claude) → fit (MAP) → compare (Laplace, then nested) loop."""

    name: ClassVar[str]
    supported_tiers: ClassVar[set[str]] = {"T1"}
    Config: ClassVar[type[StrategyConfig]] = LlmConfig

    def fit(
        self,
        obs: ObservationSet,
        meta: PublicMeta,
        config: StrategyConfig,
        rng_seed: int,
        workdir: Path,
    ) -> BaseHypothesis:
        assert isinstance(config, LlmConfig)
        return PerBodyHypothesis(
            {b: self._fit_body(obs.for_body(b), b, meta, config, rng_seed) for b in obs.body_ids}
        )

    def _fit_body(
        self, ob: ObservationSet, bid: str, meta: PublicMeta, cfg: LlmConfig, seed: int
    ) -> PosteriorHypothesis:
        t_start = time.perf_counter()
        setup = setup_body(ob, meta)
        cands: dict[str, Candidate] = {}
        history: list[dict[str, Any]] = []
        for name in cfg.start_families:
            model, mp = _fit(
                make_bayes_family(name, setup.t0, setup.gm, setup.span), setup, cfg, seed
            )
            cands[name] = Candidate(name, model, mp, None)
            history.append({"iteration": 0, "family": name, "source": "start"})
        precession = _precession(ob, meta, seed, cfg)
        usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        for it in range(1, cfg.max_iterations + 1):
            best = max(cands.values(), key=lambda c: c.lnz)
            summary = summarize(best, setup, precession)
            fitted = [
                {"family": c.name, "ln_evidence": _sig(c.lnz, 7), "law": _law_text(c)}
                for c in sorted(cands.values(), key=lambda c: -c.lnz)
            ]
            reply = cached_call(build_request(summary, fitted, cfg), cfg)
            usage["calls"] += 1
            for k in ("input_tokens", "output_tokens"):
                usage[k] += int(reply.get("usage", {}).get(k, 0))
            parsed = reply["parsed"]
            entry: dict[str, Any] = {
                "iteration": it,
                "analysis": parsed["analysis"],
                "stop": bool(parsed["stop"]),
                "proposals": [],
            }
            for prop in parsed["proposals"][: cfg.max_proposals]:
                rec: dict[str, Any] = {"proposal": prop}
                try:
                    fam = _family(prop, setup)
                    if fam.name in cands:
                        raise ValueError("already fitted")
                    model, mp = _fit(fam, setup, cfg, seed)
                    cand = Candidate(fam.name, model, mp, prop)
                    cands[fam.name] = cand
                    rec.update({"family": fam.name, "ln_evidence_laplace": cand.lnz})
                except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                entry["proposals"].append(rec)
            history.append(entry)
            if parsed["stop"] or not any("family" in p for p in entry["proposals"]):
                break
        # Nested-sampling evidence for the leading candidates (bayes.nested's box method).
        ncfg = NestedConfig.model_validate(
            {"nlive": cfg.nlive, "likelihood": cfg.likelihood, "nu": cfg.nu}
        )
        lead = sorted(cands.values(), key=lambda c: -c.lnz)[: cfg.final_nested]
        for c in lead:
            try:
                r = run_box(c.model, c.mp.x, c.mp.cov, ncfg, seed)
                c.lnz, c.method, c.samples = r["lnZ"], "nested", r["samples"]
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                pass
        ranked = sorted(cands.values(), key=lambda c: -c.lnz)
        lnz = np.array([c.lnz for c in ranked])
        prob = np.exp(lnz - lnz.max())
        prob /= prob.sum()
        scores = [
            ModelScore(model_name=c.name, log_evidence=float(c.lnz), posterior_prob=float(p))
            for c, p in zip(ranked, prob, strict=True)
        ]
        best = ranked[0]
        samples = best.samples
        if samples is None:
            g = np.random.default_rng(seed)
            w, v = np.linalg.eigh(0.5 * (best.mp.cov + best.mp.cov.T))
            samples = (
                best.mp.x
                + g.standard_normal((2000, len(best.mp.x))) @ (v * np.sqrt(np.clip(w, 0, None))).T
            )
        info = {
            "history": history,
            "candidates": [
                {
                    "family": c.name,
                    "ln_evidence": float(c.lnz),
                    "method": c.method,
                    "posterior_prob": float(p),
                    "loglike_map": float(c.mp.loglike),
                    "n_params": len(c.mp.x),
                    "law": _law_text(c),
                    "iteration": _iteration(history, c.name),
                }
                for c, p in zip(ranked, prob, strict=True)
            ],
            "best_family": best.name,
            "llm_model": cfg.model,
            "usage": usage,
            "fit_seconds": time.perf_counter() - t_start,
            "notes": f"LLM-proposed families ({usage['calls']} calls, cached in {cfg.cache_dir}); "
            f"best {best.name} (p={prob[0]:.3g}, {best.method} evidence).",
        }
        model = best.model
        return PosteriorHypothesis(
            model.family,
            best.mp.x,
            samples,
            bid,
            setup.observer,
            model.sigma_free,
            sigma_known=None if model.sigma_free else setup.data.median_sigma(),
            n_predictive=cfg.n_predictive_samples,
            seed=seed,
            info=info,
            scores=scores,
        )


def _law_text(c: Candidate) -> str | None:
    ls = getattr(c.model.family, "law_strings", None)
    if ls is None:
        return "Newton (-GM/r**2)" if c.model.family.conic else None
    _, consts, radial = ls(c.mp.x[: c.model.family.n_params])
    shown = {k: _sig(v) for k, v in consts.items() if k != "GM"}
    return f"f(r) = {radial}" + (f" with {shown}" if shown else "")


def _iteration(history: list[dict[str, Any]], name: str) -> int:
    for h in history:
        if h.get("family") == name:
            return int(h["iteration"])
        for p in h.get("proposals", []):
            if p.get("family") == name:
                return int(h["iteration"])
    return 0
