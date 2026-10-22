"""Phase 5: `llm.hypothesize` — safe parsing of proposed laws, caching, the loop (fake LLM)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from jan.models.radial import ParamSpec, RadialSpec, parse_radial, radial_family
from jt_core.constants import GM_SUN

N = ParamSpec("n", 1.5, 2.5, 2.0)


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').system('true')",
        "r.__class__",
        "exec('1')",
        "lambda: 1",
        "-GM*r**(-n); 1",
        "-GM*q",
    ],
)
def test_parse_radial_rejects_anything_but_arithmetic(text: str) -> None:
    with pytest.raises(ValueError):
        parse_radial(text, [N])


def test_radial_family_reproduces_newton() -> None:
    cls = radial_family(RadialSpec("power", "-GM*r^(-n)", (N,)), GM_SUN)
    fam = cls(t0=0.0, gm=GM_SUN, t_span=1.0)
    phi = jnp.asarray([np.log(1.5), 0.1, 1.0, 0.99, 0.5, 2.0, 2.0])
    x = jnp.asarray([[1.2, 0.3, 0.1]])
    a = np.asarray(fam.acceleration(phi, x, jnp.zeros(1)))[0]
    r = np.linalg.norm(x[0])
    assert np.allclose(a, -GM_SUN * np.asarray(x[0]) / r**3, rtol=1e-12)
    _, consts, radial = fam.law_strings(np.asarray(phi))
    assert consts["p_n"] == 2.0 and "p_n" in radial


def _reply(payload: dict[str, Any]) -> Any:
    text = SimpleNamespace(type="text", text=json.dumps(payload))
    usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    return SimpleNamespace(content=[text], stop_reason="end_turn", model="fake", usage=usage)


class FakeClient:
    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies, self.requests = replies, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        return _reply(self.replies[len(self.requests) - 1])


POWER = {
    "kind": "central_radial",
    "family": "",
    "slug": "power",
    "radial": "-GM*mu*r**(-n)",
    "params": [
        {"name": "n", "lower": 1.5, "upper": 2.5, "initial": 2.0, "log_scale": False},
        {"name": "mu", "lower": 0.5, "upper": 2.0, "initial": 1.0, "log_scale": True},
    ],
    "rationale": "perihelion precession suggests a non-inverse-square law",
}
BAD = {**POWER, "slug": "evil", "radial": "__import__('os').getcwd()"}


@pytest.mark.slow
@pytest.mark.bayes
def test_llm_loop_finds_power_law_and_replays_from_cache(tmp_path, monkeypatch) -> None:
    from jan.strategies.llm import hypothesize as hyp
    from jt_bench.harness import ensure_run, fit, score
    from tycho import load_scenario

    from .conftest import SCENARIOS

    fake = FakeClient(
        [
            {"analysis": "precession", "stop": False, "proposals": [POWER, BAD]},
            {"analysis": "fits", "stop": True, "proposals": []},
        ]
    )
    monkeypatch.setattr(hyp, "_client", lambda: fake)
    sc = load_scenario(SCENARIOS / "precessing_power_law.yaml")
    d = ensure_run(sc, 0, run_dir=tmp_path / "r")
    cfg = {"cache_dir": str(tmp_path / "cache"), "final_nested": 2, "nlive": 100}
    res = fit(d, "llm.hypothesize", cfg)
    s = score(d, res.label, plots=False)["bodies"]["target"]
    assert s["structural"]["symbolic"] in ("exact", "up_to_constants")
    info = json.loads((d / "jan" / res.label / "hypothesis.json").read_text())
    hist = info["hypothesis"]["bodies"]["target"]["info"]["history"]
    errors = [p.get("error", "") for h in hist for p in h.get("proposals", [])]
    assert any("unknown name" in e for e in errors)  # the injected expression was refused
    assert len(fake.requests) == 2 and len(list((tmp_path / "cache").glob("*.json"))) == 2
    # A re-run is served from the cache: no API call, identical choice.
    monkeypatch.setattr(hyp, "_client", lambda: pytest.fail("cache miss"))
    res2 = fit(d, "llm.hypothesize", {**cfg, "offline": True}, label="replay")
    info2 = json.loads((d / "jan" / res2.label / "hypothesis.json").read_text())
    best = info["hypothesis"]["bodies"]["target"]["info"]["best_family"]
    assert info2["hypothesis"]["bodies"]["target"]["info"]["best_family"] == best == "llm:power"
    from rudolf.viz.manifest import FigureManifest

    score(d, res2.label, plots=True)
    assert FigureManifest.load(d / "rudolf" / res2.label).get("hypothesis_history") is not None
