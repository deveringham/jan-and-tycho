"""Model-selection metric (SPEC §7): posterior probability given to the true family."""

from __future__ import annotations

from typing import Any


def selection_metrics(
    truth_body: dict[str, Any], comparison: list[dict[str, Any]] | None
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "true_family": truth_body.get("family"),
        "p_true_family": None,
        "selected_family": None,
        "selected_correct": None,
        "family_probs": None,
    }
    if not comparison:
        return out
    probs = {c["model_name"]: c.get("posterior_prob") for c in comparison}
    out["family_probs"] = probs
    known = {k: v for k, v in probs.items() if v is not None}
    if not known:
        return out
    best = max(known, key=lambda k: known[k])
    out["selected_family"] = best
    fam = truth_body.get("family")
    if fam in probs:
        out["p_true_family"] = probs[fam]
        out["selected_correct"] = best == fam
    return out
