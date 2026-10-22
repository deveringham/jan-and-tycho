"""Probabilistic predictive metrics on the holdout (SPEC §7, Phase 2).

* NLPD — negative log predictive density of the *noisy* holdout observations under
  N(mean, C + σ̂² I) in the tangent plane at the predicted mean, where C is the predictive
  (parameter) covariance and σ̂ the strategy's own noise estimate. Lower is better; in nats
  per observation for directions measured in radians.
* Energy score — from predictive samples (plus σ̂ noise), in arcsec; lower is better.
* Predictive coverage — fraction of noise-free holdout directions inside the strategy's
  68% / 95% predictive ellipses (from C alone): calibration of the predictive uncertainty.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from jt_core.angles import tangent_residuals
from jt_core.constants import ARCSEC
from jt_core.frames import lonlat_to_unit
from rudolf.data import RunView

FloatArray = NDArray[np.float64]

CHI2_2DOF_68 = 2.2788
CHI2_2DOF_95 = 5.9915
_KEYS = ("nlpd", "energy_score_arcsec", "pred_cover68", "pred_cover95")


def _sample_cov(samples: FloatArray, mean: FloatArray) -> FloatArray:
    r = np.stack([tangent_residuals(s, mean) for s in samples])  # (S, N, 2)
    return np.einsum("sna,snb->nab", r, r) / max(len(samples), 1)


def probabilistic_metrics(view: RunView, rng_seed: int = 0) -> dict[str, Any]:
    out: dict[str, Any] = dict.fromkeys(_KEYS)
    if not view.sealed.is_sky:
        return out
    bid = view.body_id
    mean = view.jan.sky("holdout", bid)
    if mean is None or len(mean) == 0:
        return out
    hold = view.sealed.holdout_body(bid)
    y = lonlat_to_unit(hold["lon"].to_numpy(), hold["lat"].to_numpy())
    truth = view.sealed.holdout_true_directions(bid)
    cov = view.jan.sky_cov("holdout", bid)
    samples = view.jan.sample_array("holdout", bid, "u")
    if cov is None and samples is not None:
        cov = _sample_cov(samples, mean)
    sig = view.jan.obs_sigma(bid)

    if sig is not None:
        c = np.zeros((len(mean), 2, 2)) if cov is None else cov
        s_mat = c + sig**2 * np.eye(2)[None]
        r = tangent_residuals(y, mean)
        inv = np.linalg.inv(s_mat)
        maha = np.einsum("na,nab,nb->n", r, inv, r)
        _, logdet = np.linalg.slogdet(2 * np.pi * s_mat)
        out["nlpd"] = float(np.mean(0.5 * maha + 0.5 * logdet))

    if samples is not None:
        g = np.random.default_rng(rng_seed)
        x = np.stack([tangent_residuals(s, y) for s in samples]) / ARCSEC  # (S, N, 2) about y
        if sig is not None:
            x = x + g.normal(0, sig / ARCSEC, x.shape)
        term1 = np.mean(np.linalg.norm(x, axis=2), axis=0)
        perm = g.permutation(len(x))
        term2 = np.mean(np.linalg.norm(x - x[perm], axis=2), axis=0)
        out["energy_score_arcsec"] = float(np.mean(term1 - 0.5 * term2))

    if cov is not None:
        r = tangent_residuals(truth, mean)
        inv = np.linalg.pinv(cov)
        chi2 = np.einsum("na,nab,nb->n", r, inv, r)
        out["pred_cover68"] = float(np.mean(chi2 <= CHI2_2DOF_68))
        out["pred_cover95"] = float(np.mean(chi2 <= CHI2_2DOF_95))
    return out
