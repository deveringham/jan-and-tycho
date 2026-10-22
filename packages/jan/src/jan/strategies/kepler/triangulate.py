"""`kepler.triangulate`: Kepler's own route to the orbit of Mars (SPEC §6.5, Phase 3).

1. Period: candidate periods from the sky track's periodogram (excluding the observer's
   year); the one whose period-separated lines of sight meet best wins, then is refined.
2. Triangulation (`jan.reconstruct`): with the observer known (T1), intersect the lines of
   sight at t, t+P, t+2P, …; with it unknown (T2), Kepler's reverse trick recovers both the
   target point and the observer's positions from the Sun stream, one scale per period
   group, and the groups are tied together by requiring a single Keplerian observer orbit
   (whose size the period fixes through GM).
3. Law or path: a conic is fitted to the triangulated points. If it explains them about as
   well as a free periodic path, the answer is dynamical (Newtonian conic); otherwise the
   strategy honestly reports a kinematic, periodic Fourier path.

Assumptions: the target is periodic and observed over ≥ 2 periods; failure modes are
period aliases, short arcs, and near-degenerate geometry (lines of sight almost parallel).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from scipy.optimize import least_squares

from jan.base import BaseHypothesis, PerBodyHypothesis, StrategyConfig, register
from jan.fitting import ConicProblem, sky_periods
from jan.inference import SkyModel, fit_map
from jan.models.conic import EllipticOrbit, canonical_elements
from jan.models.conic_hypothesis import ConicHypothesis
from jan.models.families import ObserverFamily
from jan.models.kinematic import (
    FourierPath,
    KinematicPath,
    PeriodicKinematicHypothesis,
    PeriodicLinearPath,
    PolygonPath,
    fit_periodic,
)
from jan.models.observer import observer_function
from jan.models.posterior import PosteriorHypothesis
from jan.reconstruct.periodic_triangulation import (
    DirectionInterpolator,
    TriPoint,
    consistency,
    refine_period,
    triangulate_known_observer,
    triangulate_unknown_observer,
)
from jan.reconstruct.polygon import Polygon, fit_polygon
from jan.setup import observer_known, observer_positions, setup_body
from jt_core.conics import ConicElements, state_to_elements
from jt_core.constants import ARCSEC
from jt_core.schema import (
    ElementsEstimate,
    HypothesisKind,
    LawEstimate,
    ObservationSet,
    PositionPrediction,
    PublicMeta,
    SkyPrediction,
)

FloatArray = NDArray[np.float64]
YEAR = 365.25


class TriangulateConfig(StrategyConfig):
    n_anchor: int = Field(72, description="Phases per period at which to triangulate.")
    period_days: float | None = Field(None, description="Override the period search.")
    kinematic_ratio: float = Field(
        2.0, description="Reject the conic when its RMS exceeds this × the periodic path's."
    )
    n_harmonics: int = Field(8, description="Order of the periodic path used to test the conic.")
    kinematic_anchors: int = Field(
        180, description="Phases re-triangulated for a kinematic answer (finer: keeps corners)."
    )
    assumed_sigma_arcsec: float = 60.0
    refine_with_observations: bool = Field(
        True,
        description="Polish a dynamical answer against all sky observations (differential "
        "correction from the triangulated orbit), as Kepler checked his orbit against the data.",
    )
    n_predictive_samples: int = 64


class TriangulationHypothesis(BaseHypothesis):
    """Wraps the final model (conic or kinematic) with the triangulation evidence and,
    under T2, the reconstructed observer orbit."""

    def __init__(
        self,
        inner: BaseHypothesis,
        body_id: str,
        period: float,
        points: list[TriPoint],
        info: dict[str, Any],
        observer: Callable[[FloatArray], FloatArray] | None = None,
        observer_el: ConicElements | None = None,
    ) -> None:
        self.inner = inner
        self.body_id = body_id
        self.period = period
        self.points = points
        self.info = info
        self.kind: HypothesisKind = inner.kind
        self._observer = observer
        self._observer_el = observer_el

    def predict_sky(self, t: FloatArray, body_id: str) -> SkyPrediction:
        return self.inner.predict_sky(t, body_id)

    def heliocentric_position(self, t: FloatArray, body_id: str) -> PositionPrediction | None:
        return self.inner.heliocentric_position(t, body_id)

    def elements(self, body_id: str) -> ElementsEstimate | None:
        return self.inner.elements(body_id)

    def law(self) -> LawEstimate | None:
        return self.inner.law()

    def observation_sigma(self, body_id: str) -> float | None:
        v = self.info.get("sigma_rad")
        return None if v is None else float(v)

    def observer_position(self, t: FloatArray) -> PositionPrediction | None:
        inner = self.inner.observer_position(t)
        if inner is not None:
            return inner
        if self._observer is None or self._observer_el is None:
            return None
        return PositionPrediction(mean=self._observer(np.atleast_1d(t)))

    def observer_elements(self) -> ElementsEstimate | None:
        inner = self.inner.observer_elements()
        if inner is not None:
            return inner
        if self._observer_el is None:
            return None
        return ElementsEstimate(point=self._observer_el.to_dict())

    def notes(self) -> str:
        return str(self.info.get("notes", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "period_days": self.period,
            "model": self.inner.to_dict(),
            "triangulation": [p.to_dict() for p in self.points],
            "info": self.info,
            "notes": self.notes(),
        }


def _candidate_periods(ob: ObservationSet, observer_period: float | None) -> list[float]:
    cands = []
    for p in sky_periods(ob, n=3):
        if observer_period and abs(p / observer_period - 1) < 0.03:
            continue  # the observer's own year
        if observer_period and abs(p / (observer_period / 2) - 1) < 0.03:
            continue
        cands.append(p)
    return cands


def _fit_observer_points(
    groups: list[TriPoint],
    gm: float,
    t0: float,
) -> tuple[ObserverFamily, FloatArray, FloatArray]:
    """T2: one Keplerian observer orbit through all groups' observer points, solving for a
    scale per group (alternating least squares). Returns (family, φ, group scales)."""
    fam = ObserverFamily(t0=t0, gm=gm)
    t_all = np.concatenate([g.times for g in groups])
    e_raw = [np.asarray(g.extra["observer"]) for g in groups]
    c = np.ones(len(groups))
    # Start: circular orbit at the scale where the mean distance is ~1 AU-equivalent.
    k0 = int(np.argmin(np.abs(t_all - t0)))
    e0 = np.concatenate(e_raw)[k0]
    phi = fam.wrap(np.array([0.0, 0.01, 0.0, float(np.arctan2(e0[1], e0[0]))]))
    for _ in range(12):
        pts = np.concatenate([ci * e for ci, e in zip(c, e_raw, strict=True)])

        def res(p: FloatArray, pts: FloatArray = pts) -> FloatArray:
            r = np.asarray(fam.position(jnp.asarray(p), jnp.asarray(t_all)))
            return (r - pts)[:, :2].ravel()

        phi = least_squares(
            res,
            phi,
            bounds=([np.log(0.3), 0, -np.inf, -np.inf], [np.log(3.0), 0.3, np.inf, np.inf]),
        ).x
        fit = np.asarray(fam.position(jnp.asarray(phi), jnp.asarray(t_all)))
        i = 0
        for j, e in enumerate(e_raw):
            f = fit[i : i + len(e)]
            c[j] = float(np.sum(f * e) / np.sum(e * e))
            i += len(e)
    return fam, fam.wrap(phi), c


def _fit_conic(
    taus: FloatArray,
    pts: FloatArray,
    sd: FloatArray | None,
    gm: float,
) -> tuple[FloatArray, ConicElements, float, EllipticOrbit]:
    """Elliptic orbit through the triangulated points (positions with phase times)."""
    order = np.argsort(taus)
    t, x = taus[order], pts[order]
    t0 = float(np.median(t))
    fam = EllipticOrbit(t0=t0, gm=gm)
    w = None if sd is None else 1.0 / np.maximum(sd[order], 1e-9)
    prob = ConicProblem(fam, t, sky=False, pos_obs=x, weights=w)
    starts = []
    for k in np.linspace(1, len(t) - 2, 6).astype(int):
        v = (x[k + 1] - x[k - 1]) / (t[k + 1] - t[k - 1])
        try:
            el = state_to_elements(x[k], v, float(t[k]), gm)
        except ValueError:
            continue
        if fam.admits(el):
            starts.append(fam.theta_from_elements(el))
    if not starts:
        # e.g. a path through the Sun: fall back to a circle at the median radius.
        r = float(np.median(np.linalg.norm(x, axis=1)))
        starts.append(fam.theta_from_elements(ConicElements(max(r, 1e-3), 0.05, 0.1, 0, 0, t0)))
    fits = []
    for s in starts:
        try:
            fits.append(
                least_squares(prob.residuals, s, jac=prob.jacobian, method="trf", x_scale="jac")  # pyright: ignore[reportArgumentType]
            )
        except ValueError:  # non-finite Jacobian: the conic cannot follow the points
            continue
    if not fits:
        theta = starts[0]
        el = canonical_elements(np.asarray(fam.elements(jnp.asarray(theta))))
        return theta, el, float("inf"), fam
    best = min(fits, key=lambda r: r.cost)
    theta = best.x
    el = canonical_elements(np.asarray(fam.elements(jnp.asarray(theta))))
    r = np.asarray(fam.position(jnp.asarray(theta), jnp.asarray(t))) - x
    return theta, el, float(np.sqrt(np.mean(r**2))), fam


def _refine_t1(
    ob: ObservationSet,
    gm: float,
    el: ConicElements,
    observer: Any,
    cfg: TriangulateConfig,
    info: dict[str, Any],
    bid: str,
) -> BaseHypothesis:
    """Differential correction (T1): sky least squares from the triangulated orbit."""
    t0 = float(np.median(ob.t))
    fam = EllipticOrbit(t0=t0, gm=gm)
    r_obs = observer(ob.t)
    w = np.full((len(ob), 2), 1.0 / (cfg.assumed_sigma_arcsec * ARCSEC))
    prob = ConicProblem(fam, ob.t, sky=True, u_obs=ob.unit_vectors(), r_obs=r_obs, weights=w)
    res = least_squares(
        prob.residuals,
        fam.theta_from_elements(el),
        jac=prob.jacobian,  # pyright: ignore[reportArgumentType]
        method="trf",
        x_scale="jac",
        max_nfev=200,
    )
    r = prob.residuals(res.x)
    jm = prob.jacobian(res.x)
    red = float(r @ r) / max(r.size - len(res.x), 1)
    cov = np.linalg.pinv(jm.T @ jm, rcond=1e-13) * red
    info["sigma_rad"] = float(np.sqrt(red)) * cfg.assumed_sigma_arcsec * ARCSEC
    return ConicHypothesis(
        fam, res.x, bid, observer, cov=cov, n_samples=cfg.n_predictive_samples, info=info
    )


def _with_polygon_vertices(
    taus: FloatArray,
    pts: FloatArray,
    period: float,
    fourier_rms: float,
) -> tuple[FloatArray, FloatArray, Polygon | None]:
    """If the points form a polygon (fits at least as well as a smooth periodic path),
    insert its vertices — recovered by intersecting edges, even inside observing gaps."""
    phase = np.mod(taus - taus.min(), period)
    order = np.argsort(phase)
    ph, pp = phase[order], pts[order]
    diam = float(np.max(np.linalg.norm(pp - pp.mean(axis=0), axis=1)))
    # Coarser tolerances merge noise-induced corners; keep the fewest vertices that fit.
    fits = [fit_polygon(pp, eps=f * diam) for f in (0.02, 0.04, 0.08)]
    ok = [p for p in fits if p is not None and p.rms <= 1.5 * fourier_rms]
    if not ok:
        return taus, pts, None
    poly = min(ok, key=lambda p: (len(p.vertices), p.rms))
    n = len(pp)
    new_t, new_x = list(ph), list(pp)
    for v in poly.vertices:
        # The gap (j, j+1) the vertex fits into best (smallest detour).
        detour = [
            np.linalg.norm(v - pp[j])
            + np.linalg.norm(pp[(j + 1) % n] - v)
            - np.linalg.norm(pp[(j + 1) % n] - pp[j])
            for j in range(n)
        ]
        j = int(np.argmin(detour))
        a, b = pp[j], pp[(j + 1) % n]
        pa, pb = ph[j], ph[(j + 1) % n] + (period if j == n - 1 else 0.0)
        w = np.linalg.norm(v - a) / max(np.linalg.norm(v - a) + np.linalg.norm(b - v), 1e-30)
        new_t.append(float(np.mod(pa + w * (pb - pa), period)))
        new_x.append(v)
    t_out = np.array(new_t) + taus.min()
    return t_out, np.array(new_x), poly


def _choose_path(
    taus: FloatArray,
    pts: FloatArray,
    period: float,
    order: int,
    path_t: FloatArray,
    path_x: FloatArray,
    poly: Polygon | None,
) -> tuple[KinematicPath, str, dict[str, float]]:
    """Pick the kinematic path (polygon, smooth Fourier, or point-to-point) with the
    lowest cross-validated error on the triangulated points. Folds are contiguous runs
    in phase: neighbouring points share interpolated directions, so their errors are
    correlated and interleaved folds would reward joining noisy points."""
    builders: dict[str, Callable[[FloatArray, FloatArray], KinematicPath]] = {
        "linear": lambda t, x: PeriodicLinearPath(t, x, period),
        "fourier": lambda t, x: FourierPath(fit_periodic(t, x, period, order)[0], period),
    }
    if poly is not None:
        builders["polygon"] = lambda t, x: PolygonPath(t, x, poly.vertices, period)
    rank = np.argsort(np.argsort(np.mod(taus - taus.min(), period)))
    folds = (rank * 10 // len(taus)) % 5
    cv: dict[str, float] = {}
    for name, build in builders.items():
        err = []
        for k in range(5):
            tr, te = folds != k, folds == k
            err.append(np.sum((build(taus[tr], pts[tr])(taus[te]) - pts[te]) ** 2, axis=1))
        cv[name] = float(np.sqrt(np.mean(np.concatenate(err))))
    best = min(cv, key=lambda n: cv[n])
    if best == "linear":  # keep the inserted polygon corners when joining points
        return PeriodicLinearPath(path_t, path_x, period), best, cv
    return builders[best](taus, pts), best, cv


@register("kepler.triangulate")
class KeplerTriangulate:
    """Periodic triangulation (Kepler's trick), then a conic or an honest kinematic path."""

    name: ClassVar[str]
    supported_tiers: ClassVar[set[str]] = {"T1", "T2"}
    Config: ClassVar[type[StrategyConfig]] = TriangulateConfig

    def fit(
        self,
        obs: ObservationSet,
        meta: PublicMeta,
        config: StrategyConfig,
        rng_seed: int,
        workdir: Path,
    ) -> BaseHypothesis:
        assert isinstance(config, TriangulateConfig)
        if not obs.is_sky:
            raise ValueError("kepler.triangulate needs sky observations")
        return PerBodyHypothesis(
            {b: self._fit_body(obs.for_body(b), b, meta, config, obs.sun()) for b in obs.body_ids}
        )

    def _refine(
        self,
        ob: ObservationSet,
        bid: str,
        meta: PublicMeta,
        sun: ObservationSet | None,
        el: ConicElements,
        observer_el: ConicElements | None,
        cfg: TriangulateConfig,
        info: dict[str, Any],
    ) -> BaseHypothesis:
        if observer_el is None:
            return _refine_t1(ob, meta.gm_sun, el, observer_function(meta), cfg, info, bid)
        # T2: joint body + observer fit (with the Sun stream), from the reconstruction.
        setup = setup_body(ob, meta, sun)
        assert setup.observer_family is not None
        model = SkyModel(setup.family("conic-elliptic"), setup.data, observer=setup.observer_family)
        x_b = model.family.from_elements(el)
        x_o = setup.observer_family.from_elements(observer_el)
        if x_b is None or x_o is None:
            raise RuntimeError("triangulated orbit outside the elliptic family")
        mp = fit_map(model, [np.concatenate([x_b, x_o])])
        g = np.random.default_rng(0)
        w, v = np.linalg.eigh(mp.cov)
        draws = (
            mp.x
            + g.standard_normal((cfg.n_predictive_samples, len(mp.x)))
            @ (v * np.sqrt(np.clip(w, 0, None))).T
        )
        info["sigma_rad"] = mp.rms_rad
        return PosteriorHypothesis(
            model.family,
            mp.x,
            draws,
            bid,
            setup.observer,
            model.sigma_free,
            n_predictive=cfg.n_predictive_samples,
            info=info,
            observer_family=model.observer,
        )

    def _fit_body(
        self,
        ob: ObservationSet,
        bid: str,
        meta: PublicMeta,
        cfg: TriangulateConfig,
        sun: ObservationSet | None,
    ) -> TriangulationHypothesis:
        t_start = time.perf_counter()
        gm = meta.gm_sun
        sigma = cfg.assumed_sigma_arcsec * ARCSEC
        interp = DirectionInterpolator.from_obs(ob)
        t_range = (float(ob.t.min()), float(ob.t.max()))
        known = observer_known(meta)
        if not known and sun is None:
            raise ValueError("kepler.triangulate needs the observer (T1) or the Sun stream (T2)")
        sun_interp = None if known else DirectionInterpolator.from_obs(sun)  # type: ignore[arg-type]
        obs_period = YEAR
        if known and meta.observer_elements is not None:
            el = ConicElements(*meta.observer_elements.as_tuple())
            obs_period = el.period(gm)
        observer = observer_function(meta) if known else None

        def make(period: float) -> list[TriPoint]:
            if observer is not None:
                return triangulate_known_observer(
                    interp, observer, period, t_range, cfg.n_anchor, sigma
                )
            assert sun_interp is not None
            return triangulate_unknown_observer(interp, sun_interp, period, t_range, cfg.n_anchor)

        def make_fine(period: float) -> list[TriPoint]:
            assert observer is not None
            return triangulate_known_observer(
                interp, observer, period, t_range, cfg.kinematic_anchors, sigma
            )

        cands = [cfg.period_days] if cfg.period_days else _candidate_periods(ob, obs_period)
        cands = [p for p in cands if p and 2 * p <= t_range[1] - t_range[0] + 1e-9]
        if not cands:
            raise RuntimeError("no period with ≥ 2 repeats in the data (arc too short)")
        tried = [(consistency(make(p)), p) for p in cands]
        _, p0 = min(tried)
        period, points = refine_period(make, p0)
        if len(points) < 6:
            raise RuntimeError(f"only {len(points)} triangulated points (period {period:.1f} d)")

        observer_el = None
        obs_fn = observer
        if not known:
            obs_fam, phi_o, scales = _fit_observer_points(points, gm, float(np.median(ob.t)))
            for p, c in zip(points, scales, strict=True):
                p.x = p.x * c
                p.origins = p.origins * c
                p.extra["observer"] = p.origins.tolist()
                p.extra["scale"] = float(c)
            observer_el = ConicElements.from_array(np.asarray(obs_fam.elements(jnp.asarray(phi_o))))
            obs_fn = observer_positions(obs_fam, phi_o)

        taus = np.array([p.tau for p in points])
        pts = np.array([p.x for p in points])
        sd = None
        if known:
            sd = np.array([np.sqrt(np.clip(np.diag(p.cov), 1e-30, None)) for p in points])
        theta, el, conic_rms, fam = _fit_conic(taus, pts, sd, gm)
        _, path_rms = fit_periodic(taus, pts, period, cfg.n_harmonics)
        dynamical = conic_rms <= cfg.kinematic_ratio * max(path_rms, 1e-12)
        info: dict[str, Any] = {
            "period_days": period,
            "candidate_periods": [p for _, p in tried],
            "n_points": len(points),
            "conic_rms_au": conic_rms,
            "path_rms_au": path_rms,
            "dynamical": bool(dynamical),
            "sigma_rad": sigma,
            "fit_seconds": time.perf_counter() - t_start,
        }
        info["triangulated_elements"] = el.to_dict()
        if observer_el is not None:
            info["triangulated_observer_elements"] = observer_el.to_dict()
        if dynamical:
            inner: BaseHypothesis = ConicHypothesis(fam, theta, bid, obs_fn, info=info)
            verdict = f"conic fits the points (RMS {conic_rms:.2e} AU vs path {path_rms:.2e} AU)"
            if cfg.refine_with_observations:
                inner = self._refine(ob, bid, meta, sun, el, observer_el, cfg, info)
                verdict += "; refined against all observations"
        else:
            # Honest kinematic answer: re-triangulate more finely and join the points.
            fine = make_fine(period) if known else points
            if len(fine) >= len(points):
                points = fine
            taus = np.array([p.tau for p in points])
            pts = np.array([p.x for p in points])
            coef, fourier_rms = fit_periodic(taus, pts, period, cfg.n_harmonics)
            path_t, path_x, poly = _with_polygon_vertices(taus, pts, period, fourier_rms)
            if poly is not None:
                info["polygon_vertices"] = poly.vertices.tolist()
                info["polygon_rms_au"] = poly.rms
            path, path_kind, cv = _choose_path(
                taus, pts, period, cfg.n_harmonics, path_t, path_x, poly
            )
            info["path_kind"] = path_kind
            info["path_cv_rms_au"] = cv
            inner = PeriodicKinematicHypothesis(bid, period, coef, obs_fn, info, path=path)
            verdict = (
                f"no conic fits (RMS {conic_rms:.2e} AU vs periodic path {path_rms:.2e} AU): "
                f"reporting a kinematic {path_kind} path from {len(points)} triangulated points"
            )
        info["notes"] = (
            f"P = {period:.2f} d from {len(points)} triangulated points"
            f"{'' if known else ' (observer reconstructed from the Sun stream)'}; {verdict}."
        )
        return TriangulationHypothesis(
            inner, bid, period, points, info, obs_fn if not known else None, observer_el
        )
