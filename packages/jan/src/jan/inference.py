"""Shared machinery for the Bayesian strategies: likelihood, MAP, Laplace, starts.

The likelihood is defined on tangent-plane residuals of the model direction in the
(east, north) basis at each observed direction (never on raw longitudes, SPEC §4):

* Gaussian:   r ~ N(0, σ² I₂) per observation;
* Student-t:  bivariate t with ν degrees of freedom and scale σ (robust to outliers).

σ is either published by Tycho (`report_uncertainties`) or a free parameter `log_sigma`
with a log-uniform prior on [0.1″, 45°]. Arrays are padded to power-of-two lengths with a
zero mask so the jitted kernels are reused across datasets.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln
from numpy.typing import NDArray
from scipy.optimize import least_squares

from jan.models.families import BayesFamily, Dim, ObserverFamily
from jt_core.angles import tangent_basis
from jt_core.conics import ConicElements, bucket, pad_rows
from jt_core.constants import ARCSEC, DEG
from jt_core.schema import ObservationSet

FloatArray = NDArray[np.float64]
Likelihood = Literal["gaussian", "student_t"]

LOG_SIGMA_DIM = Dim("log_sigma", float(np.log(0.1 * ARCSEC)), float(np.log(45.0 * DEG)))


def _pad_block(
    t: FloatArray,
    u: FloatArray,
    sigma: FloatArray | None,
    r_obs: FloatArray | None,
    minimum: int = 8,
) -> tuple[jax.Array, ...]:
    """Padded (t, r_obs, u, east, north, mask, σ) arrays for one observation stream."""
    n = len(t)
    nb = bucket(n, minimum)
    if n == 0:  # an absent stream: a single masked dummy row, padded
        t = np.zeros(1)
        u = np.array([[1.0, 0.0, 0.0]])
        sigma = None
        r_obs = None
    east, north = tangent_basis(u)
    mask = np.concatenate([np.ones(n), np.zeros(nb - n)])
    sig = np.ones(len(t)) if sigma is None else np.asarray(sigma)
    r = np.zeros((len(t), 3)) if r_obs is None else np.asarray(r_obs)
    cols = (t, r, u, east, north)
    return (
        *(jnp.asarray(pad_rows(a, nb)) for a in cols),
        jnp.asarray(mask),
        jnp.asarray(pad_rows(sig, nb)),
    )


@dataclass
class SkyData:
    """Padded, jit-friendly view of one body's sky observations (+ the Sun stream, T2).

    `r_obs` is the known observer position (T1) or None when the observer is fitted.
    """

    t: FloatArray
    u_obs: FloatArray
    r_obs: FloatArray | None
    sigma: FloatArray | None = None
    """Published per-observation σ (radians), or None."""
    sun_t: FloatArray | None = None
    sun_u: FloatArray | None = None
    sun_sigma: FloatArray | None = None
    n: int = field(init=False)
    n_sun: int = field(init=False)
    args: tuple[jax.Array, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.n = len(self.t)
        self.n_sun = 0 if self.sun_t is None else len(self.sun_t)
        body = _pad_block(self.t, self.u_obs, self.sigma, self.r_obs)
        if self.sun_t is not None and self.sun_u is not None:
            sun = _pad_block(self.sun_t, self.sun_u, self.sun_sigma, None)
        else:
            sun = _pad_block(np.zeros(0), np.zeros((0, 3)), None, None)
        self.args = body + sun[:1] + sun[2:]  # the Sun stream needs no observer column

    @classmethod
    def from_obs(
        cls, ob: ObservationSet, r_obs: FloatArray | None, sun: ObservationSet | None = None
    ) -> SkyData:
        def sig(o: ObservationSet) -> FloatArray | None:
            s = o.sigmas()
            return None if s is None else np.sqrt(0.5 * (s[:, 0] ** 2 + s[:, 1] ** 2))

        if sun is None:
            return cls(ob.t, ob.unit_vectors(), r_obs, sig(ob))
        return cls(ob.t, ob.unit_vectors(), r_obs, sig(ob), sun.t, sun.unit_vectors(), sig(sun))

    @property
    def sigma_known(self) -> bool:
        return self.sigma is not None

    def median_sigma(self) -> float | None:
        return None if self.sigma is None else float(np.median(self.sigma))


def _tangent(u: jax.Array, u_obs: jax.Array, east: jax.Array, north: jax.Array) -> jax.Array:
    d = jnp.sum(u * u_obs, axis=1)
    return jnp.stack([jnp.sum(u * east, 1) / d, jnp.sum(u * north, 1) / d], axis=1)


def _residuals(
    fam: BayesFamily,
    obs_fam: ObserverFamily,
    xf: jax.Array,
    p: int,
    observer_free: bool,
    t: jax.Array,
    r_obs: jax.Array,
    u_obs: jax.Array,
    east: jax.Array,
    north: jax.Array,
    ts: jax.Array,
    us: jax.Array,
    es: jax.Array,
    ns: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """(body residuals (N, 2), Sun-stream residuals (Ns, 2)) for the fitted vector xf."""
    phi = xf[:p]
    if observer_free:
        phi_o = xf[p : p + OBS_DIMS]
        r_obs = obs_fam.position(phi_o, t)
        r_sun = obs_fam.position(phi_o, ts)
        sun_dir = -r_sun / jnp.linalg.norm(r_sun, axis=1, keepdims=True)
        sun_res = _tangent(sun_dir, us, es, ns)
    else:
        sun_res = jnp.zeros((ts.shape[0], 2))
    return _tangent(fam.sky(phi, t, r_obs), u_obs, east, north), sun_res


def _logpdf(r: jax.Array, sigma: jax.Array, kind: str, nu: float) -> jax.Array:
    q = jnp.sum(r**2, axis=1) / sigma**2
    if kind == "gaussian":
        lp = -0.5 * q - 2.0 * jnp.log(sigma) - jnp.log(2 * jnp.pi)
    else:
        lp = (
            gammaln((nu + 2) / 2)
            - gammaln(nu / 2)
            - jnp.log(nu * jnp.pi)
            - 2.0 * jnp.log(sigma)
            - (nu + 2) / 2 * jnp.log1p(q / nu)
        )
    return jnp.where(jnp.isfinite(lp), lp, -1e30)


def _loglike_x(
    fam: BayesFamily,
    obs_fam: ObserverFamily,
    x: jax.Array,
    p: int,
    observer_free: bool,
    sigma_free: bool,
    kind: str,
    nu: float,
    t: jax.Array,
    r_obs: jax.Array,
    u_obs: jax.Array,
    east: jax.Array,
    north: jax.Array,
    mask: jax.Array,
    sig_obs: jax.Array,
    ts: jax.Array,
    us: jax.Array,
    es: jax.Array,
    ns: jax.Array,
    mask_s: jax.Array,
    sig_s: jax.Array,
) -> jax.Array:
    """Log-likelihood of the full vector: (φ, observer φ when free, log σ when free)."""
    nf = p + (OBS_DIMS if observer_free else 0)
    log_sigma = x[nf] if sigma_free else jnp.zeros(())
    rb, rs = _residuals(
        fam, obs_fam, x[:nf], p, observer_free, t, r_obs, u_obs, east, north, ts, us, es, ns
    )
    scale = jnp.exp(log_sigma)  # published σ × scale, or exp(log_sigma) on a unit σ
    lp = jnp.sum(_logpdf(rb, sig_obs * scale, kind, nu) * mask)
    return lp + jnp.sum(_logpdf(rs, sig_s * scale, kind, nu) * mask_s)


OBS_DIMS = 4
_STATIC_RES = (3, 4)
_res_j = jax.jit(_residuals, static_argnums=_STATIC_RES)
_jac_j = jax.jit(jax.jacfwd(_residuals, argnums=2), static_argnums=_STATIC_RES)
_STATIC_LL = (3, 4, 5, 6, 7)
_ll_j = jax.jit(_loglike_x, static_argnums=_STATIC_LL)
_ll_batch_j = jax.jit(
    jax.vmap(_loglike_x, in_axes=(None, None, 0, None, None, None, None, None) + (None,) * 13),
    static_argnums=_STATIC_LL,
)


@dataclass
class SkyModel:
    """A family bound to data and a likelihood; `observer` set when the observer's orbit is
    fitted too (T2/T3, with the Sun stream in the likelihood when present)."""

    family: BayesFamily
    data: SkyData
    likelihood: Likelihood = "gaussian"
    nu: float = 4.0
    observer: ObserverFamily | None = None

    @property
    def sigma_free(self) -> bool:
        return not self.data.sigma_known

    @property
    def observer_free(self) -> bool:
        return self.observer is not None

    @property
    def _obs_fam(self) -> ObserverFamily:
        return self.observer or ObserverFamily(t0=self.family.t0, gm=self.family.gm)

    @property
    def fit_dims(self) -> tuple[Dim, ...]:
        """Dimensions fitted by least squares: family (+ observer)."""
        d = self.family.dims()
        return (*d, *self.observer.dims()) if self.observer is not None else d

    @property
    def dims(self) -> tuple[Dim, ...]:
        """Sampled dimensions: fitted ones (+ log_sigma when σ is unknown)."""
        d = self.fit_dims
        return (*d, LOG_SIGMA_DIM) if self.sigma_free else d

    @property
    def n_fit(self) -> int:
        return len(self.fit_dims)

    def _res_args(self) -> tuple[jax.Array, ...]:
        a = self.data.args
        return (*a[:5], *a[7:11])

    def residuals(self, xf: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Tangent-plane residuals in radians: body (N, 2) and Sun stream (Ns, 2)."""
        rb, rs = _res_j(
            self.family,
            self._obs_fam,
            jnp.asarray(xf[: self.n_fit]),
            self.family.n_params,
            self.observer_free,
            *self._res_args(),
        )
        return np.asarray(rb)[: self.data.n], np.asarray(rs)[: self.data.n_sun]

    def jacobian(self, xf: FloatArray) -> tuple[FloatArray, FloatArray]:
        jb, js = _jac_j(
            self.family,
            self._obs_fam,
            jnp.asarray(xf[: self.n_fit]),
            self.family.n_params,
            self.observer_free,
            *self._res_args(),
        )
        return np.asarray(jb)[: self.data.n], np.asarray(js)[: self.data.n_sun]

    def _static(self) -> tuple[int, bool, bool, str, float]:
        return (self.family.n_params, self.observer_free, self.sigma_free, self.likelihood, self.nu)

    def loglike(self, x: FloatArray) -> float:
        xa = np.asarray(x, dtype=np.float64)
        return float(_ll_j(self.family, self._obs_fam, xa, *self._static(), *self.data.args))

    def loglike_batch(self, xs: FloatArray) -> FloatArray:
        xa = np.asarray(xs, dtype=np.float64)
        m = len(xa)
        out = _ll_batch_j(
            self.family, self._obs_fam, pad_rows(xa, bucket(m, 4)), *self._static(), *self.data.args
        )
        return np.asarray(out)[:m]

    def loglike_jax(self, x: jax.Array) -> jax.Array:
        """Differentiable log-likelihood for HMC."""
        return _loglike_x(self.family, self._obs_fam, x, *self._static(), *self.data.args)

    # ---- prior (uniform box with periodic wrap) on the full vector
    def lo_hi(self, fit_only: bool = False) -> tuple[FloatArray, FloatArray]:
        d = self.fit_dims if fit_only else self.dims
        return np.array([x.lo for x in d]), np.array([x.hi for x in d])

    def periodic(self, fit_only: bool = False) -> NDArray[np.bool_]:
        return np.array([x.periodic for x in (self.fit_dims if fit_only else self.dims)])

    def log_prior_density(self) -> float:
        return -float(sum(np.log(x.width) for x in self.dims))

    def wrap(self, x: FloatArray) -> FloatArray:
        n = np.shape(x)[-1]
        lo, hi = self.lo_hi(fit_only=n == self.n_fit)
        per = self.periodic(fit_only=n == self.n_fit)
        out = np.array(x, dtype=np.float64, copy=True)
        out[..., per] = lo[per] + np.mod(out[..., per] - lo[per], (hi - lo)[per])
        return out

    def log_prob_batch(self, xs: FloatArray) -> FloatArray:
        """Unnormalized log posterior for a batch (−inf outside the prior support)."""
        lo, hi = self.lo_hi()
        per = self.periodic()
        inside = np.all(((xs >= lo) & (xs <= hi)) | per, axis=-1)
        out = np.full(len(xs), -np.inf)
        if inside.any():
            ll = self.loglike_batch(self.wrap(xs[inside]))
            out[inside] = np.where(np.isfinite(ll), ll, -np.inf)
        return out


# ----------------------------------------------------------------------------- MAP & Laplace


@dataclass
class MapResult:
    x: FloatArray
    """Full parameter vector at the MAP (φ, observer φ when free, log σ when free)."""
    loglike: float
    cov: FloatArray
    """Laplace covariance of the full vector."""
    rms_rad: float


def fit_map(model: SkyModel, starts: list[FloatArray], n_refine: int = 4) -> MapResult:
    """Least-squares MAP over the fitted vector from several starts (σ profiled), then the
    Laplace covariance. Starts may omit the observer part only when it is not free.

    `rms_rad` is the per-axis residual RMS: radians when σ is free, units of σ otherwise.
    With a uniform box prior the MAP is the maximum-likelihood point inside the box.
    """
    lo, hi = model.lo_hi(fit_only=True)
    per = model.periodic(fit_only=True)
    blo = np.where(per, -np.inf, lo)
    bhi = np.where(per, np.inf, hi)
    if not starts:
        raise ValueError(f"no starting points for family {model.family.name}")
    xs = np.array([model.wrap(np.asarray(s, dtype=np.float64)) for s in starts])
    xs = np.where(per, xs, np.clip(xs, lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo)))
    d = model.data
    sig_b = d.sigma if d.sigma is not None else np.ones(d.n)
    sig_s = d.sun_sigma if d.sun_sigma is not None else np.ones(d.n_sun)

    def res(xf: FloatArray) -> FloatArray:
        rb, rs = model.residuals(xf)
        return np.concatenate([(rb / sig_b[:, None]).ravel(), (rs / sig_s[:, None]).ravel()])

    def jac(xf: FloatArray) -> FloatArray:
        jb, js = model.jacobian(xf)
        k = len(xf)
        return np.concatenate(
            [(jb / sig_b[:, None, None]).reshape(-1, k), (js / sig_s[:, None, None]).reshape(-1, k)]
        )

    def cost(x: FloatArray) -> float:
        r = res(x)
        return float(np.sum(r**2)) if np.all(np.isfinite(r)) else np.inf

    costs = np.array([cost(x) for x in xs])
    best = None
    for j in np.argsort(costs)[:n_refine]:
        try:
            r = least_squares(
                res,
                xs[j],
                jac=jac,  # pyright: ignore[reportArgumentType]
                bounds=(blo, bhi),
                method="trf",
                x_scale="jac",
                max_nfev=200,
            )
        except (ValueError, FloatingPointError):
            continue
        if np.isfinite(r.cost) and (best is None or r.cost < best.cost):
            best = r
    if best is None:
        raise RuntimeError(f"MAP search failed for family {model.family.name}")
    xf = model.wrap(best.x)
    r = res(xf)
    n_res = r.size // 2
    rms = float(np.sqrt(np.mean(r**2)))  # per axis, in units of σ (or radians)
    jm = jac(xf)
    if model.sigma_free:
        s2 = max(rms**2, 1e-30)
        cov_f = np.linalg.pinv(jm.T @ jm / s2, rcond=1e-12)
        x = np.append(xf, np.clip(0.5 * np.log(s2), LOG_SIGMA_DIM.lo, LOG_SIGMA_DIM.hi))
        cov = np.zeros((len(x), len(x)))
        cov[:-1, :-1] = cov_f
        cov[-1, -1] = 1.0 / (4.0 * n_res)  # Var(log σ̂) ≈ 1/(2·dof), dof = 2N
    else:
        x = xf
        cov = np.linalg.pinv(jm.T @ jm, rcond=1e-12)
    cov = 0.5 * (cov + cov.T)
    return MapResult(x, model.loglike(x), cov, rms)


def fit_observer_from_sun(sun: ObservationSet, gm: float, t0: float) -> FloatArray:
    """Observer φ (log a, e, ϖ, λ0) from the Sun-direction stream alone (T2 start).

    The Sun's geocentric direction is minus the observer's heliocentric direction; with GM
    known, the angular motion fixes the period, hence a, and the equation of centre fixes e.
    """
    fam = ObserverFamily(t0=t0, gm=gm)
    u = sun.unit_vectors()
    east, north = tangent_basis(u)
    t = sun.t
    k0 = int(np.argmin(np.abs(t - t0)))
    lam_obs = float(np.arctan2(-u[k0, 1], -u[k0, 0]))  # observer longitude ≈ Sun's + π
    lam0 = lam_obs + 2 * np.pi / 365.25 * (t0 - t[k0])

    tj, uj, ej, nj = (jnp.asarray(a) for a in (t, u, east, north))

    def res_j(phi: jax.Array) -> jax.Array:
        r = fam.position(phi, tj)
        return _tangent(-r / jnp.linalg.norm(r, axis=1, keepdims=True), uj, ej, nj).ravel()

    res_jit, jac_jit = jax.jit(res_j), jax.jit(jax.jacfwd(res_j))

    def res(phi: FloatArray) -> FloatArray:
        return np.asarray(res_jit(jnp.asarray(phi)))

    def jac(phi: FloatArray) -> FloatArray:
        return np.asarray(jac_jit(jnp.asarray(phi)))

    best = None
    for varpi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        x0 = np.array([0.0, 0.02, varpi, lam0])
        r = least_squares(
            res,
            x0,
            jac=jac,  # pyright: ignore[reportArgumentType]
            bounds=([np.log(0.3), 0.0, -np.inf, -np.inf], [np.log(3.0), 0.3, np.inf, np.inf]),
        )
        if best is None or r.cost < best.cost:
            best = r
    assert best is not None
    return fam.wrap(best.x)


# ----------------------------------------------------------------------------- starts


def conic_starts(
    obs: ObservationSet, r_obs: FloatArray, gm: float, t0: float, seed: int
) -> list[ConicElements]:
    """Candidate conics: Gauss IOD + circular grid, polished by elliptic least squares.

    The three cheapest polished candidates per family come first. Then up to `N_EXTRA`
    further orbits that differ from them, drawn from a wider Gauss IOD search: on a
    non-Keplerian truth the cheapest polishes often converge to one whole-arc alias, and
    an integrated family needs a start in its own basin (e.g. Yukawa: 1 seed in 10
    recovered the law without them, 9 in 10 with).
    """
    from jan.fitting import ConicProblem, circular_grid_starts, period_grid_starts, sky_periods
    from jan.models.conic import ConicOrbit, EllipticOrbit, OrbitFamily, canonical_elements
    from jan.strategies.classical.gauss_iod import gauss_candidates

    raw = gauss_candidates(obs, r_obs, gm, 12, (15.0, 30.0, 60.0), np.random.default_rng(seed))
    wide = gauss_candidates(
        obs, r_obs, gm, N_WIDE_TRIPLETS, (15.0, 30.0, 60.0), np.random.default_rng(seed + 1)
    )
    periods = sky_periods(obs)
    out: list[ConicElements] = []
    extra: list[ConicElements] = []
    for fam in (EllipticOrbit(t0=t0, gm=gm), ConicOrbit(t0=t0, gm=gm)):
        prob = ConicProblem(fam, obs.t, sky=True, u_obs=obs.unit_vectors(), r_obs=r_obs)

        def admitted(cs: list[ConicElements], fam: OrbitFamily = fam) -> list[ConicElements]:
            cs = [c for c in cs if fam.admits(c)]
            return cs if fam.name == "ellipse" else [c for c in cs if c.e > 1.0]

        def polish(cands: list[ConicElements], prob: Any = prob, fam: OrbitFamily = fam):
            thetas = np.array([fam.theta_from_elements(c) for c in cands])
            for j in np.argsort(prob.costs(thetas)):
                try:
                    r = least_squares(
                        prob.residuals,
                        thetas[j],
                        jac=prob.jacobian,  # pyright: ignore[reportArgumentType]
                        method="trf",
                        x_scale="jac",
                        max_nfev=200,
                    )
                except (ValueError, FloatingPointError):
                    continue
                el = canonical_elements(np.asarray(fam.elements(jnp.asarray(r.x))))
                if np.all(np.isfinite(el.as_array())):
                    yield el

        cands = admitted(raw)
        if fam.name == "ellipse":
            cands += circular_grid_starts(prob, gm, t0)
            cands += period_grid_starts(prob, gm, t0, periods)
        if not cands:
            continue
        main = list(itertools.islice(polish(cands[:]), 3))
        out += main
        more: list[ConicElements] = []
        for k, el in enumerate(polish(cands + admitted(wide))):
            if len(more) >= N_EXTRA or k >= N_POLISH:
                break
            if not any(_same_orbit(el, p) for p in main + more):
                more.append(el)
        extra += more
    return out + extra


N_WIDE_TRIPLETS = 96  # Gauss IOD triplets per spacing in the wider search
N_POLISH = 16  # candidates polished in the wider search
N_EXTRA = 5  # distinct extra orbits kept per family


def _same_orbit(a: ConicElements, b: ConicElements) -> bool:
    """Polished starts that converged to one solution (sky aliases differ in q and e)."""
    return abs(a.q - b.q) <= 0.02 * max(a.q, b.q) and abs(a.e - b.e) <= 0.02


def epicycle_starts(obs: ObservationSet, t0: float) -> list[FloatArray]:
    """Epicycle-family starts from an order-1, two-frequency epicycle fit of the sky track."""
    from jan.base import make_config
    from jan.strategies.ptolemy.fourier import PtolemyFourier, design, epicycle_fit

    strat = PtolemyFourier()
    cfg = make_config(strat, {"n_frequencies": 2, "n_harmonics": 1})
    h = strat._fit_body(obs, "x", cfg)  # pyright: ignore[reportPrivateUsage]
    starts: list[FloatArray] = []
    f = np.asarray(h.freqs)
    x = design(obs.t - t0, f)
    coef, _ = epicycle_fit(x, obs.unit_vectors(), 1e-8)
    terms = []
    for j in range(len(f)):
        a3, b3 = coef[1 + j], coef[1 + len(f) + j]
        terms.append((f[j], a3, b3))
    # Plane from the largest term; try both normals and keep the prograde-deferent one.
    _, a3, b3 = max(terms, key=lambda q: np.linalg.norm(q[1]) + np.linalg.norm(q[2]))
    nrm0 = np.cross(a3, b3)
    nrm0 = nrm0 / np.linalg.norm(nrm0) if np.linalg.norm(nrm0) > 0 else np.array([0, 0, 1.0])
    for nrm in (nrm0, -nrm0):
        Omega = float(np.arctan2(nrm[0], -nrm[1]))
        node = np.array([np.cos(Omega), np.sin(Omega), 0.0])
        yax = np.cross(nrm, node)
        circles = []
        for fj, a3, b3 in terms:
            ac = complex(a3 @ node, a3 @ yax)
            bc = complex(b3 @ node, b3 @ yax)
            A, B = (ac - 1j * bc) / 2, (ac + 1j * bc) / 2
            if abs(A) >= abs(B):
                circles.append((abs(A), fj, float(np.angle(A))))
            else:
                circles.append((abs(B), -fj, float(np.angle(B))))
        circles.sort(key=lambda c: -c[0])
        (r_d, f_d, ph_d), (r_e, f_e, ph_e) = circles[0], circles[1]
        if f_d <= 0:
            continue
        ratio = min(r_e / r_d, 1.49)
        for e_c, e_q in ((0.0, 0.0), (0.05, 0.1)):
            starts.append(
                np.array([f_d, f_e, ratio, e_c, e_q, 0.0, ph_d, ph_e, float(nrm[2]), Omega])
            )
    return starts
