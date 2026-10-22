# Jan and Tycho — Project Specification

> **Repository name:** `jan-and-tycho`
> **Audience:** this document is written to be handed to a code-generation model (or a human developer). It is prescriptive where decisions have been made and explicitly marks open choices. Build in the phase order given in §10; do not start a later phase before the previous phase's acceptance criteria pass.

---

## 1. Concept

In ~1600, Tycho Brahe produced the most precise naked-eye astronomical observations of his time, and Johannes Kepler used them to discover the laws of planetary motion. This project recreates that partnership as a testbed for **automated model and law discovery from data**:

- **Tycho** is a *forward simulator*. It holds a secret ground-truth model of how a body moves and emits noisy, incomplete observations of that body *as seen from a moving Earth* (a direction on the sky plus a time).
- **Jan** is a family of *inverse-problem solvers* ("strategies"). Each strategy consumes Tycho's observations and tries to recover the truth: a predictive model, a 3D trajectory, physical parameters, and ideally the underlying law.
- **Rudolf** (named after Emperor Rudolf II, patron of both men) is the *referee*. It is the only component allowed to open the sealed ground truth. It scores Jan's hypotheses on several levels of "recovery".
- **The bench** orchestrates sweeps: many scenarios × noise profiles × strategies × random seeds, followed by reports comparing strategies.

The goal is educational and exploratory: to compare classical orbit determination, Bayesian inference, equation discovery, and physics-informed ML across problems that range from easy (circular orbits) to historically hard (Kepler's Mars problem with an unknown Earth orbit) to non-physical (rectangular orbits).

### 1.1 Guiding principles

1. **Information boundary is sacred.** Jan never sees Tycho's code, config, or truth. Enforced by package structure and an automated import-lint check (§3.2).
2. **Reproducibility.** Every scenario is a serializable config plus an integer seed. Same config + seed ⇒ bit-identical outputs.
3. **Extrapolation is the real test.** A law should predict the *future*, not just interpolate the past. Held-out data is always a later time window.
4. **Honest uncertainty and honest failure.** Strategies are rewarded for calibrated uncertainty and for correctly reporting "no model in my class fits".
5. **Composability.** Truth models, observer models, schedules and noise models are independent, swappable components.
6. **See it, don't just score it.** Every phase ships plots and animations (produced by Rudolf, §7.1) that make it possible to visually judge how each Jan strategy compares to the truth and to the others, across any Tycho or Jan parameter.

### 1.2 Non-goals (for now)

- High-fidelity real-world ephemerides (no JPL DE files, no relativistic time scales, no nutation/precession).
- Ingesting real observational catalogs (a possible Phase 5+ extension).
- A GUI. CLI plus generated reports and notebooks only.

---

## 2. Technology stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Python ≥ 3.11 | |
| Environment / workspace | `uv` (workspace with multiple packages) | |
| Lint / format | `ruff` | |
| Type checking | `pyright` (strict for `jt_core`, basic elsewhere) | |
| Tests | `pytest`, `hypothesis` (property tests) | |
| Import boundary | `import-linter` | Run in CI and as a pytest test |
| Config schemas | `pydantic` v2 + YAML (`pyyaml`) | Discriminated unions on `type` field |
| CLI | `typer` | Single entry point `jt` |
| Numerics | `numpy`, `scipy` | |
| Differentiable numerics | `jax` (**x64 enabled globally**), `diffrax` (ODEs), `optax` (optimizers), `equinox` (neural nets) | |
| Symbolic math | `sympy` | Law representation and structural scoring |
| Tabular I/O | `pyarrow` (Parquet) + `pandas` for convenience | JSON for metadata |
| Plotting / reports | `matplotlib`, `arviz`, `corner`, `seaborn` (optional, for heatmaps/facets) | Static HTML report generation via Jinja2 |
| Animations | `matplotlib.animation` → MP4 via `ffmpeg` (preferred), GIF via `Pillow` (fallback when ffmpeg is absent) | Detect ffmpeg at runtime; never fail a run because ffmpeg is missing |
| Bayesian | `emcee`, `numpyro`, `dynesty` | Optional extra `jan[bayes]` |
| Equation discovery | `pysindy`, `pysr` | Optional extra `jan[discovery]`; PySR requires a Julia runtime — document install, skip tests if unavailable |
| Gaussian processes | `tinygp` (JAX) or `scikit-learn` | Pick `tinygp` if JAX stack is working |
| Simulation-based inference | `sbi` (PyTorch) | Optional extra `jan[sbi]`, Phase 5 |
| LLM hypothesis generation | Anthropic Python SDK | Optional extra `jan[llm]`, Phase 5 |

Heavy dependencies must be **optional extras**. The core loop (Tycho Keplerian → Jan least squares → Rudolf) must install and run with only `numpy`, `scipy`, `jax`, `pydantic`, `pyyaml`, `pyarrow`, `pandas`, `sympy`, `typer`, `matplotlib`.

---

## 3. Repository layout and package boundaries

### 3.1 Layout

```
jan-and-tycho/
├── pyproject.toml                # uv workspace root
├── SPEC.md
├── README.md
├── .importlinter                 # boundary contracts
├── packages/
│   ├── jt-core/src/jt_core/      # shared, truth-agnostic utilities
│   │   ├── constants.py          # GM_SUN, AU, DAY, gaussian gravitational constant
│   │   ├── units.py              # arcsec/deg/rad helpers
│   │   ├── frames.py             # ecliptic cartesian <-> (lon, lat), unit vectors, rotations
│   │   ├── conics.py             # universal-variable Kepler propagation, elements <-> state
│   │   ├── angles.py             # great-circle separation, wrapping
│   │   ├── schema.py             # ObservationSet, PublicMeta, Prediction, Hypothesis protocol
│   │   ├── io.py                 # read/write Parquet + JSON bundles
│   │   └── rng.py                # seed -> numpy Generator / JAX PRNGKey derivation
│   ├── tycho/src/tycho/
│   │   ├── scenario.py           # pydantic Scenario config
│   │   ├── truth/                # keplerian.py, central_force.py, nbody.py, kinematic.py, epicycle.py
│   │   ├── observer.py           # Earth models
│   │   ├── operator.py           # geometry -> observed quantities
│   │   ├── schedule.py           # when observations happen; visibility
│   │   ├── noise.py              # composable noise models
│   │   └── generate.py           # Scenario + seed -> public bundle + sealed bundle
│   ├── jan/src/jan/
│   │   ├── base.py               # Strategy protocol, registry, Hypothesis base classes
│   │   ├── models/               # hypothesis families shared across strategies (conic, circle, equant, central-force, ...)
│   │   ├── reconstruct/          # 3D trajectory reconstruction from lines of sight
│   │   └── strategies/           # ptolemy/, gp/, classical/, bayes/, kepler/, discovery/, piml/, sbi/, llm/
│   ├── rudolf/src/rudolf/
│   │   ├── score.py              # entry point
│   │   ├── metrics/              # predictive.py, trajectory.py, parameters.py, structural.py, calibration.py
│   │   ├── viz/                  # all plotting and animation (§7.1)
│   │   │   ├── style.py          # shared palette, strategy→color map, figure conventions
│   │   │   ├── run_plots.py      # per-run static diagnostics
│   │   │   ├── compare_plots.py  # cross-strategy / cross-parameter comparison plots
│   │   │   ├── animations.py     # per-run and cross-run animations
│   │   │   └── manifest.py       # index of produced figures for reports
│   │   └── report.py
│   └── jt-bench/src/jt_bench/
│       ├── cli.py                # `jt` command
│       ├── harness.py            # runs one (scenario, seed, strategy) cell end to end
│       ├── sweep.py              # grid sweeps, parallel execution, result aggregation
│       └── plots.py
├── scenarios/                    # library of scenario YAMLs (§8)
├── bench/                        # sweep configs
├── notebooks/                    # exploratory notebooks, one per phase
├── runs/                         # generated outputs (gitignored)
└── tests/                        # cross-package tests (boundary, determinism, e2e)
```

### 3.2 Dependency rules (enforced by `import-linter`)

- `jt_core` imports nothing from the other packages.
- `tycho` → may import `jt_core` only.
- `jan` → may import `jt_core` only. **Must never import `tycho` or `rudolf`.**
- `rudolf` → may import `jt_core` only.
- `jt_bench` → may import everything (it is the harness).

`jt_core` must contain only **generic, truth-agnostic** physics and math (conic propagation, frames, schema). It must not contain any scenario-specific truth models.

### 3.3 Runtime information boundary

The harness, not Jan, decides which files Jan can read. Jan strategies receive an in-memory `ObservationSet` and `PublicMeta` and must not take filesystem paths to the run directory. Strategies that need scratch space receive a dedicated working directory.

---

## 4. Conventions

- **Frame:** heliocentric, inertial, ecliptic-like Cartesian frame. +x toward a fixed reference direction ("vernal equinox"), +z toward the ecliptic north pole. The Sun is at the origin (for N-body scenarios, use the barycenter and document it).
- **Units:** distance in AU, time in days, angles in **radians internally** (configs may use degrees or arcseconds with explicit suffixed field names, e.g. `sigma_arcsec`). GM of the Sun = k², k = 0.01720209895 AU^(3/2)/day.
- **Time:** a single uniform time scale `t` in days since the scenario epoch (`t = 0`). No UTC/TT distinctions. An optional cosmetic `epoch_calendar` string may be carried in metadata.
- **Observed quantities:** geocentric ecliptic longitude λ ∈ [0, 2π) and latitude β ∈ [−π/2, π/2]. Likelihoods and errors must be computed on **unit vectors** or via great-circle distance, never on raw λ differences (wraparound bugs).
- **Orbital elements:** use the conic parametrization `(q, e, i, Ω, ω, t_p)` (perihelion distance, eccentricity, inclination, ascending node, argument of perihelion, time of perihelion) so ellipses, parabolas and hyperbolas share one code path. Provide conversions to `(a, e, i, Ω, ω, M0)` for e < 1.
- **Precision:** float64 everywhere. Call `jax.config.update("jax_enable_x64", True)` at import of `jt_core`.
- **Randomness:** never use global RNG state. Derive all randomness from the scenario seed via `jt_core.rng` (named sub-streams: `"schedule"`, `"noise"`, `"truth"`, etc., so changing one component does not reshuffle others).

---

## 5. Component: Tycho (the observer)

### 5.1 Scenario configuration

A `Scenario` is a pydantic model loaded from YAML. All sub-components use discriminated unions keyed by `type`.

```yaml
name: mars_like
description: Mars-like ellipse observed with Tycho-level precision.
duration_days: 7300          # total simulated span
holdout_fraction: 0.25       # last 25% of the span is sealed for scoring
knowledge_tier: T1           # see §5.6

observer:
  type: keplerian
  elements: {q: 0.9833, e: 0.0167, i_deg: 0.0, Omega_deg: 0.0, omega_deg: 102.9, t_p: 2.0}

bodies:
  - id: target
    truth:
      type: keplerian
      elements: {q: 1.381, e: 0.0934, i_deg: 1.85, Omega_deg: 49.6, omega_deg: 286.5, t_p: 120.0}
    photometry: {H: -1.5}     # optional; used by magnitude-based visibility

schedule:
  type: regular
  cadence_days: 10
  jitter_days: 2
  visibility:
    min_solar_elongation_deg: 15
    max_magnitude: 6.0       # naked-eye limit; requires photometry
    dropout_prob: 0.2        # weather

noise:
  - {type: gaussian, sigma_arcsec: 60}
  - {type: bias, lon_arcsec: 20, lat_arcsec: 0}
  - {type: quantize, step_arcsec: 30}

report_uncertainties: false  # if true, Tycho publishes per-observation sigma columns
```

### 5.2 Truth models (`tycho.truth`)

Every truth model implements:

```python
class TruthModel(Protocol):
    kind: Literal["dynamical", "kinematic"]

    def position(self, t: Array) -> Array: ...  # (N, 3) heliocentric AU
    def velocity(
        self, t: Array
    ) -> Array: ...  # (N, 3) AU/day (finite difference allowed for kinematic)
    def describe(
        self,
    ) -> TruthDescription: ...  # serializable: params, elements, symbolic law if any
```

Required truth model types:

| `type` | Phase | Description |
|---|---|---|
| `keplerian` | 1 | Two-body conic (elliptic/parabolic/hyperbolic) via universal variables. Law: a = −GM r/\|r\|³. |
| `circular` | 1 | Convenience wrapper: e = 0, i configurable. |
| `central_force` | 4 | Integrate r̈ = f(\|r\|) r̂ with `diffrax` (symplectic or high-order adaptive, tight tolerances). Built-in laws: `power_law` (exponent n, GM), `newton_plus_inverse_cube` (GR-like precession, coefficient ε), `hooke` (linear, ellipse centered on Sun), `yukawa` (range λ), `custom` (sympy expression string in `r`). |
| `nbody` | 4 | Sun + N bodies with mutual gravity; some bodies may be marked `hidden: true` (never observed) for the hidden-perturber scenario. |
| `kinematic_polygon` | 4 | Closed polygonal path (e.g. rectangle) in a plane with orientation (i, Ω, ω-analog), center offset, period, and `speed_profile ∈ {constant_speed, constant_areal_velocity}`. |
| `kinematic_parametric` | 4 | Lissajous, figure-eight, or a custom parametric curve `r(φ)` with φ(t) = 2πt/P. |
| `epicycle` | 2 | Deferent + one or more epicycles (Ptolemaic), optionally with equant. Useful as a "pre-Keplerian truth" and for testing model selection. |

The `describe()` output must include the symbolic acceleration law (sympy string with symbols `x, y, z, vx, vy, vz, t`) for dynamical models, or `null` for kinematic ones.

### 5.3 Observer models (`tycho.observer`)

- `keplerian` (default: Earth-like elements). Phase 1.
- `fixed` (observer at a fixed heliocentric point; debugging). Phase 1.
- `heliocentric` (observer at the Sun; T0 sanity tier). Phase 1.
- Any truth model may be reused as the observer trajectory (e.g. an Earth on a non-Keplerian orbit). Phase 4.

### 5.4 Observation operator (`tycho.operator`)

Computes geocentric direction `u = (r_body − r_obs) / |r_body − r_obs|` and converts it to (λ, β). Also computes the geocentric distance Δ and the heliocentric distance r for visibility and magnitude, but **never publishes them**. Optional realism effects (Phase 5, off by default): light-time correction, stellar aberration.

For tier T0, the operator publishes heliocentric Cartesian positions instead of angles (debug tier for testing discovery methods independently of reconstruction).

For tier T2, Tycho additionally publishes a noisy Sun-direction stream (body id `sun`), i.e. the geocentric direction of the Sun at each observation night.

### 5.5 Schedules and visibility (`tycho.schedule`)

- `regular` (cadence + jitter), `poisson` (random times at a rate), `explicit` (list of times). Phase 1.
- `seasons`: observing windows (e.g. object only observable part of the year). Phase 3.
- Visibility filters, applied after time generation:
  - `min_solar_elongation_deg` (object too close to the Sun). Phase 3.
  - `max_magnitude` using the H–G-free simple magnitude model m = H + 5 log₁₀(r Δ). For comets, an optional activity term m = H + 5 log₁₀Δ + 2.5 κ log₁₀ r. Phase 3.
  - `dropout_prob` (weather). Phase 1.

### 5.6 Knowledge tiers

The tier controls what goes into `PublicMeta` and which streams are published:

| Tier | Jan receives | Notes |
|---|---|---|
| `T0` | Heliocentric 3D positions of the body + times | Debug; isolates law discovery from reconstruction. |
| `T1` | Sky directions + times, **and the observer's orbital elements** (Earth known) | Classical orbit determination. |
| `T2` | Sky directions of all observed bodies + noisy Sun-direction stream; Earth's orbit **unknown** | Kepler's actual problem. Scenarios may include several observed bodies. |
| `T3` | Sky directions only; no Sun stream; Earth unknown | Stretch goal. Jan may assume Earth is Keplerian. Document identifiability limits. |

### 5.7 Noise models (`tycho.noise`)

Composable list, applied in order. Each acts on unit vectors or (λ, β) and/or on times.

| `type` | Phase | Parameters |
|---|---|---|
| `gaussian` | 1 | `sigma_arcsec` (isotropic on the sphere: perturb the unit vector in the tangent plane) |
| `bias` | 3 | constant `lon_arcsec`, `lat_arcsec` |
| `drift` | 3 | linear bias drift per year |
| `ar1` | 3 | temporally correlated error (`sigma_arcsec`, `rho` per day) |
| `heteroscedastic` | 3 | sigma depends on altitude-proxy or magnitude |
| `outliers` | 3 | `fraction`, `scale_arcsec` (e.g. 10′–5° gross errors) |
| `quantize` | 1 | `step_arcsec` (rounding to instrument graduations) |
| `clock` | 3 | Gaussian time error `sigma_days` and/or constant offset |

The realized noise (per-observation perturbations, outlier flags) is recorded in the sealed bundle.

### 5.8 Generation output (run directory)

`tycho.generate(scenario, seed) -> (PublicBundle, SealedBundle)`, written as:

```
runs/<scenario>/<seed>/
├── public/
│   ├── observations.parquet   # columns: obs_id, body_id, t, lon, lat [, sigma_lon, sigma_lat] (T0: x, y, z)
│   └── meta.json              # tier, units, epoch, t_range, observer elements (T1 only), body ids, report_uncertainties
└── sealed/
    ├── scenario.yaml          # full config
    ├── truth.json             # TruthDescription per body incl. symbolic law and elements
    ├── trajectory.parquet     # dense truth positions/velocities on a uniform grid over the full span
    ├── holdout.parquet        # noisy AND noise-free observations in the holdout window
    └── noise_realization.parquet
```

The holdout window is the final `holdout_fraction` of `duration_days`. The holdout observations follow the same schedule, visibility and noise rules as the public ones.

---

## 6. Component: Jan (the theorist)

### 6.1 Strategy interface

```python
class Strategy(Protocol):
    name: str  # registry key, e.g. "classical.lsq"
    supported_tiers: set[str]

    def fit(
        self,
        obs: ObservationSet,
        meta: PublicMeta,
        config: StrategyConfig,
        rng_seed: int,
        workdir: Path,
    ) -> Hypothesis: ...
```

Strategies are registered via a decorator (`@register("classical.lsq")`) and configured by pydantic models (YAML-loadable). A strategy must not read any file outside `workdir`.

### 6.2 Hypothesis interface

```python
class Hypothesis(Protocol):
    kind: Literal["dynamical", "kinematic", "nonparametric", "rejected"]

    def predict_sky(self, t: Array, body_id: str) -> SkyPrediction: ...
    # Optional capabilities; return None if unsupported:
    def heliocentric_position(self, t: Array, body_id: str) -> PositionPrediction | None: ...
    def elements(self, body_id: str) -> ElementsEstimate | None: ...  # point + optional samples
    def law(self) -> LawEstimate | None: ...  # sympy string + fitted constants
    def model_comparison(
        self,
    ) -> list[ModelScore] | None: ...  # (model_name, log_evidence or IC, posterior_prob)
    def trace(self) -> list[TraceFrame] | None: ...  # fit history for animations (§7.1.6)
    def notes(self) -> str: ...  # free-text rationale
    def to_dict(self) -> dict: ...  # JSON-serializable archive
```

- `SkyPrediction`: mean unit vectors `(N, 3)`, plus optionally `samples (S, N, 3)` or a per-point 2×2 tangent-plane covariance.
- `PositionPrediction`: mean `(N, 3)` AU, plus optional samples.
- `kind = "rejected"` means "no model in my class explains these data." The strategy must still return its best nonparametric predictive model so that predictive scoring works.

**Scoring flow:** the harness obtains a *query sheet* from the sealed side containing **only times** (holdout times and dense-grid times, no values). It calls the hypothesis's prediction methods on those times and serializes the results to `runs/.../jan/<strategy>/predictions.parquet` plus `hypothesis.json`. Rudolf scores these files. Rudolf never executes Jan code.

### 6.3 Shared hypothesis families (`jan.models`)

Reusable parametric families, each providing a forward model `params, t -> sky directions` (JAX, differentiable), priors, and parameter transforms:

- `CircularOrbit`, `ConicOrbit` (q, e, i, Ω, ω, t_p), `EquantCircle` (Kepler's "vicarious hypothesis": eccentric circle with equant point), `Epicycles(K)`.
- `CentralForceOrbit(law_family)` integrated with `diffrax` for power-law / Hooke / Yukawa / Newton+ε/r³ families.
- `PerturbedConic`: conic + one hidden perturber (mass, orbit) for the Neptune-style scenario.
- `PeriodicKinematic`: Fourier series in 3D with a shared period (non-physical but periodic paths).

Under tier T2/T3, families are extended with the observer's unknown Keplerian elements as additional parameters.

### 6.4 Trajectory reconstruction (`jan.reconstruct`)

Several discovery strategies need 3D positions first. Provide:

- `from_dynamical_fit`: positions from any fitted dynamical hypothesis.
- `periodic_triangulation`: **Kepler's trick**. Given a period estimate P (from a periodogram of the sky track or from the synodic period), group observations at times t, t+P, t+2P, … at which the body is at the same heliocentric point. Intersect the lines of sight from different observer positions (least squares) to get that point. Works for any periodic trajectory, physical or not. Under T2, use the same idea in reverse to triangulate the observer's orbit.
- Outputs positions with uncertainty estimates (propagated from line-of-sight geometry).

### 6.5 Strategy catalogue

| Registry name | Phase | Method | Output kind |
|---|---|---|---|
| `ptolemy.fourier` | 1 | Fit λ(t), β(t) (via unit vectors) with a truncated Fourier series / epicycles, period(s) from a Lomb–Scargle periodogram. The "curve-fitting" baseline: great interpolation, poor extrapolation. | nonparametric |
| `gp.sky` | 2 | Gaussian process on unit-vector components with periodic + RBF kernels. | nonparametric |
| `classical.gauss_iod` | 1 | Gauss's method for an initial orbit from 3 observations (T1). Used as initialization by other strategies. | dynamical |
| `classical.lsq` | 1 | Batch nonlinear least squares (differential correction) on `ConicOrbit`, initialized from Gauss IOD or a multi-start grid. Robust loss option (Huber/Cauchy). Covariance from the Jacobian. | dynamical |
| `bayes.mcmc` | 2 | `emcee` posterior over a chosen family. Gaussian or Student-t likelihood on tangent-plane residuals, optional noise-scale parameter. | dynamical |
| `bayes.hmc` | 2 | `numpyro` NUTS with the differentiable JAX forward model. | dynamical |
| `bayes.nested` | 2 | `dynesty` nested sampling over **several families** (circle, equant-circle, conic-elliptic, conic-hyperbolic, central-force variants), reporting log-evidence and posterior model probabilities. | dynamical |
| `kepler.triangulate` | 3 | Periodic triangulation (§6.4), then fit a conic to the reconstructed 3D points; reports kinematic description if the conic fit fails. | dynamical or kinematic |
| `discovery.sindy` | 4 | Two stage: reconstruct 3D positions, then sparse regression of acceleration onto a library (polynomials in x, y, z and terms r^k x̂ for k ∈ {−4…2}). Use weak-form SINDy to avoid noisy numerical derivatives. | dynamical |
| `discovery.pysr` | 4 | Symbolic regression of the radial acceleration magnitude vs r (and of the full vector field as a stretch goal). Reports a Pareto front of complexity vs loss; selects by a stated criterion. | dynamical |
| `piml.neural_ode` | 4 | Neural ODE for acceleration a = NN(r, v), fitted end to end through the observation operator. | dynamical (black box) |
| `piml.hnn` | 4 | Hamiltonian neural network (learned H(q, p)); energy conserved by construction. | dynamical (black box) |
| `piml.ude` | 4 | Universal differential equation: a = −GM r/\|r\|³ + NN(r). Then symbolic distillation of the NN term with PySR. Primary tool for detecting small deviations and hidden perturbers. | dynamical |
| `sbi.npe` | 5 | Neural posterior estimation, trained on simulations **from Jan's own hypothesis families** (never from Tycho). | dynamical |
| `llm.hypothesize` | 5 | An LLM proposes candidate functional forms or model families from residual plots and summaries; proposals are then fitted and compared with `bayes.nested`. | any |

Each strategy ships with a default config and a short docstring explaining its assumptions and failure modes.

---

## 7. Component: Rudolf (the referee)

Rudolf reads the sealed bundle plus Jan's `predictions.parquet` and `hypothesis.json` and writes `runs/.../rudolf/<strategy>/score.json`. Metrics are grouped by level; any metric that the hypothesis cannot support is `null` (not zero).

| Level | Metric | Phase |
|---|---|---|
| **Predictive (holdout)** | Great-circle residuals vs **noise-free** holdout directions: RMS, median, 95th percentile, max (arcsec). Also residual growth vs time since the end of training (extrapolation curve). | 1 |
| **Predictive (probabilistic)** | Negative log predictive density of noisy holdout observations; energy score when samples are given. | 2 |
| **Trajectory** | RMS and max 3D position error (AU) over the training window and the holdout window, on the dense grid. | 1 |
| **Parameters** | Absolute/relative error per orbital element (angles wrapped). For posteriors: z-scores and whether truth lies within the 68/95% credible intervals. | 1 / 2 |
| **Model selection** | Posterior probability assigned to the true family (when the truth is in the candidate set). | 2 |
| **Structural** | (a) Symbolic: sympy-simplify the difference between the recovered and true law after fitting constants; report exact match / match up to constants / mismatch. (b) Numerical: relative L2 error of the acceleration field on points sampled near the true trajectory (works for black-box laws too). | 4 |
| **Rejection** | For kinematic truths: credit if `kind ∈ {"kinematic", "rejected"}` and predictive metrics are good; penalty flag if a dynamical law is claimed with a large structural error. | 4 |
| **Calibration (across seeds)** | Empirical coverage of credible intervals for parameters and predictions; simulation-based calibration rank histograms. Computed by the bench over many seeds. | 2 |

### 7.1 Visualization and animation

Visual evaluation is a first-class output of the project, not an afterthought. Rudolf owns **all** plotting and animation, because only Rudolf may see the ground truth, and nearly every useful figure compares Jan against the truth. Jan strategies may write their own internal debug plots into `workdir`, but those are not part of the evaluation output.

#### 7.1.1 Conventions

- **Truth is always visible.** Every figure that shows a Jan estimate also shows the ground truth (truth in a fixed neutral color, e.g. black or dark gray; observations as small markers; holdout observations in a distinct marker style).
- **Consistent strategy colors.** `viz/style.py` maintains a stable strategy-name → color map (and line style) used in every figure and animation, so a strategy looks the same everywhere.
- **Training/holdout boundary.** Every time-axis plot draws a vertical line at the end of the training window and shades the holdout region.
- **Uncertainty.** When a hypothesis provides samples or covariance, draw credible bands (68% and 95%) or sample spaghetti, not just the mean.
- **Units on every axis.** Errors in arcsec (log scale by default for error axes), distances in AU, time in days or years.
- **Output formats.** Static figures are saved as PNG (for reports) and SVG (for editing). Animations are saved as MP4 (ffmpeg) or GIF (fallback). Default animation length is ≤ 20 s at 30 fps; frame count is configurable, and a `--quick` flag renders low-resolution, short versions for CI.
- **Manifest.** Every produced figure or animation is recorded in `figures.json` (path, kind, title, one-line caption, the scenario/strategy/seed it depicts). Reports are built from manifests, never by globbing.
- **CLI control.** `jt rudolf score` produces static plots by default; `--animate` adds animations; `--no-plots` disables all visualization (for large sweeps). Sweep configs have a `viz` section (§9.2) that picks which cells get full visualization.

#### 7.1.2 Per-run static plots (`viz/run_plots.py`)

Written to `runs/<scenario>/<seed>/rudolf/<strategy>/figures/`.

| Figure | Content | Phase |
|---|---|---|
| `sky_track` | Observations, holdout observations, Jan's predicted track and true noise-free track on a (λ, β) plot (with wraparound handling), retrograde loops visible. | 1 |
| `residuals_vs_time` | Great-circle error of Jan's prediction vs noise-free truth over the full span (log scale), with the noise level drawn as a horizontal reference line. Shows the extrapolation blow-up directly. | 1 |
| `orbit_topdown` | Heliocentric x–y view: true orbit, recovered orbit (if available), observer's orbit, Sun; plus a small x–z inset. | 1 |
| `orbit_3d` | Static 3D view of true vs recovered trajectory. | 1 |
| `element_errors` | Bar or dot chart of per-element errors, normalized by Jan's reported uncertainty when available. | 1 |
| `observation_coverage` | Timeline of when observations exist (and why gaps exist: elongation, magnitude, weather), plus orbital-phase coverage (fraction of the orbit observed). | 1 (basic), 3 (gap reasons) |
| `corner` | Posterior corner plot with truth marked. | 2 |
| `posterior_predictive` | Sky-track and residual plots with posterior-predictive bands. | 2 |
| `model_probabilities` | Bar chart of posterior model probabilities / evidence per candidate family, true family highlighted. | 2 |
| `noise_diagnostics` | Normalized residual histogram vs the assumed likelihood; Q–Q plot; outliers flagged (Rudolf knows the true outlier flags, so it shows hits and misses). | 3 |
| `triangulation_geometry` | For `kepler.triangulate`: observer positions, lines of sight at period-separated epochs, and the triangulated points vs the true points. | 3 |
| `force_law` | Radial acceleration vs r: true law, recovered law(s), and the region of r actually visited by the orbit shaded. | 4 |
| `acceleration_field` | In-plane quiver/streamline plot of the true vs recovered acceleration field, plus a heatmap of the relative field error. | 4 |
| `symbolic_pareto` | For symbolic regression: complexity vs loss Pareto front, with the selected expression and the true expression marked. | 4 |
| `conserved_quantities` | Energy and angular momentum vs time along the true trajectory, evaluated under the recovered model (reveals non-conservative or wrong laws). | 4 |

#### 7.1.3 Per-run animations (`viz/animations.py`)

| Animation | Content | Phase |
|---|---|---|
| `orbit_animation` | Top-down heliocentric view over time: Sun, observer, true body, Jan's predicted body (with uncertainty cloud when available), and the line of sight from observer to body. Trail lengths configurable; a clock and a training/holdout indicator in the corner. | 1 |
| `sky_animation` | The body's apparent motion across the sky as seen from Earth, true vs predicted, building up the retrograde loops over time; observations pop in as they happen. | 1 |
| `fit_progress` | How Jan's estimate evolves: optimizer iterations (least squares, neural fits) or growing training data (refit with the first k observations, k increasing). Shows predicted orbit converging (or not) on the truth. Requires strategies to optionally emit a trace (§7.1.6). | 2 |
| `posterior_samples` | Spaghetti of orbits drawn from the posterior, animated over time or over MCMC/nested-sampling progress. | 2 |
| `triangulation_animation` | Kepler's trick step by step: lines of sight at t, t+P, t+2P… intersecting to reveal the trajectory point by point. | 3 |
| `law_discovery` | For discovery strategies: the recovered acceleration field or force-law curve morphing across iterations / Pareto front entries toward the final answer, with the truth as a fixed overlay. | 4 |

#### 7.1.4 Cross-run comparison plots (`viz/compare_plots.py`)

These consume the bench's aggregated `results.parquet` (§9.2) and are the main tool for comparing methods and models. They are written to `<results_dir>/figures/`.

| Figure | Content | Phase |
|---|---|---|
| `leaderboard` | Per scenario: strategies ranked by holdout RMS (median over seeds, with interquartile range), with additional columns for trajectory error and runtime. Rendered as a figure and as an HTML table. | 1 |
| `error_vs_tycho_param` | Error metric (y, log scale) vs any swept **Tycho** parameter (x): noise sigma, cadence, number of observations, duration/arc coverage, eccentricity, outlier fraction, bias, etc. One line per strategy, median with IQR band across seeds. | 2 |
| `error_vs_jan_param` | Error metric vs any swept **Jan** hyperparameter (Fourier order, SINDy threshold, network width, number of MCMC walkers, robust-loss scale…). Shows over/underfitting regimes. | 2 |
| `heatmap_2d` | Error metric over a 2D grid of two swept parameters (Tycho × Tycho, or Tycho × Jan), one panel per strategy (small multiples with a shared color scale), plus a panel showing which strategy wins in each cell. | 2 |
| `extrapolation_curves` | Holdout error vs time since the end of training, one curve per strategy (median over seeds). | 1 |
| `calibration` | Coverage plot (nominal vs empirical credible-interval coverage) per strategy, and SBC rank histograms. | 2 |
| `model_selection_vs_noise` | Posterior probability of the true family vs noise level (the "eight arcminutes" curve). | 2 |
| `accuracy_vs_runtime` | Scatter of error vs wall-clock time per strategy (Pareto view of cost vs quality). | 2 |
| `structural_success` | Fraction of seeds where each strategy recovers the exact law / the law up to constants / fails, per scenario. | 4 |
| `rejection_matrix` | For each truth type (dynamical laws and kinematic paths) × strategy: what kind of answer was given (correct law, wrong law, kinematic, rejected), as a colored matrix. | 4 |

#### 7.1.5 Cross-run animations

| Animation | Content | Phase |
|---|---|---|
| `sweep_animation` | Animate a per-run figure (e.g. `orbit_topdown` or `sky_track`) across one swept parameter, e.g. noise rising from 10″ to 20′, showing every strategy's recovered orbit degrade side by side against the truth. Configurable: which figure, which parameter, which seed. | 2 |
| `strategy_race` | Several strategies' `orbit_animation`s in a synchronized grid on the same scenario and seed. | 2 |

#### 7.1.6 Fit traces (Jan addendum)

To support `fit_progress` and `law_discovery` animations, `Hypothesis` gains an optional method:

```python
def trace(self) -> list[TraceFrame] | None: ...
```

A `TraceFrame` holds an iteration label and a snapshot sufficient to predict (e.g. parameter vector for a parametric family, or a serialized law). Strategies record traces only when `config.record_trace = True`, keeping at most `trace_max_frames` (default 100, evenly subsampled). The harness evaluates each frame's predictions on the query sheet, just like the final hypothesis, so Rudolf never executes Jan code.

---

## 8. Scenario library (`scenarios/`)

Ship these YAMLs. The phase in which each must work is listed.

| File | Phase | What it tests |
|---|---|---|
| `circle_coplanar.yaml` | 1 | Sanity: circular, i = 0, low noise, T1. |
| `mars_like.yaml` | 1 | Mars-like ellipse, Tycho-level noise (~60″), T1. |
| `eight_arcminutes.yaml` | 2 | Sweep template: Mars-like truth, noise from 10″ to 20′; question: when can `bayes.nested` distinguish equant-circle from ellipse? (Recreates the discrepancy that led Kepler to reject circles.) |
| `mars_unknown_earth.yaml` | 3 | T2: Mars-like body + Sun stream, several bodies optional. Kepler's actual problem. |
| `sedna_like.yaml` | 3 | e ≈ 0.85, high inclination, period of thousands of years, observed over a short arc; expect strong period–eccentricity degeneracy. |
| `hyperbolic_comet.yaml` | 3 | e ≈ 1.2, single apparition, magnitude-limited visibility near perihelion. Tests e < 1 vs e > 1 selection. |
| `outliers_and_bias.yaml` | 3 | Mars-like with 5% gross outliers and a constant bias; tests robust likelihoods. |
| `hooke_trap.yaml` | 4 | Linear central force: closed ellipse **centered** on the Sun. Looks Keplerian at a glance but isn't. |
| `precessing_power_law.yaml` | 4 | f ∝ r^(−2.05): rosette orbit. |
| `gr_like_precession.yaml` | 4 | Newton + small ε/r³ term; tests UDE sensitivity to tiny deviations. |
| `yukawa.yaml` | 4 | Yukawa-modified gravity with range comparable to the orbit. |
| `hidden_perturber.yaml` | 4 | Uranus-like target perturbed by an unobserved massive outer body (Neptune-style). |
| `rectangle.yaml` | 4 | Kinematic rectangular orbit, constant speed. Tests periodic triangulation and honest rejection of smooth force laws. |
| `figure_eight.yaml` | 4 | Kinematic figure-eight / Lissajous path. |
| `epicycle_truth.yaml` | 2 | Ptolemaic truth; can Jan correctly prefer epicycles over conics here? |

---

## 9. Bench, CLI and reporting

### 9.1 CLI (`jt`)

```
jt tycho generate <scenario.yaml> --seed 1 [--out runs/]
jt jan fit <run_dir> --strategy classical.lsq [--config strategy.yaml]
jt rudolf score <run_dir> --strategy classical.lsq [--animate] [--no-plots] [--quick]
jt rudolf report <run_dir>                                     # single-run HTML report across strategies
jt rudolf animate <run_dir> --strategy classical.lsq --which orbit_animation
jt run <scenario.yaml> --strategy classical.lsq --seed 1 [--animate]   # generate + fit + score + plots
jt bench sweep <bench/sweep.yaml> [--workers 8]
jt bench report <results_dir> --out report.html
jt list scenarios | strategies
```

### 9.2 Sweep config

```yaml
name: phase2_noise_sweep
scenarios: [scenarios/mars_like.yaml]
overrides:                      # dotted-path grid over scenario fields
  noise.0.sigma_arcsec: [10, 60, 240, 480, 1200]
strategies: [ptolemy.fourier, classical.lsq, bayes.nested]
strategy_overrides:             # dotted-path grid over strategy config fields (Jan hyperparameters)
  ptolemy.fourier:
    n_harmonics: [2, 4, 8, 16]
seeds: {start: 0, count: 20}
timeout_minutes: 30
viz:
  per_run: first_seed           # none | first_seed | all — which cells get full per-run figures
  animations: first_seed        # none | first_seed | all
  comparison:                   # cross-run figures to produce (§7.1.4–7.1.5)
    - {figure: error_vs_tycho_param, x: noise.0.sigma_arcsec, metric: holdout_rms_arcsec}
    - {figure: error_vs_jan_param, strategy: ptolemy.fourier, x: n_harmonics, metric: holdout_rms_arcsec}
    - {figure: extrapolation_curves}
    - {figure: sweep_animation, base_figure: orbit_topdown, x: noise.0.sigma_arcsec, seed: 0}
```

Every swept parameter (Tycho or Jan) is recorded as its own column in `results.parquet`, so any comparison figure can use any swept parameter as an axis or facet.

The bench runs each (scenario override, seed, strategy) cell in a separate process (`concurrent.futures`), caches completed cells (keyed by a hash of scenario + seed + strategy config + code version), and aggregates all `score.json` files into `results.parquet`.

### 9.3 Reports

`jt bench report` calls Rudolf's comparison plotting (§7.1.4–7.1.5) and assembles a static HTML report (Jinja2) from the figure manifests. The report contains the leaderboard, every requested comparison figure with its caption, embedded animations (`<video>` for MP4, `<img>` for GIF), and links to each cell's per-run figures and animations. A single-run report (`jt rudolf report <run_dir>`) does the same for one scenario + seed across all strategies that were run on it.

---

## 10. Implementation phases

Each phase ends with a notebook in `notebooks/` demonstrating the new capabilities, **every phase delivers the visualizations listed under its "Visualization" bullet (from §7.1), produced by Rudolf for every scenario introduced in that phase**, and the acceptance criteria below pass in CI (heavy optional-extra tests may be marked and run separately). Numerical thresholds are initial targets: if one proves unrealistic, record the measured value, adjust the threshold, and explain the change in the notebook.

### Phase 0 — Scaffolding
- uv workspace with the five packages; ruff, pyright, pytest configured; GitHub Actions CI.
- `.importlinter` contracts from §3.2 plus a pytest test that runs them.
- `jt_core`: constants, units, frames, angles, rng, schema, io.
- `jt` CLI skeleton with `list` commands.
- **Visualization:** `rudolf.viz.style` (palette, strategy color map, figure conventions), the figure manifest, and the ffmpeg/Pillow animation-writer abstraction with runtime detection.

**Acceptance:** CI green; a deliberate `import tycho` inside `jan` makes the boundary test fail; a trivial test animation renders to MP4 when ffmpeg is present and to GIF otherwise.

### Phase 1 — The minimal loop
- `jt_core.conics`: universal-variable propagation; elements ↔ state conversions for all e.
- Tycho: `keplerian` and `circular` truths; `keplerian`, `fixed`, `heliocentric` observers; T0/T1 tiers; `regular`/`poisson`/`explicit` schedules with dropout; `gaussian` and `quantize` noise; full public/sealed bundle output.
- Jan: strategy registry; `ConicOrbit` and `CircularOrbit` families; `classical.gauss_iod`, `classical.lsq`, `ptolemy.fourier`.
- Rudolf: predictive (holdout), trajectory and parameter metrics; diagnostic plots.
- Bench: `jt run`; a minimal sweep (scenario overrides only, no caching yet) sufficient for the leaderboard and extrapolation curves.
- Scenarios: `circle_coplanar`, `mars_like`.
- **Visualization:** per-run `sky_track`, `residuals_vs_time`, `orbit_topdown`, `orbit_3d`, `element_errors`, basic `observation_coverage`; animations `orbit_animation` and `sky_animation`; comparison `leaderboard` and `extrapolation_curves`; `jt rudolf report`.

**Acceptance:**
- Conic tests: state → elements → state round-trip error < 1e-10 (relative) for e ∈ {0, 0.5, 0.99, 1.0, 1.5, 5}; specific orbital energy and angular momentum conserved to < 1e-10 over 100 periods.
- Determinism: identical bundles (byte-equal Parquet contents) for identical config + seed.
- `mars_like`, T1, 60″ noise: `classical.lsq` recovers a within 0.5%, e within 0.005, angles within 0.5°; holdout RMS < 2× the noise sigma.
- `ptolemy.fourier` has holdout RMS at least 5× worse than `classical.lsq` on `mars_like` (demonstrates the extrapolation gap).
- `jt run scenarios/mars_like.yaml --strategy classical.lsq --animate` produces every Phase 1 per-run figure and animation, listed in `figures.json`; the `extrapolation_curves` figure visibly shows `ptolemy.fourier` diverging after the training boundary.

### Phase 2 — Bayesian inference and model selection
- Jan: `EquantCircle`, `Epicycles` families; `bayes.mcmc`, `bayes.hmc`, `bayes.nested`; `gp.sky`.
- Tycho: `epicycle` truth.
- Rudolf: probabilistic predictive metrics, posterior parameter metrics, model-selection metric; corner plots.
- Bench: `jt bench sweep` with Tycho and Jan (`strategy_overrides`) parameter grids, caching, aggregation, calibration across seeds, `viz` section, HTML report.
- Jan: optional fit traces (§7.1.6) for `classical.lsq`, `bayes.mcmc` and `bayes.hmc`.
- Scenarios: `eight_arcminutes`, `epicycle_truth`.
- **Visualization:** per-run `corner`, `posterior_predictive`, `model_probabilities`; animations `fit_progress` and `posterior_samples`; comparison `error_vs_tycho_param`, `error_vs_jan_param`, `heatmap_2d`, `calibration`, `model_selection_vs_noise`, `accuracy_vs_runtime`; cross-run animations `sweep_animation` and `strategy_race`.

**Acceptance:**
- On `mars_like` (T1, 60″), 95% credible intervals from `bayes.hmc` cover the true elements in 90–99% of 50 seeds.
- In the `eight_arcminutes` sweep, `bayes.nested` assigns > 0.95 posterior probability to the ellipse at 60″ noise and is indecisive (< 0.9) at the largest noise levels; the report shows the transition.
- On `epicycle_truth`, `bayes.nested` prefers the epicycle family.
- The `eight_arcminutes` sweep report contains `model_selection_vs_noise`, `error_vs_tycho_param` (noise on x), a `heatmap_2d` of noise × number of observations, and a `sweep_animation` of `orbit_topdown` across noise levels.
- A sweep over `ptolemy.fourier.n_harmonics` produces an `error_vs_jan_param` figure showing the training-vs-holdout error divergence (overfitting).

### Phase 3 — Realistic observing and Kepler's problem
- Tycho: T2/T3 tiers and Sun stream; multi-body scenarios; `seasons` schedule; solar elongation and magnitude visibility; `bias`, `drift`, `ar1`, `heteroscedastic`, `outliers`, `clock` noise.
- Jan: `jan.reconstruct.periodic_triangulation`; `kepler.triangulate`; unknown-observer parameters in families; robust likelihoods (Student-t) for all Bayesian strategies.
- Scenarios: `mars_unknown_earth`, `sedna_like`, `hyperbolic_comet`, `outliers_and_bias`.
- **Visualization:** `observation_coverage` with gap reasons, `noise_diagnostics`, `triangulation_geometry`; animation `triangulation_animation`; `orbit_animation` extended to show the *recovered* observer orbit alongside the true one under T2/T3.

**Acceptance:**
- `mars_unknown_earth` (T2): Earth's e recovered within 0.005 and Mars's a within 1% (in AU) by at least one strategy.
- `hyperbolic_comet`: `bayes.nested` assigns > 0.95 probability to e > 1.
- `sedna_like`: posterior shows the expected degeneracy (documented with a corner plot) and credible intervals still cover the truth.
- `outliers_and_bias`: the robust likelihood cuts holdout RMS by ≥ 2× vs the Gaussian likelihood.
- `triangulation_animation` on `mars_unknown_earth` shows both the body's and the observer's orbits being reconstructed; `noise_diagnostics` on `outliers_and_bias` marks true and detected outliers.

### Phase 4 — Alien physics and law discovery
- Tycho: `central_force` (all built-in laws + custom), `nbody` with hidden bodies, `kinematic_polygon`, `kinematic_parametric`; non-Keplerian observer support.
- Jan: `CentralForceOrbit`, `PerturbedConic`, `PeriodicKinematic` families; `discovery.sindy`, `discovery.pysr`, `piml.neural_ode`, `piml.hnn`, `piml.ude` with symbolic distillation.
- Rudolf: structural metrics (symbolic + numerical field error) and rejection scoring.
- Scenarios: `hooke_trap`, `precessing_power_law`, `gr_like_precession`, `yukawa`, `hidden_perturber`, `rectangle`, `figure_eight`.
- Jan: fit traces for discovery and PIML strategies.
- **Visualization:** per-run `force_law`, `acceleration_field`, `symbolic_pareto`, `conserved_quantities`; animation `law_discovery`; comparison `structural_success` and `rejection_matrix`; `orbit_animation` for `hidden_perturber` also shows the true hidden body and Jan's inferred position for it.

**Acceptance:**
- T0 tier: `discovery.sindy` and `discovery.pysr` recover the inverse-square law exactly (symbolic match up to constants) on `mars_like` and the linear law on `hooke_trap`.
- T1 tier: at least one strategy recovers the exponent on `precessing_power_law` within ±0.01.
- `hidden_perturber`: `piml.ude` or `PerturbedConic` + `bayes.nested` localizes the hidden body's heliocentric longitude within 10° at the epoch of maximum perturbation.
- `rectangle`: `kepler.triangulate` reconstructs the rectangle's vertices within 2% of its side length, and returns `kind ∈ {"kinematic", "rejected"}`; no strategy that claims a dynamical law scores well on the rejection metric.
- A Phase 4 sweep report contains `rejection_matrix` across all Phase 4 scenarios and strategies, and `force_law` on `precessing_power_law` shows the recovered exponent curve overlapping the truth within the visited range of r.

### Phase 5 — Extensions (open-ended)
- `sbi.npe` (amortized inference trained on Jan's own families).
- `llm.hypothesize` loop (propose → fit → compare), with all LLM calls cached for reproducibility.
- Realism effects: light-time, aberration, equatorial coordinates with obliquity.
- **Visualization:** posterior plots for `sbi.npe` (amortized vs per-scenario posteriors compared); a hypothesis-history view for `llm.hypothesize` (proposed families, their evidence, and the chosen one); an optional interactive HTML explorer (e.g. Plotly) over `results.parquet` for slicing comparisons by any parameter.
- New domains reusing the Tycho/Jan/Rudolf split: pendulum (with damping), Lotka–Volterra, chemical kinetics, a double pendulum (chaos as a hard case). Refactor the observation operator and schema to be domain-generic if this is pursued.
- Optional: ingest a real public observation set as a "Tycho" stand-in.

---

## 11. Testing requirements (all phases)

- **Unit tests** for every math function in `jt_core`, with `hypothesis` property tests for frames and conics.
- **Integrator tests**: central-force integrations conserve energy/angular momentum (where applicable) to stated tolerances; `keplerian` and `central_force(power_law n=2)` truths agree to < 1e-9 AU over 10 periods.
- **Noise tests**: empirical great-circle residual distribution of `gaussian` noise matches the configured sigma (KS test).
- **Boundary test**: import-linter contracts (§3.2).
- **Determinism test**: same config + seed ⇒ identical outputs; different seeds ⇒ different noise.
- **End-to-end smoke test**: `jt run scenarios/circle_coplanar.yaml --strategy classical.lsq` completes in < 60 s on a laptop.
- **Visualization tests**: for every figure and animation type, a smoke test renders it (`--quick` mode, `Agg` backend) from a small fixture run and checks that the file exists, is non-empty, and is listed in `figures.json`. Figures must handle missing optional data (e.g. no posterior samples, no recovered trajectory) by omitting that layer and noting it in the caption, never by crashing.
- Mark slow and optional-extra tests (`@pytest.mark.slow`, `@pytest.mark.bayes`, `@pytest.mark.discovery`) so the default suite stays fast.

---

## 12. Known pitfalls to handle explicitly

- **Range ambiguity:** angles-only data does not constrain distance without a dynamical or periodicity assumption; absolute scale is only defined relative to the observer's orbit (AU). Document this in Jan strategies that make such assumptions.
- **Kepler's equation** at high e and near e = 1: use universal variables with a robust root finder (Laguerre–Conway or Newton with safeguards); test near-parabolic cases.
- **Angle wraparound:** always use unit vectors / great-circle distance.
- **Circular / equatorial degeneracies:** ω undefined at e = 0, Ω undefined at i = 0. Rudolf must skip or merge such elements (e.g. compare ϖ = Ω + ω, or mean longitude) rather than report spurious errors.
- **Multimodality** in Bayesian fits (period aliases from sparse sampling): use multi-start initialization and report it.
- **Numerical drift** in long integrations masquerading as new physics: tight tolerances, and conservation checks in tests.
- **Noisy derivatives** in SINDy: prefer weak-form or fit-based derivatives.
- **Fairness:** all strategies in a comparison get identical data, and the bench records wall-clock time and the compute budget per cell.
- **Leakage via simulation:** SBI and any simulator-based strategy must simulate from Jan's own families, never from Tycho.
