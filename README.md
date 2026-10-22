# Jan and Tycho

A testbed for **automated model and law discovery from data**, modelled on the partnership
between Tycho Brahe (precise observations) and Johannes Kepler (the laws behind them).
See [`SPEC.md`](SPEC.md) for the full design.

- **Tycho** (`packages/tycho`) holds a secret ground-truth model and emits noisy,
  incomplete sky observations as seen from a moving Earth.
- **Jan** (`packages/jan`) is a family of inverse-problem strategies that see *only* the
  public observations and try to recover predictions, trajectories, elements and laws.
- **Rudolf** (`packages/rudolf`) is the referee: the only component that opens the sealed
  truth. It scores Jan's serialized predictions and draws every figure and animation.
- **The bench** (`packages/jt-bench`) runs cells and sweeps and provides the `jt` CLI.
- **`jt_core`** (`packages/jt-core`) holds shared, truth-agnostic math: frames, angles,
  universal-variable conic propagation (JAX, differentiable), schema and I/O.

The information boundary is enforced by `import-linter` (`.importlinter`): Jan may import
only `jt_core`, never `tycho` or `rudolf`.

## Status

| Phase | State |
|---|---|
| 0 — Scaffolding | done |
| 1 — Minimal loop | done (see notes on the `ptolemy.fourier` criteria below) |
| 2 — Bayesian inference and model selection | done |
| 3 — Realism and Kepler's problem | done (sedna coverage met by emcee in state coordinates; see notes) |
| 4 — Alien physics and law discovery | done (see notes) |
| 5 — Extensions | realism effects, `sbi.npe`, `llm.hypothesize`, results explorer (see notes; new domains and real data not started) |

Each phase has a notebook in `notebooks/` (built by `notebooks/build_notebooks.py`) that
checks its acceptance criteria against **measured** values from the sweeps in `bench/`.

## Quick start

```bash
uv sync                                   # Python ≥ 3.11; installs the whole workspace
uv run jt list scenarios
uv run jt list strategies

# generate + fit + score + figures (+ HTML report) for one scenario and seed
uv run jt run scenarios/mars_like.yaml --strategy classical.lsq --strategy ptolemy.fourier --animate
# → runs/mars_like/1/rudolf/report.html

# step by step
uv run jt tycho generate scenarios/mars_like.yaml --seed 1
uv run jt jan fit runs/mars_like/1 --strategy classical.lsq
uv run jt rudolf score runs/mars_like/1 --strategy classical.lsq --animate
uv run jt rudolf report runs/mars_like/1

# sweeps (cached per cell; a cell re-runs when its scenario, seed, config or code changes)
uv run jt bench sweep bench/phase1_minimal.yaml --workers 8
uv run jt bench report runs/sweeps/phase1_minimal

# phase notebooks (read runs/sweeps/*)
uv run python notebooks/build_notebooks.py phase5
```

Optional extras: `jan[bayes]` (emcee, NumPyro, dynesty), `jan[gp]`, `jan[piml]` (equinox,
optax), `jan[discovery]` (PySR — needs Julia, installed on first use), `jan[sbi]` (sbi +
CPU PyTorch), `jan[llm]` (Anthropic SDK) and `rudolf[explorer]` (plotly). The root dev
environment installs all of them.

Animations are written as MP4 (H.264) via `ffmpeg`. If `ffmpeg` is missing they fall back
to GIF (Pillow) rather than failing the run; re-rendering removes any stale file of the other
format.

## Development

```bash
uv run pytest                 # fast suite (slow / optional-extra tests are deselected)
uv run pytest -m slow         # slow acceptance tests
uv run pytest -m "sbi or bayes or discovery"   # optional-extra tests
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run lint-imports           # information-boundary contracts
```

## Run directory layout

```
runs/<scenario>/<seed>/
├── public/       observations.parquet, meta.json             ← all Jan ever sees
├── sealed/       scenario.yaml, truth.json, trajectory.parquet, holdout.parquet,
│                 noise_realization.parquet, schedule.parquet, run.json
├── jan/<label>/  hypothesis.json, predictions.parquet, prediction_samples.parquet,
│                 trace_predictions.parquet, query.parquet (times only), work/
└── rudolf/       report.html, figures/ (cross-strategy); <label>/score.json, figures/, figures.json
```

## Phase 1 notes and deviations

- **`ptolemy.fourier` is a real epicycle model.** The geocentric vector
  d(t) = r_planet − r_earth is exactly a sum of two periodic vectors, so the strategy models
  d (not the unit vector) as harmonics of two Lomb–Scargle base periods and solves the
  linear condition u × d = 0. Fitting the unit vector component-wise was tried first and
  fails badly (degrees of error even in-sample) because |d| varies by ×6 around opposition.
- Because the truth is exactly bi-periodic, well-estimated epicycles **extrapolate well**
  (just as Ptolemy's did). On `mars_like` the holdout RMS ratio `fourier / lsq` is
  3.2–16× across seeds 0–4 (median 6.5×), so the "≥ 5× worse" criterion is checked as a
  median over seeds (`tests/test_jan.py::test_fourier_extrapolates_worse_than_lsq`,
  marked slow). The extrapolation curve does **not** visibly diverge after the training
  boundary at the default order; the divergence is expected to show up in the Phase 2
  `n_harmonics` overfitting sweep and on non-periodic truths.
- `classical.lsq` fits a non-singular elliptic parametrization by default
  (`family: ellipse`); `family: conic` (raw `q, e, i, Ω, ω, t_p`, any e) and
  `family: circle` are available. Gauss's method degenerates for coplanar geometry, so
  starts also come from a circular multi-start grid.

## Phase 2 notes

- `bayes.nested` runs dynesty in a box of ±8 Laplace σ around each family's MAP (orbit
  posteriors are needles in wide physical priors); with uniform physical priors,
  ln Z = ln Z_box + ln V_box + ln π. Families trailing the best by > 200 nats get a Laplace
  evidence.
- The equant circle's period is tied to Kepler's third law, like the conics'; with a free
  period it paid an Occam penalty that turned the eight-arcminutes question into one
  about the period prior.
- Eight arcminutes: with ~360 observations and a known Earth the ellipse stays decisive
  (P > 0.99) far beyond 8′; it only becomes indecisive near 1° with a 90-day cadence. The
  sweep therefore extends to 1°.

## Phase 3 notes

- `kepler.triangulate` is Kepler's method: pair observations one period apart and intersect
  the lines of sight. At T2 it first recovers the Earth's orbit from Mars sightings
  (Kepler's reverse trick: one unknown scale per group, all tied together by a single
  Keplerian observer orbit), then triangulates, fits a conic and refines against all data.
- `sedna_like`: in orbital elements the short-arc posterior is a curved banana that the
  Laplace approximation, NUTS and emcee all undercover (emcee in φ: τ ≈ 2,000 steps, and
  20,000-step chains still covered q, e, ω, t_p at 95 % in only 10–11/20 seeds).
  `bayes.mcmc` with `coords: state` samples the same posterior in the mid-arc state vector
  (x, v), where the ridge is nearly straight (τ ≈ 75 steps): 95 % coverage 19–20/20 for
  every element.

## Phase 4 notes

- **Judging a law.** A recovered law is "correct" only if its relative error is within
  10⁻³ *and* within half of the truth's own departure from Newton (otherwise plain Newton
  would count as recovering the GR-like 2×10⁻⁴ term). "Right form" means the correct
  expression with wrong constants. For Newton + a hidden planet, a claimed perturber within
  10° of the true longitude counts as correct.
- **hidden_perturber.** `classical.lsq` and `bayes.nested` with the `perturbed-conic` family
  localize the planet within ~1°, but not its distance: over a 70-year arc mass and
  distance trade off.
- **Kinematic paths.** `kepler.triangulate` answers "kinematic" for the rectangle and the
  figure-eight. Rectangle vertices are recovered by intersecting fitted edge lines (so
  corners hidden in observing gaps are recovered too): within 1 % of the shortest side
  against the 2 % target. Rejection credit accepts either a median holdout sky error
  within 5σ or a holdout path within 1 % of the orbit size, because the rectangle passes
  close to the Earth.
- **Discovery at T1** regresses on positions from the best central-family fit and inherits
  its bias; T0 (positions) is the native tier, where SINDy and PySR recover 1/r² and
  Hooke's law exactly.
- **Black-box PIML.** `piml.neural_ode` and `piml.hnn` cannot integrate a 20-year orbit to
  arcsecond precision (a small MLP matches Newton to ~10⁻³); `piml.ude` (Newton + learned
  correction, distilled to a formula) does not have that problem.

## Phase 5 notes

- **Realism** (`scenarios/realism_mars.yaml`, `bench/realism.yaml`). Tycho can publish
  apparent directions (light-time, annual aberration) in ecliptic or equatorial
  coordinates; `PublicMeta.realism` says which. Jan's `RealismAdapter` wraps every strategy:
  exact frame rotation and aberration removal, a two-pass differential correction for
  light-time (and for aberration when the observer is inferred), and apparent predictions
  in the published frame. On 1″ data the corrected fits stay at ~0.1″ holdout RMS; ignoring
  the effects (`realism: ignore`) leaves ~9″ (light-time), ~16″ (aberration), ~25″ (both).
- **`sbi.npe`** (`bench/sbi_npe.yaml`, mars_like, 10 seeds). Simulations come from Jan's
  conic family only. The amortized round (whole prior) is ~10⁴× too wide and miscalibrated
  (median inclination 30–100° vs 1.9°). Truncated sequential rounds (6–7, ~130k
  simulations) match emcee's accuracy (6.0″ vs 6.1″ holdout) but are overconfident: 95 %
  element intervals cover the truth in 78–89 % of seeds (emcee 100 %), predictive 72 % (emcee
  96 %). One seed exceeded the 30-minute timeout while sharing the CPU.
- **`llm.hypothesize`** (`bench/llm_hypothesize.yaml`). Claude (`claude-opus-5-5`,
  structured JSON output, server-side refusal fallback enabled) proposes registered families
  or new central laws f(r) from rounded numerical residual summaries; Jan fits them (MAP,
  Laplace evidence) and compares the leaders by nested sampling. Proposed expressions are
  token-whitelisted before sympy parses them. Every call is cached in `runs/llm_cache`
  (keyed by a hash of the request; the one part of `runs/` that is committed), so runs
  replay offline. A first run reads `ANTHROPIC_API_KEY`. Tested end to end with a
  scripted client (power law recovered on `precessing_power_law`, an injected expression
  refused, an offline replay identical); **the sweep has not been run against the real API**
  — a first run needs Anthropic credentials and is billed.
- **Explorer.** `jt bench report` also writes `explorer.html` (plotly, self-contained): any
  result column against any other, coloured and filtered at will.
- Not started: new domains (pendulum, Lotka–Volterra, …) and real-data ingestion.
