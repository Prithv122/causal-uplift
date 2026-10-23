# Design and module contracts

Written before any estimator code. Every module is built against this file; a change to a
contract is a dated amendment at the bottom, never a silent edit.

## The question

On simulated users whose true individual treatment effect is known, how far do the estimators
(naive difference, response model, S/T/X/DR-learners, causal forest) fall from the truth, in ATE
bias and CATE ranking (Qini, PEHE), as confounding strength rises from 0 (randomised) to strong,
and at what strength does each stop being usable?

## Data-generating process (`causal_uplift/dgp.py`)

```python
simulate(n: int, seed: int, *, gamma: float = 0.0, hidden: float = 0.0,
         effect: float = 1.0, noise: float = 0.0) -> Sim
```

Covariates, 8 columns, names `x0..x7`:

| column | distribution | role |
|---|---|---|
| x0 | N(0, 1) | baseline conversion driver and the observed confounder |
| x1 | N(0, 1) | responsiveness to the offer (sign decides help vs no help) |
| x2 | N(0, 1) | baseline |
| x3 | N(0, 1) | smooth effect modifier |
| x4 | N(0, 1) | baseline |
| x5 | N(0, 1) | pure noise column (no role) |
| x6 | Bernoulli(0.5) | baseline |
| x7 | Bernoulli(0.3) | decides sleeping dogs |

Hidden confounder `u ~ N(0, 1)`, never in `X`.

Segments, decided by observed covariates (integer codes, in this order):

| code | name | rule | logit lift `s` |
|---|---|---|---|
| 0 | persuadable | x1 > 0 and x0 < 0.5 | +1.5 |
| 1 | sure_thing | x1 > 0 and x0 >= 0.5 | +0.3 |
| 2 | lost_cause | x1 <= 0 and x7 == 0 | 0.0 |
| 3 | sleeping_dog | x1 <= 0 and x7 == 1 | -1.0 |

Logits and probabilities:

```
eps   ~ N(0, noise)                       (per user, independent of everything; 0 when noise == 0)
b(x)  = -1.5 + 1.2*x0 + 0.4*x2 - 0.3*x4 + 0.3*x6 + 1.2*hidden*u + eps
l(x)  = effect * (s(segment) + 0.3*x3 * [segment in {persuadable, sure_thing}])
p0    = sigmoid(b)
p1    = sigmoid(b + l)
tau   = p1 - p0                            (exact, per user)
e(x)  = sigmoid(gamma * (x0 + hidden*u))   (true propensity; not clipped)
t     ~ Bernoulli(e)
y     ~ Bernoulli(p1 if t == 1 else p0)
```

Why this shape: `x0` raises both conversion and the chance of being treated, so a naive
comparison credits the offer with conversions that would have happened anyway (positive bias that
grows with `gamma`). `hidden` moves part of that confounding into `u`, which no estimator sees.
Sure things have a high baseline, so their probability lift is small even with a positive logit
lift; sleeping dogs are hurt by the offer.

Randomness: one `np.random.default_rng(seed)`, draws in a fixed order (X columns in order, u,
eps, t, y), so the same `(n, seed, knobs)` gives identical arrays.

```python
@dataclass(frozen=True)
class Sim:
    X: np.ndarray            # (n, 8) float64
    t: np.ndarray            # (n,) int8 in {0, 1}
    y: np.ndarray            # (n,) int8 in {0, 1}
    p0: np.ndarray           # (n,) float64
    p1: np.ndarray           # (n,) float64
    tau: np.ndarray          # (n,) float64, p1 - p0
    propensity: np.ndarray   # (n,) float64, true e(x)
    segment: np.ndarray      # (n,) int8 codes 0..3
    u: np.ndarray            # (n,) float64, hidden confounder (oracle use only)
    feature_names: tuple[str, ...]  # ("x0", ..., "x7")
    knobs: dict              # {"n", "seed", "gamma", "hidden", "effect", "noise"}
    @property true_ate -> float   # mean(tau)
    def to_frame() -> pd.DataFrame  # x0..x7, t, y, p0, p1, tau, propensity, segment
```

`SEGMENT_NAMES = ("persuadable", "sure_thing", "lost_cause", "sleeping_dog")`.

## Metrics (`causal_uplift/metrics.py`) — arrays in, floats out, no fitting

| function | definition |
|---|---|
| `pehe(tau_hat, tau)` | `sqrt(mean((tau_hat - tau)^2))` |
| `spearman(score, tau)` | Spearman rank correlation; NaN if either is constant |
| `uplift_curve_true(score, tau)` | cumulative true gain `V(k) = sum of tau over the top k by score`, k = 0..n, ties averaged (users with equal score share the group's mean tau, as if randomly ordered) |
| `qini_true(score, tau)` | `(A(score) - A(random)) / (A(oracle) - A(random))`, A = area under `V(k)/n` against k/n by the trapezoid rule; oracle = ranking by tau; random = the straight line to `sum(tau)`. 1 = oracle ranking, 0 = no better than random, can be negative. Constant score → 0 |
| `qini_observed(y, t, score)` | Radcliffe Qini from outcomes only, for randomised real data: `Q(k) = Yt(k) - Yc(k) * Nt(k)/Nc(k)`, ties grouped; returns the area between the curve and the random line, normalised by `n` |
| `policy_value(score, tau, frac, cost)` | treat the top `frac` by score (ties averaged): `(sum of tau over treated - cost * n_treated) / n`, per user, in conversions |
| `oracle_policy_value(tau, frac, cost)`, `random_policy_value(tau, frac, cost)` | same for ranking by true tau and for random targeting (`frac * (mean(tau) - cost)`) |
| `covered(lo, hi, truth)` | `lo <= truth <= hi`; False if either bound is NaN |

## Estimators (`causal_uplift/estimators/`)

Every estimator is a class with:

```python
name: str                      # the row label in results, e.g. "x_learner"
estimates_tau: bool            # False for the response model (its score is not an effect)
def fit(self, X, t, y) -> Self
def predict_tau(self, X) -> np.ndarray     # (n,) score used for ranking; an effect estimate when estimates_tau
def ate(self) -> AteResult | None          # after fit; None if the estimator has no ATE
```

```python
@dataclass(frozen=True)
class AteResult:
    estimate: float
    se: float
    lo: float  # 95% interval
    hi: float
```

Base learners come from `estimators/base.py: make_classifier(kind, seed)` and
`make_regressor(kind, seed)`, `kind in {"lgbm", "logistic"}` (logistic/linear for fast tests).
LightGBM settings, fixed for every learner: `n_estimators=200, learning_rate=0.05, num_leaves=15,
min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=1,
deterministic=True, force_col_wise=True, verbose=-1, random_state=seed`. Estimated propensities
are clipped to `[0.01, 0.99]` wherever they are divided by (`PROPENSITY_CLIP`).

| name | file | what it does | ATE |
|---|---|---|---|
| `naive` | `naive.py` | constant tau = mean(y | t=1) - mean(y | t=0) | difference in means, Welch SE |
| `response` | `response.py` | P(y=1 | x) fitted on treated users only; score = that probability ("treat who is most likely to convert") | None |
| `s_learner` | `metalearners.py` | one classifier on `[X, t]`; tau = p(x, 1) - p(x, 0) | mean of tau_hat, SE NaN |
| `t_learner` | `metalearners.py` | classifiers mu0 on controls, mu1 on treated; tau = mu1 - mu0 | mean of tau_hat, SE NaN |
| `x_learner` | `metalearners.py` | Künzel et al. 2019: T-learner stage, imputed effects `D1 = y - mu0(x)` on treated, `D0 = mu1(x) - y` on controls, regressors tau1, tau0, propensity g; tau = g*tau0 + (1-g)*tau1 | mean of tau_hat, SE NaN |
| `dr_learner` | `dr.py` | Kennedy 2023: K=2 cross-fitted nuisances mu0, mu1, e; pseudo-outcome `phi = mu1 - mu0 + t(y - mu1)/e - (1-t)(y - mu0)/(1-e)`; regress phi on X | AIPW: mean(phi), SE = sd(phi)/sqrt(n) |
| `ipw` | `dr.py` | ATE only: Hajek-normalised inverse-propensity weighting with a cross-fitted propensity; `predict_tau` returns the constant ATE | influence-function SE |
| `causal_forest` | `econml_arm.py` | EconML `CausalForestDML` reference arm, optional extra, skipped when not installed | EconML's ATE interval |

`ESTIMATORS: dict[str, Callable[[int, str], Estimator]]` in `estimators/__init__.py` maps name →
factory `(seed, base_kind)`. The S/T/X mean-of-tau ATE has no SE because none of them is built
for inference; coverage is reported only where an SE exists.

## Experiments (`causal_uplift/experiments.py`)

A **cell** is one knob setting `(gamma, hidden, n, effect, noise)`; a **replication** is one fresh
simulation of that cell. Seeds: `np.random.SeedSequence([BASE_SEED, cell_index, rep])` with
`BASE_SEED = 20260830`; the simulation seed and each estimator's seed are drawn from that
sequence, so a row's numbers do not depend on worker count or run order.

One results row per (cell, rep, estimator), written to `results/grid/<cell_id>.csv` (one file per
cell, so an interrupted run resumes by skipping finished cells), then concatenated to
`results/grid.csv`. Columns, in order:

```
cell_id, gamma, hidden, n, effect, noise, rep, sim_seed, estimator, base,
true_ate, ate_hat, ate_se, ate_lo, ate_hi, covered,
pehe, spearman, qini_true,
policy_top10, policy_top20, policy_top30,
oracle_top20, random_top20, cost, fit_seconds
```

NaN where a metric does not apply (`pehe` and `ate_*` for `response`; ranking metrics for `ipw`).
Metrics are computed on a **held-out evaluation sample** of the same cell (fresh draw, same knobs,
`n_eval = 20,000` by default): estimators fit on the training draw, then score the evaluation
users, whose true tau is known. `cost` is the treatment cost in conversions per treated user
(default 0.02).

`report` reads `results/grid.csv` and writes `results/REPORT.md`: per cell and estimator the mean
and Monte-Carlo SE of ATE bias, coverage, PEHE, Spearman, qini_true, policy value; plus the
mechanical H1–H3 verdicts defined in `PREREGISTRATION.md`. The report states only what the
numbers say; interpretation goes in the README after it.

## Difference-in-differences (`causal_uplift/did.py`)

A separate design question (a policy rolled out to one group at one date), kept small.

```python
simulate_panel(n_units, seed, *, selection=1.0, trend_gap=0.0, effect=0.05) -> Panel
```

Two periods (pre, post). Unit level `alpha_i ~ N(0, 1)`; group `g_i ~ Bernoulli(sigmoid(selection
* alpha_i))` (high-level units adopt more often); common time shock `lambda_post = 0.10`;
outcome `y_it = 0.5 + 0.2*alpha_i + lambda_t + effect*g_i*post + trend_gap*g_i*post + N(0, 0.1)`.
`trend_gap` is the parallel-trends violation: DiD is biased by exactly `trend_gap` in expectation.

Estimators: `post_difference` (treated vs control after; biased by selection), `pre_post`
(treated after vs before; biased by the time shock), `did_2x2` (difference of the two groups'
mean change; unit-level Welch SE). Each returns `AteResult`. CLI `did` sweeps `trend_gap` and
prints a table.

## Real-data loader (`causal_uplift/data.py`)

`criteo` only (licence verified CC BY-NC-SA 4.0; RCT; no true tau, so Qini only). A `download`
command fetches it on the owner's say-so, records sha256, and never puts it in git (`data/raw/` is
ignored). `load_criteo(path, n=1_000_000, seed=20260830)` → seeded subsample with columns
`f0..f11, treatment, conversion, visit, exposure`; schema checked. Hillstrom: not included
(licence unverified).

## CLI (`causal-uplift`)

| command | does |
|---|---|
| `simulate --n --seed [--gamma --hidden --effect --noise] [--out]` | prints a summary (n, treated share, true ATE, naive ATE, segment shares); `--out` writes the frame as CSV |
| `run-grid [--smoke] [--config FILE] [--out DIR] [--workers N] [--reps R]` | runs the grid; `--smoke` = n=2,000, n_eval=2,000, R=3, two cells, logistic base |
| `report [--results DIR]` | writes `REPORT.md` |
| `did [--seed --units]` | DiD sweep table |
| `download criteo [--dest]` | real-data fetch (needs network, owner-approved) |

## Conventions

Python 3.12, numpy/pandas/scikit-learn/LightGBM/scipy only in the core; EconML optional. Seeds
explicit everywhere; no global RNG. Tests use small `n` and the logistic base so the suite stays
under a minute.
