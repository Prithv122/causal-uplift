# Causal Uplift — D4

**Tier:** 3 · **Category:** Data Science · **Wave:** 4

The project's own engineering guide. Keep it short and current.

## What this is

A retention team has a discount budget and wants to spend it on users whose conversion the offer
actually changes. This repo simulates users whose true individual treatment effect is known, then
measures how far common estimators (naive difference, response model, S/T/X/DR-learners, causal
forest) fall from that truth as confounding rises from none (randomised) to strong. A small
difference-in-differences module covers the panel design question.

## Stack

Python 3.12, uv, numpy, pandas, scipy, scikit-learn, LightGBM. EconML is an optional extra for the
causal-forest reference arm. Learners are implemented here, on top of scikit-learn/LightGBM base
models, not imported from a causal library.

## Contracts

`DESIGN.md` is the contract for every module (simulator formulas, estimator interface, metric
definitions, results columns). `PREREGISTRATION.md` fixes the grid and the hypotheses before the
full grid runs. A change to either is a dated amendment, never a silent edit.

## Acceptance criteria

- [ ] Simulator gives exact per-user truth (`p0`, `p1`, `tau`, propensity) and is seed-deterministic
- [ ] Every estimator scored on ATE bias, coverage (where an SE exists), PEHE, Spearman, Qini against
      the truth, and policy value at a fixed treatment cost
- [ ] Full grid (R = 200 per cell) run locally from the committed pre-registration; H1–H3 verdicts
      in `results/REPORT.md` are mechanical
- [ ] DiD module shows the parallel-trends violation as bias of the expected size
- [ ] Runs from a clean clone using only the documented steps
- [ ] Tests, lint and CI green; README results come from real runs

## Conventions

- `uv` for dependencies; `ruff check` + `ruff format`; `pytest`.
- Seeds explicit everywhere (`BASE_SEED = 20260830`); no global RNG.
- No dataset in git. Real data (Criteo Uplift, CC BY-NC-SA 4.0) is downloaded by a command and
  lives in `data/raw/`, which is ignored.

## Project-specific notes

- CI runs a smoke grid (n = 2,000, R = 3, logistic base) only. Reported numbers come from the full
  local grid.
- LightGBM runs with `n_jobs=1, deterministic=True`; parallelism is across replications, so
  results do not depend on the worker count.
