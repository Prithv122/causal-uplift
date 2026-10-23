# Pre-registration

**Status: APPROVED by the owner, 2026-10-23, before any full-grid run.** The full grid runs only from
this committed version. Changes are dated amendments at the bottom, made before the runs they
govern.

## Question

On simulated users whose true individual effect is known (`DESIGN.md`), how far do the
estimators fall from the truth as confounding rises, and at what strength does each stop being
usable?

## Grid

| knob | values |
|---|---|
| `gamma` (observed confounding) | 0, 0.1, 0.25, 0.5, 1.0, 2.0, with `hidden = 0` |
| `hidden` (unobserved confounding) | 0.5 and 1.0, at `gamma` = 0.5 and 1.0 |
| `n` (training users per replication) | 10,000 |
| `n_eval` (fresh evaluation users, same knobs) | 20,000 |
| `effect`, `noise` | 1.0, 0.0 (fixed) |
| replications `R` per cell | 200 |
| base learner | LightGBM, settings fixed in `DESIGN.md` |
| treatment cost | 0.02 conversions per treated user |

10 cells × 200 replications. Estimators: `naive`, `response`, `s_learner`, `t_learner`,
`x_learner`, `dr_learner`, `ipw`, and `causal_forest` if the optional extra is installed (a
reference arm; no hypothesis depends on it). Seeds: `SeedSequence([20260830, cell_index, rep])`.

Monte-Carlo SE (MC SE) below means the standard deviation across the 200 replications divided by
sqrt(200). "Own SE" means the mean, across replications, of the standard error the estimator
reports for itself — the uncertainty an analyst would actually see.

## Hypotheses and how each is decided

**H1 — uplift models beat the response model when there is no confounding.**
In the cell `gamma = 0, hidden = 0`, for each of `x_learner` and `dr_learner` separately: the mean
paired difference `qini_true(learner) − qini_true(response)` across replications is greater than
2 × its MC SE. Reported per learner: supported / not supported.

**H2 — observed confounding breaks the naive estimate but not the adjusted ones.**
`gamma* = 0.5`. In every `hidden = 0` cell with `gamma >= gamma*`:
- naive is **biased**: |mean bias| > 2 × naive's own SE;
- `ipw` and `dr_learner` (AIPW) are **usable**: |mean bias| <= 1 × their own SE, and 95% interval
  coverage >= 0.90.
H2 is supported if all three conditions hold in all three cells (gamma 0.5, 1.0, 2.0). Each
condition is reported per cell regardless.

**H3 — an expected failure: no observed-covariate method removes hidden confounding.**
In every `hidden > 0` cell, `ipw` and `dr_learner` are **biased**: |mean bias| > 2 × own SE.
Reported as "supported" when they fail, because the failure is the prediction.

## Breaking points (descriptive, no hypothesis)

For each estimator, the smallest grid `gamma` (hidden = 0) at which it stops being usable:
- ATE estimators: the usable rule from H2 (|mean bias| <= own SE and coverage >= 0.90). Naive has
  an SE, so it is judged the same way. S/T/X-learner ATEs report no SE and are judged on
  |mean bias| <= 0.01 (one percentage point) instead.
- Ranking: mean `qini_true` no longer greater than the response model's by 2 × MC SE of the paired
  difference.

## What a null looks like

If every learner's mean `qini_true` is within 2 MC SE of zero, or the response model ties the
uplift learners at `gamma = 0`, the README reports "uplift modelling did not pay here". If the
adjusted estimators are biased at observed confounding, H2 is reported as not supported, with the
cells.

## What is not decided by this file

The real-data check (Criteo, randomised, Qini from outcomes only) is a sanity check with no
hypothesis. The DiD module has its own unit-tested expectations and no hypothesis here.

## Amendments

_None yet._
