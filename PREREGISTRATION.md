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
`gamma* = 0.5`. For **every** `hidden = 0` cell with `gamma >= gamma*` — that is, each of the three
cells `gamma = 0.5`, `gamma = 1.0` and `gamma = 2.0` — all of the following must hold:
- naive is **biased**: |mean bias| > 2 × naive's own SE;
- `ipw` is **usable**: |mean bias| <= 1 × its own SE, and 95% interval coverage >= 0.90;
- `dr_learner` (AIPW) is **usable**: |mean bias| <= 1 × its own SE, and 95% interval coverage
  >= 0.90.
H2 is supported only if every condition holds in every one of the three cells; a failure in any
single cell means not supported. Each condition is reported per cell regardless.

**H3 — an expected failure: no observed-covariate method removes hidden confounding.**
In **each** of the four hidden-confounding cells — `(gamma, hidden)` = (0.5, 0.5), (0.5, 1.0),
(1.0, 0.5) and (1.0, 1.0) — both `ipw` and `dr_learner` are **biased**: |mean bias| > 2 × own SE.
H3 is supported only if both estimators are biased in all four cells. Reported as "supported" when
they fail, because the failure is the prediction; each cell is reported regardless.

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

**A0 — 2026-10-23, before any run (smoke or full).** Wording only, on the owner's review: H2 now
names its three cells and says every condition must hold in every cell; H3 names its four cells
and requires both estimators to fail in all of them. No threshold, cell or estimator changed. The
"own SE" yardstick is kept deliberately: with R = 200 the Monte-Carlo SE of the mean bias is about
1/14 of the per-replication spread, so "|bias| <= 1 × MC SE" would fail an unbiased estimator about
a third of the time by chance and any small finite-sample bias always. MC SEs are still reported
next to every mean.

**A1 — 2026-10-23, before the full grid.** An independent code review ran a small pilot
(R = 40, n = 10,000, LightGBM base, outside the committed results). It found no formula error,
but IPW with a LightGBM propensity over-corrected at gamma 1.0 and 2.0 (mean bias -0.015 and
-0.023 against own SEs of 0.012 and 0.020, coverage 0.83 and 0.80), while IPW with a logistic
or the true propensity did not. Because that pilot saw cells H2 judges, **nothing about H2
changes**: `ipw` keeps the LightGBM propensity and is judged exactly as registered. Two
additions, both outside every hypothesis:
1. An exploratory arm `ipw_logit` (the same Hajek IPW with a logistic-regression propensity) runs
   in every cell. It is reported next to `ipw` and labelled exploratory; no verdict reads it.
2. The main grid runs from the default install, without the optional EconML extra, so
   `causal_forest` is not in it (one forest fit at n = 10,000 would roughly double the grid's
   run time). The causal forest gets a separate, smaller reference run (cells gamma 0 and 1.0,
   hidden 0, R = 20), reported apart from the grid, with no hypothesis.

The review also noted that IPW's SE treats the propensity as known, which makes it conservative
(wider). That is left as implemented and stated in the README.
