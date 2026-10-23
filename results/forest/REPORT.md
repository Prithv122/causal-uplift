# Results report

- rows: 120
- cells: 2
- replications per cell: 20
- base learner: lgbm
- commit at run start: 7157116039e028371fe19d84ee32564a67c69ae1 (clean)
- run started: 2026-10-23T13:37:46.531137+00:00

## 1. Summary by cell and estimator

### g0.00_h0.00_n10000 (gamma=0, hidden=0)

| estimator | reps | ATE bias ± MC SE | own SE | coverage | PEHE | Spearman | Qini ± MC SE | policy top20 | oracle top20 | random top20 | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| causal_forest | 20 | -0.0010 ± 0.0015 | 0.0088 | 1.000 | 0.0700 | 0.884 | +0.920 ± 0.002 | 0.0580 | 0.0624 | 0.0118 | 28.75 |
| x_learner | 20 | -0.0007 ± 0.0014 | - | - | 0.0872 | 0.837 | +0.879 ± 0.004 | 0.0559 | 0.0624 | 0.0118 | 3.37 |
| dr_learner | 20 | -0.0024 ± 0.0016 | 0.0087 | 0.950 | 0.1181 | 0.762 | +0.803 ± 0.004 | 0.0518 | 0.0624 | 0.0118 | 3.67 |

### g1.00_h0.00_n10000 (gamma=1, hidden=0)

| estimator | reps | ATE bias ± MC SE | own SE | coverage | PEHE | Spearman | Qini ± MC SE | policy top20 | oracle top20 | random top20 | fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| causal_forest | 20 | -0.0026 ± 0.0026 | 0.0106 | 0.950 | 0.0770 | 0.867 | +0.901 ± 0.004 | 0.0563 | 0.0623 | 0.0117 | 26.74 |
| x_learner | 20 | +0.0018 ± 0.0022 | - | - | 0.0962 | 0.823 | +0.861 ± 0.005 | 0.0545 | 0.0623 | 0.0117 | 3.10 |
| dr_learner | 20 | -0.0009 ± 0.0025 | 0.0108 | 0.950 | 0.1582 | 0.700 | +0.741 ± 0.007 | 0.0471 | 0.0623 | 0.0117 | 3.50 |

## 2. Pre-registered verdicts (mechanical)

### H1 - uplift models vs the response model at gamma=0, hidden=0

| learner | mean paired diff | MC SE | 2 x MC SE | pairs | verdict |
|---|---|---|---|---|---|
| x_learner | - | - | - | - | not evaluable (missing: response in g0.00_h0.00_n10000) |
| dr_learner | - | - | - | - | not evaluable (missing: response in g0.00_h0.00_n10000) |

### H2 - observed confounding, hidden=0, gamma = 0.5, 1.0, 2.0

Every row must hold: naive biased; ipw and dr_learner usable.

Verdict: **not evaluable (missing: cell gamma=0.5, hidden=0; naive ATE with SE in g1.00_h0.00_n10000; ipw ATE with SE in g1.00_h0.00_n10000; cell gamma=2, hidden=0)**

### H3 - hidden confounding, (gamma, hidden) = (0.5, 0.5), (0.5, 1.0), (1.0, 0.5), (1.0, 1.0)

Every row must hold: ipw and dr_learner biased.

Verdict: **not evaluable (missing: cell gamma=0.5, hidden=0.5; cell gamma=0.5, hidden=1; cell gamma=1, hidden=0.5; cell gamma=1, hidden=1)**

## 3. Breaking points (descriptive)

Smallest gamma at which the estimator stops being usable (hidden = 0). ATE rule: |mean bias| <= own SE and coverage >= 0.90 (S/T/X-learner ATEs, which report no SE: |mean bias| <= 0.01). Ranking rule: mean qini_true no longer above the response model's by 2 x MC SE of the paired difference.

hidden = 0 gammas in these results: 0, 1

| estimator | ATE | ranking |
|---|---|---|
| causal_forest | never in grid | n/a (no ranking) |
| x_learner | never in grid | n/a (no ranking) |
| dr_learner | never in grid | n/a (no ranking) |
