# Outcome-only real-data sanity check (Criteo Uplift, randomised)

- file: criteo-research-uplift-v2.1.csv.gz
- sha256: 2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc
- licence: CC BY-NC-SA 4.0
- subsample: 1000000 rows, seed 20260830
- splits: 5 seeded 50/50 splits, stratified on treatment
- base learner: lgbm
- metric: qini_observed on the held-out half; sd is across splits; beats random counts splits where the estimator's value exceeds the random baseline's on the same split

## visit

| estimator | mean qini_observed | sd across splits | splits beating random |
|---|---|---|---|
| random | 0.000005 | 0.000199 | - |
| response | 0.003023 | 0.000127 | 5/5 |
| t_learner | 0.002614 | 0.000134 | 5/5 |
| x_learner | 0.002603 | 0.000267 | 5/5 |
| dr_learner | 0.002698 | 0.000236 | 5/5 |

Test-half ATE (difference in means, mean over splits): 0.010237 (mean Welch SE 0.000775)

## conversion

| estimator | mean qini_observed | sd across splits | splits beating random |
|---|---|---|---|
| random | -0.000007 | 0.000042 | - |
| response | 0.000493 | 0.000048 | 5/5 |
| t_learner | 0.000363 | 0.000057 | 5/5 |
| x_learner | 0.000280 | 0.000082 | 5/5 |
| dr_learner | 0.000271 | 0.000025 | 5/5 |

Test-half ATE (difference in means, mean over splits): 0.001344 (mean Welch SE 0.000174)

No hypothesis reads this check; it does not affect H1–H3.
