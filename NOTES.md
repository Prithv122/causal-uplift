# Notes — Causal Uplift

What broke, what was tried, and why one option won. Newest at the bottom of each section.

## Decisions

- **Own learners, EconML only as a reference.** S/T/X/DR are short enough to write and test here,
  and an interviewer can ask about any line. EconML 0.17.0 installs cleanly next to numpy 2.5,
  pandas 3.1 and scikit-learn 1.9.1 (it caps scikit-learn below 1.10), so it is an optional extra
  (`forest`) rather than a core dependency.
- **Metrics on a fresh evaluation draw.** Each replication fits on 10,000 users and is scored on
  20,000 new users from the same settings. Scoring on the training users would reward a learner
  that memorises noise.
- **Qini against the true effect.** With confounded assignment the usual outcome-based Qini is
  itself biased, so the main ranking metric uses the simulator's `tau`. The outcome-based Qini is
  kept for the randomised Criteo data, where it is valid.
- **ATE estimand.** ATEs are compared with the mean true effect of the *training* users, since
  that is the sample every ATE estimator averages over.
- **Seeds.** `SeedSequence([20260830, cell, rep])` for the two draws; each estimator's seed adds
  a CRC32 of its name, so adding or removing an arm does not change the others' numbers. Seeds are
  masked to 31 bits because LightGBM reduces larger seeds modulo 2^31.
- **"Own SE", not Monte-Carlo SE, for the usable rule** (pre-registration A0). With 200
  replications the MC SE of a mean bias is about 1/14 of the per-run spread, so a ±1 MC SE rule
  would fail an unbiased estimator a third of the time.

## What broke

- `ProcessPoolExecutor` failed intermittently on Windows under pytest with `WinError 6: handle is
  invalid`. A `multiprocessing` spawn pool with `imap_unordered` passed every run; results are
  keyed by (cell, rep), so the scheduling order does not matter.
- A pilot in the code review (R = 40) showed IPW with a LightGBM propensity over-correcting at
  gamma ≥ 1: the out-of-fold propensity was calibrated (logit slope 1.02) but noisy (correlation
  0.925 with the truth), and noise in the logit spreads the 1/e weights. The same IPW with a
  logistic or the true propensity was fine, and AIPW was fine with LightGBM. Because the pilot saw
  cells that H2 judges, H2 was left exactly as registered; an exploratory `ipw_logit` arm was
  added (amendment A1), outside every hypothesis.
- The first test suite let 17 of 19 deliberate formula mutations pass (50/50 test data made the
  X-learner's propensity weight ≈ 0.5, correctly specified models hid weighting errors, SE tests
  were one-sided). New tests rebuild each estimator independently, check clipping with a stub
  model and check SE calibration over 50 replications; all 25 mutations tried (the 19 plus 6
  more) now fail at least one test. They were applied by hand, one at a time, to a copy of the
  repo: X-learner blend swapped or g fixed at 0.5; D1 built from mu1; tau1 fitted on the wrong
  arm; DR control term divided by e; DR nuisances fitted in-sample; DR final stage fitted on
  mu1 - mu0; DR SE without /sqrt(n) or with ddof = 0; DR phi using mu0 in the treated residual;
  IPW influence function uncentred, SE x 3, Horvitz-Thompson instead of Hajek, no cross-fitting;
  S-learner sign flipped; T-learner arms swapped; propensity clipping removed (DR, IPW, X);
  simulator segment lifts changed, x3 term removed, u/eps draw order swapped, propensity
  ignoring u, baseline x0 coefficient changed.

## Known limits

- IPW's SE treats the estimated propensity as known, so it is conservative (too wide).
- At gamma = 2, 2.2% of users have a true propensity outside the [0.01, 0.99] clip; clipping
  alone moves Hajek IPW by about +0.004 on 4M users.
