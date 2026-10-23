# Resume Bullets — Causal Uplift

Form: **action → technical specifics → measured outcome.** Numbers or it doesn't go on the resume.

---

## Bullets

- Benchmarked 8 causal estimators (own S/T/X/DR-learners, IPW, AIPW) against exact per-user truth
  from a seeded simulator across 10 confounding settings × 200 replications, under pre-registered
  rules; uplift learners ranked users at 0.80–0.94 of oracle Qini vs 0.48 for a response model,
  and AIPW stayed within 0.0011 of the true effect while a naive comparison reported up to 3.6×
  the true effect.
- Diagnosed a pre-registered failure: IPW with a LightGBM propensity model lost coverage (0.86) at
  moderate confounding while the same weighting with a logistic propensity stayed within 0.005,
  and reported 2 of 3 hypotheses as not supported rather than moving the thresholds.
- Hardened the test suite with independent re-implementations and SE-calibration checks so that all
  25 injected formula mutations (pseudo-outcome, weighting, cross-fitting, clipping) fail a test.

## Which roles this supports

- [x] Data Scientist / ML
- [ ] AI Engineer (LLM/NLP/CV)
- [ ] Data Engineer
- [x] Data Analyst / Python Developer

## Keywords this project earns

Causal inference, uplift modelling, meta-learners (S/T/X), DR-learner, AIPW, inverse propensity
weighting, propensity scores, confounding, Qini/AUUC, PEHE, difference-in-differences, Monte-Carlo
simulation, pre-registration, cross-fitting, LightGBM, scikit-learn, EconML (reference arm).
