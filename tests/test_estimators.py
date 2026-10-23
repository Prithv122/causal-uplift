"""Estimator tests on small synthetic data built here (independent of the simulator module)."""

from __future__ import annotations

import functools
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from causal_uplift.estimators import (
    ESTIMATORS,
    IPW,
    DRLearner,
    Naive,
    ResponseModel,
    SLearner,
    TLearner,
    XLearner,
    available_estimators,
)


@pytest.fixture(autouse=True)
def single_threaded_blas():
    """Small fits run faster, and steadier on a busy machine, without BLAS thread fan-out."""
    with threadpool_limits(limits=1):
        yield


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_data(n: int, seed: int, *, confounded: bool = False, gated: bool = False):
    """Users with known tau.

    Baseline logit ``-0.5 + x0``; the offer adds ``x1`` to the logit. ``confounded`` treats
    users with propensity ``sigmoid(1.5 x0)``; ``gated`` makes the effect zero for users with
    high baseline (``x0 > 0``) and positive for the rest when ``x1 > 0``.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    e = sigmoid(1.5 * X[:, 0]) if confounded else np.full(n, 0.5)
    t = rng.binomial(1, e).astype(np.int8)
    lift = 1.5 * (X[:, 1] > 0) * (X[:, 0] < 0) if gated else X[:, 1]
    p0 = sigmoid(-0.5 + X[:, 0])
    p1 = sigmoid(-0.5 + X[:, 0] + lift)
    y = rng.binomial(1, np.where(t == 1, p1, p0)).astype(np.int8)
    return SimpleNamespace(X=X, t=t, y=y, tau=p1 - p0)


def test_interface_for_every_estimator_and_base():
    d = make_data(600, seed=1)
    for base in ("logistic", "lgbm"):
        for key, factory in ESTIMATORS.items():
            est = factory(3, base).fit(d.X, d.t, d.y)
            tau_hat = est.predict_tau(d.X)
            assert est.name == key
            assert tau_hat.shape == (600,)
            assert np.isfinite(tau_hat).all()
            if key == "response":
                assert est.ate() is None
                assert not est.estimates_tau
            else:
                assert est.estimates_tau
                assert np.isfinite(est.ate().estimate)


def test_ate_se_present_only_where_defined():
    d = make_data(600, seed=2)
    for key in ("naive", "dr_learner", "ipw"):
        assert ESTIMATORS[key](0, "logistic").fit(d.X, d.t, d.y).ate().se > 0
    for key in ("s_learner", "t_learner", "x_learner"):
        assert np.isnan(ESTIMATORS[key](0, "logistic").fit(d.X, d.t, d.y).ate().se)


@pytest.mark.parametrize("key", list(ESTIMATORS))
def test_predict_before_fit_raises(key):
    est = ESTIMATORS[key](0, "logistic")
    with pytest.raises(RuntimeError, match="not fitted"):
        est.predict_tau(np.zeros((3, 4)))
    if key != "response":
        with pytest.raises(RuntimeError, match="not fitted"):
            est.ate()


@pytest.mark.parametrize("key", list(ESTIMATORS))
def test_empty_arm_raises(key):
    d = make_data(200, seed=3)
    with pytest.raises(ValueError, match="arm 1"):
        ESTIMATORS[key](0, "logistic").fit(d.X, np.zeros(200, dtype=np.int8), d.y)


@pytest.mark.parametrize("key", list(ESTIMATORS))
def test_same_seed_gives_identical_predictions(key):
    d = make_data(1500, seed=4)
    a = ESTIMATORS[key](7, "lgbm").fit(d.X, d.t, d.y).predict_tau(d.X)
    b = ESTIMATORS[key](7, "lgbm").fit(d.X, d.t, d.y).predict_tau(d.X)
    np.testing.assert_array_equal(a, b)


def test_naive_matches_welch_by_hand_and_recovers_randomised_ate():
    d = make_data(20_000, seed=5)
    res = Naive().fit(d.X, d.t, d.y).ate()
    y1, y0 = d.y[d.t == 1].astype(float), d.y[d.t == 0].astype(float)
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    assert res.estimate == pytest.approx(y1.mean() - y0.mean())
    assert res.se == pytest.approx(se)
    assert res.lo == pytest.approx(res.estimate - 1.959964 * se)
    assert res.hi == pytest.approx(res.estimate + 1.959964 * se)
    assert abs(res.estimate - d.tau.mean()) < 4 * res.se


def test_naive_predict_tau_is_constant():
    d = make_data(500, seed=6)
    tau_hat = Naive().fit(d.X, d.t, d.y).predict_tau(d.X[:7])
    assert tau_hat.shape == (7,)
    assert np.ptp(tau_hat) == 0


def test_t_learner_with_linear_models_ranks_a_logit_linear_effect():
    train, test = make_data(20_000, seed=7), make_data(5_000, seed=8)
    tau_hat = TLearner(0, "logistic").fit(train.X, train.t, train.y).predict_tau(test.X)
    assert spearmanr(tau_hat, test.tau).statistic > 0.9


def test_confounded_data_biases_naive_but_not_ipw_or_dr():
    d = make_data(20_000, seed=9, confounded=True)
    truth = d.tau.mean()
    naive = Naive().fit(d.X, d.t, d.y).ate()
    ipw = IPW(0, "logistic").fit(d.X, d.t, d.y).ate()
    dr = DRLearner(0, "logistic").fit(d.X, d.t, d.y).ate()
    print(
        f"bias in own SEs: naive {(naive.estimate - truth) / naive.se:.1f}, "
        f"ipw {(ipw.estimate - truth) / ipw.se:.1f}, dr {(dr.estimate - truth) / dr.se:.1f}"
    )
    assert naive.estimate - truth > 3 * naive.se
    assert abs(ipw.estimate - truth) < 3 * ipw.se
    assert abs(dr.estimate - truth) < 3 * dr.se


def test_dr_ate_interval_and_pseudo_outcome_se():
    d = make_data(3_000, seed=10, confounded=True)
    dr = DRLearner(0, "logistic").fit(d.X, d.t, d.y)
    res = dr.ate()
    assert res.se > 0
    assert res.lo < res.estimate < res.hi
    assert res.hi - res.lo == pytest.approx(2 * 1.959964 * res.se)


def test_ipw_predict_tau_is_the_constant_ate():
    d = make_data(1_000, seed=11)
    ipw = IPW(0, "logistic").fit(d.X, d.t, d.y)
    tau_hat = ipw.predict_tau(d.X[:9])
    np.testing.assert_allclose(tau_hat, ipw.ate().estimate)


def test_ipw_equals_hajek_difference_with_true_weights_when_propensity_is_flat():
    # Randomised 50/50 data: a fitted propensity is close to 0.5, so IPW ~ difference in means.
    d = make_data(20_000, seed=12)
    ipw = IPW(0, "logistic").fit(d.X, d.t, d.y).ate()
    naive = Naive().fit(d.X, d.t, d.y).ate()
    assert abs(ipw.estimate - naive.estimate) < 0.5 * naive.se


def test_effect_aware_learners_beat_the_response_model_when_high_baseline_users_gain_nothing():
    train, test = make_data(8_000, seed=13, gated=True), make_data(5_000, seed=14, gated=True)
    scores = {}
    for key in ("response", "x_learner", "dr_learner"):
        est = ESTIMATORS[key](0, "lgbm").fit(train.X, train.t, train.y)
        scores[key] = spearmanr(est.predict_tau(test.X), test.tau).statistic
    print("spearman with true tau:", scores)
    assert scores["x_learner"] > scores["response"] + 0.2
    assert scores["dr_learner"] > scores["response"] + 0.2


def test_x_learner_is_constructed_with_distinct_child_seeds():
    d = make_data(2_000, seed=15)
    a = XLearner(0, "lgbm").fit(d.X, d.t, d.y).predict_tau(d.X)
    b = XLearner(1, "lgbm").fit(d.X, d.t, d.y).predict_tau(d.X)
    assert not np.array_equal(a, b)


def test_response_model_score_is_a_probability():
    d = make_data(2_000, seed=16)
    score = ResponseModel(0, "logistic").fit(d.X, d.t, d.y).predict_tau(d.X)
    assert ((score >= 0) & (score <= 1)).all()


def test_available_estimators_works_without_econml_arm(monkeypatch):
    monkeypatch.setitem(sys.modules, "causal_uplift.estimators.econml_arm", None)
    assert available_estimators().keys() == ESTIMATORS.keys()
    assert available_estimators() is not ESTIMATORS


def test_available_estimators_adds_causal_forest_when_available(monkeypatch):
    fake = SimpleNamespace(is_available=lambda: True, CausalForest=object)
    monkeypatch.setitem(sys.modules, "causal_uplift.estimators.econml_arm", fake)
    assert available_estimators()["causal_forest"] is object
    fake.is_available = lambda: False
    assert "causal_forest" not in available_estimators()


# --- Independent re-computation of every formula ---------------------------------------------
#
# Each reference below rebuilds an estimator from the DESIGN.md formulas using only scikit-learn
# and numpy (same folds, same clipping), so a changed formula in the package cannot also change
# the reference. ``clf(X, target)`` returns a function ``X -> P(target = 1)``.

Z95 = 1.959964
CLIP_LO, CLIP_HI = 0.01, 0.99


def logistic_clf(X, target):
    model = LogisticRegression(max_iter=1000).fit(X, target)
    return lambda Xnew: model.predict_proba(Xnew)[:, 1]


def make_weighted_data(
    n: int,
    seed: int,
    *,
    gamma: float = 1.0,
    intercept: float = 0.0,
    curvature: float = 0.0,
    shift: float = 0.0,
    slope: float = 1.0,
):
    """Confounded users with known tau.

    Treatment propensity ``sigmoid(intercept + gamma x0)``; baseline logit
    ``-0.5 + x0 + curvature (x0^2 - 1)`` (a logistic outcome model is misspecified when
    ``curvature != 0``); the offer adds ``shift + slope x1`` to the logit.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    e = sigmoid(intercept + gamma * X[:, 0])
    t = rng.binomial(1, e).astype(np.int8)
    base = -0.5 + X[:, 0] + curvature * (X[:, 0] ** 2 - 1)
    p0, p1 = sigmoid(base), sigmoid(base + shift + slope * X[:, 1])
    y = rng.binomial(1, np.where(t == 1, p1, p0)).astype(np.int8)
    return SimpleNamespace(X=X, t=t, y=y, tau=p1 - p0, e=e)


@functools.cache
def _population_ate(knobs: tuple) -> float:
    return float(make_weighted_data(2_000_000, 987_654, **dict(knobs)).tau.mean())


def population_ate(**knobs) -> float:
    """E[tau] of ``make_weighted_data`` from a very large draw (tau needs no outcomes)."""
    return _population_ate(tuple(sorted(knobs.items())))


def folds(t, seed):
    return StratifiedKFold(n_splits=2, shuffle=True, random_state=seed).split(np.zeros(len(t)), t)


def dr_reference(X, t, y, seed, clf=logistic_clf, *, in_sample=False):
    n = len(X)
    mu0, mu1, e = (np.empty(n) for _ in range(3))
    splits = [(np.arange(n), np.arange(n))] if in_sample else folds(t, seed)
    for fit, pred in splits:
        Xf, tf, yf = X[fit], t[fit], y[fit]
        mu0[pred] = clf(Xf[tf == 0], yf[tf == 0])(X[pred])
        mu1[pred] = clf(Xf[tf == 1], yf[tf == 1])(X[pred])
        e[pred] = clf(Xf, tf)(X[pred])
    raw = e
    e = np.clip(e, CLIP_LO, CLIP_HI)
    phi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    return SimpleNamespace(phi=phi, mu0=mu0, mu1=mu1, e=e, e_raw=raw)


def ipw_reference(X, t, y, seed, clf=logistic_clf, *, in_sample=False):
    n = len(X)
    e = np.empty(n)
    splits = [(np.arange(n), np.arange(n))] if in_sample else folds(t, seed)
    for fit, pred in splits:
        e[pred] = clf(X[fit], t[fit])(X[pred])
    raw = e
    e = np.clip(e, CLIP_LO, CLIP_HI)
    w1, w0 = t / e, (1 - t) / (1 - e)
    m1, m0 = (w1 * y).sum() / w1.sum(), (w0 * y).sum() / w0.sum()
    psi = w1 * (y - m1) / w1.mean() - w0 * (y - m0) / w0.mean()
    return SimpleNamespace(
        estimate=m1 - m0,
        se=psi.std(ddof=1) / np.sqrt(n),
        e=e,
        e_raw=raw,
        horvitz_thompson=(w1 * y).mean() - (w0 * y).mean(),
        uncentred_se=(w1 * y / w1.mean() - w0 * y / w0.mean()).std(ddof=1) / np.sqrt(n),
    )


def x_learner_reference(X, t, y, Xnew, clf=logistic_clf):
    treated, control = t == 1, t == 0
    mu0 = clf(X[control], y[control])
    mu1 = clf(X[treated], y[treated])
    d1 = y[treated] - mu0(X[treated])
    d0 = mu1(X[control]) - y[control]
    tau1 = LinearRegression().fit(X[treated], d1).predict(Xnew)
    tau0 = LinearRegression().fit(X[control], d0).predict(Xnew)
    g = np.clip(clf(X, t)(Xnew), CLIP_LO, CLIP_HI)
    return SimpleNamespace(tau=g * tau0 + (1 - g) * tau1, tau0=tau0, tau1=tau1, g=g)


class ExtremeProba:
    """A classifier stub that ignores its training data and returns extreme probabilities."""

    def fit(self, X, target):
        return self

    @staticmethod
    def p(X):
        return np.where(X[:, 0] < -0.5, 1e-4, np.where(X[:, 0] > 0.5, 1 - 1e-4, 0.5))

    def predict_proba(self, X):
        p = self.p(X)
        return np.column_stack([1 - p, p])


def stub_clf(X, target):
    return ExtremeProba().p


def rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def test_x_learner_matches_the_documented_construction_under_unbalanced_treatment():
    # Treated share ~ 0.2 and varying with x0: g is far from 0.5, so the blend direction matters.
    d = make_weighted_data(6_000, 20, gamma=1.5, intercept=-1.5, curvature=0.5)
    test = make_weighted_data(2_000, 21, gamma=1.5, intercept=-1.5, curvature=0.5)
    assert 0.1 < d.t.mean() < 0.35
    ref = x_learner_reference(d.X, d.t, d.y, test.X)
    est = XLearner(0, "logistic").fit(d.X, d.t, d.y)
    np.testing.assert_allclose(est.predict_tau(test.X), ref.tau, rtol=0, atol=1e-10)
    train_ref = x_learner_reference(d.X, d.t, d.y, d.X)
    assert est.ate().estimate == pytest.approx(train_ref.tau.mean(), abs=1e-10)
    # Power check: the plausible mutations of this formula move the answer far above the tolerance.
    swapped = (1 - ref.g) * ref.tau0 + ref.g * ref.tau1
    half = 0.5 * ref.tau0 + 0.5 * ref.tau1
    assert rms(swapped, ref.tau) > 1e-3
    assert rms(half, ref.tau) > 1e-3


def test_x_learner_imputes_treated_effects_with_the_control_model():
    # D1 = y - mu0(x) on treated users. Built from mu1 instead it would be a residual with
    # mean ~0, and the effect estimate would collapse towards zero.
    d = make_weighted_data(8_000, 22, gamma=1.0, intercept=-1.5, shift=1.0, slope=0.0)
    est = XLearner(0, "logistic").fit(d.X, d.t, d.y)
    assert est.predict_tau(d.X).mean() == pytest.approx(d.tau.mean(), abs=0.03)
    assert est.predict_tau(d.X).mean() > 0.1


def test_t_and_s_learners_match_the_documented_construction():
    d = make_weighted_data(5_000, 23, gamma=0.5, shift=1.0, slope=0.5)
    test = make_weighted_data(1_500, 24, gamma=0.5, shift=1.0, slope=0.5)
    mu0 = logistic_clf(d.X[d.t == 0], d.y[d.t == 0])
    mu1 = logistic_clf(d.X[d.t == 1], d.y[d.t == 1])
    t_hat = TLearner(0, "logistic").fit(d.X, d.t, d.y).predict_tau(test.X)
    np.testing.assert_allclose(t_hat, mu1(test.X) - mu0(test.X), rtol=0, atol=1e-10)
    joint = logistic_clf(np.column_stack([d.X, d.t]), d.y)
    ones, zeros = np.ones((len(test.X), 1)), np.zeros((len(test.X), 1))
    s_ref = joint(np.hstack([test.X, ones])) - joint(np.hstack([test.X, zeros]))
    s_hat = SLearner(0, "logistic").fit(d.X, d.t, d.y).predict_tau(test.X)
    np.testing.assert_allclose(s_hat, s_ref, rtol=0, atol=1e-10)


def test_s_learner_has_the_right_sign_and_size_for_a_positive_constant_logit_lift():
    # The offer adds +1.2 to every user's logit, so every true tau is positive and a model on
    # [X, t] is correctly specified.
    train = make_weighted_data(30_000, 25, gamma=0.0, shift=1.2, slope=0.0)
    test = make_weighted_data(5_000, 26, gamma=0.0, shift=1.2, slope=0.0)
    assert (test.tau > 0).all()
    est = SLearner(0, "logistic").fit(train.X, train.t, train.y)
    tau_hat = est.predict_tau(test.X)
    assert (tau_hat > 0).mean() > 0.99
    assert rms(tau_hat, test.tau) < 0.03
    assert est.ate().estimate == pytest.approx(train.tau.mean(), abs=0.02)


def test_dr_learner_matches_the_documented_cross_fitted_construction():
    d = make_weighted_data(4_000, 27, gamma=1.0, intercept=-0.5, curvature=0.8)
    seed = 5
    ref = dr_reference(d.X, d.t, d.y, seed)
    dr = DRLearner(seed, "logistic").fit(d.X, d.t, d.y)
    res = dr.ate()
    assert res.estimate == pytest.approx(ref.phi.mean(), rel=0, abs=1e-10)
    assert res.se == pytest.approx(ref.phi.std(ddof=1) / np.sqrt(len(d.X)), rel=1e-10)
    assert res.lo == pytest.approx(res.estimate - Z95 * res.se, abs=1e-10)
    # The final stage regresses phi (not the outcome-model contrast) on X.
    final = LinearRegression().fit(d.X, ref.phi)
    np.testing.assert_allclose(dr.predict_tau(d.X[:300]), final.predict(d.X[:300]), atol=1e-9)
    # Power checks: each plausible mutation moves a checked quantity by far more than 1e-10.
    in_sample = dr_reference(d.X, d.t, d.y, seed, in_sample=True)
    assert abs(in_sample.phi.mean() - ref.phi.mean()) > 1e-5
    assert rms(in_sample.mu1, ref.mu1) > 1e-4
    wrong_control = (
        ref.mu1 - ref.mu0 + d.t * (d.y - ref.mu1) / ref.e - (1 - d.t) * (d.y - ref.mu0) / ref.e
    )
    assert abs(wrong_control.mean() - ref.phi.mean()) > 1e-4
    contrast = LinearRegression().fit(d.X, ref.mu1 - ref.mu0).predict(d.X[:300])
    assert rms(contrast, final.predict(d.X[:300])) > 1e-4


def test_dr_learner_nuisances_are_cross_fitted_not_fitted_in_sample():
    # Fitted and held-out predictions differ, so an in-sample fit gives a different phi.
    d = make_weighted_data(600, 28, gamma=1.0, curvature=0.8)
    out_of_fold = dr_reference(d.X, d.t, d.y, 3)
    in_sample = dr_reference(d.X, d.t, d.y, 3, in_sample=True)
    assert rms(out_of_fold.e, in_sample.e) > 1e-4
    est = DRLearner(3, "logistic").fit(d.X, d.t, d.y).ate().estimate
    assert est == pytest.approx(out_of_fold.phi.mean(), abs=1e-10)
    assert abs(est - in_sample.phi.mean()) > 1e-6


def test_ipw_matches_the_documented_hajek_cross_fitted_construction():
    d = make_weighted_data(4_000, 29, gamma=1.0, intercept=-0.5)
    seed = 4
    ref = ipw_reference(d.X, d.t, d.y, seed)
    res = IPW(seed, "logistic").fit(d.X, d.t, d.y).ate()
    assert res.estimate == pytest.approx(ref.estimate, rel=0, abs=1e-10)
    assert res.se == pytest.approx(ref.se, rel=1e-10)
    assert res.hi - res.lo == pytest.approx(2 * Z95 * ref.se, rel=1e-10)
    # Power checks: Horvitz-Thompson, an uncentred influence function and an in-sample
    # propensity each land measurably away from the Hajek / cross-fitted numbers.
    assert abs(ref.horvitz_thompson - ref.estimate) > 1e-5
    assert abs(ref.uncentred_se - ref.se) > 1e-7
    in_sample = ipw_reference(d.X, d.t, d.y, seed, in_sample=True)
    assert abs(in_sample.estimate - ref.estimate) > 1e-6
    assert rms(in_sample.e, ref.e) > 1e-4


def test_ipw_estimate_is_unbiased_with_a_misspecified_outcome_surface():
    # IPW never uses an outcome model: with the propensity right it is unbiased no matter how
    # curved the outcome surface is. Reversed or mis-normalised weights would not be.
    knobs = {"gamma": 1.5, "intercept": -0.5, "curvature": 1.0}
    truth = population_ate(**knobs)
    d = make_weighted_data(30_000, 30, **knobs)
    res = IPW(0, "logistic").fit(d.X, d.t, d.y).ate()
    assert abs(res.estimate - truth) < 3.5 * res.se


def test_dr_estimate_is_unbiased_with_a_misspecified_outcome_surface():
    # The logistic outcome models are wrong (the baseline is curved in x0), so the estimate rests
    # on the propensity weights: a control term divided by e instead of 1 - e is off by ~0.7.
    knobs = {"gamma": 1.5, "intercept": -0.5, "curvature": 1.0}
    truth = population_ate(**knobs)
    d = make_weighted_data(30_000, 31, **knobs)
    dr = DRLearner(0, "logistic").fit(d.X, d.t, d.y).ate()
    assert abs(dr.estimate - truth) < 3.5 * dr.se


def test_aipw_and_ipw_standard_errors_are_calibrated_across_replications():
    # 50 independent confounded samples, curved outcome surface, logistic nuisances. The mean
    # reported SE must match the Monte Carlo sd of the estimates, and the 95% interval must cover.
    knobs = {"gamma": 1.0, "curvature": 1.0}
    truth = population_ate(**knobs)
    reps, n = 50, 3_000
    out = {"dr_learner": [], "ipw": []}
    for rep in range(reps):
        d = make_weighted_data(n, 1_000 + rep, **knobs)
        for name, cls in (("dr_learner", DRLearner), ("ipw", IPW)):
            r = cls(rep, "logistic").fit(d.X, d.t, d.y).ate()
            out[name].append((r.estimate, r.se, r.lo <= truth <= r.hi))
    for name, rows in out.items():
        est, se, hit = (np.array(col) for col in zip(*rows, strict=True))
        ratio = se.mean() / est.std(ddof=1)
        print(
            f"{name}: coverage {hit.mean():.2f}, SE/MC sd {ratio:.2f}, "
            f"bias {est.mean() - truth:+.4f}"
        )
        assert 0.7 < ratio < 1.6, name
        assert hit.mean() >= 0.86, name
        assert abs(est.mean() - truth) < 3.5 * est.std(ddof=1) / np.sqrt(reps), name


@pytest.mark.parametrize("name", ["dr_learner", "ipw", "x_learner"])
def test_extreme_propensities_are_clipped_before_dividing(name, monkeypatch):
    # The stub propensity is 1e-4 / 0.9999 for a large share of users. Unclipped, weights of
    # 1e4 would dominate; clipped to [0.01, 0.99] the weights are at most 100.
    from causal_uplift.estimators import dr, metalearners

    monkeypatch.setattr(dr, "make_classifier", lambda kind, seed: ExtremeProba())
    monkeypatch.setattr(metalearners, "make_classifier", lambda kind, seed: ExtremeProba())
    d = make_weighted_data(3_000, 32, gamma=0.5)
    p = ExtremeProba.p(d.X)
    assert ((p < CLIP_LO) | (p > CLIP_HI)).mean() > 0.4

    if name == "x_learner":
        est = XLearner(0, "logistic").fit(d.X, d.t, d.y)
        ref = x_learner_reference(d.X, d.t, d.y, d.X, stub_clf)
        np.testing.assert_allclose(est.predict_tau(d.X), ref.tau, rtol=0, atol=1e-10)
        assert ref.g.min() == CLIP_LO and ref.g.max() == CLIP_HI
        return

    if name == "ipw":
        ref = ipw_reference(d.X, d.t, d.y, 0, stub_clf)
        res = IPW(0, "logistic").fit(d.X, d.t, d.y).ate()
        assert res.estimate == pytest.approx(ref.estimate, rel=0, abs=1e-10)
        assert res.se == pytest.approx(ref.se, rel=1e-10)
    else:
        ref = dr_reference(d.X, d.t, d.y, 0, stub_clf)
        res = DRLearner(0, "logistic").fit(d.X, d.t, d.y).ate()
        assert res.estimate == pytest.approx(ref.phi.mean(), rel=0, abs=1e-10)
        assert res.se == pytest.approx(ref.phi.std(ddof=1) / np.sqrt(len(d.X)), rel=1e-10)
        assert np.abs(ref.phi).max() <= 1 + 1 / CLIP_LO + 1 / (1 - CLIP_HI)
    assert ref.e.min() == CLIP_LO and ref.e.max() == CLIP_HI
    assert (1 / ref.e).max() <= 100 + 1e-9 and (1 / (1 - ref.e)).max() <= 100 + 1e-9
