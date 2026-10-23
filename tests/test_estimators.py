"""Estimator tests on small synthetic data built here (independent of the simulator module)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import spearmanr

from causal_uplift.estimators import (
    ESTIMATORS,
    IPW,
    DRLearner,
    Naive,
    ResponseModel,
    TLearner,
    XLearner,
    available_estimators,
)


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
