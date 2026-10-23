import numpy as np
import pytest

from causal_uplift import metrics
from causal_uplift.dgp import simulate

TAU = np.array([0.4, 0.2, 0.0, -0.1])


def _numerator(score: np.ndarray, tau: np.ndarray) -> float:
    """A(score) - A(random), recomputed independently of the module's helpers."""
    curve = metrics.uplift_curve_true(score, tau)
    n = tau.size
    return float(np.trapezoid(curve / n, dx=1 / n) - tau.sum() / (2 * n))


def test_pehe_zero_for_perfect_estimate_and_hand_example():
    assert metrics.pehe(TAU, TAU) == 0.0
    assert metrics.pehe(np.array([1.0, 3.0]), np.array([0.0, 0.0])) == pytest.approx(np.sqrt(5.0))


def test_spearman_monotone_reversed_and_constant():
    assert metrics.spearman(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 35.0])) == 1.0
    assert metrics.spearman(np.array([3.0, 2.0, 1.0]), np.array([10.0, 20.0, 35.0])) == -1.0
    assert np.isnan(metrics.spearman(np.ones(3), np.array([1.0, 2.0, 3.0])))
    assert np.isnan(metrics.spearman(np.array([1.0, 2.0, 3.0]), np.ones(3)))


def test_uplift_curve_endpoints_and_hand_values():
    curve = metrics.uplift_curve_true(TAU, TAU)
    assert curve[0] == 0.0
    assert curve[-1] == pytest.approx(TAU.sum())
    np.testing.assert_allclose(curve, [0.0, 0.4, 0.6, 0.6, 0.5])


def test_uplift_curve_ties_share_the_group_mean():
    score = np.array([2.0, 1.0, 1.0, 0.0])
    tau = np.array([0.4, 0.3, -0.1, 0.0])
    np.testing.assert_allclose(metrics.uplift_curve_true(score, tau), [0.0, 0.4, 0.5, 0.6, 0.6])


def test_qini_true_oracle_is_one_and_reversed_is_minus_one():
    rng = np.random.default_rng(0)
    tau = rng.normal(0.05, 0.1, 500)
    assert metrics.qini_true(tau, tau) == pytest.approx(1.0)
    assert metrics.qini_true(-tau, tau) == pytest.approx(-1.0)


def test_qini_true_hand_example():
    tau = np.array([3.0, 1.0, 0.0, -1.0])
    score = np.array([3.0, 0.0, 1.0, -1.0])  # swaps the middle two users
    # V = 0, 3, 3, 4, 3: area/n = 0.71875, random 0.375, oracle 0.78125.
    assert metrics.qini_true(score, tau) == pytest.approx((0.71875 - 0.375) / (0.78125 - 0.375))


def test_qini_true_constant_score_is_zero():
    tau = np.random.default_rng(1).normal(0.05, 0.1, 1_000)
    assert metrics.qini_true(np.zeros(1_000), tau) == pytest.approx(0.0, abs=1e-12)


def test_qini_true_constant_tau_is_zero():
    assert metrics.qini_true(np.arange(10.0), np.full(10, 0.1)) == 0.0


def test_qini_true_random_scores_near_zero():
    rng = np.random.default_rng(2)
    tau = rng.normal(0.05, 0.1, 50_000)
    assert abs(metrics.qini_true(rng.normal(size=50_000), tau)) < 0.02


def test_oracle_ranking_beats_every_other_ranking():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        tau = rng.normal(0.0, 0.2, 200)
        noisy = tau + rng.normal(0.0, 0.1, 200)
        shuffled = rng.normal(size=200)
        oracle = metrics.qini_true(tau, tau)
        assert oracle >= metrics.qini_true(noisy, tau) - 1e-12
        assert oracle >= metrics.qini_true(shuffled, tau) - 1e-12
        assert metrics.qini_true(noisy, tau) > metrics.qini_true(-noisy, tau)


def test_all_tied_score_equals_the_random_line():
    tau = np.random.default_rng(3).normal(0.05, 0.1, 100)
    curve = metrics.uplift_curve_true(np.full(100, 7.0), tau)
    np.testing.assert_allclose(curve, np.linspace(0.0, tau.sum(), 101), atol=1e-12)


def test_tie_handling_is_order_invariant():
    rng = np.random.default_rng(4)
    score = rng.integers(0, 5, 300).astype(float)
    tau = rng.normal(size=300)
    perm = rng.permutation(300)
    np.testing.assert_allclose(
        metrics.uplift_curve_true(score, tau), metrics.uplift_curve_true(score[perm], tau[perm])
    )


def test_policy_value_hand_examples_including_fractional_boundary():
    assert metrics.policy_value(TAU, TAU, 0.5, 0.05) == pytest.approx((0.6 - 0.1) / 4)
    assert metrics.policy_value(TAU, TAU, 0.625, 0.05) == pytest.approx((0.6 + 0.0 - 0.125) / 4)
    assert metrics.policy_value(TAU, TAU, 0.875, 0.05) == pytest.approx((0.55 - 0.175) / 4)
    assert metrics.policy_value(TAU, TAU, 0.0, 0.05) == 0.0
    assert metrics.policy_value(TAU, TAU, 1.0, 0.0) == pytest.approx(TAU.mean())


def test_policy_value_with_ties_matches_random_targeting():
    got = metrics.policy_value(np.zeros(4), TAU, 0.5, 0.05)
    assert got == pytest.approx(metrics.random_policy_value(TAU, 0.5, 0.05))


def test_random_policy_value_formula():
    assert metrics.random_policy_value(TAU, 0.2, 0.02) == pytest.approx(0.2 * (TAU.mean() - 0.02))


def test_oracle_policy_value_dominates_any_score():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        tau = rng.normal(0.02, 0.15, 300)
        score = tau + rng.normal(0.0, 0.2, 300)
        for frac in (0.1, 0.2, 0.33, 0.5):
            oracle = metrics.oracle_policy_value(tau, frac, 0.02)
            assert oracle >= metrics.policy_value(score, tau, frac, 0.02) - 1e-12
            assert oracle >= metrics.random_policy_value(tau, frac, 0.02) - 1e-12


def test_policy_value_rejects_bad_fraction():
    with pytest.raises(ValueError, match="frac"):
        metrics.policy_value(TAU, TAU, 1.5, 0.0)


def test_qini_observed_hand_example():
    y = np.array([1, 0, 0, 1])
    t = np.array([1, 0, 1, 0])
    score = np.array([4.0, 3.0, 2.0, 1.0])
    # Q = 0, 1, 1, 1, 0 on k = 0..4; the random line is flat at 0; trapezoid area 0.75, over n.
    assert metrics.qini_observed(y, t, score) == pytest.approx(0.1875)


def test_qini_observed_groups_ties():
    y = np.array([1, 0, 0, 0])
    t = np.array([1, 0, 1, 0])
    score = np.array([1.0, 1.0, 0.0, 0.0])
    # Q = 0, 1, 1 at k = 0, 2, 4 against the line 0, 0.5, 1: gap 0, 0.5, 0.
    assert metrics.qini_observed(y, t, score) == pytest.approx(0.0625)


def test_qini_observed_needs_both_arms():
    with pytest.raises(ValueError, match="treated and control"):
        metrics.qini_observed(np.ones(3), np.ones(3), np.arange(3.0))


def test_qini_observed_near_zero_for_random_and_positive_for_true_tau():
    sim = simulate(200_000, 21, gamma=0.0)
    rng = np.random.default_rng(5)
    assert abs(metrics.qini_observed(sim.y, sim.t, rng.normal(size=sim.y.size))) < 0.003
    informed = metrics.qini_observed(sim.y, sim.t, sim.tau)
    assert informed > 0.01
    # Q counts treated users, so it is the true numerator scaled by the treated share.
    assert informed == pytest.approx(sim.t.mean() * _numerator(sim.tau, sim.tau), rel=0.1)


def test_covered_including_nan():
    assert metrics.covered(0.0, 1.0, 0.5)
    assert metrics.covered(0.0, 1.0, 1.0)
    assert not metrics.covered(0.0, 1.0, 1.5)
    assert not metrics.covered(float("nan"), 1.0, 0.5)
    assert not metrics.covered(0.0, float("nan"), 0.5)
    assert not metrics.covered(0.0, 1.0, float("nan"))
