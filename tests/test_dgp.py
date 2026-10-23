import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from causal_uplift.dgp import SEGMENT_NAMES, Sim, simulate


def _baseline_logit(sim: Sim) -> np.ndarray:
    x = sim.X
    return -1.5 + 1.2 * x[:, 0] + 0.4 * x[:, 2] - 0.3 * x[:, 4] + 0.3 * x[:, 6]


def _naive_ate(sim: Sim) -> tuple[float, float]:
    y1, y0 = sim.y[sim.t == 1], sim.y[sim.t == 0]
    se = np.sqrt(y1.var(ddof=1) / y1.size + y0.var(ddof=1) / y0.size)
    return float(y1.mean() - y0.mean()), float(se)


def test_same_arguments_give_identical_arrays():
    a = simulate(500, 7, gamma=1.0, hidden=0.5, noise=0.3)
    b = simulate(500, 7, gamma=1.0, hidden=0.5, noise=0.3)
    for name in ("X", "t", "y", "p0", "p1", "tau", "propensity", "segment", "u"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_different_seed_gives_different_data():
    a, b = simulate(500, 1), simulate(500, 2)
    assert not np.array_equal(a.X, b.X)
    assert not np.array_equal(a.y, b.y)


def test_shapes_dtypes_and_metadata():
    n = 300
    sim = simulate(n, 3, gamma=0.5, effect=1.5, noise=0.2)
    assert sim.X.shape == (n, 8) and sim.X.dtype == np.float64
    for name in ("p0", "p1", "tau", "propensity", "u"):
        arr = getattr(sim, name)
        assert arr.shape == (n,) and arr.dtype == np.float64
    for name in ("t", "y", "segment"):
        arr = getattr(sim, name)
        assert arr.shape == (n,) and arr.dtype == np.int8
    assert set(np.unique(sim.t)) <= {0, 1}
    assert set(np.unique(sim.y)) <= {0, 1}
    assert set(np.unique(sim.segment)) <= {0, 1, 2, 3}
    assert sim.feature_names == tuple(f"x{i}" for i in range(8))
    assert sim.knobs == {
        "n": n,
        "seed": 3,
        "gamma": 0.5,
        "hidden": 0.0,
        "effect": 1.5,
        "noise": 0.2,
    }
    assert set(np.unique(sim.X[:, 6])) <= {0.0, 1.0}
    assert set(np.unique(sim.X[:, 7])) <= {0.0, 1.0}
    assert len(SEGMENT_NAMES) == 4


def test_propensity_strictly_inside_unit_interval():
    sim = simulate(20_000, 4, gamma=2.0, hidden=1.0)
    assert (sim.propensity > 0).all() and (sim.propensity < 1).all()


def test_propensity_is_half_without_confounding():
    sim = simulate(1_000, 5, gamma=0.0, hidden=1.0)
    assert (sim.propensity == 0.5).all()


def test_tau_is_exactly_p1_minus_p0():
    sim = simulate(2_000, 6, gamma=1.0, hidden=0.5, effect=2.0, noise=0.4)
    np.testing.assert_array_equal(sim.tau, sim.p1 - sim.p0)
    assert sim.true_ate == pytest.approx(sim.tau.mean())


def test_segment_rules_match_covariates():
    sim = simulate(20_000, 8)
    x0, x1, x7 = sim.X[:, 0], sim.X[:, 1], sim.X[:, 7]
    np.testing.assert_array_equal(sim.segment == 0, (x1 > 0) & (x0 < 0.5))
    np.testing.assert_array_equal(sim.segment == 1, (x1 > 0) & (x0 >= 0.5))
    np.testing.assert_array_equal(sim.segment == 2, (x1 <= 0) & (x7 == 0))
    np.testing.assert_array_equal(sim.segment == 3, (x1 <= 0) & (x7 == 1))


def test_segments_have_the_intended_sign_of_effect():
    sim = simulate(50_000, 9)
    means = [sim.tau[sim.segment == k].mean() for k in range(4)]
    assert means[0] > means[1] > 0
    assert means[2] == 0
    assert means[3] < 0


def test_difference_in_means_is_unbiased_when_randomised():
    sim = simulate(200_000, 10, gamma=0.0)
    estimate, se = _naive_ate(sim)
    assert abs(estimate - sim.true_ate) < 4 * se


def test_confounding_inflates_the_naive_estimate():
    randomised = simulate(200_000, 11, gamma=0.0)
    confounded = simulate(200_000, 11, gamma=1.0)
    gap_random = _naive_ate(randomised)[0] - randomised.true_ate
    gap_confounded = _naive_ate(confounded)[0] - confounded.true_ate
    assert abs(gap_random) < 0.01
    assert gap_confounded > 0.05


def test_hidden_zero_removes_the_effect_of_u():
    base = simulate(2_000, 12, hidden=0.0)
    np.testing.assert_allclose(base.p0, expit(_baseline_logit(base)))
    with_u = simulate(2_000, 12, hidden=0.5)
    np.testing.assert_array_equal(base.X, with_u.X)
    np.testing.assert_array_equal(base.u, with_u.u)
    np.testing.assert_allclose(with_u.p0, expit(_baseline_logit(with_u) + 0.6 * with_u.u))


def test_hidden_enters_the_propensity():
    sim = simulate(2_000, 13, gamma=1.0, hidden=1.0)
    np.testing.assert_allclose(sim.propensity, expit(sim.X[:, 0] + sim.u))


def test_noise_zero_means_no_shock_and_noise_adds_a_shock_with_that_sd():
    quiet = simulate(50_000, 14, noise=0.0)
    np.testing.assert_allclose(quiet.p0, expit(_baseline_logit(quiet)))
    loud = simulate(50_000, 14, noise=0.5)
    np.testing.assert_array_equal(quiet.X, loud.X)
    shock = np.log(loud.p0 / (1 - loud.p0)) - _baseline_logit(loud)
    assert shock.std() == pytest.approx(0.5, rel=0.03)
    assert shock.mean() == pytest.approx(0.0, abs=0.02)


def test_effect_scales_the_logit_lift():
    none = simulate(1_000, 15, effect=0.0)
    np.testing.assert_allclose(none.tau, 0.0, atol=1e-15)
    assert simulate(20_000, 15, effect=2.0).tau.max() > simulate(20_000, 15, effect=1.0).tau.max()


def test_to_frame_columns_and_content():
    sim = simulate(100, 16, gamma=1.0, hidden=0.5)
    frame = sim.to_frame()
    expected = [f"x{i}" for i in range(8)] + ["t", "y", "p0", "p1", "tau", "propensity", "segment"]
    assert list(frame.columns) == expected
    assert isinstance(frame, pd.DataFrame) and len(frame) == 100
    np.testing.assert_array_equal(frame["x3"].to_numpy(), sim.X[:, 3])
    np.testing.assert_array_equal(frame["tau"].to_numpy(), sim.tau)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n": 0}, "n must"),
        ({"n": 10, "noise": -0.1}, "noise"),
        ({"n": 10, "hidden": -1.0}, "hidden"),
    ],
)
def test_invalid_arguments_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        simulate(seed=0, **kwargs)


# --- Formula pins: the DESIGN.md formulas written out again, independently of dgp.py ---------

LIFT = {"persuadable": 1.5, "sure_thing": 0.3, "lost_cause": 0.0, "sleeping_dog": -1.0}


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def _reference(n: int, seed: int, *, gamma=0.0, hidden=0.0, effect=1.0, noise=0.0) -> dict:
    """Redraw a sample from the documented draw order and formulas, with its own code.

    Order (DESIGN.md): x0..x5 standard normal, x6 Bernoulli(0.5), x7 Bernoulli(0.3), then u,
    eps, then the treatment uniforms, then the outcome uniforms.
    """
    rng = np.random.default_rng(seed)
    cols = [rng.standard_normal(n) for _ in range(6)]
    cols.append(rng.binomial(1, 0.5, n).astype(float))
    cols.append(rng.binomial(1, 0.3, n).astype(float))
    X = np.column_stack(cols)
    u = rng.standard_normal(n)
    eps = noise * rng.standard_normal(n)
    x0, x1, x3, x7 = X[:, 0], X[:, 1], X[:, 3], X[:, 7]

    persuadable = (x1 > 0) & (x0 < 0.5)
    sure_thing = (x1 > 0) & (x0 >= 0.5)
    lost_cause = (x1 <= 0) & (x7 == 0)
    sleeping_dog = (x1 <= 0) & (x7 == 1)
    segment = (
        0 * persuadable + 1 * sure_thing + 2 * lost_cause + 3 * sleeping_dog
    )  # codes in DESIGN order
    s = (
        LIFT["persuadable"] * persuadable
        + LIFT["sure_thing"] * sure_thing
        + LIFT["lost_cause"] * lost_cause
        + LIFT["sleeping_dog"] * sleeping_dog
    )
    smooth = 0.3 * x3 * (persuadable | sure_thing)
    b = -1.5 + 1.2 * x0 + 0.4 * X[:, 2] - 0.3 * X[:, 4] + 0.3 * X[:, 6] + 1.2 * hidden * u + eps
    p0 = expit(b)
    p1 = expit(b + effect * (s + smooth))
    e = expit(gamma * (x0 + hidden * u))
    t = (rng.random(n) < e).astype(int)
    y = (rng.random(n) < np.where(t == 1, p1, p0)).astype(int)
    return {
        "X": X, "u": u, "eps": eps, "segment": segment, "p0": p0, "p1": p1, "e": e, "t": t, "y": y
    }  # fmt: skip


@pytest.mark.parametrize(
    "knobs",
    [
        {},
        {"gamma": 1.2, "hidden": 0.8, "effect": 1.3, "noise": 0.7},
        {"gamma": 0.4, "hidden": 1.5, "effect": 0.6, "noise": 0.25},
    ],
)
def test_every_array_matches_an_independent_redraw_of_the_documented_formulas(knobs):
    n, seed = 4_000, 31
    sim = simulate(n, seed, **knobs)
    ref = _reference(n, seed, **knobs)
    np.testing.assert_array_equal(sim.X, ref["X"])
    np.testing.assert_array_equal(sim.u, ref["u"])  # u is drawn before eps
    np.testing.assert_array_equal(sim.segment, ref["segment"])
    np.testing.assert_allclose(sim.p0, ref["p0"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(sim.p1, ref["p1"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(sim.tau, ref["p1"] - ref["p0"], rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(sim.propensity, ref["e"], rtol=1e-12, atol=0)
    np.testing.assert_array_equal(sim.t, ref["t"])  # t uniforms follow u and eps
    np.testing.assert_array_equal(sim.y, ref["y"])


def test_draw_order_puts_u_before_eps():
    # With noise > 0 and hidden > 0 both consume a standard-normal stream; swapping them would
    # make the hidden confounder equal to the noise draw and vice versa.
    n, seed = 3_000, 32
    sim = simulate(n, seed, hidden=1.0, noise=1.0)
    rng = np.random.default_rng(seed)
    for _ in range(6):
        rng.standard_normal(n)
    rng.binomial(1, 0.5, n)
    rng.binomial(1, 0.3, n)
    first, second = rng.standard_normal(n), rng.standard_normal(n)
    np.testing.assert_array_equal(sim.u, first)
    shock = _logit(sim.p0) - (_baseline_logit(sim) + 1.2 * sim.u)
    np.testing.assert_allclose(shock, second, atol=1e-9)


@pytest.mark.parametrize("effect", [1.0, 1.5])
def test_logit_lift_is_the_documented_constant_per_segment_plus_the_x3_term(effect):
    sim = simulate(20_000, 33, effect=effect)
    lift = (_logit(sim.p1) - _logit(sim.p0)) / effect
    x3 = sim.X[:, 3]
    for code, name in enumerate(SEGMENT_NAMES):
        seg = sim.segment == code
        assert seg.sum() > 1_000
        if name in ("persuadable", "sure_thing"):
            smooth = (lift[seg] - LIFT[name]) / x3[seg]
            np.testing.assert_allclose(smooth, 0.3, atol=1e-8)
        else:
            np.testing.assert_allclose(lift[seg], LIFT[name], atol=1e-9)


def test_segment_shares_follow_the_covariate_distributions():
    n = 200_000
    sim = simulate(n, 34)
    # P(x1 > 0) = 1/2, P(x0 < 0.5) = Phi(0.5), P(x7 = 1) = 0.3
    phi = 0.6914624612740131
    expected = np.array([0.5 * phi, 0.5 * (1 - phi), 0.5 * 0.7, 0.5 * 0.3])
    observed = np.bincount(sim.segment, minlength=4) / n
    se = np.sqrt(expected * (1 - expected) / n)
    assert np.all(np.abs(observed - expected) < 5 * se)


def test_segment_mean_tau_and_baseline_match_an_independent_redraw():
    # Two independent large samples from the same documented process must agree per segment,
    # within sampling error, on the mean effect and the mean baseline conversion.
    n = 200_000
    sim = simulate(n, 35, hidden=0.5, noise=0.3)
    ref = _reference(n, 36, hidden=0.5, noise=0.3)
    ref_tau = ref["p1"] - ref["p0"]
    for code in range(4):
        a, b = sim.segment == code, ref["segment"] == code
        for ours, theirs in ((sim.tau[a], ref_tau[b]), (sim.p0[a], ref["p0"][b])):
            se = np.sqrt(ours.var() / ours.size + theirs.var() / theirs.size)
            assert abs(ours.mean() - theirs.mean()) <= 5 * se + 1e-12  # lost_cause tau is exactly 0
    # The documented signs and sizes, so a rescaled lift cannot hide behind both draws agreeing.
    mean_tau = [sim.tau[sim.segment == k].mean() for k in range(4)]
    assert 0.2 < mean_tau[0] < 0.3
    assert 0.03 < mean_tau[1] < 0.1
    assert mean_tau[2] == 0.0
    assert -0.16 < mean_tau[3] < -0.09
