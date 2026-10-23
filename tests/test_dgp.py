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
