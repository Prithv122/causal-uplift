"""The EconML causal-forest arm; skipped when the optional extra is not installed."""

import numpy as np
import pytest

pytest.importorskip("econml")

from causal_uplift.estimators.econml_arm import CausalForest, is_available

TRUE_ATE = 0.15


def _randomised_data(n: int, seed: int):
    """A coin-flip trial where treatment lifts conversion by 0.10 or 0.20 depending on x1."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    t = rng.integers(0, 2, size=n)
    tau = np.where(X[:, 1] > 0, 0.20, 0.10)
    p = 0.25 + 0.05 * np.tanh(X[:, 0]) + tau * t
    y = (rng.random(n) < p).astype(int)
    return X, t, y, tau


def test_is_available() -> None:
    assert is_available()


@pytest.fixture(scope="module")
def fitted():
    X, t, y, tau = _randomised_data(3_000, seed=1)
    return CausalForest(seed=0, base="logistic").fit(X, t, y), X, tau


def test_predict_tau_shape_and_signal(fitted) -> None:
    model, X, tau = fitted
    scores = model.predict_tau(X)
    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all()
    assert 0.0 < scores.mean() < 0.3
    assert np.corrcoef(scores, tau)[0, 1] > 0


def test_ate_interval_contains_the_truth(fitted) -> None:
    model, _, _ = fitted
    ate = model.ate()
    assert ate is not None
    assert 0 < ate.se < 0.05
    assert ate.lo < ate.estimate < ate.hi
    assert ate.lo <= TRUE_ATE <= ate.hi


def test_attributes_and_unfitted_error() -> None:
    model = CausalForest()
    assert model.name == "causal_forest"
    assert model.estimates_tau
    assert model.ate() is None
    with pytest.raises(RuntimeError):
        model.predict_tau(np.zeros((2, 4)))
