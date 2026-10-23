"""S-, T- and X-learners (Kunzel et al., 2019) on top of the shared base learners.

None of these is built for inference, so their ATE is the mean of the estimated individual
effects over the training users and carries no standard error.
"""

from __future__ import annotations

from typing import Self

import numpy as np

from .base import AteResult, BaseKind, clip_propensity, make_classifier, make_regressor


def prepare_inputs(
    X: np.ndarray, t: np.ndarray, y: np.ndarray, min_per_arm: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a training sample and return ``X`` as float and ``t``, ``y`` as 0/1 ints."""
    X = np.asarray(X, dtype=float)
    t = np.asarray(t)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if not len(X) == len(t) == len(y):
        raise ValueError("X, t and y must have the same number of rows")
    for name, v in (("t", t), ("y", y)):
        if not np.isin(v, (0, 1)).all():
            raise ValueError(f"{name} must contain only 0 and 1")
    t = t.astype(np.int64)
    for arm in (0, 1):
        if (t == arm).sum() < min_per_arm:
            raise ValueError(f"treatment arm {arm} has fewer than {min_per_arm} samples")
    return X, t, y.astype(np.int64)


def ensure_fitted(value, estimator: str):
    """Return ``value`` unless it is still ``None`` (the estimator has not been fitted)."""
    if value is None:
        raise RuntimeError(f"{estimator} is not fitted; call fit(X, t, y) first")
    return value


def proba(model, X: np.ndarray) -> np.ndarray:
    """P(target = 1) from a fitted classifier."""
    return model.predict_proba(X)[:, 1]


class _MetaLearner:
    """Shared plumbing: seed and base storage, fitted checks and the mean-of-tau ATE."""

    name: str
    estimates_tau = True

    def __init__(self, seed: int = 0, base: BaseKind = "lgbm") -> None:
        self.seed = seed
        self.base = base
        self._ate: AteResult | None = None

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _finish_fit(self, X: np.ndarray) -> Self:
        self._ate = AteResult.point(float(self.predict_tau(X).mean()))
        return self

    def ate(self) -> AteResult:
        return ensure_fitted(self._ate, self.name)


class SLearner(_MetaLearner):
    """One outcome model on ``[X, t]``; ``tau(x) = p(x, 1) - p(x, 0)``."""

    name = "s_learner"
    _model = None

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> Self:
        X, t, y = prepare_inputs(X, t, y)
        self._model = make_classifier(self.base, self.seed).fit(np.column_stack([X, t]), y)
        return self._finish_fit(X)

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        model = ensure_fitted(self._model, self.name)
        X = np.asarray(X, dtype=float)
        ones = np.ones((len(X), 1))
        return proba(model, np.hstack([X, ones])) - proba(model, np.hstack([X, 0 * ones]))


class TLearner(_MetaLearner):
    """Separate outcome models per arm; ``tau(x) = mu1(x) - mu0(x)``."""

    name = "t_learner"
    _mu0 = None
    _mu1 = None

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> Self:
        X, t, y = prepare_inputs(X, t, y)
        self._mu0 = make_classifier(self.base, self.seed).fit(X[t == 0], y[t == 0])
        self._mu1 = make_classifier(self.base, self.seed + 1).fit(X[t == 1], y[t == 1])
        return self._finish_fit(X)

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        mu0 = ensure_fitted(self._mu0, self.name)
        X = np.asarray(X, dtype=float)
        return proba(self._mu1, X) - proba(mu0, X)


class XLearner(_MetaLearner):
    """X-learner: impute individual effects with the other arm's model, then blend.

    ``D1 = y - mu0(x)`` on treated users and ``D0 = mu1(x) - y`` on controls are regressed on
    ``x`` to give ``tau1`` and ``tau0``; ``tau(x) = g(x) tau0(x) + (1 - g(x)) tau1(x)`` where
    ``g`` is the (clipped) propensity of treatment.
    """

    name = "x_learner"
    _tau0 = None
    _tau1 = None
    _g = None

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> Self:
        X, t, y = prepare_inputs(X, t, y)
        treated, control = t == 1, t == 0
        mu0 = make_classifier(self.base, self.seed).fit(X[control], y[control])
        mu1 = make_classifier(self.base, self.seed + 1).fit(X[treated], y[treated])
        d1 = y[treated] - proba(mu0, X[treated])
        d0 = proba(mu1, X[control]) - y[control]
        self._tau1 = make_regressor(self.base, self.seed + 2).fit(X[treated], d1)
        self._tau0 = make_regressor(self.base, self.seed + 3).fit(X[control], d0)
        self._g = make_classifier(self.base, self.seed + 4).fit(X, t)
        return self._finish_fit(X)

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        g_model = ensure_fitted(self._g, self.name)
        X = np.asarray(X, dtype=float)
        g = clip_propensity(proba(g_model, X))
        return g * self._tau0.predict(X) + (1.0 - g) * self._tau1.predict(X)
