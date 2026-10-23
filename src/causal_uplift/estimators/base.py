"""Shared pieces for every estimator: the ATE result type and the base learners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Self

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

BaseKind = Literal["lgbm", "logistic"]

# Estimated propensities are clipped to this range wherever they are divided by.
PROPENSITY_CLIP = (0.01, 0.99)

_LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_jobs": 1,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}


@dataclass(frozen=True)
class AteResult:
    """An average-treatment-effect estimate with its 95% interval (NaN when there is no SE)."""

    estimate: float
    se: float
    lo: float
    hi: float

    @classmethod
    def from_se(cls, estimate: float, se: float, z: float = 1.959964) -> AteResult:
        return cls(float(estimate), float(se), float(estimate - z * se), float(estimate + z * se))

    @classmethod
    def point(cls, estimate: float) -> AteResult:
        nan = float("nan")
        return cls(float(estimate), nan, nan, nan)


class Estimator(Protocol):
    name: str
    estimates_tau: bool

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> Self: ...

    def predict_tau(self, X: np.ndarray) -> np.ndarray: ...

    def ate(self) -> AteResult | None: ...


def make_classifier(kind: BaseKind, seed: int):
    """A probability model for a binary target."""
    if kind == "lgbm":
        return LGBMClassifier(random_state=seed, **_LGBM_PARAMS)
    if kind == "logistic":
        return LogisticRegression(max_iter=1000)
    raise ValueError(f"unknown base learner: {kind!r}")


def make_regressor(kind: BaseKind, seed: int):
    """A model for a continuous target (imputed effects, pseudo-outcomes)."""
    if kind == "lgbm":
        return LGBMRegressor(random_state=seed, **_LGBM_PARAMS)
    if kind == "logistic":
        return LinearRegression()
    raise ValueError(f"unknown base learner: {kind!r}")


def clip_propensity(e: np.ndarray) -> np.ndarray:
    return np.clip(e, *PROPENSITY_CLIP)
