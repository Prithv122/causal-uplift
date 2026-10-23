"""The naive baseline: compare converting shares of treated and untreated users."""

from __future__ import annotations

from typing import Self

import numpy as np

from .base import AteResult, BaseKind
from .metalearners import ensure_fitted, prepare_inputs


class Naive:
    """Constant effect ``mean(y | t=1) - mean(y | t=0)`` with a Welch standard error.

    ``SE = sqrt(s1^2 / n1 + s0^2 / n0)`` with sample variances (ddof=1). Unbiased only when
    treatment is randomised. ``base`` is accepted for interface symmetry and unused.
    """

    name = "naive"
    estimates_tau = True

    def __init__(self, seed: int = 0, base: BaseKind = "lgbm") -> None:
        self.seed = seed
        self.base = base
        self._ate: AteResult | None = None

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> Self:
        _, t, y = prepare_inputs(X, t, y, min_per_arm=2)
        y1, y0 = y[t == 1], y[t == 0]
        se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
        self._ate = AteResult.from_se(y1.mean() - y0.mean(), se)
        return self

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.ate().estimate)

    def ate(self) -> AteResult:
        return ensure_fitted(self._ate, self.name)
