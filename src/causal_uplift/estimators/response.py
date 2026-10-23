"""The response model: rank users by their chance of converting when treated."""

from __future__ import annotations

from typing import Self

import numpy as np

from .base import AteResult, BaseKind, make_classifier
from .metalearners import ensure_fitted, prepare_inputs, proba


class ResponseModel:
    """``P(y = 1 | x)`` fitted on treated users only; the score is that probability.

    This is the common "treat whoever is most likely to convert" targeting rule. Its score is a
    conversion probability, not an effect, so ``estimates_tau`` is False and there is no ATE.
    """

    name = "response"
    estimates_tau = False

    def __init__(self, seed: int = 0, base: BaseKind = "lgbm") -> None:
        self.seed = seed
        self.base = base
        self._model = None

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> Self:
        X, t, y = prepare_inputs(X, t, y)
        self._model = make_classifier(self.base, self.seed).fit(X[t == 1], y[t == 1])
        return self

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        model = ensure_fitted(self._model, self.name)
        return proba(model, np.asarray(X, dtype=float))

    def ate(self) -> AteResult | None:
        return None
