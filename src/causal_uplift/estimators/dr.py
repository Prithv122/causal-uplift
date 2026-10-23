"""Doubly robust estimation: the DR-learner (Kennedy, 2023) and Hajek-normalised IPW.

Both cross-fit their nuisance models with K = 2 folds, stratified on treatment, so that no
user's nuisance prediction comes from a model that saw that user.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Self

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .base import AteResult, BaseKind, clip_propensity, make_classifier, make_regressor
from .metalearners import ensure_fitted, prepare_inputs, proba

N_FOLDS = 2


def _folds(t: np.ndarray, seed: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Seeded (fit, predict) index pairs, stratified on treatment."""
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    yield from splitter.split(np.zeros(len(t)), t)


class DRLearner:
    """Regress the doubly robust pseudo-outcome on ``X``.

    ``phi = mu1 - mu0 + t (y - mu1) / e - (1 - t)(y - mu0) / (1 - e)`` with cross-fitted ``mu0``,
    ``mu1`` and clipped ``e``. Then ``tau(x) = E[phi | x]`` is fitted by a regressor on all users
    and ``ATE = mean(phi)`` (the AIPW estimator) with ``SE = sd(phi) / sqrt(n)``.
    """

    name = "dr_learner"
    estimates_tau = True

    def __init__(self, seed: int = 0, base: BaseKind = "lgbm") -> None:
        self.seed = seed
        self.base = base
        self._model = None
        self._ate: AteResult | None = None

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> Self:
        X, t, y = prepare_inputs(X, t, y, min_per_arm=N_FOLDS)
        mu0, mu1, e = (np.empty(len(X)) for _ in range(3))
        for k, (fit_idx, pred_idx) in enumerate(_folds(t, self.seed)):
            Xf, tf, yf = X[fit_idx], t[fit_idx], y[fit_idx]
            s = self.seed + 3 * k
            m0 = make_classifier(self.base, s).fit(Xf[tf == 0], yf[tf == 0])
            m1 = make_classifier(self.base, s + 1).fit(Xf[tf == 1], yf[tf == 1])
            prop = make_classifier(self.base, s + 2).fit(Xf, tf)
            mu0[pred_idx] = proba(m0, X[pred_idx])
            mu1[pred_idx] = proba(m1, X[pred_idx])
            e[pred_idx] = proba(prop, X[pred_idx])
        e = clip_propensity(e)
        phi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
        self._model = make_regressor(self.base, self.seed + 3 * N_FOLDS).fit(X, phi)
        self._ate = AteResult.from_se(phi.mean(), phi.std(ddof=1) / np.sqrt(len(phi)))
        return self

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        model = ensure_fitted(self._model, self.name)
        return model.predict(np.asarray(X, dtype=float))

    def ate(self) -> AteResult:
        return ensure_fitted(self._ate, self.name)


class IPW:
    """Hajek-normalised inverse-propensity weighting, an ATE-only estimator.

    ``ATE = sum(w1 y) / sum(w1) - sum(w0 y) / sum(w0)`` with ``w1 = t / e`` and
    ``w0 = (1 - t) / (1 - e)`` from a cross-fitted, clipped propensity. The SE is
    ``sd(psi) / sqrt(n)`` for the estimator's influence function ``psi``.

    There is no individual effect: ``predict_tau`` returns the constant ATE, so any ranking
    metric computed from it is degenerate by construction.
    """

    name = "ipw"
    estimates_tau = True

    def __init__(self, seed: int = 0, base: BaseKind = "lgbm") -> None:
        self.seed = seed
        self.base = base
        self._ate: AteResult | None = None

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> Self:
        X, t, y = prepare_inputs(X, t, y, min_per_arm=N_FOLDS)
        e = np.empty(len(X))
        for k, (fit_idx, pred_idx) in enumerate(_folds(t, self.seed)):
            prop = make_classifier(self.base, self.seed + k).fit(X[fit_idx], t[fit_idx])
            e[pred_idx] = proba(prop, X[pred_idx])
        e = clip_propensity(e)
        w1, w0 = t / e, (1 - t) / (1 - e)
        mu1 = (w1 * y).sum() / w1.sum()
        mu0 = (w0 * y).sum() / w0.sum()
        psi = w1 * (y - mu1) / w1.mean() - w0 * (y - mu0) / w0.mean()
        self._ate = AteResult.from_se(mu1 - mu0, psi.std(ddof=1) / np.sqrt(len(psi)))
        return self

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.ate().estimate)

    def ate(self) -> AteResult:
        return ensure_fitted(self._ate, self.name)
