"""Causal-forest reference arm built on EconML (an optional extra).

The module imports without EconML; the dependency is loaded when a `CausalForest` is fitted.
"""

from __future__ import annotations

import importlib.util
from typing import Self

import numpy as np

from causal_uplift.estimators.base import AteResult, BaseKind, make_classifier

N_TREES = 200
CROSS_FIT_FOLDS = 2


def is_available() -> bool:
    """True when `econml` can be imported."""
    return importlib.util.find_spec("econml") is not None


class CausalForest:
    """EconML `CausalForestDML`: orthogonalised outcome and treatment, then a causal forest.

    The outcome and treatment nuisance models are the project's base classifiers, cross-fitted
    over two folds. The ATE and its interval are EconML's doubly robust ATE inference.
    """

    name = "causal_forest"
    estimates_tau = True

    def __init__(self, seed: int = 0, base: BaseKind = "lgbm") -> None:
        self.seed = seed
        self.base = base
        self._model = None
        self._ate: AteResult | None = None

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> Self:
        if not is_available():
            raise ImportError("causal_forest needs the optional dependency 'econml'")
        from econml.dml import CausalForestDML

        self._model = CausalForestDML(
            model_y=make_classifier(self.base, self.seed),
            model_t=make_classifier(self.base, self.seed),
            discrete_outcome=True,
            discrete_treatment=True,
            n_estimators=N_TREES,
            cv=CROSS_FIT_FOLDS,
            n_jobs=1,
            random_state=self.seed,
        )
        self._model.fit(y, t, X=X)
        # The doubly robust ATE: its SE is the sampling error of the average, whereas
        # `ate_inference(X)` aggregates per-user forest variances and is several times wider.
        summary = self._model.ate__inference()
        lo, hi = (float(np.squeeze(v)) for v in summary.conf_int(alpha=0.05))
        self._ate = AteResult(
            estimate=float(np.squeeze(summary.point_estimate)),
            se=float(np.squeeze(summary.stderr)),
            lo=lo,
            hi=hi,
        )
        return self

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        """Estimated effect of treatment for each row of `X`, shape `(n,)`."""
        if self._model is None:
            raise RuntimeError("call fit before predict_tau")
        return np.asarray(self._model.effect(X), dtype=float).reshape(-1)

    def ate(self) -> AteResult | None:
        return self._ate
