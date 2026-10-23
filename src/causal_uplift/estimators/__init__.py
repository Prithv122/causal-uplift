"""Treatment-effect estimators, all sharing the interface in DESIGN.md."""

from __future__ import annotations

from collections.abc import Callable

from .base import AteResult, BaseKind, Estimator
from .dr import IPW, DRLearner, IPWLogit
from .metalearners import SLearner, TLearner, XLearner
from .naive import Naive
from .response import ResponseModel

# Name -> factory(seed, base_kind). Every class is its own factory.
ESTIMATORS: dict[str, Callable[[int, BaseKind], Estimator]] = {
    "naive": Naive,
    "response": ResponseModel,
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
    "dr_learner": DRLearner,
    "ipw": IPW,
    "ipw_logit": IPWLogit,
}


def available_estimators() -> dict[str, Callable[[int, BaseKind], Estimator]]:
    """``ESTIMATORS`` plus the causal forest when the optional EconML extra is installed."""
    estimators = dict(ESTIMATORS)
    try:
        from .econml_arm import CausalForest, is_available
    except ImportError:
        return estimators
    if is_available():
        estimators["causal_forest"] = CausalForest
    return estimators


__all__ = [
    "ESTIMATORS",
    "IPW",
    "AteResult",
    "DRLearner",
    "Estimator",
    "IPWLogit",
    "Naive",
    "ResponseModel",
    "SLearner",
    "TLearner",
    "XLearner",
    "available_estimators",
]
