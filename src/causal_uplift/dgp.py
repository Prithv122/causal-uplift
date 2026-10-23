"""Data-generating process: simulated users with known per-user treatment effects.

Every user has two potential conversion probabilities, ``p0`` (untreated) and ``p1`` (treated),
so the true individual effect ``tau = p1 - p0`` is exact. Treatment assignment depends on the
observed covariate ``x0`` (and, optionally, on a hidden confounder ``u`` that no estimator sees).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

SEGMENT_NAMES = ("persuadable", "sure_thing", "lost_cause", "sleeping_dog")
FEATURE_NAMES = tuple(f"x{i}" for i in range(8))

_SEGMENT_LIFT = np.array([1.5, 0.3, 0.0, -1.0])
_SMOOTH_SEGMENTS = 2  # codes below this (persuadable, sure_thing) also respond to x3


@dataclass(frozen=True, eq=False)
class Sim:
    """One simulated sample with the exact per-user truth."""

    X: np.ndarray
    t: np.ndarray
    y: np.ndarray
    p0: np.ndarray
    p1: np.ndarray
    tau: np.ndarray
    propensity: np.ndarray
    segment: np.ndarray
    u: np.ndarray
    feature_names: tuple[str, ...]
    knobs: dict

    @property
    def true_ate(self) -> float:
        """Mean of the per-user true effect."""
        return float(self.tau.mean())

    def to_frame(self) -> pd.DataFrame:
        """Covariates, treatment, outcome and truth as one frame (the hidden ``u`` is omitted)."""
        frame = pd.DataFrame(self.X, columns=list(self.feature_names))
        for name in ("t", "y", "p0", "p1", "tau", "propensity", "segment"):
            frame[name] = getattr(self, name)
        return frame


def _segments(X: np.ndarray) -> np.ndarray:
    """Segment code per user from the observed covariates."""
    x0, x1, x7 = X[:, 0], X[:, 1], X[:, 7]
    responsive = x1 > 0
    code = np.where(responsive, np.where(x0 < 0.5, 0, 1), np.where(x7 == 0, 2, 3))
    return code.astype(np.int8)


def simulate(
    n: int,
    seed: int,
    *,
    gamma: float = 0.0,
    hidden: float = 0.0,
    effect: float = 1.0,
    noise: float = 0.0,
) -> Sim:
    """Simulate ``n`` users.

    ``gamma`` is the confounding strength (propensity ``sigmoid(gamma * (x0 + hidden * u))``),
    ``hidden`` moves part of that confounding into the unobserved ``u``, ``effect`` scales the
    logit lift of the offer, and ``noise`` is the sd of a per-user logit shock on the baseline.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if noise < 0:
        raise ValueError(f"noise must be non-negative, got {noise}")
    if hidden < 0:
        raise ValueError(f"hidden must be non-negative, got {hidden}")

    rng = np.random.default_rng(seed)
    X = np.empty((n, len(FEATURE_NAMES)))
    for j in range(6):
        X[:, j] = rng.standard_normal(n)
    X[:, 6] = rng.binomial(1, 0.5, n)
    X[:, 7] = rng.binomial(1, 0.3, n)
    u = rng.standard_normal(n)
    eps = noise * rng.standard_normal(n)

    segment = _segments(X)
    baseline = -1.5 + 1.2 * X[:, 0] + 0.4 * X[:, 2] - 0.3 * X[:, 4] + 0.3 * X[:, 6]
    b = baseline + 1.2 * hidden * u + eps
    smooth = 0.3 * X[:, 3] * (segment < _SMOOTH_SEGMENTS)
    lift = effect * (_SEGMENT_LIFT[segment] + smooth)

    p0 = expit(b)
    p1 = expit(b + lift)
    propensity = expit(gamma * (X[:, 0] + hidden * u))
    t = (rng.random(n) < propensity).astype(np.int8)
    y = (rng.random(n) < np.where(t == 1, p1, p0)).astype(np.int8)

    return Sim(
        X=X,
        t=t,
        y=y,
        p0=p0,
        p1=p1,
        tau=p1 - p0,
        propensity=propensity,
        segment=segment,
        u=u,
        feature_names=FEATURE_NAMES,
        knobs={
            "n": n,
            "seed": seed,
            "gamma": gamma,
            "hidden": hidden,
            "effect": effect,
            "noise": noise,
        },
    )
