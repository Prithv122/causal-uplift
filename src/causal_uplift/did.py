"""Difference-in-differences on a simulated two-period panel.

A policy is rolled out to one group at one date. Three ways of reading the effect are compared,
and each fails differently: comparing groups after the rollout is biased by who adopts, comparing
the treated group before and after is biased by the common time shock, and the difference of
changes is unbiased exactly when the two groups' trends are parallel.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from causal_uplift.estimators.base import AteResult

TIME_SHOCK = 0.10  # lambda_post: the change every unit sees between the two periods
NOISE_SD = 0.10
BASELINE = 0.5
LEVEL_LOADING = 0.2  # how much a unit's level alpha_i moves its outcome


@dataclass(frozen=True)
class Panel:
    """Two-period outcomes for `n` units; `g` marks the group that receives the policy."""

    unit: np.ndarray  # (n,) int64 unit ids
    g: np.ndarray  # (n,) int8 in {0, 1}
    y_pre: np.ndarray  # (n,) float64
    y_post: np.ndarray  # (n,) float64
    effect: float  # the true effect of the policy on the treated
    knobs: dict[str, Any]  # {"n_units", "seed", "selection", "trend_gap", "effect"}

    def to_frame(self) -> pd.DataFrame:
        """Long format: one row per unit and period, columns unit, period, g, post, y."""
        n = len(self.unit)
        period = np.repeat(np.array([0, 1], dtype=np.int8), n)
        return pd.DataFrame(
            {
                "unit": np.tile(self.unit, 2),
                "period": period,
                "g": np.tile(self.g, 2),
                "post": period,
                "y": np.concatenate([self.y_pre, self.y_post]),
            }
        )


def simulate_panel(
    n_units: int,
    seed: int,
    *,
    selection: float = 1.0,
    trend_gap: float = 0.0,
    effect: float = 0.05,
) -> Panel:
    """Simulate the panel described in DESIGN.md.

    Units with a higher level `alpha` adopt more often (strength `selection`). `trend_gap` adds
    an extra post-period change to the treated group alone, which breaks parallel trends: the
    difference-in-differences estimate is then off by `trend_gap` in expectation.
    """
    rng = np.random.default_rng(seed)
    alpha = rng.normal(size=n_units)
    g = (rng.random(n_units) < expit(selection * alpha)).astype(np.int8)
    noise_pre = rng.normal(0.0, NOISE_SD, size=n_units)
    noise_post = rng.normal(0.0, NOISE_SD, size=n_units)

    level = BASELINE + LEVEL_LOADING * alpha
    y_pre = level + noise_pre
    y_post = level + TIME_SHOCK + (effect + trend_gap) * g + noise_post
    return Panel(
        unit=np.arange(n_units),
        g=g,
        y_pre=y_pre,
        y_post=y_post,
        effect=effect,
        knobs={
            "n_units": n_units,
            "seed": seed,
            "selection": selection,
            "trend_gap": trend_gap,
            "effect": effect,
        },
    )


def _welch(a: np.ndarray, b: np.ndarray) -> AteResult:
    """Difference in means `a - b` with the Welch standard error."""
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each group needs at least two units")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return AteResult.from_se(a.mean() - b.mean(), se)


def post_difference(panel: Panel) -> AteResult:
    """Treated minus control after the rollout; biased when adoption depends on the unit level."""
    treated = panel.g == 1
    return _welch(panel.y_post[treated], panel.y_post[~treated])


def pre_post(panel: Panel) -> AteResult:
    """Treated group after minus before; picks up the time shock as if it were the effect."""
    change = (panel.y_post - panel.y_pre)[panel.g == 1]
    if len(change) < 2:
        raise ValueError("the treated group needs at least two units")
    return AteResult.from_se(change.mean(), change.std(ddof=1) / np.sqrt(len(change)))


def did_2x2(panel: Panel) -> AteResult:
    """Difference of the two groups' mean change, with a Welch SE on the per-unit changes."""
    change = panel.y_post - panel.y_pre
    treated = panel.g == 1
    return _welch(change[treated], change[~treated])


ESTIMATORS: dict[str, Callable[[Panel], AteResult]] = {
    "post_difference": post_difference,
    "pre_post": pre_post,
    "did_2x2": did_2x2,
}


def sweep(
    trend_gaps: Sequence[float],
    n_units: int,
    reps: int,
    seed: int,
    *,
    selection: float = 1.0,
    effect: float = 0.05,
) -> pd.DataFrame:
    """Monte-Carlo each estimator at each `trend_gap`.

    Returns one row per (trend_gap, estimator) with the mean estimate, mean bias against the
    true effect, the Monte-Carlo SE of that bias, and the share of 95% intervals covering the
    true effect. Replication `rep` of gap index `i` is seeded by `SeedSequence([seed, i, rep])`.
    """
    rows = []
    for i, gap in enumerate(trend_gaps):
        draws: dict[str, list[AteResult]] = {name: [] for name in ESTIMATORS}
        for rep in range(reps):
            sim_seed = int(np.random.SeedSequence([seed, i, rep]).generate_state(1)[0])
            panel = simulate_panel(
                n_units, sim_seed, selection=selection, trend_gap=gap, effect=effect
            )
            for name, estimator in ESTIMATORS.items():
                draws[name].append(estimator(panel))
        for name, results in draws.items():
            bias = np.array([r.estimate for r in results]) - effect
            rows.append(
                {
                    "trend_gap": gap,
                    "estimator": name,
                    "mean_estimate": bias.mean() + effect,
                    "mean_bias": bias.mean(),
                    "mc_se_bias": bias.std(ddof=1) / np.sqrt(reps) if reps > 1 else np.nan,
                    "coverage": float(np.mean([r.lo <= effect <= r.hi for r in results])),
                }
            )
    return pd.DataFrame(rows)
