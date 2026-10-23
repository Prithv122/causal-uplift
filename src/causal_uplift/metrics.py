"""Evaluation metrics against known truth (and one outcome-only Qini for randomised data).

Arrays in, floats (or curves) out; nothing is fitted here. Users with equal scores cannot be
ordered, so every ranking metric treats a tie group as randomly ordered: each member contributes
the group's mean, which makes a constant score exactly equivalent to random targeting.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

_EPS = 1e-12


def _as_vector(a: np.ndarray, name: str) -> np.ndarray:
    v = np.asarray(a, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    return v


def _check_pair(a: np.ndarray, b: np.ndarray, names: tuple[str, str]) -> tuple[np.ndarray, ...]:
    va, vb = _as_vector(a, names[0]), _as_vector(b, names[1])
    if va.shape != vb.shape:
        raise ValueError(f"{names[0]} and {names[1]} must have the same length")
    return va, vb


def _tie_groups(score: np.ndarray, *values: np.ndarray) -> tuple[np.ndarray, ...]:
    """Group sizes and per-group sums of ``values``, groups ordered by descending score."""
    if not np.isfinite(score).all():
        raise ValueError("score must be finite")
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = [np.bincount(inverse, weights=v) for v in values]
    return (counts[::-1], *(s[::-1] for s in sums))


def pehe(tau_hat: np.ndarray, tau: np.ndarray) -> float:
    """sqrt(mean((tau_hat - tau)^2))."""
    tau_hat, tau = _check_pair(tau_hat, tau, ("tau_hat", "tau"))
    return float(np.sqrt(np.mean((tau_hat - tau) ** 2)))


def spearman(score: np.ndarray, tau: np.ndarray) -> float:
    """Spearman rank correlation between score and tau; NaN if either is constant."""
    score, tau = _check_pair(score, tau, ("score", "tau"))
    if np.ptp(score) == 0 or np.ptp(tau) == 0:
        return float("nan")
    return float(spearmanr(score, tau).statistic)


def uplift_curve_true(score: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """V(k) = sum of tau over the top k users by score, for k = 0..n, ties averaged."""
    score, tau = _check_pair(score, tau, ("score", "tau"))
    sizes, tau_sums = _tie_groups(score, tau)
    per_user = np.repeat(tau_sums / sizes, sizes)
    return np.concatenate(([0.0], np.cumsum(per_user)))


def _area(curve: np.ndarray) -> float:
    """Area under V(k)/n against k/n by the trapezoid rule."""
    n = curve.size - 1
    return float(np.trapezoid(curve / n, dx=1.0 / n))


def qini_true(score: np.ndarray, tau: np.ndarray) -> float:
    """(A(score) - A(random)) / (A(oracle) - A(random)), A = area under V(k)/n vs k/n.

    1 is the oracle ranking, 0 is random targeting, negative is worse than random. Returns 0 when
    tau is constant, since no ranking can then beat random.
    """
    score, tau = _check_pair(score, tau, ("score", "tau"))
    area_random = float(tau.sum() / (2 * tau.size))
    gap_oracle = _area(uplift_curve_true(tau, tau)) - area_random
    if gap_oracle < _EPS:
        return 0.0
    return (_area(uplift_curve_true(score, tau)) - area_random) / gap_oracle


def qini_observed(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    """Radcliffe Qini area from outcomes only, for randomised data.

    Q(k) = Yt(k) - Yc(k) * Nt(k) / Nc(k) over the top k by score (cut only at tie boundaries).
    Returns the area between Q(k)/n and the straight line to Q(n)/n, against k/n.
    """
    y, t, score = _as_vector(y, "y"), _as_vector(t, "t"), _as_vector(score, "score")
    if not y.shape == t.shape == score.shape:
        raise ValueError("y, t and score must have the same length")
    if t.sum() in (0, t.size):
        raise ValueError("qini_observed needs both treated and control users")
    sizes, treated, y_treated, y_control = _tie_groups(score, t, y * t, y * (1 - t))
    zero = np.zeros(1)
    k = np.concatenate((zero, np.cumsum(sizes)))
    n_t = np.concatenate((zero, np.cumsum(treated)))
    n_c = k - n_t
    y_t = np.concatenate((zero, np.cumsum(y_treated)))
    y_c = np.concatenate((zero, np.cumsum(y_control)))
    ratio = np.divide(n_t, n_c, out=np.zeros_like(n_t), where=n_c > 0)
    q = y_t - y_c * ratio
    n = k[-1]
    return float(np.trapezoid((q - q[-1] * k / n) / n, x=k / n))


def _value_at(curve: np.ndarray, frac: float, cost: float) -> float:
    """Per-user net value of treating the top ``frac`` of users along an uplift curve."""
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"frac must be in [0, 1], got {frac}")
    n = curve.size - 1
    k = frac * n
    return float((np.interp(k, np.arange(n + 1), curve) - cost * k) / n)


def policy_value(score: np.ndarray, tau: np.ndarray, frac: float, cost: float) -> float:
    """(sum of tau over treated - cost * n_treated) / n, treating the top ``frac`` by score.

    ``frac * n`` may be fractional: the boundary user is split proportionally. Ties are averaged.
    """
    return _value_at(uplift_curve_true(score, tau), frac, cost)


def oracle_policy_value(tau: np.ndarray, frac: float, cost: float) -> float:
    """policy_value when ranking by the true tau."""
    return policy_value(tau, tau, frac, cost)


def random_policy_value(tau: np.ndarray, frac: float, cost: float) -> float:
    """frac * (mean(tau) - cost): the expected value of treating a random ``frac`` of users."""
    tau = _as_vector(tau, "tau")
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"frac must be in [0, 1], got {frac}")
    return float(frac * (tau.mean() - cost))


def covered(lo: float, hi: float, truth: float) -> bool:
    """lo <= truth <= hi; False if either bound (or the truth) is NaN."""
    return bool(lo <= truth <= hi)
