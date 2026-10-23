import numpy as np
import pandas as pd
import pytest

from causal_uplift.did import (
    TIME_SHOCK,
    did_2x2,
    post_difference,
    pre_post,
    simulate_panel,
    sweep,
)

N_UNITS = 2_000
REPS = 200
EFFECT = 0.05


def _bias_and_coverage(estimator, reps: int = REPS, **knobs) -> tuple[float, float, float]:
    """Mean bias, its Monte-Carlo SE, and 95% coverage of the true effect over `reps` panels."""
    results = [
        estimator(simulate_panel(N_UNITS, seed=rep, effect=EFFECT, **knobs)) for rep in range(reps)
    ]
    bias = np.array([r.estimate for r in results]) - EFFECT
    covered = np.mean([r.lo <= EFFECT <= r.hi for r in results])
    return bias.mean(), bias.std(ddof=1) / np.sqrt(reps), covered


def test_simulation_is_deterministic() -> None:
    a = simulate_panel(500, seed=3, selection=1.5, trend_gap=0.02)
    b = simulate_panel(500, seed=3, selection=1.5, trend_gap=0.02)
    c = simulate_panel(500, seed=4, selection=1.5, trend_gap=0.02)
    np.testing.assert_array_equal(a.g, b.g)
    np.testing.assert_array_equal(a.y_pre, b.y_pre)
    np.testing.assert_array_equal(a.y_post, b.y_post)
    assert not np.array_equal(a.y_post, c.y_post)
    assert a.knobs["trend_gap"] == 0.02


def test_to_frame_is_long_format() -> None:
    panel = simulate_panel(100, seed=0)
    frame = panel.to_frame()
    assert list(frame.columns) == ["unit", "period", "g", "post", "y"]
    assert len(frame) == 200
    assert frame.groupby("unit").size().eq(2).all()
    pre = frame[frame["post"] == 0].set_index("unit")["y"]
    np.testing.assert_allclose(pre.to_numpy(), panel.y_pre)


def test_did_is_unbiased_and_covers_under_parallel_trends() -> None:
    bias, mc_se, coverage = _bias_and_coverage(did_2x2)
    assert abs(bias) < 3 * mc_se
    assert 0.88 <= coverage <= 0.99


def test_post_difference_is_biased_by_selection() -> None:
    bias, mc_se, _ = _bias_and_coverage(post_difference, reps=50)
    assert bias > 0.05
    assert bias > 3 * mc_se
    unbiased, mc_se0, _ = _bias_and_coverage(post_difference, reps=50, selection=0.0)
    assert abs(unbiased) < 3 * mc_se0


def test_pre_post_is_biased_by_the_time_shock() -> None:
    bias, mc_se, _ = _bias_and_coverage(pre_post, reps=50)
    assert bias == pytest.approx(TIME_SHOCK, abs=3 * mc_se)


def test_did_bias_equals_the_trend_gap() -> None:
    bias, mc_se, coverage = _bias_and_coverage(did_2x2, trend_gap=0.03)
    assert bias == pytest.approx(0.03, abs=3 * mc_se)
    assert coverage < 0.5  # intervals are centred on the wrong value


def test_estimators_need_both_groups() -> None:
    panel = simulate_panel(50, seed=0, selection=60.0)  # almost everyone high-level adopts
    one_group = type(panel)(
        unit=panel.unit,
        g=np.ones_like(panel.g),
        y_pre=panel.y_pre,
        y_post=panel.y_post,
        effect=panel.effect,
        knobs=panel.knobs,
    )
    with pytest.raises(ValueError, match="at least two"):
        did_2x2(one_group)


def test_sweep_output() -> None:
    table = sweep([0.0, 0.03], n_units=500, reps=5, seed=1)
    assert list(table.columns) == [
        "trend_gap",
        "estimator",
        "mean_estimate",
        "mean_bias",
        "mc_se_bias",
        "coverage",
    ]
    assert len(table) == 6
    assert set(table["estimator"]) == {"post_difference", "pre_post", "did_2x2"}
    assert table["coverage"].between(0, 1).all()
    np.testing.assert_allclose(table["mean_estimate"] - table["mean_bias"], EFFECT)
    pd.testing.assert_frame_equal(table, sweep([0.0, 0.03], n_units=500, reps=5, seed=1))
