"""Report: verdict logic on controlled numbers, missing-cell handling, smoke output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causal_uplift.estimators import ESTIMATORS
from causal_uplift.experiments import COLUMNS, Cell, GridConfig, run_grid
from causal_uplift.report import (
    NOT_EVALUABLE,
    NOT_SUPPORTED,
    SUPPORTED,
    breaking_points,
    build_report,
    h1_verdicts,
    h2_verdict,
    h3_verdict,
    write_report,
)

REPS = 20
ALL_CELLS = [(g, 0.0) for g in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)] + [
    (0.5, 0.5),
    (0.5, 1.0),
    (1.0, 0.5),
    (1.0, 1.0),
]


def _alternating(mean: float, spread: float) -> np.ndarray:
    """REPS values with exactly this mean, alternating +- spread."""
    return mean + spread * np.where(np.arange(REPS) % 2 == 0, 1.0, -1.0)


def make_rows(
    gamma: float,
    hidden: float,
    estimator: str,
    *,
    bias: float = 0.0,
    se: float = 0.01,
    covered: float = 0.95,
    qini: float = 0.0,
) -> pd.DataFrame:
    """REPS rows for one (cell, estimator) with a controlled mean bias, own SE, coverage, qini."""
    rows = pd.DataFrame({column: [np.nan] * REPS for column in COLUMNS})
    rows["cell_id"] = f"g{gamma:.2f}_h{hidden:.2f}_n10000"
    rows["gamma"], rows["hidden"], rows["n"] = gamma, hidden, 10000
    rows["rep"] = np.arange(REPS)
    rows["estimator"] = estimator
    rows["base"] = "lgbm"
    rows["true_ate"] = 0.10
    rows["ate_hat"] = 0.10 + _alternating(bias, 0.02)
    if estimator == "response":
        rows["ate_hat"] = np.nan  # the response model has no ATE
    rows["ate_se"] = se
    if not np.isnan(se):
        n_covered = round(covered * REPS)
        rows["covered"] = [1.0] * n_covered + [0.0] * (REPS - n_covered)
    rows["qini_true"] = _alternating(qini, 0.01)
    rows["policy_top20"] = 0.01
    rows["oracle_top20"], rows["random_top20"] = 0.02, 0.005
    rows["fit_seconds"] = 0.1
    return rows


def make_results(
    h2_ok: bool = True, h3_ok: bool = True, h1_gain: float = 0.30, cells=None
) -> pd.DataFrame:
    """Every cell the hypotheses need, with numbers tuned to pass or fail H2 / H3 / H1."""
    frames = []
    for gamma, hidden in cells or ALL_CELLS:
        bias, coverage = 0.0, 0.95
        if hidden > 0:
            # biased (5 x own SE) when H3 should hold, within own SE when it should fail
            bias, coverage = (0.05, 0.2) if h3_ok else (0.005, 0.95)
        elif not h2_ok and gamma == 1.0:
            bias = 0.03  # adjusted estimators fail the usable rule at gamma = 1
        frames += [
            make_rows(gamma, hidden, "naive", bias=0.08 * gamma, covered=0.0),
            make_rows(gamma, hidden, "response", qini=0.10),
            make_rows(gamma, hidden, "x_learner", bias=0.005, se=np.nan, qini=0.10 + h1_gain),
            make_rows(gamma, hidden, "dr_learner", bias=bias, covered=coverage,
                      qini=0.10 + h1_gain),
            make_rows(gamma, hidden, "ipw", bias=bias, covered=coverage, qini=np.nan),
        ]  # fmt: skip
    return pd.concat(frames, ignore_index=True)


def test_h1_supported_and_not_supported() -> None:
    good = h1_verdicts(make_results(h1_gain=0.30))
    assert good["x_learner"].status == SUPPORTED
    assert good["dr_learner"].status == SUPPORTED
    tied = h1_verdicts(make_results(h1_gain=0.0))
    assert tied["x_learner"].status == NOT_SUPPORTED
    assert tied["dr_learner"].status == NOT_SUPPORTED


def test_h1_threshold_is_two_mc_se_of_the_paired_difference() -> None:
    noise = _alternating(0.0, 0.02)
    expected_se = noise.std(ddof=1) / np.sqrt(REPS)
    for gain, status in ((2.05 * expected_se, SUPPORTED), (1.95 * expected_se, NOT_SUPPORTED)):
        df = make_results(h1_gain=0.0)
        is_response = df["estimator"] == "response"
        is_x = (df["estimator"] == "x_learner") & (df["gamma"] == 0.0)
        base = df.loc[is_response & (df["gamma"] == 0.0), "qini_true"].to_numpy()
        df.loc[is_x, "qini_true"] = base + gain + noise
        row = h1_verdicts(df)["x_learner"].rows[0]
        assert abs(row["mc_se"] - expected_se) < 1e-12
        assert abs(row["threshold"] - 2 * expected_se) < 1e-12
        assert h1_verdicts(df)["x_learner"].status == status


def test_h2_supported_and_each_failure_branch() -> None:
    assert h2_verdict(make_results()).status == SUPPORTED
    failing = h2_verdict(make_results(h2_ok=False))
    assert failing.status == NOT_SUPPORTED
    # every condition in all three cells is listed: 3 x (1 naive + 2 x 2 adjusted)
    assert len(failing.rows) == 15
    bad = {(r["cell"], r["estimator"], r["condition"]) for r in failing.rows if not r["met"]}
    assert ("g1.00_h0.00_n10000", "ipw", "|mean bias| <= 1 x own SE") in bad
    assert {cell for cell, _, _ in bad} == {"g1.00_h0.00_n10000"}


def test_h2_naive_must_be_biased() -> None:
    df = make_results()
    mask = (df["estimator"] == "naive") & (df["gamma"] == 2.0) & (df["hidden"] == 0.0)
    df.loc[mask, "ate_hat"] = 0.10 + _alternating(0.0, 0.02)
    verdict = h2_verdict(df)
    assert verdict.status == NOT_SUPPORTED
    assert [r["estimator"] for r in verdict.rows if not r["met"]] == ["naive"]


def test_h2_coverage_floor() -> None:
    df = make_results()
    mask = (df["estimator"] == "dr_learner") & (df["gamma"] == 0.5) & (df["hidden"] == 0.0)
    df.loc[mask, "covered"] = [1.0] * 17 + [0.0] * 3  # 0.85 < 0.90
    verdict = h2_verdict(df)
    assert verdict.status == NOT_SUPPORTED
    failed = [(r["estimator"], r["condition"]) for r in verdict.rows if not r["met"]]
    assert failed == [("dr_learner", "coverage >= 0.90")]


def test_h3_supported_and_not_supported() -> None:
    ok = h3_verdict(make_results(h3_ok=True))
    assert ok.status == SUPPORTED and len(ok.rows) == 8
    bad = h3_verdict(make_results(h3_ok=False))
    assert bad.status == NOT_SUPPORTED
    assert not any(r["met"] for r in bad.rows)


def test_h3_one_failing_estimator_in_one_cell_is_enough() -> None:
    df = make_results()
    mask = (df["estimator"] == "ipw") & (df["gamma"] == 1.0) & (df["hidden"] == 1.0)
    df.loc[mask, "ate_hat"] = 0.10 + _alternating(0.0, 0.02)
    verdict = h3_verdict(df)
    assert verdict.status == NOT_SUPPORTED
    assert sum(not r["met"] for r in verdict.rows) == 1


def test_missing_cells_are_not_evaluable_not_a_crash() -> None:
    df = make_results(cells=[(0.0, 0.0), (1.0, 0.0)])
    assert h1_verdicts(df)["x_learner"].status == SUPPORTED
    for verdict in (h2_verdict(df), h3_verdict(df)):
        assert verdict.status == NOT_EVALUABLE
        assert verdict.headline.startswith("not evaluable (missing: ")
        assert "gamma=" in verdict.headline
    assert h1_verdicts(df[df["estimator"] != "response"])["x_learner"].status == NOT_EVALUABLE


def test_single_replication_h1_not_evaluable() -> None:
    df = make_results()
    assert h1_verdicts(df[df["rep"] == 0])["x_learner"].status == NOT_EVALUABLE


def test_breaking_points() -> None:
    frames = []
    for gamma in (0.0, 0.5, 1.0):
        frames += [
            make_rows(gamma, 0.0, "response", qini=0.10),
            # naive: fine at 0, biased from 0.5
            make_rows(gamma, 0.0, "naive", bias=0.08 * (gamma > 0), covered=0.0 if gamma else 0.95),
            # x_learner point ATE: breaks at 1.0 (|bias| > 0.01)
            make_rows(gamma, 0.0, "x_learner", bias=0.02 if gamma == 1.0 else 0.001,
                      se=np.nan, qini=0.5),
            # dr_learner: ATE never breaks; ranking stops beating response at 1.0
            make_rows(gamma, 0.0, "dr_learner", qini=0.5 if gamma < 1.0 else 0.10),
        ]  # fmt: skip
    table, gammas = breaking_points(pd.concat(frames, ignore_index=True))
    rows = table.set_index("estimator")
    assert gammas == [0.0, 0.5, 1.0]
    assert rows.loc["naive", "ate"] == "gamma = 0.5"
    assert rows.loc["x_learner", "ate"] == "gamma = 1"
    assert rows.loc["dr_learner", "ate"] == "never in grid"
    assert rows.loc["dr_learner", "ranking"] == "gamma = 1"
    assert rows.loc["x_learner", "ranking"] == "never in grid"
    assert rows.loc["response", "ranking"] == "reference"
    assert rows.loc["response", "ate"].startswith("n/a")


def test_report_text_and_write(tmp_path: Path) -> None:
    make_results().to_csv(tmp_path / "grid.csv", index=False)
    (tmp_path / "run_meta.json").write_text('{"git_commit": "abc123", "git_tree_clean": true}')
    text = build_report(tmp_path / "grid.csv")
    assert "commit: abc123" in text
    assert "### H1 - " in text and "### H2 - " in text and "### H3 - " in text
    assert "Verdict: **supported**" in text
    path = write_report(tmp_path)
    assert path.name == "REPORT.md" and path.read_text(encoding="utf-8") == text


def test_report_builds_on_smoke_style_output(tmp_path: Path) -> None:
    cells = tuple(Cell(gamma=g, hidden=0.0, n=400, n_eval=400) for g in (0.0, 1.0))
    run_grid(
        GridConfig(cells=cells, reps=2, base="logistic", estimators=tuple(ESTIMATORS)),
        tmp_path,
        progress=False,
    )
    text = build_report(tmp_path / "grid.csv")
    assert "not evaluable (missing" in text  # H2 / H3 cells are absent from a smoke run
    assert "## 1. Summary" in text and "## 3. Breaking points" in text
