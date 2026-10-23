"""Outcome-only Criteo check on a synthetic Criteo-shaped file (no real data is touched)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_uplift.cli import main
from causal_uplift.data import CRITEO_COLUMNS
from causal_uplift.realcheck import (
    COLUMNS,
    FIXED_LINE,
    RANDOM_NAME,
    run_criteo_check,
    write_criteo_report,
)

N_ROWS = 6_000
SMALL = {"n": 4_000, "splits": 2, "base": "logistic"}


def _write_synthetic(path: Path, n: int = N_ROWS, seed: int = 7) -> Path:
    """A randomised Criteo-shaped file: visit depends on f0, the effect depends on f1."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"f{i}": rng.normal(size=n).astype("float32") for i in range(12)})
    t = (rng.random(n) < 0.85).astype("int8")
    logit = -2.0 + 0.8 * frame["f0"].to_numpy() + t * (0.1 + 1.5 * frame["f1"].to_numpy())
    visit = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8")
    frame["treatment"] = t
    frame["visit"] = visit
    frame["conversion"] = (visit & (rng.random(n) < 0.2)).astype("int8")
    frame["exposure"] = (t & (rng.random(n) < 0.7)).astype("int8")
    frame = frame[list(CRITEO_COLUMNS)]
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_synthetic(tmp_path_factory.mktemp("criteo") / "criteo-synthetic.csv")


@pytest.fixture(scope="module")
def result(synthetic: Path) -> pd.DataFrame:
    return run_criteo_check(synthetic, **SMALL)


def test_columns_and_row_count(result: pd.DataFrame) -> None:
    assert list(result.columns) == COLUMNS
    # 2 splits x 2 outcomes x (4 estimators + the random baseline)
    assert len(result) == 2 * 2 * 5
    assert set(result["estimator"]) == {
        RANDOM_NAME,
        "response",
        "t_learner",
        "x_learner",
        "dr_learner",
    }
    assert set(result["outcome"]) == {"visit", "conversion"}
    assert (result["n_train"] + result["n_test"] == SMALL["n"]).all()
    assert result["qini_observed"].notna().all()
    assert (result["fit_seconds"] >= 0).all()


def test_deterministic(synthetic: Path, result: pd.DataFrame) -> None:
    again = run_criteo_check(synthetic, **SMALL)
    columns = [c for c in COLUMNS if c != "fit_seconds"]
    pd.testing.assert_frame_equal(result[columns], again[columns])


def test_uplift_models_beat_random_on_a_signal(result: pd.DataFrame) -> None:
    sub = result[result["outcome"] == "visit"]
    mean = sub.groupby("estimator")["qini_observed"].mean()
    assert mean["t_learner"] > mean[RANDOM_NAME]
    assert mean["x_learner"] > mean[RANDOM_NAME]


def test_restricting_estimators_and_outcomes(synthetic: Path) -> None:
    frame = run_criteo_check(
        synthetic, n=2_000, splits=1, base="logistic", outcomes=("visit",), estimators=("response",)
    )
    assert set(frame["estimator"]) == {RANDOM_NAME, "response"}
    assert set(frame["outcome"]) == {"visit"}


def test_unknown_estimator_is_rejected(synthetic: Path) -> None:
    with pytest.raises(ValueError, match="unknown estimators"):
        run_criteo_check(synthetic, n=1_000, splits=1, base="logistic", estimators=("nope",))


def test_report_files(result: pd.DataFrame, tmp_path: Path) -> None:
    md = write_criteo_report(result, tmp_path / "out")
    assert md == tmp_path / "out" / "CRITEO.md"
    text = md.read_text(encoding="utf-8")
    assert text.startswith("# Outcome-only real-data sanity check (Criteo Uplift, randomised)")
    assert FIXED_LINE in text
    assert "CC BY-NC-SA 4.0" in text and "criteo-synthetic.csv" in text
    assert "seed 20260830" in text and "4000 rows" in text
    for outcome in ("visit", "conversion"):
        assert f"## {outcome}" in text
    assert "splits beating random" in text and "Test-half ATE" in text
    assert "H1" in text.splitlines()[-1] and "H3" in text.splitlines()[-1]


def test_report_reads_sha256_beside_the_file(tmp_path: Path) -> None:
    path = _write_synthetic(tmp_path / "criteo-synthetic.csv")
    digest = "ab" * 32
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
    frame = run_criteo_check(path, n=1_000, splits=1, base="logistic", estimators=("response",))
    md = write_criteo_report(frame, tmp_path / "out").read_text(encoding="utf-8")
    assert digest in md


def test_csv_holds_aggregates_only(result: pd.DataFrame, tmp_path: Path) -> None:
    write_criteo_report(result, tmp_path)
    saved = pd.read_csv(tmp_path / "criteo.csv")
    assert list(saved.columns) == COLUMNS
    assert len(saved) == len(result) == 20
    assert not {f"f{i}" for i in range(12)} & set(saved.columns)
    assert not {"treatment", "visit", "exposure"} & set(saved.columns)


def test_cli_criteo_check(
    synthetic: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "res"
    argv = ["criteo-check", "--path", str(synthetic), "--n", "2000", "--splits", "1"]
    assert main([*argv, "--base", "logistic", "--out", str(out)]) == 0
    assert "CRITEO.md" in capsys.readouterr().out
    assert FIXED_LINE in (out / "CRITEO.md").read_text(encoding="utf-8")
    assert (out / "criteo.csv").exists()


def test_cli_criteo_check_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["criteo-check", "--path", str(tmp_path / "none.csv.gz"), "--out", str(tmp_path)])
    assert code == 2
    assert "not found" in capsys.readouterr().err
