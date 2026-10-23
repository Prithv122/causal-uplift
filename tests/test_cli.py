"""Command-line interface."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from causal_uplift.cli import main


def test_package_imports() -> None:
    import causal_uplift

    assert causal_uplift.__version__


def test_simulate_prints_summary_and_writes_csv(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out = tmp_path / "sim.csv"
    assert main(["simulate", "--n", "500", "--seed", "1", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    for label in ("true ATE", "naive ATE", "treated share", "persuadable"):
        assert label in printed
    frame = pd.read_csv(out)
    assert len(frame) == 500 and {"x0", "t", "y", "tau"} <= set(frame.columns)


def test_run_grid_smoke_then_report(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    out = tmp_path / "res"
    assert main(["run-grid", "--smoke", "--out", str(out), "--reps", "1"]) == 0
    assert (out / "grid.csv").exists()
    assert len(list((out / "grid").glob("*.csv"))) == 2
    assert main(["report", "--results", str(out)]) == 0
    assert "REPORT.md" in capsys.readouterr().out
    assert "# Results report" in (out / "REPORT.md").read_text(encoding="utf-8")


def test_report_without_results_fails_cleanly(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["report", "--results", str(tmp_path / "nothing")]) == 2
    assert "run the grid first" in capsys.readouterr().err


def test_missing_config_fails_cleanly(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code = main(["run-grid", "--config", str(tmp_path / "none.toml"), "--out", str(tmp_path)])
    assert code == 2
    assert "config not found" in capsys.readouterr().err


def test_did_prints_table(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("causal_uplift.did")
    assert main(["did", "--units", "200", "--reps", "3"]) == 0
    assert "did_2x2" in capsys.readouterr().out
