"""Replication and grid runner: columns, determinism, seeds, resume, worker independence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_uplift import experiments
from causal_uplift.estimators import ESTIMATORS
from causal_uplift.experiments import (
    COLUMNS,
    SMOKE_CONFIG,
    Cell,
    GridConfig,
    load_config,
    run_grid,
    run_replication,
)

DESIGN_COLUMNS = [
    "cell_id", "gamma", "hidden", "n", "effect", "noise", "rep", "sim_seed", "estimator", "base",
    "true_ate", "ate_hat", "ate_se", "ate_lo", "ate_hi", "covered",
    "pehe", "spearman", "qini_true",
    "policy_top10", "policy_top20", "policy_top30",
    "oracle_top20", "random_top20", "cost", "fit_seconds",
]  # fmt: skip

TINY = Cell(gamma=0.5, hidden=0.0, n=500, n_eval=500)


def tiny_config(estimators: tuple[str, ...] | None = None, reps: int = 2) -> GridConfig:
    names = tuple(ESTIMATORS) if estimators is None else estimators
    cells = (TINY, Cell(gamma=0.0, hidden=0.0, n=500, n_eval=500))
    return GridConfig(cells=cells, reps=reps, base="logistic", estimators=names)


def test_columns_match_design() -> None:
    assert list(COLUMNS) == DESIGN_COLUMNS


def test_replication_columns_and_rows() -> None:
    config = tiny_config()
    frame = pd.DataFrame(run_replication(TINY, 0, 0, config))
    assert list(frame.columns) == DESIGN_COLUMNS
    assert list(frame["estimator"]) == list(config.estimators)
    assert frame["cell_id"].eq(TINY.cell_id).all()
    assert frame["true_ate"].nunique() == 1


def test_replication_nan_pattern() -> None:
    frame = pd.DataFrame(run_replication(TINY, 0, 0, tiny_config())).set_index("estimator")
    assert frame.loc["response", ["ate_hat", "pehe", "covered"]].isna().all()
    assert frame.loc["response", ["spearman", "qini_true"]].notna().all()
    ipw_nan = ["pehe", "spearman", "qini_true", "policy_top10", "policy_top20", "policy_top30"]
    assert frame.loc["ipw", ipw_nan].isna().all()
    assert frame.loc["ipw", ["ate_hat", "ate_se", "covered"]].notna().all()
    assert not np.isnan(frame.loc["naive", "pehe"])
    assert frame.loc["naive", "qini_true"] == pytest.approx(0.0, abs=1e-9)
    for name in ("s_learner", "t_learner", "x_learner"):
        assert frame.loc[name, ["ate_se", "ate_lo", "ate_hi", "covered"]].isna().all()
        assert not np.isnan(frame.loc[name, "ate_hat"])
    assert frame.loc[["naive", "dr_learner", "ipw"], "covered"].isin([0.0, 1.0]).all()


def test_replication_is_deterministic() -> None:
    config = tiny_config()
    first = pd.DataFrame(run_replication(TINY, 3, 4, config)).drop(columns="fit_seconds")
    second = pd.DataFrame(run_replication(TINY, 3, 4, config)).drop(columns="fit_seconds")
    pd.testing.assert_frame_equal(first, second)
    other = pd.DataFrame(run_replication(TINY, 3, 5, config)).drop(columns="fit_seconds")
    assert not first["true_ate"].equals(other["true_ate"])


def test_estimator_seed_independent_of_list_composition() -> None:
    both = pd.DataFrame(run_replication(TINY, 1, 2, tiny_config(("t_learner", "dr_learner"))))
    alone = pd.DataFrame(run_replication(TINY, 1, 2, tiny_config(("dr_learner",))))
    a = both[both["estimator"] == "dr_learner"].drop(columns="fit_seconds").reset_index(drop=True)
    b = alone.drop(columns="fit_seconds").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_cell_id_stable_and_filename_safe() -> None:
    assert Cell(0.5, 0.0, n=10_000).cell_id == "g0.50_h0.00_n10000"
    assert Cell(0.25, 1.0, effect=2.0, noise=0.5).cell_id == "g0.25_h1.00_n10000_e2_s0.5"


def test_config_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown"):
        GridConfig(cells=(TINY,), estimators=("nope",))
    with pytest.raises(ValueError, match="duplicate"):
        GridConfig(cells=(TINY, TINY))
    bad = tmp_path / "bad.toml"
    bad.write_text("reps = 1\n[[cells]]\ngamma = 0.0\n")
    with pytest.raises(ValueError, match="bad cell"):
        load_config(bad)


def test_shipped_configs_load() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    grid = load_config(root / "grid.toml")
    assert len(grid.cells) == 10 and grid.reps == 200 and grid.base == "lgbm"
    assert (grid.cost, grid.base_seed) == (0.02, 20260830)
    pairs = [(c.gamma, c.hidden) for c in grid.cells]
    assert pairs[:6] == [(0.0, 0.0), (0.1, 0.0), (0.25, 0.0), (0.5, 0.0), (1.0, 0.0), (2.0, 0.0)]
    assert pairs[6:] == [(0.5, 0.5), (0.5, 1.0), (1.0, 0.5), (1.0, 1.0)]
    assert all(c.n == 10_000 and c.n_eval == 20_000 for c in grid.cells)
    assert load_config(root / "smoke.toml") == SMOKE_CONFIG


def test_run_grid_writes_resumes_and_matches_across_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tiny_config(("naive", "response", "ipw"))
    serial = run_grid(config, tmp_path / "w1", workers=1, progress=False)
    assert len(serial) == 2 * config.reps * 3
    assert list(serial.columns) == DESIGN_COLUMNS
    grid_dir = tmp_path / "w1" / "grid"
    assert sorted(p.name for p in grid_dir.glob("*.csv")) == sorted(
        f"{c.cell_id}.csv" for c in config.cells
    )
    assert (tmp_path / "w1" / "run_meta.json").exists()
    on_disk = pd.read_csv(tmp_path / "w1" / "grid.csv")
    assert len(on_disk) == len(serial)
    cell_file = pd.read_csv(grid_dir / f"{TINY.cell_id}.csv")
    assert list(cell_file["rep"]) == sorted(cell_file["rep"])

    # Resume: a finished run must not call the replication function again.
    def boom(*args, **kwargs):
        raise AssertionError("replication re-run on resume")

    monkeypatch.setattr(experiments, "run_replication", boom)
    resumed = run_grid(config, tmp_path / "w1", workers=1, progress=False)
    pd.testing.assert_frame_equal(resumed, on_disk)
    monkeypatch.undo()

    # Resume with one cell file missing re-runs only that cell, with identical numbers.
    (grid_dir / f"{TINY.cell_id}.csv").unlink()
    repaired = run_grid(config, tmp_path / "w1", workers=1, progress=False)
    pd.testing.assert_frame_equal(
        repaired.drop(columns="fit_seconds"), on_disk.drop(columns="fit_seconds")
    )

    parallel = run_grid(config, tmp_path / "w2", workers=2, progress=False)
    pd.testing.assert_frame_equal(
        serial.drop(columns="fit_seconds").reset_index(drop=True),
        parallel.drop(columns="fit_seconds").reset_index(drop=True),
    )


def test_run_meta_records_the_source_state_from_before_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_git(*args: str) -> str:
        calls.append(args)
        return "" if args[0] == "status" else "abc123"

    monkeypatch.setattr(experiments, "_git", fake_git)
    config = tiny_config(("naive",))
    real_task = experiments._run_task

    def task_after_git(task):
        assert calls, "source state must be read before any replication runs"
        return real_task(task)

    monkeypatch.setattr(experiments, "_run_task", task_after_git)
    run_grid(config, tmp_path, workers=1, progress=False)
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["git_commit"] == "abc123" and meta["git_tracked_changes"] is False
    assert ("status", "--porcelain", "--untracked-files=no") in calls
