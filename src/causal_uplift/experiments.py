"""The experiment grid: simulate cells, fit every estimator, score it against the known truth.

A *cell* is one knob setting; a *replication* is one fresh simulation of that cell. Every random
number in a replication is derived from ``SeedSequence([base_seed, cell_index, rep])``, so a row
depends only on its own (cell, rep, estimator) and never on worker count, run order or which
other estimators are in the list.
"""

from __future__ import annotations

import json
import multiprocessing
import platform
import subprocess
import sys
import time
import tomllib
import zlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .dgp import simulate
from .estimators import ESTIMATORS, available_estimators
from .estimators.base import AteResult, BaseKind

BASE_SEED = 20260830
DEFAULT_COST = 0.02

COLUMNS = (
    "cell_id",
    "gamma",
    "hidden",
    "n",
    "effect",
    "noise",
    "rep",
    "sim_seed",
    "estimator",
    "base",
    "true_ate",
    "ate_hat",
    "ate_se",
    "ate_lo",
    "ate_hi",
    "covered",
    "pehe",
    "spearman",
    "qini_true",
    "policy_top10",
    "policy_top20",
    "policy_top30",
    "oracle_top20",
    "random_top20",
    "cost",
    "fit_seconds",
)

# The ipw arms' "score" is a constant ATE, so it has no individual-level estimate to score.
_NO_INDIVIDUAL_METRICS = frozenset({"ipw", "ipw_logit"})
_POLICY_FRACTIONS = {"policy_top10": 0.10, "policy_top20": 0.20, "policy_top30": 0.30}
_NAN = float("nan")


@dataclass(frozen=True)
class Cell:
    """One knob setting. ``n`` training users per replication, ``n_eval`` fresh test users."""

    gamma: float
    hidden: float
    n: int = 10_000
    n_eval: int = 20_000
    effect: float = 1.0
    noise: float = 0.0

    @property
    def cell_id(self) -> str:
        """Stable, filename-safe label; effect and noise appear only when not the default."""
        label = f"g{self.gamma:.2f}_h{self.hidden:.2f}_n{self.n}"
        if self.effect != 1.0:
            label += f"_e{self.effect:g}"
        if self.noise != 0.0:
            label += f"_s{self.noise:g}"
        return label


@dataclass(frozen=True)
class GridConfig:
    """A full grid: cells, replications per cell, base learner, cost and estimator names."""

    cells: tuple[Cell, ...]
    reps: int = 200
    base: BaseKind = "lgbm"
    cost: float = DEFAULT_COST
    estimators: tuple[str, ...] = field(default_factory=lambda: tuple(available_estimators()))
    base_seed: int = BASE_SEED

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("the grid needs at least one cell")
        ids = [c.cell_id for c in self.cells]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate cells in the grid: {sorted(ids)}")
        if self.reps < 1:
            raise ValueError(f"reps must be at least 1, got {self.reps}")
        if self.base not in ("lgbm", "logistic"):
            raise ValueError(f"base must be 'lgbm' or 'logistic', got {self.base!r}")
        known = available_estimators()
        unknown = [name for name in self.estimators if name not in known]
        if unknown:
            raise ValueError(f"unknown or unavailable estimators {unknown}; have {sorted(known)}")
        if not self.estimators:
            raise ValueError("the grid needs at least one estimator")

    def with_reps(self, reps: int) -> GridConfig:
        """A copy with a different number of replications."""
        return GridConfig(self.cells, reps, self.base, self.cost, self.estimators, self.base_seed)


# Embedded so ``run-grid --smoke`` works from any directory; configs/smoke.toml mirrors it.
SMOKE_CONFIG = GridConfig(
    cells=(
        Cell(gamma=0.0, hidden=0.0, n=2_000, n_eval=2_000),
        Cell(gamma=1.0, hidden=0.0, n=2_000, n_eval=2_000),
    ),
    reps=3,
    base="logistic",
    estimators=tuple(ESTIMATORS),  # the core arms; the optional forest is too slow for a smoke run
)


def load_config(path: str | Path) -> GridConfig:
    """Read a grid from TOML: top-level ``reps``, ``base``, ``cost``, ``estimators``, ``base_seed``
    and one ``[[cells]]`` table per cell (``gamma`` and ``hidden`` required)."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    cell_keys = {"gamma", "hidden", "n", "n_eval", "effect", "noise"}
    cells = []
    for entry in raw.get("cells", []):
        extra = set(entry) - cell_keys
        if extra or "gamma" not in entry or "hidden" not in entry:
            raise ValueError(f"bad cell {entry}: need gamma and hidden, allowed keys {cell_keys}")
        cells.append(Cell(**entry))
    top_keys = {"reps", "base", "cost", "estimators", "base_seed", "cells"}
    if set(raw) - top_keys:
        raise ValueError(f"unknown config keys: {sorted(set(raw) - top_keys)}")
    options = {k: raw[k] for k in ("reps", "base", "cost", "base_seed") if k in raw}
    if "estimators" in raw:
        options["estimators"] = tuple(raw["estimators"])
    return GridConfig(cells=tuple(cells), **options)


def _seed31(entropy: Sequence[int]) -> int:
    """A 31-bit seed (safe for every library) derived from a SeedSequence entropy list."""
    return int(np.random.SeedSequence(list(entropy)).generate_state(1)[0] & 0x7FFFFFFF)


def run_replication(cell: Cell, cell_index: int, rep: int, config: GridConfig) -> list[dict]:
    """Simulate one training and one evaluation draw and score every estimator.

    The estimator seed is derived from the estimator's name (CRC32), not its position, so a row
    is the same whichever other estimators are run alongside it.
    """
    root = [config.base_seed, cell_index, rep]
    train_seed, eval_seed = (_seed31([*root, tag]) for tag in (0, 1))
    knobs = {"gamma": cell.gamma, "hidden": cell.hidden, "effect": cell.effect, "noise": cell.noise}
    train = simulate(cell.n, train_seed, **knobs)
    test = simulate(cell.n_eval, eval_seed, **knobs)
    tau = test.tau
    true_ate = train.true_ate
    factories = available_estimators()

    oracle_top20 = metrics.oracle_policy_value(tau, 0.20, config.cost)
    random_top20 = metrics.random_policy_value(tau, 0.20, config.cost)

    rows = []
    for name in config.estimators:
        seed = _seed31([*root, 2, zlib.crc32(name.encode())])
        estimator = factories[name](seed, config.base)
        start = time.perf_counter()
        estimator.fit(train.X, train.t, train.y)
        fit_seconds = time.perf_counter() - start
        score = np.asarray(estimator.predict_tau(test.X), dtype=float)
        ate = estimator.ate()

        row = {
            "cell_id": cell.cell_id,
            "gamma": cell.gamma,
            "hidden": cell.hidden,
            "n": cell.n,
            "effect": cell.effect,
            "noise": cell.noise,
            "rep": rep,
            "sim_seed": train_seed,
            "estimator": name,
            "base": config.base,
            "true_ate": true_ate,
            **_ate_columns(ate, true_ate),
        }
        scored = name not in _NO_INDIVIDUAL_METRICS
        row["pehe"] = metrics.pehe(score, tau) if scored and estimator.estimates_tau else _NAN
        row["spearman"] = metrics.spearman(score, tau) if scored else _NAN
        row["qini_true"] = metrics.qini_true(score, tau) if scored else _NAN
        for column, frac in _POLICY_FRACTIONS.items():
            row[column] = metrics.policy_value(score, tau, frac, config.cost) if scored else _NAN
        row["oracle_top20"] = oracle_top20
        row["random_top20"] = random_top20
        row["cost"] = config.cost
        row["fit_seconds"] = fit_seconds
        rows.append({column: row[column] for column in COLUMNS})
    return rows


def _ate_columns(ate: AteResult | None, true_ate: float) -> dict[str, float]:
    """ATE columns; ``covered`` is 1.0 / 0.0, or NaN when there is no interval."""
    if ate is None:
        return dict.fromkeys(("ate_hat", "ate_se", "ate_lo", "ate_hi", "covered"), _NAN)
    has_interval = not (np.isnan(ate.lo) or np.isnan(ate.hi))
    return {
        "ate_hat": ate.estimate,
        "ate_se": ate.se,
        "ate_lo": ate.lo,
        "ate_hi": ate.hi,
        "covered": float(metrics.covered(ate.lo, ate.hi, true_ate)) if has_interval else _NAN,
    }


def _run_task(task: tuple[Cell, int, int, GridConfig]) -> tuple[int, int, list[dict]]:
    """Worker entry point (top level so it pickles on Windows)."""
    cell, cell_index, rep, config = task
    return cell_index, rep, run_replication(cell, cell_index, rep, config)


def _write_cell(path: Path, rows: list[dict]) -> None:
    """Write a finished cell atomically, so an interrupted run never leaves a partial file."""
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    tmp = path.with_suffix(".csv.part")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def run_grid(
    config: GridConfig,
    out_dir: str | Path,
    workers: int = 1,
    progress: bool = True,
) -> pd.DataFrame:
    """Run every unfinished cell, write ``out_dir/grid/<cell_id>.csv``, then combine.

    Cells whose CSV already exists are skipped (resume). Returns the combined frame, also
    written to ``out_dir/grid.csv``, and records ``out_dir/run_meta.json``.
    """
    out = Path(out_dir)
    source = _source_state()  # read before any result is written, so it names the code that ran
    grid_dir = out / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    clock = time.perf_counter()

    paths = {i: grid_dir / f"{cell.cell_id}.csv" for i, cell in enumerate(config.cells)}
    todo = [i for i in paths if not paths[i].exists()]
    skipped = [config.cells[i].cell_id for i in paths if i not in todo]
    tasks = [(config.cells[i], i, rep, config) for i in todo for rep in range(config.reps)]

    pending: dict[int, dict[int, list[dict]]] = {i: {} for i in todo}
    finished = 0

    def collect(cell_index: int, rep: int, rows: list[dict]) -> None:
        nonlocal finished
        pending[cell_index][rep] = rows
        if len(pending[cell_index]) < config.reps:
            return
        ordered = [row for r in sorted(pending[cell_index]) for row in pending[cell_index][r]]
        _write_cell(paths[cell_index], ordered)
        del pending[cell_index]
        finished += 1
        if progress:
            elapsed = time.perf_counter() - clock
            label = config.cells[cell_index].cell_id
            print(
                f"[{finished}/{len(todo)}] {label} done ({config.reps} reps), {elapsed:.0f}s",
                file=sys.stderr,
                flush=True,
            )

    if progress and skipped:
        print(f"skipping {len(skipped)} finished cell(s): {', '.join(skipped)}", file=sys.stderr)
    if workers > 1 and tasks:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            for result in pool.imap_unordered(_run_task, tasks):
                collect(*result)
    else:
        for task in tasks:
            collect(*_run_task(task))

    frames = [pd.read_csv(p) for p in paths.values() if p.exists()]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out / "grid.csv", index=False)
    _write_meta(out / "run_meta.json", config, workers, started, skipped, source)
    return combined


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def _source_state() -> dict:
    """The commit and whether any tracked file differs from it (untracked outputs don't count)."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_tracked_changes": None if status is None else status != "",
    }


def _write_meta(
    path: Path,
    config: GridConfig,
    workers: int,
    started: datetime,
    skipped: list[str],
    source: dict,
) -> None:
    """Record what was run, with what, and from which commit (as it was when the run started)."""
    meta = {
        "config": asdict(config),
        "workers": workers,
        "started": started.isoformat(),
        "finished": datetime.now(UTC).isoformat(),
        "cells_skipped_as_already_done": skipped,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "scipy")
        },
        **source,
    }
    path.write_text(json.dumps(meta, indent=2) + "\n")
