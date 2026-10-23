"""Command-line interface: ``causal-uplift simulate | run-grid | report | did | download``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .dgp import SEGMENT_NAMES, simulate
from .experiments import BASE_SEED, SMOKE_CONFIG, load_config, run_grid
from .report import write_report

DEFAULT_GRID_CONFIG = Path("configs/grid.toml")
SMOKE_CONFIG_FILE = Path("configs/smoke.toml")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causal-uplift", description="Uplift estimators scored against a known true effect."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="simulate users and print a summary")
    sim.add_argument("--n", type=int, default=10_000)
    sim.add_argument("--seed", type=int, default=BASE_SEED)
    sim.add_argument("--gamma", type=float, default=0.0)
    sim.add_argument("--hidden", type=float, default=0.0)
    sim.add_argument("--effect", type=float, default=1.0)
    sim.add_argument("--noise", type=float, default=0.0)
    sim.add_argument("--out", type=Path, help="write the simulated frame to this CSV")

    grid = sub.add_parser("run-grid", help="run the experiment grid")
    grid.add_argument("--config", type=Path, default=DEFAULT_GRID_CONFIG)
    grid.add_argument("--smoke", action="store_true", help="tiny grid for a quick check")
    grid.add_argument("--out", type=Path, default=Path("results"))
    grid.add_argument("--workers", type=int, default=1)
    grid.add_argument("--reps", type=int, help="override replications per cell")

    rep = sub.add_parser("report", help="write REPORT.md from a finished run")
    rep.add_argument("--results", type=Path, default=Path("results"))

    did = sub.add_parser("did", help="difference-in-differences sweep over trend gaps")
    did.add_argument("--units", type=int, default=2000)
    did.add_argument("--reps", type=int, default=200)
    did.add_argument("--seed", type=int, default=BASE_SEED)

    dl = sub.add_parser("download", help="fetch a real dataset (needs network)")
    dl.add_argument("dataset", choices=["criteo"])
    dl.add_argument("--dest", type=Path, default=Path("data/raw"))
    return parser


def _cmd_simulate(args: argparse.Namespace) -> int:
    sim = simulate(
        args.n,
        args.seed,
        gamma=args.gamma,
        hidden=args.hidden,
        effect=args.effect,
        noise=args.noise,
    )
    treated = sim.t == 1
    both_arms = treated.any() and not treated.all()
    naive = sim.y[treated].mean() - sim.y[~treated].mean() if both_arms else float("nan")
    print(f"n:             {args.n}")
    print(f"treated share: {treated.mean():.4f}")
    print(f"true ATE:      {sim.true_ate:.4f}")
    print(f"naive ATE:     {naive:.4f}")
    for code, name in enumerate(SEGMENT_NAMES):
        print(f"segment {name + ':':<14} {(sim.segment == code).mean():.4f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        sim.to_frame().to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


def _cmd_run_grid(args: argparse.Namespace) -> int:
    if args.smoke:
        config = load_config(SMOKE_CONFIG_FILE) if SMOKE_CONFIG_FILE.exists() else SMOKE_CONFIG
    elif args.config.exists():
        config = load_config(args.config)
    else:
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2
    if args.reps is not None:
        config = config.with_reps(args.reps)
    if args.workers < 1:
        print("--workers must be at least 1", file=sys.stderr)
        return 2
    run_grid(config, args.out, workers=args.workers)
    print(f"wrote {args.out / 'grid.csv'}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    try:
        path = write_report(args.results)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 2
    print(f"wrote {path}")
    return 0


def _cmd_did(args: argparse.Namespace) -> int:
    try:
        from .did import sweep
    except ImportError:
        print("the difference-in-differences module is not available yet", file=sys.stderr)
        return 1
    table = sweep([0.0, 0.01, 0.03, 0.05], args.units, args.reps, args.seed)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    try:
        from .data import download_criteo
    except ImportError:
        print("the data loader is not available yet", file=sys.stderr)
        return 1
    download_criteo(args.dest)
    return 0


_COMMANDS = {
    "simulate": _cmd_simulate,
    "run-grid": _cmd_run_grid,
    "report": _cmd_report,
    "did": _cmd_did,
    "download": _cmd_download,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = _build_parser().parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
