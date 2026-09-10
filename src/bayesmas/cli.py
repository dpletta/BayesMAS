"""Command-line entry points for the two CPU experiments."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from bayesmas import __version__
from bayesmas.puzzles.planner import run_puzzle_sweep
from bayesmas.simulation import ComplexitySetting, run_experiment


def _print_rows(rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    print("\t".join(keys))
    for row in rows:
        print("\t".join(_fmt(row[k]) for k in keys))


def _fmt(value: float | str | int) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _cmd_simulate(args: argparse.Namespace) -> int:
    settings = (
        [getattr(ComplexitySetting, args.setting)()]
        if args.setting != "all"
        else list(ComplexitySetting.all_regimes())
    )
    summary = run_experiment(
        settings=settings,
        trials=args.trials,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    rows = summary.rows()
    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_rows(rows)
    return 0


def _cmd_puzzles(args: argparse.Namespace) -> int:
    puzzles = (args.puzzle,) if args.puzzle != "all" else ("hanoi", "checkers", "river", "blocks")
    sizes = tuple(args.n) if args.n else (2, 3)
    rows = run_puzzle_sweep(
        puzzles=puzzles,
        sizes=sizes,
        trials=args.trials,
        seed=args.seed,
        n_agents=args.agents,
        n_byzantine=args.byzantine,
        noise=args.noise,
    )
    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_rows(rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayesmas",
        description="Empirical Bayes multi-agent collaboration (CPU experiments).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="opinion-dynamics Monte Carlo")
    sim.add_argument("--setting", choices=("low", "medium", "high", "all"), default="all")
    sim.add_argument("--trials", type=int, default=100)
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--max-steps", type=int, default=15)
    sim.add_argument("--json", action="store_true")
    sim.set_defaults(func=_cmd_simulate)

    puz = sub.add_parser("puzzles", help="Apple-style puzzle complexity sweep")
    puz.add_argument(
        "--puzzle",
        choices=("hanoi", "checkers", "river", "blocks", "all"),
        default="all",
    )
    puz.add_argument(
        "--n",
        type=int,
        nargs="*",
        default=None,
        help="complexity values (default: 2 3)",
    )
    puz.add_argument("--trials", type=int, default=10)
    puz.add_argument("--seed", type=int, default=0)
    puz.add_argument("--agents", type=int, default=5)
    puz.add_argument("--byzantine", type=int, default=1)
    puz.add_argument("--noise", type=float, default=1.0)
    puz.add_argument("--json", action="store_true")
    puz.set_defaults(func=_cmd_puzzles)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
