"""Command line entry point.

    uv run memco-harness run --tasks 30 --seed 42 --paired
    uv run memco-harness report [--run RUN_ID]

`--tasks` defaults to 30, the quick first look; 100 or more shows the fuller
convergence. `report` with no argument reports the latest run and pairs it with
its control arm if one is there.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import __version__
from .report import (
    find_control,
    find_run,
    latest_run,
    reconcile,
    render_reconciliation,
    render_report,
    write_html,
)
from .runner import DEFAULT_RESULTS_DIR, RunConfig
from .runner import run as run_episode

__all__ = ["main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memco-harness",
        description="Learn a business's unwritten policies from the corrections "
        "its reviewers already make.",
    )
    parser.add_argument("--version", action="version", version=f"memco-harness {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run a set of tasks and record the results")
    run_parser.add_argument(
        "--tasks", type=int, default=30, help="how many tasks to run (default 30)"
    )
    run_parser.add_argument("--seed", type=int, default=42, help="fixes task and variant order")
    run_parser.add_argument(
        "--paired",
        action="store_true",
        help="answer every task twice from the same email, once with memory and once "
             "blind, and report the difference. Replaces running the two arms separately",
    )
    run_parser.add_argument(
        "--no-memory",
        action="store_true",
        help="a single blind arm: no memory search, no reflection. Prefer --paired",
    )
    run_parser.add_argument(
        "--resume",
        dest="resume_from",
        default="",
        metavar="RUN_ID",
        help="continue a paired run that stopped part way, by its run id. Needs the "
             "same --tasks and --seed, and a memory store that has not been cleared "
             "since",
    )
    run_parser.add_argument(
        "--all-variants",
        action="store_true",
        help="run every variant of every task in file order, ignoring --tasks and --seed",
    )
    run_parser.add_argument(
        "--label",
        default="",
        help="a word to put in the run id, e.g. pass1, to tell repeated runs apart",
    )
    run_parser.add_argument(
        "--pace",
        type=float,
        default=10.0,
        dest="pace_seconds",
        metavar="SECONDS",
        help="gap between episodes, to stay under the memory server's rate limit "
        "(default 10; 0 turns it off; ignored with --no-memory)",
    )
    run_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)

    report_parser = sub.add_parser("report", help="summarise a run and render its HTML report")
    report_parser.add_argument("--run", dest="run_id", help="defaults to the latest run")
    report_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)

    reconcile_parser = sub.add_parser(
        "reconcile",
        help="ask the store what it kept of a run's lessons (read-only, needs the network)",
    )
    reconcile_parser.add_argument("--run", dest="run_id", help="defaults to the latest run")
    reconcile_parser.add_argument(
        "--pace",
        type=float,
        default=6.0,
        dest="pace_seconds",
        metavar="SECONDS",
        help="gap between probes, to stay under the server's rate limit (default 6)",
    )
    reconcile_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "reconcile":
        return _reconcile(args)
    return _report(args)


def _run(args: argparse.Namespace) -> int:
    config = RunConfig(
        task_count=args.tasks,
        seed=args.seed,
        memory_enabled=not args.no_memory,
        paired=args.paired,
        resume_from=args.resume_from,
        all_variants=args.all_variants,
        label=args.label,
        pace_seconds=args.pace_seconds,
        results_dir=args.results_dir,
    )
    if config.paired and args.no_memory:
        print("--paired runs both arms; --no-memory has nothing to add to it")
        return 2
    result = run_episode(config)
    print(f"\nwrote {result.jsonl_path}")
    if config.paired:
        print(f"       {result.jsonl_path.with_suffix('.pairs.jsonl')}")
    print(f"report it with: uv run memco-harness report --run {result.run_id}")
    if config.memory_enabled:
        print(f"reconcile it with: uv run memco-harness reconcile --run {result.run_id}")
    return 0


def _reconcile(args: argparse.Namespace) -> int:
    """Ask the store what it kept. Read-only: no writes, no feedback."""
    from time import sleep

    from .memco_client import build_client

    results_dir: Path = args.results_dir
    run = find_run(results_dir, args.run_id) if args.run_id else latest_run(results_dir)
    if run is None:
        print(f"no results found for {args.run_id or 'any run'}", file=sys.stderr)
        return 1

    client = build_client()
    result = reconcile(run, client.search, pace=sleep, pace_seconds=args.pace_seconds)
    print(f"run {run.run_id}")
    print(render_reconciliation(result))
    return 0


def _report(args: argparse.Namespace) -> int:
    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        print(f"no results directory at {results_dir}", file=sys.stderr)
        return 1

    run = find_run(results_dir, args.run_id) if args.run_id else latest_run(results_dir)
    if run is None:
        target = args.run_id or "any run"
        print(f"no results found for {target} in {results_dir}", file=sys.stderr)
        return 1

    control = find_control(results_dir, run)
    # The page belongs to the run, not to the arm the report is written from:
    # splitting renames the run, and the file name must not follow it.
    page = results_dir / f"{run.run_id}.report.html"
    if run.paired:
        # The file holds both arms; the report is about the memory arm, with the
        # control arm beside it.
        run = run.as_arm("memory")
    print(render_report(run, control))
    html_path = write_html(run, control, page)
    print(f"\nwrote {html_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
