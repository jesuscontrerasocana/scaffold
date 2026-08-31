"""Command line entry points. Provided by us. Do not change the interface.

    uv run python -m pipeline train    --data data/history.csv --model-dir models/
    uv run python -m pipeline simulate --data data/history.csv --model-dir models/ \
                                --from 2026-06-01 --to 2026-06-30 --out out/june.csv

`train` fits your forecaster and persists it. Slow is fine.

`simulate` rolls your pipeline forward, quarter hour by quarter hour, over the window you
name. Everything in the file before `--from` is history your forecaster may look at; nothing
there is scored. At each step it applies your decision against what the site actually did,
bills the result and draws it.

We run the second command on a month you have never seen.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from pipeline.data import load_timeseries
from pipeline.forecaster import Forecaster
from pipeline.harness import RunConfig, run_backtest
from pipeline.optimizer import Optimizer
from pipeline.report import write_results
from pipeline.specs import SiteSpecs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="fit the forecaster and persist it")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--model-dir", type=Path, default=Path("models"))
    train.add_argument("--site", type=Path, default=Path("site.yaml"))

    simulate = sub.add_parser(
        "simulate", help="roll the pipeline forward over a window"
    )
    simulate.add_argument("--data", type=Path, required=True)
    simulate.add_argument("--model-dir", type=Path, default=Path("models"))
    simulate.add_argument("--site", type=Path, default=Path("site.yaml"))
    simulate.add_argument("--out", type=Path, default=Path("out/simulation.csv"))
    simulate.add_argument(
        "--from",
        dest="from_time",
        default=None,
        help="first decision, e.g. 2026-06-01. Earlier rows are history.",
    )
    simulate.add_argument(
        "--to", dest="to_time", default=None, help="last decision, e.g. 2026-06-30"
    )
    simulate.add_argument(
        "--decision-interval-minutes",
        type=int,
        default=RunConfig.decision_interval_minutes,
    )
    simulate.add_argument("--horizon-steps", type=int, default=RunConfig.horizon_steps)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    specs = SiteSpecs.from_yaml(args.site)
    df = load_timeseries(args.data, timezone=specs.timezone)

    if args.command == "train":
        args.model_dir.mkdir(parents=True, exist_ok=True)
        forecaster = Forecaster(specs)
        started = time.perf_counter()
        forecaster.fit(df)
        forecaster.save(args.model_dir)
        print(
            f"trained on {len(df)} rows in {time.perf_counter() - started:.1f}s -> {args.model_dir}"
        )
        return 0

    def stamp(value: str | None, end_of_day: bool = False) -> pd.Timestamp | None:
        if value is None:
            return None
        ts = pd.Timestamp(value, tz=specs.timezone)
        if end_of_day and ts == ts.normalize():
            # "--to 2026-06-30" means all of the 30th, not one decision at midnight
            ts = ts + pd.Timedelta(hours=23, minutes=45)
        return ts

    config = RunConfig(
        decision_interval_minutes=args.decision_interval_minutes,
        horizon_steps=args.horizon_steps,
        first_decision=stamp(args.from_time),
        last_decision=stamp(args.to_time, end_of_day=True),
    )
    forecaster = Forecaster.load(args.model_dir, specs)
    optimizer = Optimizer(specs)

    started = time.perf_counter()
    schedule, summary = run_backtest(df, specs, forecaster, optimizer, config)
    summary["wall_clock_seconds"] = round(time.perf_counter() - started, 1)
    write_results(schedule, summary, args.out)
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {args.out} and {args.out.with_name(args.out.stem + '_plot.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
