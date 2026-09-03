"""Select a forecast model and training lookback with monthly rolling origins."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from pipeline.data import STEP, load_timeseries
from pipeline.forecaster import Forecaster
from pipeline.harness import RunConfig
from pipeline.specs import SiteSpecs
if __package__:
    from scripts.evaluate_forecast import (
        calculate_metrics,
        collect_forecasts,
        component_metrics,
    )
else:
    from evaluate_forecast import calculate_metrics, collect_forecasts, component_metrics


DEFAULT_MODELS = ["ridge", "ridge_decomposed", "ridge_hgbr_decomposed"]
DEFAULT_LOOKBACKS = ["1m", "3m", "expanding"]
METRIC_COLUMNS = [
    "net_mae_kw",
    "net_rmse_kw",
    "net_bias_kw",
    "net_nmae",
    "load_mae_kw",
    "pv_mae_kw",
    "pv_rmse_kw",
    "lead_1_mae_kw",
    "first_hour_mae_kw",
    "first_4_hours_mae_kw",
    "full_horizon_mae_kw",
]


def monthly_folds(
    from_time: pd.Timestamp, to_time: pd.Timestamp
) -> list[pd.Timestamp]:
    """Return local-time month starts touched by an inclusive validation range."""

    if from_time.tzinfo is None or to_time.tzinfo is None:
        raise ValueError("Validation timestamps must be timezone-aware")
    if from_time > to_time:
        raise ValueError("Validation start must not be after validation end")
    starts = pd.date_range(
        from_time.normalize().replace(day=1),
        to_time.normalize().replace(day=1),
        freq="MS",
    )
    if any(start.year == 2026 and start.month == 8 for start in starts):
        raise ValueError("August 2026 is reserved as the final holdout")
    return list(starts)


def training_window(
    data: pd.DataFrame, validation_start: pd.Timestamp, lookback: str
) -> pd.DataFrame:
    """Select leakage-safe training observations for one validation month."""

    training = data.loc[data.index < validation_start]
    if lookback == "expanding":
        return training
    months = {"1m": 1, "3m": 3}.get(lookback)
    if months is None:
        raise ValueError(f"Unknown lookback: {lookback}")
    start = validation_start - pd.DateOffset(months=months)
    return training.loc[training.index >= start]


def horizon_metrics(comparisons: pd.DataFrame) -> dict[str, float]:
    """Calculate pooled net-load MAE for the requested lead ranges."""

    def mae(max_lead: int | None) -> float:
        rows = comparisons
        if max_lead is not None:
            rows = rows.loc[rows["lead_steps"] <= max_lead]
        return calculate_metrics(rows["actual_kw"], rows["predicted_kw"])["mae_kw"]

    return {
        "lead_1_mae_kw": mae(1),
        "first_hour_mae_kw": mae(4),
        "first_4_hours_mae_kw": mae(16),
        "full_horizon_mae_kw": mae(None),
    }


def fold_metrics(comparisons: pd.DataFrame) -> dict[str, float | int]:
    """Flatten the required accuracy and runtime diagnostics for one fold."""

    components = component_metrics(comparisons)
    net = components["net"]
    result: dict[str, float | int] = {
        "net_mae_kw": net["mae_kw"],
        "net_rmse_kw": net["rmse_kw"],
        "net_bias_kw": net["bias_kw"],
        "net_nmae": net["nmae"],
        **horizon_metrics(comparisons),
    }
    for component in ("load", "pv"):
        values = components.get(component, {})
        result[f"{component}_mae_kw"] = values.get("mae_kw", float("nan"))
    result["pv_rmse_kw"] = components.get("pv", {}).get("rmse_kw", float("nan"))
    seconds = float(comparisons.attrs.get("forecast_seconds", 0.0))
    decisions = int(comparisons.attrs.get("n_decisions", 0))
    result.update(
        forecast_seconds=seconds,
        n_decisions=decisions,
        forecast_seconds_per_decision=seconds / decisions if decisions else float("nan"),
    )
    return result


def aggregate_results(folds: pd.DataFrame) -> pd.DataFrame:
    """Produce one mean-validation row per model and lookback."""

    if folds.empty:
        raise ValueError("No fold results to aggregate")
    rows = []
    for (model, lookback), group in folds.groupby(["model", "lookback"], sort=False):
        row: dict[str, object] = {
            "model": model,
            "lookback": lookback,
            "n_folds": len(group),
        }
        for column in METRIC_COLUMNS:
            row[column] = group[column].mean()
        row["net_mae_std_kw"] = group["net_mae_kw"].std(ddof=0)
        row["forecast_seconds"] = group["forecast_seconds"].sum()
        row["n_decisions"] = int(group["n_decisions"].sum())
        row["forecast_seconds_per_decision"] = (
            row["forecast_seconds"] / row["n_decisions"]
            if row["n_decisions"]
            else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def select_configuration(results: pd.DataFrame) -> pd.Series:
    """Apply the deterministic quality-first selection rule from issue #34."""

    ranked = results.assign(_absolute_bias=results["net_bias_kw"].abs()).sort_values(
        [
            "net_mae_kw",
            "lead_1_mae_kw",
            "first_hour_mae_kw",
            "net_rmse_kw",
            "_absolute_bias",
            "net_mae_std_kw",
            "forecast_seconds_per_decision",
            "model",
            "lookback",
        ],
        kind="stable",
    )
    return ranked.iloc[0].drop(labels="_absolute_bias")


def run_selection(
    data: pd.DataFrame,
    specs: SiteSpecs,
    validation_starts: list[pd.Timestamp],
    models: list[str],
    lookbacks: list[str],
    horizon_steps: int,
    decision_interval_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit each candidate once per monthly fold and evaluate rolling forecasts."""

    rows = []
    for model_name in models:
        for lookback in lookbacks:
            for validation_start in validation_starts:
                next_month = validation_start + pd.DateOffset(months=1)
                validation_end = next_month - STEP
                training = training_window(data, validation_start, lookback)
                if training.empty:
                    raise ValueError(
                        f"No training data for {model_name} + {lookback} before "
                        f"{validation_start:%Y-%m}"
                    )
                fold_data = data.loc[data.index < next_month]
                last_decision = min(validation_end, fold_data.index[-1])
                if last_decision < validation_start:
                    raise ValueError(f"No validation data for {validation_start:%Y-%m}")

                forecaster = Forecaster(specs, model_name=model_name)
                started = time.perf_counter()
                forecaster.fit(training)
                training_seconds = time.perf_counter() - started
                comparisons = collect_forecasts(
                    fold_data,
                    forecaster,
                    RunConfig(
                        first_decision=validation_start,
                        last_decision=last_decision,
                        horizon_steps=horizon_steps,
                        decision_interval_minutes=decision_interval_minutes,
                    ),
                )
                rows.append(
                    {
                        "model": model_name,
                        "lookback": lookback,
                        "validation_month": validation_start.strftime("%Y-%m"),
                        "training_start": training.index[0],
                        "training_end": training.index[-1],
                        "training_rows": len(training),
                        "training_seconds": training_seconds,
                        **fold_metrics(comparisons),
                    }
                )
    folds = pd.DataFrame(rows)
    return folds, aggregate_results(folds)


def _timestamp(value: str, timezone: str, end_of_day: bool = False) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    timestamp = (
        timestamp.tz_localize(timezone)
        if timestamp.tzinfo is None
        else timestamp.tz_convert(timezone)
    )
    if end_of_day and timestamp == timestamp.normalize():
        timestamp += pd.Timedelta(hours=23, minutes=45)
    return timestamp


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/history.csv"))
    parser.add_argument("--from", dest="from_time", default="2026-04-01")
    parser.add_argument("--to", dest="to_time", default="2026-06-30")
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--lookbacks", nargs="+", choices=DEFAULT_LOOKBACKS, default=DEFAULT_LOOKBACKS)
    parser.add_argument("--site", type=Path, default=Path("site.yaml"))
    parser.add_argument("--out", type=Path, default=Path("out/model_selection.csv"))
    parser.add_argument("--horizon-steps", type=int, default=RunConfig.horizon_steps)
    parser.add_argument(
        "--decision-interval-minutes",
        type=int,
        default=RunConfig.decision_interval_minutes,
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    specs = SiteSpecs.from_yaml(args.site)
    data = load_timeseries(args.data, timezone=specs.timezone)
    starts = monthly_folds(
        _timestamp(args.from_time, specs.timezone),
        _timestamp(args.to_time, specs.timezone, end_of_day=True),
    )
    folds, results = run_selection(
        data,
        specs,
        starts,
        args.models,
        args.lookbacks,
        args.horizon_steps,
        args.decision_interval_minutes,
    )
    folds_path = args.out.with_name(f"{args.out.stem}_folds{args.out.suffix}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    folds.to_csv(folds_path, index=False)

    selected = select_configuration(results)
    print(f"Selected model: {selected['model']}")
    print(f"Selected lookback: {selected['lookback']}")
    print(f"Mean validation net MAE: {selected['net_mae_kw']:.4f} kW")
    print(f"Lead-1 MAE: {selected['lead_1_mae_kw']:.4f} kW")
    print(
        "Forecast seconds/decision: "
        f"{selected['forecast_seconds_per_decision']:.6f}"
    )
    print(f"Aggregate results: {args.out}")
    print(f"Per-fold results: {folds_path}")


if __name__ == "__main__":
    main()
