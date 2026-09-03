"""Evaluate a persisted forecaster without running battery optimization."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.data import STEP, load_timeseries
from pipeline.forecaster import Forecaster
from pipeline.harness import (
    FORECAST_COL,
    RunConfig,
    decision_times,
    prices_known_until,
)
from pipeline.specs import SiteSpecs


EXOG_COLUMNS = ["most_recent_load_factor_forecast"]
PRICE_COLUMNS = ["offtake_price_eur_per_mwh", "injection_price_eur_per_mwh"]
COMPONENT_COLUMNS = {
    "net": ("actual_kw", "predicted_kw"),
    "load": ("actual_load_kw", "forecast_load_kw"),
    "pv": ("actual_pv_kw", "forecast_pv_kw"),
}


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Calculate forecast errors from finite, aligned comparison pairs."""

    pairs = pd.DataFrame({"actual": actual, "predicted": predicted}).replace(
        [np.inf, -np.inf], np.nan
    )
    if pairs.empty:
        raise ValueError("No actual and predicted values to evaluate")
    if pairs.isna().any().any():
        raise ValueError("Actual and predicted values must be finite")

    errors = pairs["predicted"] - pairs["actual"]
    mae = float(errors.abs().mean())
    mean_absolute_actual = float(pairs["actual"].abs().mean())
    return {
        "mae_kw": mae,
        "rmse_kw": float(math.sqrt((errors**2).mean())),
        "bias_kw": float(errors.mean()),
        "nmae": mae / mean_absolute_actual if mean_absolute_actual else math.nan,
    }


def component_metrics(comparisons: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Calculate metrics for each forecast component with available actuals."""

    results = {}
    for component, (actual_column, forecast_column) in COMPONENT_COLUMNS.items():
        if not {actual_column, forecast_column}.issubset(comparisons.columns):
            continue
        pairs = comparisons[[actual_column, forecast_column]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if not pairs.empty:
            results[component] = calculate_metrics(
                pairs[actual_column], pairs[forecast_column]
            )
    return results


def collect_forecasts(
    data: pd.DataFrame,
    forecaster: Forecaster,
    config: RunConfig,
) -> pd.DataFrame:
    """Roll through decision times using the harness information boundary."""

    rows: list[pd.DataFrame] = []
    forecast_seconds = 0.0
    n_decisions = 0
    for at_time in decision_times(data.index, config):
        if at_time.minute == 0 and at_time.hour == 0:
            print(at_time)

        history = data.loc[data.index < at_time]
        horizon_index = pd.date_range(
            at_time, periods=config.horizon_steps, freq=STEP, tz=data.index.tz
        )
        horizon_index = horizon_index[horizon_index <= data.index[-1]]
        if horizon_index.empty:
            continue

        known_until = prices_known_until(at_time, config.price_publication_hour)
        prices = data.loc[horizon_index, PRICE_COLUMNS].copy()
        prices.loc[horizon_index > known_until] = np.nan
        future_exog = data.loc[horizon_index, EXOG_COLUMNS].copy()
        future_exog[PRICE_COLUMNS] = prices

        started = time.perf_counter()
        forecast = forecaster.predict(
            at_time=at_time,
            history=history,
            future_exog=future_exog,
            horizon=len(horizon_index),
        ).reindex(horizon_index)
        forecast_seconds += time.perf_counter() - started
        n_decisions += 1
        if FORECAST_COL not in forecast:
            raise ValueError(f"Forecaster output is missing '{FORECAST_COL}'")
        if forecast[FORECAST_COL].isna().any():
            raise ValueError("Forecaster returned NaN or an incomplete horizon")

        comparison = pd.DataFrame(
            {
                "decision_time": at_time,
                "target_time": horizon_index,
                "lead_steps": np.arange(1, len(horizon_index) + 1),
                "actual_kw": data.loc[horizon_index, "grid_net_kw"].to_numpy(),
                "predicted_kw": forecast[FORECAST_COL].to_numpy(),
            }
        )
        if "load_kw" in forecast:
            comparison["forecast_load_kw"] = forecast["load_kw"].to_numpy()
            if "pv_production_kw" in data:
                actual_pv = data.loc[horizon_index, "pv_production_kw"]
                comparison["actual_load_kw"] = (
                    data.loc[horizon_index, "grid_net_kw"] + actual_pv
                ).to_numpy()
                comparison["load_error_kw"] = (
                    comparison["forecast_load_kw"] - comparison["actual_load_kw"]
                )
        if "pv_kw" in forecast:
            comparison["forecast_pv_kw"] = forecast["pv_kw"].to_numpy()
            if "pv_production_kw" in data:
                actual_pv = data.loc[horizon_index, "pv_production_kw"]
                comparison["actual_pv_kw"] = actual_pv.to_numpy()
                comparison["pv_error_kw"] = (
                    comparison["forecast_pv_kw"] - comparison["actual_pv_kw"]
                )
        rows.append(comparison)

    if not rows:
        raise ValueError("No forecasts were produced for the requested window")
    comparisons = pd.concat(rows, ignore_index=True)
    comparisons.attrs["forecast_seconds"] = forecast_seconds
    comparisons.attrs["n_decisions"] = n_decisions
    return comparisons


def lead_metrics(comparisons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for component, (actual_column, forecast_column) in COMPONENT_COLUMNS.items():
        if not {actual_column, forecast_column}.issubset(comparisons.columns):
            continue
        for lead_step, group in comparisons.groupby("lead_steps", sort=True):
            pairs = group[[actual_column, forecast_column]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if pairs.empty:
                continue
            metrics = calculate_metrics(pairs[actual_column], pairs[forecast_column])
            rows.append(
                {
                    "component": component,
                    "lead_step": int(lead_step),
                    **{
                        key: metrics[key]
                        for key in ("mae_kw", "rmse_kw", "bias_kw")
                    },
                }
            )
    return pd.DataFrame(rows)


def _print_metrics(metrics: dict[str, dict[str, float]]) -> None:
    if list(metrics) == ["net"]:
        values = metrics["net"]
        print(f"MAE (kW): {values['mae_kw']:.4f}")
        print(f"RMSE (kW): {values['rmse_kw']:.4f}")
        print(f"Bias (kW): {values['bias_kw']:.4f}")
        print(f"nMAE: {values['nmae']:.4f}")
        return

    labels = {"net": "Net load", "load": "Load", "pv": "PV"}
    for component, values in metrics.items():
        print(f"{labels[component]}:")
        print(f"  MAE (kW): {values['mae_kw']:.4f}")
        print(f"  RMSE (kW): {values['rmse_kw']:.4f}")
        print(f"  Bias (kW): {values['bias_kw']:.4f}")
        print(f"  nMAE: {values['nmae']:.4f}")
        print()


def _timestamp(value: str, timezone: str, end_of_day: bool = False) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone)
    else:
        timestamp = timestamp.tz_convert(timezone)
    if end_of_day and timestamp == timestamp.normalize():
        timestamp += pd.Timedelta(hours=23, minutes=45)
    return timestamp


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--from", dest="from_time", required=True)
    parser.add_argument("--to", dest="to_time", required=True)
    parser.add_argument("--site", type=Path, default=Path("site.yaml"))
    parser.add_argument("--out", type=Path, default=None)
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
    forecaster = Forecaster.load(args.model_dir, specs)
    config = RunConfig(
        first_decision=_timestamp(args.from_time, specs.timezone),
        last_decision=_timestamp(args.to_time, specs.timezone, end_of_day=True),
        horizon_steps=args.horizon_steps,
        decision_interval_minutes=args.decision_interval_minutes,
    )

    comparisons = collect_forecasts(data, forecaster, config)
    metrics = component_metrics(comparisons)
    per_lead = lead_metrics(comparisons)
    out_path = args.out or args.model_dir / "forecast_lead_metrics.csv"
    comparisons_path = out_path.with_name("forecast_comparisons.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_lead.to_csv(out_path, index=False)
    comparisons.to_csv(comparisons_path, index=False)

    _print_metrics(metrics)
    print(f"Lead-step metrics: {out_path}")
    print(f"Forecast comparisons: {comparisons_path}")


if __name__ == "__main__":
    main()
