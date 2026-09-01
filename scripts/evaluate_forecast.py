"""Evaluate a persisted forecaster without running battery optimization."""

from __future__ import annotations

import argparse
import math
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


def collect_forecasts(
    data: pd.DataFrame,
    forecaster: Forecaster,
    config: RunConfig,
) -> pd.DataFrame:
    """Roll through decision times using the harness information boundary."""

    rows: list[pd.DataFrame] = []
    for at_time in decision_times(data.index, config):
        if at_time.hour == 0 and at_time.minute == 0:
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

        forecast = forecaster.predict(
            at_time=at_time,
            history=history,
            future_exog=future_exog,
            horizon=len(horizon_index),
        ).reindex(horizon_index)
        if FORECAST_COL not in forecast:
            raise ValueError(f"Forecaster output is missing '{FORECAST_COL}'")
        if forecast[FORECAST_COL].isna().any():
            raise ValueError("Forecaster returned NaN or an incomplete horizon")

        rows.append(
            pd.DataFrame(
                {
                    "decision_time": at_time,
                    "target_time": horizon_index,
                    "lead_steps": np.arange(1, len(horizon_index) + 1),
                    "actual_kw": data.loc[horizon_index, "grid_net_kw"].to_numpy(),
                    "predicted_kw": forecast[FORECAST_COL].to_numpy(),
                }
            )
        )

    if not rows:
        raise ValueError("No forecasts were produced for the requested window")
    return pd.concat(rows, ignore_index=True)


def lead_metrics(comparisons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lead_step, group in comparisons.groupby("lead_steps", sort=True):
        metrics = calculate_metrics(group["actual_kw"], group["predicted_kw"])
        rows.append(
            {
                "lead_steps": int(lead_step),
                **{
                    key: metrics[key]
                    for key in ("mae_kw", "rmse_kw", "bias_kw")
                },
            }
        )
    return pd.DataFrame(rows)


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
    metrics = calculate_metrics(comparisons["actual_kw"], comparisons["predicted_kw"])
    per_lead = lead_metrics(comparisons)
    out_path = args.out or args.model_dir / "forecast_lead_metrics.csv"
    comparisons_path = out_path.with_name("forecast_comparisons.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_lead.to_csv(out_path, index=False)
    comparisons.to_csv(comparisons_path, index=False)

    print(f"MAE (kW): {metrics['mae_kw']:.4f}")
    print(f"RMSE (kW): {metrics['rmse_kw']:.4f}")
    print(f"Bias (kW): {metrics['bias_kw']:.4f}")
    print(f"nMAE: {metrics['nmae']:.4f}")
    print(f"Lead-step metrics: {out_path}")
    print(f"Forecast comparisons: {comparisons_path}")


if __name__ == "__main__":
    main()
