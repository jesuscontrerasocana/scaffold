from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.forecaster import build_timeseries_features


def _history(periods: int = 8 * 96) -> pd.DataFrame:
    index = pd.date_range(
        "2026-01-05", periods=periods, freq="15min", tz="Europe/Brussels"
    )
    return pd.DataFrame(
        {
            "grid_net_kw": np.arange(periods, dtype=float),
            "most_recent_load_factor_forecast": np.linspace(0.0, 1.0, periods),
        },
        index=index,
    )


def test_calendar_and_known_future_features_are_built_for_requested_times() -> None:
    history = _history()
    timestamps = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-10 12:30", tz="Europe/Brussels")]
    )
    known_future = pd.DataFrame(
        {
            "most_recent_load_factor_forecast": [0.75],
            "temperature_forecast": [12.0],
        },
        index=timestamps,
    )

    features = build_timeseries_features(
        history, timestamps, "grid_net_kw", known_future
    )

    assert features.loc[timestamps[0], "quarter_of_day"] == 50
    assert features.loc[timestamps[0], "day_of_week"] == 5
    assert features.loc[timestamps[0], "is_weekend"] == 1
    assert features.loc[timestamps[0], "month"] == 1
    assert features.loc[timestamps[0], "most_recent_load_factor_forecast"] == 0.75
    assert features.loc[timestamps[0], "temperature_forecast"] == 12.0


def test_lags_use_exact_past_timestamps_and_leave_missing_history_unavailable() -> None:
    history = _history()
    timestamp = history.index[-1]
    history = history.drop(timestamp - pd.Timedelta(days=1))

    features = build_timeseries_features(
        history, pd.DatetimeIndex([timestamp]), "grid_net_kw"
    )

    assert features.loc[timestamp, "lag_15min"] == history.loc[
        timestamp - pd.Timedelta(minutes=15), "grid_net_kw"
    ]
    assert pd.isna(features.loc[timestamp, "lag_1day"])
    assert features.loc[timestamp, "lag_1week"] == history.loc[
        timestamp - pd.Timedelta(days=7), "grid_net_kw"
    ]


def test_lag_and_rolling_features_never_use_current_or_future_values() -> None:
    history = _history()
    timestamp = history.index[-2]
    targets = pd.DatetimeIndex([timestamp])
    before = build_timeseries_features(history, targets, "grid_net_kw")

    changed = history.copy()
    changed.loc[timestamp:, "grid_net_kw"] = 1_000_000.0
    after = build_timeseries_features(changed, targets, "grid_net_kw")

    historical_columns = [
        "lag_15min",
        "lag_1h",
        "lag_1day",
        "lag_1week",
        "rolling_mean_1h",
        "rolling_mean_4h",
    ]
    pd.testing.assert_series_equal(
        before.loc[timestamp, historical_columns],
        after.loc[timestamp, historical_columns],
    )


def test_rolling_feature_is_missing_when_its_history_window_is_incomplete() -> None:
    history = _history()
    timestamp = history.index[-1]
    history = history.drop(timestamp - pd.Timedelta(minutes=30))

    features = build_timeseries_features(
        history, pd.DatetimeIndex([timestamp]), "grid_net_kw"
    )

    assert pd.isna(features.loc[timestamp, "rolling_mean_1h"])


def test_daily_and_weekly_lags_preserve_local_time_across_dst() -> None:
    timestamp = pd.Timestamp("2026-03-29 12:00", tz="Europe/Brussels")
    previous_day = timestamp - pd.DateOffset(days=1)
    previous_week = timestamp - pd.DateOffset(weeks=1)
    elapsed_day = timestamp - pd.Timedelta(days=1)
    elapsed_week = timestamp - pd.Timedelta(days=7)
    history = pd.DataFrame(
        {
            "grid_net_kw": [70.0, 700.0, 10.0, 100.0],
            "most_recent_load_factor_forecast": 0.5,
        },
        index=pd.DatetimeIndex(
            [previous_week, elapsed_week, previous_day, elapsed_day]
        ),
    ).sort_index()
    known_future = pd.DataFrame(
        {"most_recent_load_factor_forecast": [0.75]}, index=[timestamp]
    )

    features = build_timeseries_features(
        history, pd.DatetimeIndex([timestamp]), "grid_net_kw", known_future
    )

    assert features.loc[timestamp, "lag_1day"] == 10.0
    assert features.loc[timestamp, "lag_1week"] == 70.0


def test_training_and_inference_use_the_same_feature_columns() -> None:
    history = _history()
    timestamps = history.index[-4:]
    future = history.loc[timestamps, ["most_recent_load_factor_forecast"]]

    training = build_timeseries_features(
        history, timestamps, "grid_net_kw", future
    )
    inference = build_timeseries_features(
        history.loc[history.index < timestamps[0]],
        timestamps,
        "grid_net_kw",
        future,
    )

    assert training.columns.tolist() == inference.columns.tolist()


def test_features_can_be_built_from_another_target_column() -> None:
    history = _history()
    history["pv_production_kw"] = history["grid_net_kw"] * 2
    timestamp = history.index[-1]

    features = build_timeseries_features(
        history, pd.DatetimeIndex([timestamp]), "pv_production_kw"
    )

    assert features.loc[timestamp, "lag_15min"] == history.loc[
        timestamp - pd.Timedelta(minutes=15), "pv_production_kw"
    ]
