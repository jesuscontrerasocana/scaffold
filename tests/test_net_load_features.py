from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.forecaster import build_net_load_features


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
        {"most_recent_load_factor_forecast": [0.75]}, index=timestamps
    )

    features = build_net_load_features(history, timestamps, known_future)

    assert features.loc[timestamps[0], "quarter_of_day"] == 50
    assert features.loc[timestamps[0], "day_of_week"] == 5
    assert features.loc[timestamps[0], "is_weekend"] == 1
    assert features.loc[timestamps[0], "month"] == 1
    assert features.loc[timestamps[0], "most_recent_load_factor_forecast"] == 0.75


def test_lags_use_exact_past_timestamps_and_leave_missing_history_unavailable() -> None:
    history = _history()
    timestamp = history.index[-1]
    history = history.drop(timestamp - pd.Timedelta(days=1))

    features = build_net_load_features(history, pd.DatetimeIndex([timestamp]))

    assert features.loc[timestamp, "lag_15min_kw"] == history.loc[
        timestamp - pd.Timedelta(minutes=15), "grid_net_kw"
    ]
    assert pd.isna(features.loc[timestamp, "lag_1day_kw"])
    assert features.loc[timestamp, "lag_1week_kw"] == history.loc[
        timestamp - pd.Timedelta(days=7), "grid_net_kw"
    ]


def test_lag_and_rolling_features_never_use_current_or_future_values() -> None:
    history = _history()
    timestamp = history.index[-2]
    targets = pd.DatetimeIndex([timestamp])
    before = build_net_load_features(history, targets)

    changed = history.copy()
    changed.loc[timestamp:, "grid_net_kw"] = 1_000_000.0
    after = build_net_load_features(changed, targets)

    historical_columns = [
        "lag_15min_kw",
        "lag_30min_kw",
        "lag_1day_kw",
        "lag_1week_kw",
        "rolling_mean_1h_kw",
        "rolling_mean_24h_kw",
    ]
    pd.testing.assert_series_equal(
        before.loc[timestamp, historical_columns],
        after.loc[timestamp, historical_columns],
    )


def test_rolling_feature_is_missing_when_its_history_window_is_incomplete() -> None:
    history = _history()
    timestamp = history.index[-1]
    history = history.drop(timestamp - pd.Timedelta(minutes=30))

    features = build_net_load_features(history, pd.DatetimeIndex([timestamp]))

    assert pd.isna(features.loc[timestamp, "rolling_mean_1h_kw"])


def test_training_and_inference_use_the_same_feature_columns() -> None:
    history = _history()
    timestamps = history.index[-4:]
    future = history.loc[timestamps, ["most_recent_load_factor_forecast"]]

    training = build_net_load_features(history, timestamps)
    inference = build_net_load_features(
        history.loc[history.index < timestamps[0]], timestamps, future
    )

    assert training.columns.tolist() == inference.columns.tolist()
