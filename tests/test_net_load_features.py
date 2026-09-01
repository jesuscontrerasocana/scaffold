from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.forecaster import (
    build_calendar_features,
    build_lag_features,
    build_rolling_features,
)


def _series(periods: int = 8 * 96) -> pd.Series:
    index = pd.date_range(
        "2026-01-05", periods=periods, freq="15min", tz="Europe/Brussels"
    )
    return pd.Series(np.arange(periods, dtype=float), index=index)


def test_calendar_features_are_selected_explicitly() -> None:
    timestamps = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-10 12:30", tz="Europe/Brussels")]
    )

    features = build_calendar_features(
        timestamps, features=["time_of_day", "is_weekend"]
    )

    assert features.columns.tolist() == ["time_of_day", "is_weekend"]
    assert features.loc[timestamps[0], "time_of_day"] == 50
    assert features.loc[timestamps[0], "is_weekend"] == 1


def test_unsupported_calendar_feature_is_rejected() -> None:
    timestamps = pd.date_range("2026-01-05", periods=1, freq="15min")

    with pytest.raises(ValueError, match="Unsupported calendar features"):
        build_calendar_features(timestamps, features=["season"])


def test_arbitrary_lags_use_exact_past_timestamps() -> None:
    series = _series()
    timestamp = series.index[-1]
    lags = [pd.Timedelta(minutes=30), pd.Timedelta(hours=2)]

    features = build_lag_features(series, pd.DatetimeIndex([timestamp]), lags)

    assert features.columns.tolist() == ["lag_30min", "lag_2h"]
    assert features.loc[timestamp, "lag_30min"] == series.loc[
        timestamp - pd.Timedelta(minutes=30)
    ]
    assert features.loc[timestamp, "lag_2h"] == series.loc[
        timestamp - pd.Timedelta(hours=2)
    ]


def test_missing_lag_observation_remains_missing() -> None:
    series = _series()
    timestamp = series.index[-1]
    series = series.drop(timestamp - pd.Timedelta(hours=1))

    features = build_lag_features(
        series, pd.DatetimeIndex([timestamp]), [pd.Timedelta(hours=1)]
    )

    assert pd.isna(features.loc[timestamp, "lag_1h"])


def test_calendar_lags_preserve_local_time_across_dst() -> None:
    timestamp = pd.Timestamp("2026-03-29 12:00", tz="Europe/Brussels")
    previous_day = timestamp - pd.DateOffset(days=1)
    previous_week = timestamp - pd.DateOffset(weeks=1)
    elapsed_day = timestamp - pd.Timedelta(days=1)
    elapsed_week = timestamp - pd.Timedelta(days=7)
    series = pd.Series(
        [70.0, 700.0, 10.0, 100.0],
        index=pd.DatetimeIndex(
            [previous_week, elapsed_week, previous_day, elapsed_day]
        ),
    ).sort_index()

    features = build_lag_features(
        series,
        pd.DatetimeIndex([timestamp]),
        [pd.DateOffset(days=1), pd.DateOffset(weeks=1)],
    )

    assert features.loc[timestamp, "lag_1day"] == 10.0
    assert features.loc[timestamp, "lag_1week"] == 70.0


def test_arbitrary_rolling_windows_exclude_current_and_future_values() -> None:
    series = _series()
    timestamp = series.index[-2]
    timestamps = pd.DatetimeIndex([timestamp])
    windows = [pd.Timedelta(minutes=30), pd.Timedelta(hours=2)]
    before = build_rolling_features(series, timestamps, windows)

    changed = series.copy()
    changed.loc[timestamp:] = 1_000_000.0
    after = build_rolling_features(changed, timestamps, windows)

    assert before.columns.tolist() == ["rolling_mean_30min", "rolling_mean_2h"]
    pd.testing.assert_series_equal(before.loc[timestamp], after.loc[timestamp])


def test_rolling_feature_is_missing_when_window_is_incomplete() -> None:
    series = _series()
    timestamp = series.index[-1]
    series = series.drop(timestamp - pd.Timedelta(minutes=30))

    features = build_rolling_features(
        series, pd.DatetimeIndex([timestamp]), [pd.Timedelta(hours=1)]
    )

    assert pd.isna(features.loc[timestamp, "rolling_mean_1h"])


def test_helpers_work_for_another_target_series() -> None:
    pv_production = _series() * 2
    timestamp = pv_production.index[-1]

    features = build_lag_features(
        pv_production,
        pd.DatetimeIndex([timestamp]),
        [pd.Timedelta(minutes=15)],
    )

    assert features.loc[timestamp, "lag_15min"] == pv_production.loc[
        timestamp - pd.Timedelta(minutes=15)
    ]


def test_calling_code_appends_known_future_explicitly() -> None:
    series = _series()
    timestamps = series.index[-2:]
    known_future = pd.DataFrame(
        {"most_recent_load_factor_forecast": [0.4, 0.5]}, index=timestamps
    )

    features = pd.concat(
        [
            build_calendar_features(timestamps, ["time_of_day"]),
            build_lag_features(series, timestamps, [pd.Timedelta(minutes=15)]),
            known_future.reindex(timestamps),
        ],
        axis=1,
    )

    assert features.columns.tolist() == [
        "time_of_day",
        "lag_15min",
        "most_recent_load_factor_forecast",
    ]
