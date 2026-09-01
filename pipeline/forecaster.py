"""YOUR CODE GOES HERE (1 of 2).

What ships here is a **deliberately weak baseline**: whatever the site did at this time
yesterday. It exists so the pipeline runs the moment you unzip it, and so you have
something to beat. It is not a starting point for your model — replace it.

Forecast what the site will do over the horizon, at quarter-hourly resolution.

Contract
--------
`train` calls:     Forecaster(specs) -> fit(history) -> save(model_dir)
`simulate` calls:  Forecaster.load(model_dir, specs) -> predict(...) once per decision

A decision is taken every quarter hour, so `predict` is called about 2,880 times over a
month and the whole run must finish in under fifteen minutes. Fit expensive things in `fit`,
persist them in `save`, keep `predict` cheap. Do not retrain inside `predict`.

Arguments to `predict`
----------------------
at_time      : the decision time. You know nothing at or after it.
history      : every row strictly before `at_time`, with all the columns from `data.py`.
               Metered values included — this is the past.
future_exog  : rows from `at_time` onwards, restricted to what is knowable in advance:
               `most_recent_load_factor_forecast` over the whole horizon, and
               `offtake_price_eur_per_mwh` and `injection_price_eur_per_mwh` only as far as
               the day-ahead auction has published them — NaN past that edge, since
               tomorrow's prices appear at 15:00 today.
horizon      : how many quarter hours to forecast, counting from `at_time`.

Return value
------------
A DataFrame indexed by exactly `future_exog.index[:horizon]`, containing at least:

    net_kw   your forecast of `grid_net_kw` over those timestamps
             (positive = the site imports, negative = the site exports)

No NaNs, and the index must cover the whole horizon — the harness rejects both.

Any extra column you add is passed straight through to your optimizer untouched. If you
want to hand the control layer more than a point forecast, that is how it travels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.data import REQUIRED_COLUMNS, STEP
from pipeline.specs import SiteSpecs

LOGGER = logging.getLogger(__name__)

# This is the one place that controls the model used by the standard training command.
SELECTED_MODEL = "scaffold"   # weekly or scaffold

# Central feature definition for time-series forecasting.
TIMESERIES_LAGS = {
    "lag_15min": pd.Timedelta(minutes=15),
    "lag_1h": pd.Timedelta(hours=1),
    "lag_1day": pd.DateOffset(days=1),
    "lag_1week": pd.DateOffset(weeks=1),
}
TIMESERIES_ROLLING_WINDOWS = {
    "rolling_mean_1h": pd.Timedelta(hours=1),
    "rolling_mean_4h": pd.Timedelta(hours=4),
}


def build_timeseries_features(
    history: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    target_column: str,
    known_future: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build model-independent, leakage-safe features for a time-series target.

    ``history`` supplies observed target values. Any explicitly supplied
    ``known_future`` columns are appended for the requested timestamps.
    """

    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("History must have a DatetimeIndex")
    if not isinstance(timestamps, pd.DatetimeIndex):
        raise ValueError("Feature timestamps must be a DatetimeIndex")
    if history.index.has_duplicates:
        raise ValueError("History contains duplicate timestamps")
    if timestamps.has_duplicates:
        raise ValueError("Feature timestamps contain duplicates")
    if target_column not in history:
        raise ValueError(f"History must contain target column {target_column}")

    features = pd.DataFrame(index=timestamps)
    features["quarter_of_day"] = timestamps.hour * 4 + timestamps.minute // 15
    features["day_of_week"] = timestamps.dayofweek
    features["is_weekend"] = (timestamps.dayofweek >= 5).astype(int)
    features["month"] = timestamps.month

    target = pd.to_numeric(history[target_column], errors="coerce").sort_index()
    for column, lag in TIMESERIES_LAGS.items():
        lagged = target.reindex(timestamps - lag)
        features[column] = lagged.to_numpy()

    # Add target timestamps as empty rows so rolling values are calculated at the
    # requested instants. ``closed="left"`` strictly excludes the target itself.
    rolling_source = target.reindex(target.index.union(timestamps)).sort_index()
    for column, window in TIMESERIES_ROLLING_WINDOWS.items():
        features[column] = (
            rolling_source.rolling(
                window, closed="left", min_periods=int(window / STEP)
            )
            .mean()
            .reindex(timestamps)
        )

    if known_future is not None:
        if not isinstance(known_future.index, pd.DatetimeIndex):
            raise ValueError("Known future data must have a DatetimeIndex")
        if known_future.index.has_duplicates:
            raise ValueError("Known future data contains duplicate timestamps")
        overlap = features.columns.intersection(known_future.columns)
        if not overlap.empty:
            raise ValueError(
                f"Known future columns conflict with features: {list(overlap)}"
            )
        for column in known_future.columns:
            features[column] = known_future[column].reindex(timestamps)

    return features


def validate_and_clean_history(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate training history and return an unchanged-value copy with a summary.

    Missing values are reported, not filled or used to remove rows. Unusual prices and
    grid values are deliberately kept for downstream feature code to interpret locally.
    """

    cleaned = history.copy()
    if not isinstance(cleaned.index, pd.DatetimeIndex):
        try:
            cleaned.index = pd.to_datetime(
                cleaned.index, errors="raise", format="mixed"
            )
        except (TypeError, ValueError) as error:
            raise ValueError("History timestamps must be parseable") from error
    if cleaned.index.isna().any():
        raise ValueError("History timestamps must be parseable")
    if not cleaned.index.is_monotonic_increasing:
        raise ValueError("History timestamps must be ordered")
    if cleaned.index.has_duplicates:
        raise ValueError("History contains duplicate timestamps")

    deltas = cleaned.index.to_series().diff().dropna()
    if not deltas.eq(STEP).all():
        raise ValueError(
            "History must have a regular 15-minute cadence with no missing intervals"
        )

    try:
        numeric = cleaned[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Required history fields must be numeric") from error
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("History contains infinite values")

    missing_by_column = {
        column: {
            "count": int(cleaned[column].isna().sum()),
            "percentage": float(cleaned[column].isna().mean() * 100),
        }
        for column in REQUIRED_COLUMNS
    }

    summary: dict[str, object] = {
        "rows": len(cleaned),
        "start": cleaned.index.min(),
        "end": cleaned.index.max(),
        "missing_by_column": missing_by_column,
    }
    return cleaned, summary


def _usable_median(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = finite.median()
    if pd.isna(median):
        raise ValueError("History has no usable grid load observations")
    return float(median)


class ScaffoldBaseline:
    """The original yesterday/last-week persistence baseline."""

    def __init__(self) -> None:
        self.fallback_kw = 0.0

    def fit(self, history: pd.DataFrame) -> None:
        self.fallback_kw = _usable_median(history["grid_net_kw"])

    def predict(self, history: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
        net = history["grid_net_kw"]
        yesterday = net.reindex(index - pd.Timedelta(days=1))
        last_week = net.reindex(index - pd.Timedelta(days=7))
        values = pd.Series(yesterday.to_numpy(), index=index).fillna(
            pd.Series(last_week.to_numpy(), index=index)
        )
        return values.fillna(self.fallback_kw)

    def state(self) -> dict[str, object]:
        return {"fallback_kw": self.fallback_kw}

    def load_state(self, state: dict[str, object]) -> None:
        self.fallback_kw = float(state["fallback_kw"])


class WeeklySeasonalBaseline:
    """Previous-week persistence with a historical time-of-week fallback."""


    def __init__(self) -> None:
        self.time_of_week_medians: dict[int, float] = {}

    @staticmethod
    def _slots(index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek * 96 + index.hour * 4 + index.minute // 15

    def _validate_time_of_week_medians(self):
        expected_slots = set(range(7 * 24 * 4))

        if set(self.time_of_week_medians) != expected_slots:
            missing = sorted(expected_slots - set(self.time_of_week_medians))
            raise ValueError(
                f"Missing time-of-week medians for slots: {missing}"
            )

        if not all(np.isfinite(value) for value in self.time_of_week_medians.values()):
            raise ValueError("Time-of-week medians contain invalid values")

    def fit(self, history: pd.DataFrame) -> None:
        net = pd.to_numeric(history["grid_net_kw"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        by_slot = net.groupby(self._slots(history.index)).median().dropna()
        self.time_of_week_medians = {
            int(slot): float(value) for slot, value in by_slot.items()
        }

        self._validate_time_of_week_medians()

    def predict(self, history: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
        weekly = pd.to_numeric(
            history["grid_net_kw"].reindex(index - pd.Timedelta(days=7)),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        weekly.index = index
        fallback = pd.Series(
            [
                self.time_of_week_medians.get(int(slot))
                for slot in self._slots(index)
            ],
            index=index,
        )

        return weekly.fillna(fallback)

    def state(self) -> dict[str, object]:
        return {
            "time_of_week_medians": self.time_of_week_medians,
        }

    def load_state(self, state: dict[str, object]) -> None:
        medians = state["time_of_week_medians"]
        if not isinstance(medians, dict):
            raise ValueError("Invalid time-of-week median state")
        self.time_of_week_medians = {
            int(slot): float(value) for slot, value in medians.items()
        }
        self._validate_time_of_week_medians()


class Forecaster:
    """Harness-facing wrapper for the selected forecasting baseline."""

    PARAMS = "forecaster.json"

    SCAFFOLD = "scaffold"
    WEEKLY = "weekly"

    def __init__(self, specs: SiteSpecs, model_name: str = SELECTED_MODEL) -> None:
        self.specs = specs
        self.model_name = model_name
        self.model = self._make_model(model_name)

    @staticmethod
    def _make_model(model_name: str) -> ScaffoldBaseline | WeeklySeasonalBaseline:
        if model_name == Forecaster.SCAFFOLD:
            return ScaffoldBaseline()
        if model_name == Forecaster.WEEKLY:
            return WeeklySeasonalBaseline()
        raise ValueError(f"Unknown forecasting model: {model_name}")

    def fit(self, history: pd.DataFrame) -> None:
        cleaned, summary = validate_and_clean_history(history)
        LOGGER.warning(
            "Training data: %d rows (%s to %s); missing=%s; ",
            summary["rows"],
            summary["start"],
            summary["end"],
            summary["missing_by_column"],
        )
        self.model.fit(cleaned)

    def save(self, path: Path) -> None:
        (Path(path) / self.PARAMS).write_text(
            json.dumps({"model_name": self.model_name, "state": self.model.state()})
        )

    @classmethod
    def load(cls, path: Path, specs: SiteSpecs) -> "Forecaster":
        params = json.loads((Path(path) / cls.PARAMS).read_text())
        self = cls(specs, model_name=params["model_name"])
        self.model.load_state(params["state"])
        return self

    def predict(
        self,
        at_time: pd.Timestamp,
        history: pd.DataFrame,
        future_exog: pd.DataFrame,
        horizon: int,
    ) -> pd.DataFrame:
        index = future_exog.index[:horizon]
        values = self.model.predict(history, index)
        return pd.DataFrame({"net_kw": values}, index=index)
