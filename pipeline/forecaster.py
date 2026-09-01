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

CALENDAR_FEATURES = ("time_of_day", "day_of_week", "is_weekend", "month")


def _offset_label(offset: pd.Timedelta | pd.DateOffset) -> str:
    """Return a concise deterministic label for a requested offset."""

    if isinstance(offset, pd.Timedelta):
        if offset <= pd.Timedelta(0):
            raise ValueError("Timedelta lags and windows must be positive")
        for unit, duration in (
            ("day", pd.Timedelta(days=1)),
            ("h", pd.Timedelta(hours=1)),
            ("min", pd.Timedelta(minutes=1)),
        ):
            if offset % duration == pd.Timedelta(0):
                return f"{int(offset / duration)}{unit}"
        raise ValueError("Timedelta lags and windows must use whole minutes")

    if isinstance(offset, pd.DateOffset) and offset.kwds:
        invalid = any(
            not isinstance(value, int) or value <= 0
            for value in offset.kwds.values()
        )
        if invalid:
            raise ValueError("Calendar lags must be positive whole units")
        parts = [
            f"{value}{unit.removesuffix('s')}"
            for unit, value in sorted(offset.kwds.items())
        ]
        return "_".join(parts)

    raise ValueError("Lags must be Timedelta or DateOffset values")


def build_calendar_features(
    timestamps: pd.DatetimeIndex,
    features: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Build the explicitly requested unencoded calendar features."""

    if not isinstance(timestamps, pd.DatetimeIndex):
        raise ValueError("Feature timestamps must be a DatetimeIndex")
    if timestamps.has_duplicates:
        raise ValueError("Feature timestamps contain duplicates")
    unsupported = set(features) - set(CALENDAR_FEATURES)
    if unsupported:
        raise ValueError(f"Unsupported calendar features: {sorted(unsupported)}")

    result = pd.DataFrame(index=timestamps)
    for feature in features:
        if feature == "time_of_day":
            result[feature] = timestamps.hour * 4 + timestamps.minute // 15
        elif feature == "day_of_week":
            result[feature] = timestamps.dayofweek
        elif feature == "is_weekend":
            result[feature] = (timestamps.dayofweek >= 5).astype(int)
        elif feature == "month":
            result[feature] = timestamps.month
    return result


def build_lag_features(
    series: pd.Series,
    timestamps: pd.DatetimeIndex,
    lags: list[pd.Timedelta | pd.DateOffset] | tuple[pd.Timedelta | pd.DateOffset, ...],
) -> pd.DataFrame:
    """Build timestamp-matched features for explicitly requested past lags."""

    _validate_feature_inputs(series, timestamps)
    numeric = pd.to_numeric(series, errors="coerce").sort_index()
    result = pd.DataFrame(index=timestamps)
    for lag in lags:
        column = f"lag_{_offset_label(lag)}"
        if column in result:
            raise ValueError(f"Duplicate lag feature: {column}")
        lagged_timestamps = timestamps - lag
        if len(timestamps) and not (lagged_timestamps < timestamps).all():
            raise ValueError("Lags must refer strictly to past timestamps")
        result[column] = numeric.reindex(lagged_timestamps).to_numpy()
    return result


def build_rolling_features(
    series: pd.Series,
    timestamps: pd.DatetimeIndex,
    windows: list[pd.Timedelta] | tuple[pd.Timedelta, ...],
) -> pd.DataFrame:
    """Build complete rolling means from observations strictly before each timestamp."""

    _validate_feature_inputs(series, timestamps)
    numeric = pd.to_numeric(series, errors="coerce").sort_index()
    rolling_source = numeric.reindex(numeric.index.union(timestamps)).sort_index()
    result = pd.DataFrame(index=timestamps)
    for window in windows:
        label = _offset_label(window)
        if not isinstance(window, pd.Timedelta) or window % STEP != pd.Timedelta(0):
            raise ValueError("Rolling windows must be whole 15-minute intervals")
        column = f"rolling_mean_{label}"
        if column in result:
            raise ValueError(f"Duplicate rolling feature: {column}")
        result[column] = (
            rolling_source.rolling(
                window, closed="left", min_periods=int(window / STEP)
            )
            .mean()
            .reindex(timestamps)
        )
    return result


def _validate_feature_inputs(
    series: pd.Series, timestamps: pd.DatetimeIndex
) -> None:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")
    if not isinstance(timestamps, pd.DatetimeIndex):
        raise ValueError("Feature timestamps must be a DatetimeIndex")
    if series.index.has_duplicates:
        raise ValueError("Series contains duplicate timestamps")
    if timestamps.has_duplicates:
        raise ValueError("Feature timestamps contain duplicates")


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
