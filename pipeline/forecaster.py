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

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from pipeline.data import REQUIRED_COLUMNS, STEP
from pipeline.specs import SiteSpecs

LOGGER = logging.getLogger(__name__)

# This is the one place that controls the model used by the standard training command.
SELECTED_MODEL = "ridge_hgbr_decomposed"   # ridge_hgbr_decomposed, ridge_decomposed, ridge, weekly, or scaffold

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


def _fit_ridge_lead(
    features: pd.DataFrame,
    target: pd.Series,
    feature_names: tuple[str, ...],
    lead_steps: int,
) -> tuple[StandardScaler, Ridge, dict[str, object]]:
    usable = (
        features.notna().all(axis=1)
        & np.isfinite(features).all(axis=1)
        & target.notna()
        & np.isfinite(target)
    )
    training_features = features.loc[usable]
    training_target = target.loc[usable]
    if len(training_target) < 12:
        raise ValueError(f"Ridge lead {lead_steps} requires at least 12 usable rows")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(training_features)
    model = Ridge(alpha=1.0).fit(scaled, training_target)
    fitted = model.predict(scaled)
    summary = {
        "lead_steps": lead_steps,
        "lead_minutes": (lead_steps - 1) * 15,
        "training_samples": len(training_target),
        "dropped_rows": int((~usable).sum()),
        "train_mae": float(mean_absolute_error(training_target, fitted)),
        "train_rmse": float(mean_squared_error(training_target, fitted) ** 0.5),
        "coefficients": {
            name: float(value)
            for name, value in zip(feature_names, model.coef_, strict=True)
        },
        "intercept": float(model.intercept_),
    }
    return scaler, model, summary


def _predict_ridge_leads(
    features: pd.DataFrame,
    scalers: list[StandardScaler],
    models: list[Ridge],
) -> np.ndarray:
    count = len(features)
    values = features.to_numpy(dtype=float)
    means = np.vstack([scaler.mean_ for scaler in scalers[:count]])
    scales = np.vstack([scaler.scale_ for scaler in scalers[:count]])
    coefficients = np.vstack([model.coef_ for model in models[:count]])
    intercepts = np.asarray([model.intercept_ for model in models[:count]])
    return np.sum(((values - means) / scales) * coefficients, axis=1) + intercepts


class DirectRidgeNetLoadModel:
    """One direct Ridge model per forecast lead time."""

    ARTIFACT = "model.joblib"
    ALPHA = 1.0
    MAX_HORIZON = 132
    EXOG_FEATURE = "most_recent_load_factor_forecast"
    HISTORY_LAGS = (pd.Timedelta(minutes=15), pd.Timedelta(hours=1))
    HISTORY_WINDOWS = (pd.Timedelta(hours=1), pd.Timedelta(hours=4))
    FEATURE_NAMES = (
        "lag_15min",
        "lag_1h",
        "rolling_mean_1h",
        "rolling_mean_4h",
        "target_minus_1day",
        "target_minus_7day",
        "time_of_day_sin",
        "time_of_day_cos",
        "dow_0",
        "dow_1",
        "dow_2",
        "dow_3",
        "dow_4",
        "dow_5",
        "dow_6",
        EXOG_FEATURE,
    )

    def __init__(self) -> None:
        self.scalers: list[StandardScaler] = []
        self.models: list[Ridge] = []
        self.fill_values: dict[str, float] = {}
        self.lead_summaries: list[dict[str, object]] = []

    @staticmethod
    def _target_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
        raw = build_calendar_features(index, ("time_of_day", "day_of_week"))
        angle = 2 * np.pi * raw["time_of_day"] / 96
        result = pd.DataFrame(
            {
                "time_of_day_sin": np.sin(angle),
                "time_of_day_cos": np.cos(angle),
            },
            index=index,
        )
        for day in range(7):
            result[f"dow_{day}"] = (raw["day_of_week"] == day).astype(float)
        return result

    @classmethod
    def _decision_features(
        cls, net: pd.Series, decisions: pd.DatetimeIndex
    ) -> pd.DataFrame:
        return build_lag_features(net, decisions, cls.HISTORY_LAGS).join(
            build_rolling_features(net, decisions, cls.HISTORY_WINDOWS)
        )

    @classmethod
    def _features_for_lead(
        cls,
        decision_features: pd.DataFrame,
        net: pd.Series,
        exog: pd.Series,
        lead_steps: int,
    ) -> pd.DataFrame:
        decisions = decision_features.index
        target_times = decisions + (lead_steps - 1) * STEP
        calendar = cls._target_calendar(target_times)
        calendar.index = decisions
        result = decision_features.join(calendar)
        seasonal = build_lag_features(
            net,
            target_times,
            (pd.Timedelta(days=1), pd.Timedelta(days=7)),
        )
        seasonal.index = decisions
        result["target_minus_1day"] = seasonal["lag_1day"]
        result.loc[
            target_times - pd.Timedelta(days=1) >= decisions,
            "target_minus_1day",
        ] = np.nan
        result["target_minus_7day"] = seasonal["lag_7day"]
        result[cls.EXOG_FEATURE] = pd.to_numeric(
            exog.reindex(target_times), errors="coerce"
        ).to_numpy(dtype=float)
        return result.loc[:, cls.FEATURE_NAMES]

    def fit(self, history: pd.DataFrame) -> None:
        net = pd.to_numeric(history["grid_net_kw"], errors="coerce")
        exog = pd.to_numeric(history[self.EXOG_FEATURE], errors="coerce")
        decisions = history.index
        decision_features = self._decision_features(net, decisions)
        calendar = self._target_calendar(decisions)
        fill_source = decision_features.join(calendar)
        fill_source["target_minus_1day"] = net.reindex(
            decisions - pd.Timedelta(days=1)
        ).to_numpy(dtype=float)
        fill_source["target_minus_7day"] = net.reindex(
            decisions - pd.Timedelta(days=7)
        ).to_numpy(dtype=float)
        fill_source[self.EXOG_FEATURE] = exog
        self.fill_values = {
            name: float(fill_source[name].median()) for name in self.FEATURE_NAMES
        }
        if not all(np.isfinite(list(self.fill_values.values()))):
            raise ValueError("Training history cannot populate all Ridge features")

        self.scalers = []
        self.models = []
        self.lead_summaries = []
        for lead_steps in range(1, self.MAX_HORIZON + 1):
            features = self._features_for_lead(
                decision_features, net, exog, lead_steps
            )
            features["target_minus_1day"] = features[
                "target_minus_1day"
            ].fillna(self.fill_values["target_minus_1day"])
            target_times = decisions + (lead_steps - 1) * STEP
            target = net.reindex(target_times)
            target.index = decisions
            scaler, model, summary = _fit_ridge_lead(
                features, target, self.FEATURE_NAMES, lead_steps
            )
            self.scalers.append(scaler)
            self.models.append(model)
            self.lead_summaries.append(summary)

    def predict(
        self,
        history: pd.DataFrame,
        future_exog: pd.DataFrame,
        index: pd.DatetimeIndex,
        at_time: pd.Timestamp,
    ) -> pd.Series:
        if len(index) > self.MAX_HORIZON:
            raise ValueError(f"Ridge horizon cannot exceed {self.MAX_HORIZON} steps")
        known_net = pd.to_numeric(
            history.loc[history.index < at_time, "grid_net_kw"], errors="coerce"
        )
        decision_features = self._decision_features(
            known_net, pd.DatetimeIndex([at_time])
        )
        historical = decision_features.iloc[0]
        calendar = self._target_calendar(index)
        features = calendar.copy()
        for name in decision_features.columns:
            features[name] = historical[name]
        seasonal = build_lag_features(
            known_net,
            index,
            (pd.Timedelta(days=1), pd.Timedelta(days=7)),
        )
        features["target_minus_1day"] = seasonal["lag_1day"]
        features.loc[
            index - pd.Timedelta(days=1) >= at_time,
            "target_minus_1day",
        ] = np.nan
        features["target_minus_7day"] = seasonal["lag_7day"]
        features[self.EXOG_FEATURE] = pd.to_numeric(
            future_exog[self.EXOG_FEATURE].reindex(index), errors="coerce"
        )
        features = features.loc[:, self.FEATURE_NAMES].fillna(self.fill_values)

        predictions = _predict_ridge_leads(features, self.scalers, self.models)
        return pd.Series(predictions, index=index)

    def state(self) -> dict[str, object]:
        return {
            "max_horizon": self.MAX_HORIZON,
            "feature_names": list(self.FEATURE_NAMES),
            "alpha": self.ALPHA,
            "fill_values": self.fill_values,
            "lead_models": self.lead_summaries,
            "coefficient_space": "standardized feature space",
        }

    def load_state(self, state: dict[str, object]) -> None:
        if state["max_horizon"] != self.MAX_HORIZON:
            raise ValueError("Ridge artifact has an incompatible forecast horizon")
        if state["feature_names"] != list(self.FEATURE_NAMES):
            raise ValueError("Ridge artifact has incompatible features")
        self.fill_values = {
            str(name): float(value) for name, value in state["fill_values"].items()
        }
        self.lead_summaries = list(state["lead_models"])

    def summary(self) -> dict[str, object]:
        samples = [int(item["training_samples"]) for item in self.lead_summaries]
        return {
            "model_count": len(self.models),
            "feature_names": list(self.FEATURE_NAMES),
            "alpha": self.ALPHA,
            "training_samples_min": min(samples),
            "training_samples_max": max(samples),
            "lead_models": self.lead_summaries,
        }

    def save_artifact(self, path: Path) -> None:
        joblib.dump(
            {"scalers": self.scalers, "models": self.models},
            Path(path) / self.ARTIFACT,
        )

    def load_artifact(self, path: Path) -> None:
        artifact = joblib.load(Path(path) / self.ARTIFACT)
        self.scalers = artifact["scalers"]
        self.models = artifact["models"]


class DecomposedRidgeNetLoadModel:
    """Direct load and PV Ridge forecasts recombined into grid net load."""

    ARTIFACT = "model.joblib"
    ALPHA = 1.0
    MAX_HORIZON = 132
    EXOG_FEATURE = "most_recent_load_factor_forecast"
    LOAD_FEATURE_NAMES = (
        "load_lag_15min",
        "load_lag_1h",
        "load_rolling_mean_1h",
        "load_rolling_mean_4h",
        "load_target_minus_1day",
        "load_target_minus_7day",
        "time_of_day_sin",
        "time_of_day_cos",
        "dow_0",
        "dow_1",
        "dow_2",
        "dow_3",
        "dow_4",
        "dow_5",
        "dow_6",
    )
    PV_FEATURE_NAMES = (
        "pv_lag_15min",
        "pv_lag_1h",
        EXOG_FEATURE,
        "time_of_day_sin",
        "time_of_day_cos",
    )

    def __init__(self) -> None:
        self.load_scalers: list[StandardScaler] = []
        self.load_models: list[Ridge] = []
        self.pv_scalers: list[StandardScaler] = []
        self.pv_models: list[Ridge] = []
        self.load_fill_values: dict[str, float] = {}
        self.pv_fill_values: dict[str, float] = {}
        self.load_summaries: list[dict[str, object]] = []
        self.pv_summaries: list[dict[str, object]] = []

    @staticmethod
    def _load_decision_features(
        load: pd.Series, decisions: pd.DatetimeIndex
    ) -> pd.DataFrame:
        recent = build_lag_features(
            load,
            decisions,
            (pd.Timedelta(minutes=15), pd.Timedelta(hours=1)),
        ).rename(
            columns={"lag_15min": "load_lag_15min", "lag_1h": "load_lag_1h"}
        )
        rolling = build_rolling_features(
            load,
            decisions,
            (pd.Timedelta(hours=1), pd.Timedelta(hours=4)),
        ).rename(
            columns={
                "rolling_mean_1h": "load_rolling_mean_1h",
                "rolling_mean_4h": "load_rolling_mean_4h",
            }
        )
        return recent.join(rolling)

    @staticmethod
    def _pv_decision_features(
        pv: pd.Series, decisions: pd.DatetimeIndex
    ) -> pd.DataFrame:
        return build_lag_features(
            pv,
            decisions,
            (pd.Timedelta(minutes=15), pd.Timedelta(hours=1)),
        ).rename(columns={"lag_15min": "pv_lag_15min", "lag_1h": "pv_lag_1h"})

    @classmethod
    def _load_features_for_lead(
        cls,
        decision_features: pd.DataFrame,
        load: pd.Series,
        lead_steps: int,
    ) -> pd.DataFrame:
        decisions = decision_features.index
        targets = decisions + (lead_steps - 1) * STEP
        calendar = DirectRidgeNetLoadModel._target_calendar(targets)
        calendar.index = decisions
        features = decision_features.join(calendar)
        seasonal = build_lag_features(
            load, targets, (pd.Timedelta(days=1), pd.Timedelta(days=7))
        )
        seasonal.index = decisions
        features["load_target_minus_1day"] = seasonal["lag_1day"]
        features.loc[
            targets - pd.Timedelta(days=1) >= decisions,
            "load_target_minus_1day",
        ] = np.nan
        features["load_target_minus_7day"] = seasonal["lag_7day"]
        return features.loc[:, cls.LOAD_FEATURE_NAMES]

    @classmethod
    def _pv_features_for_lead(
        cls,
        decision_features: pd.DataFrame,
        exog: pd.Series,
        lead_steps: int,
    ) -> pd.DataFrame:
        decisions = decision_features.index
        targets = decisions + (lead_steps - 1) * STEP
        calendar = DirectRidgeNetLoadModel._target_calendar(targets).loc[
            :, ["time_of_day_sin", "time_of_day_cos"]
        ]
        calendar.index = decisions
        features = decision_features.join(calendar)
        features[cls.EXOG_FEATURE] = pd.to_numeric(
            exog.reindex(targets), errors="coerce"
        ).to_numpy(dtype=float)
        return features.loc[:, cls.PV_FEATURE_NAMES]

    def fit(self, history: pd.DataFrame) -> None:
        net = pd.to_numeric(history["grid_net_kw"], errors="coerce")
        pv = pd.to_numeric(history["pv_production_kw"], errors="coerce")
        load = net + pv
        exog = pd.to_numeric(history[self.EXOG_FEATURE], errors="coerce")
        decisions = history.index
        load_decision = self._load_decision_features(load, decisions)
        pv_decision = self._pv_decision_features(pv, decisions)

        load_fill = load_decision.join(
            DirectRidgeNetLoadModel._target_calendar(decisions)
        )
        load_fill["load_target_minus_1day"] = load.reindex(
            decisions - pd.Timedelta(days=1)
        ).to_numpy(dtype=float)
        load_fill["load_target_minus_7day"] = load.reindex(
            decisions - pd.Timedelta(days=7)
        ).to_numpy(dtype=float)
        self.load_fill_values = {
            name: float(load_fill[name].median()) for name in self.LOAD_FEATURE_NAMES
        }
        pv_fill = pv_decision.join(
            DirectRidgeNetLoadModel._target_calendar(decisions).loc[
                :, ["time_of_day_sin", "time_of_day_cos"]
            ]
        )
        pv_fill[self.EXOG_FEATURE] = exog
        self.pv_fill_values = {
            name: float(pv_fill[name].median()) for name in self.PV_FEATURE_NAMES
        }
        if not all(
            np.isfinite(
                list(self.load_fill_values.values())
                + list(self.pv_fill_values.values())
            )
        ):
            raise ValueError("Training history cannot populate decomposed Ridge features")

        self.load_scalers = []
        self.load_models = []
        self.pv_scalers = []
        self.pv_models = []
        self.load_summaries = []
        self.pv_summaries = []
        for lead_steps in range(1, self.MAX_HORIZON + 1):
            targets = decisions + (lead_steps - 1) * STEP
            load_target = load.reindex(targets)
            load_target.index = decisions
            load_features = self._load_features_for_lead(
                load_decision, load, lead_steps
            )
            load_features["load_target_minus_1day"] = load_features[
                "load_target_minus_1day"
            ].fillna(self.load_fill_values["load_target_minus_1day"])
            scaler, model, summary = _fit_ridge_lead(
                load_features, load_target, self.LOAD_FEATURE_NAMES, lead_steps
            )
            self.load_scalers.append(scaler)
            self.load_models.append(model)
            self.load_summaries.append(summary)

            pv_target = pv.reindex(targets)
            pv_target.index = decisions
            pv_features = self._pv_features_for_lead(
                pv_decision, exog, lead_steps
            )
            scaler, model, summary = _fit_ridge_lead(
                pv_features, pv_target, self.PV_FEATURE_NAMES, lead_steps
            )
            self.pv_scalers.append(scaler)
            self.pv_models.append(model)
            self.pv_summaries.append(summary)

    def _prediction_features(
        self,
        history: pd.DataFrame,
        future_exog: pd.DataFrame,
        index: pd.DatetimeIndex,
        at_time: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if len(index) > self.MAX_HORIZON:
            raise ValueError(f"Ridge horizon cannot exceed {self.MAX_HORIZON} steps")
        recent_index = pd.date_range(end=at_time - STEP, periods=16, freq=STEP)
        recent = history.reindex(recent_index)
        recent_pv = pd.to_numeric(recent["pv_production_kw"], errors="coerce")
        recent_load = (
            pd.to_numeric(recent["grid_net_kw"], errors="coerce") + recent_pv
        )
        load_recent = {
            "load_lag_15min": recent_load.iloc[-1],
            "load_lag_1h": recent_load.iloc[-4],
            "load_rolling_mean_1h": (
                recent_load.iloc[-4:].mean()
                if recent_load.iloc[-4:].notna().all()
                else np.nan
            ),
            "load_rolling_mean_4h": (
                recent_load.mean() if recent_load.notna().all() else np.nan
            ),
        }
        load_features = DirectRidgeNetLoadModel._target_calendar(index)
        for name, value in load_recent.items():
            load_features[name] = value

        one_day_index = index - pd.Timedelta(days=1)
        seven_day_index = index - pd.Timedelta(days=7)
        seasonal_index = one_day_index[one_day_index < at_time].union(
            seven_day_index[seven_day_index < at_time]
        )
        seasonal = history.reindex(seasonal_index)
        seasonal_pv = pd.to_numeric(
            seasonal["pv_production_kw"], errors="coerce"
        )
        seasonal_load = (
            pd.to_numeric(seasonal["grid_net_kw"], errors="coerce") + seasonal_pv
        )
        load_features["load_target_minus_1day"] = seasonal_load.reindex(
            one_day_index
        ).to_numpy()
        load_features["load_target_minus_7day"] = seasonal_load.reindex(
            seven_day_index
        ).to_numpy()
        load_features = load_features.loc[:, self.LOAD_FEATURE_NAMES].fillna(
            self.load_fill_values
        )

        pv_recent = {
            "pv_lag_15min": recent_pv.iloc[-1],
            "pv_lag_1h": recent_pv.iloc[-4],
        }
        pv_features = DirectRidgeNetLoadModel._target_calendar(index).loc[
            :, ["time_of_day_sin", "time_of_day_cos"]
        ]
        for name, value in pv_recent.items():
            pv_features[name] = value
        pv_features[self.EXOG_FEATURE] = pd.to_numeric(
            future_exog[self.EXOG_FEATURE].reindex(index), errors="coerce"
        )
        pv_features = pv_features.loc[:, self.PV_FEATURE_NAMES].fillna(
            self.pv_fill_values
        )

        return load_features, pv_features

    def _predict_components(
        self,
        history: pd.DataFrame,
        future_exog: pd.DataFrame,
        index: pd.DatetimeIndex,
        at_time: pd.Timestamp,
    ) -> tuple[pd.Series, pd.Series]:
        load_features, pv_features = self._prediction_features(
            history, future_exog, index, at_time
        )

        load_prediction = _predict_ridge_leads(
            load_features, self.load_scalers, self.load_models
        )
        pv_prediction = np.maximum(
            _predict_ridge_leads(pv_features, self.pv_scalers, self.pv_models), 0.0
        )
        return (
            pd.Series(load_prediction, index=index),
            pd.Series(pv_prediction, index=index),
        )

    def predict(
        self,
        history: pd.DataFrame,
        future_exog: pd.DataFrame,
        index: pd.DatetimeIndex,
        at_time: pd.Timestamp,
    ) -> pd.Series:
        load, pv = self._predict_components(history, future_exog, index, at_time)
        return load - pv

    def state(self) -> dict[str, object]:
        return {
            "max_horizon": self.MAX_HORIZON,
            "alpha": self.ALPHA,
            "load": {
                "feature_names": list(self.LOAD_FEATURE_NAMES),
                "fill_values": self.load_fill_values,
                "lead_models": self.load_summaries,
            },
            "pv": {
                "feature_names": list(self.PV_FEATURE_NAMES),
                "fill_values": self.pv_fill_values,
                "lead_models": self.pv_summaries,
                "prediction_clip_min_kw": 0.0,
            },
            "coefficient_space": "standardized feature space",
        }

    def load_state(self, state: dict[str, object]) -> None:
        if state["max_horizon"] != self.MAX_HORIZON:
            raise ValueError("Ridge artifact has an incompatible forecast horizon")
        if state["load"]["feature_names"] != list(self.LOAD_FEATURE_NAMES):
            raise ValueError("Ridge artifact has incompatible load features")
        if state["pv"]["feature_names"] != list(self.PV_FEATURE_NAMES):
            raise ValueError("Ridge artifact has incompatible PV features")
        self.load_fill_values = {
            str(name): float(value)
            for name, value in state["load"]["fill_values"].items()
        }
        self.pv_fill_values = {
            str(name): float(value)
            for name, value in state["pv"]["fill_values"].items()
        }
        self.load_summaries = list(state["load"]["lead_models"])
        self.pv_summaries = list(state["pv"]["lead_models"])

    def summary(self) -> dict[str, object]:
        return self.state()

    def save_artifact(self, path: Path) -> None:
        joblib.dump(
            {
                "load_scalers": self.load_scalers,
                "load_models": self.load_models,
                "pv_scalers": self.pv_scalers,
                "pv_models": self.pv_models,
            },
            Path(path) / self.ARTIFACT,
        )

    def load_artifact(self, path: Path) -> None:
        artifact = joblib.load(Path(path) / self.ARTIFACT)
        self.load_scalers = artifact["load_scalers"]
        self.load_models = artifact["load_models"]
        self.pv_scalers = artifact["pv_scalers"]
        self.pv_models = artifact["pv_models"]


class RidgeHgbrDecomposedNetLoadModel(DecomposedRidgeNetLoadModel):
    """Existing Ridge load forecast combined with direct HGBR PV forecasts."""

    HGBR_PARAMETERS = {
        "learning_rate": 0.08,
        "max_iter": 50,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
        "random_state": 0,
    }

    def fit(self, history: pd.DataFrame) -> None:
        # Reuse the established decomposed training path so the load models and their
        # fill values remain exactly the same as ``ridge_decomposed``.
        super().fit(history)

        pv = pd.to_numeric(history["pv_production_kw"], errors="coerce")
        exog = pd.to_numeric(history[self.EXOG_FEATURE], errors="coerce")
        decisions = history.index
        pv_decision = self._pv_decision_features(pv, decisions)
        self.pv_scalers = []
        self.pv_models = []
        self.pv_summaries = []

        for lead_steps in range(1, self.MAX_HORIZON + 1):
            targets = decisions + (lead_steps - 1) * STEP
            target = pv.reindex(targets)
            target.index = decisions
            features = self._pv_features_for_lead(
                pv_decision, exog, lead_steps
            ).fillna(self.pv_fill_values)
            usable = (
                features.notna().all(axis=1)
                & np.isfinite(features).all(axis=1)
                & target.notna()
                & np.isfinite(target)
            )
            if usable.sum() < 12:
                raise ValueError(f"HGBR PV lead {lead_steps} requires 12 usable rows")

            model = HistGradientBoostingRegressor(**self.HGBR_PARAMETERS)
            model.fit(features.loc[usable], target.loc[usable])
            fitted = model.predict(features.loc[usable])
            self.pv_models.append(model)
            self.pv_summaries.append(
                {
                    "lead_steps": lead_steps,
                    "lead_minutes": (lead_steps - 1) * 15,
                    "training_samples": int(usable.sum()),
                    "dropped_rows": int((~usable).sum()),
                    "train_mae": float(mean_absolute_error(target.loc[usable], fitted)),
                    "train_rmse": float(
                        mean_squared_error(target.loc[usable], fitted) ** 0.5
                    ),
                }
            )

    def _predict_components(
        self,
        history: pd.DataFrame,
        future_exog: pd.DataFrame,
        index: pd.DatetimeIndex,
        at_time: pd.Timestamp,
    ) -> tuple[pd.Series, pd.Series]:
        load_features, pv_features = self._prediction_features(
            history, future_exog, index, at_time
        )
        load_prediction = _predict_ridge_leads(
            load_features, self.load_scalers, self.load_models
        )
        pv_prediction = np.maximum(
            np.array(
                [
                    model.predict(pv_features.iloc[[lead]])[0]
                    for lead, model in enumerate(self.pv_models[: len(index)])
                ]
            ),
            0.0,
        )
        return (
            pd.Series(load_prediction, index=index),
            pd.Series(pv_prediction, index=index),
        )

    def state(self) -> dict[str, object]:
        state = super().state()
        state["pv"]["model"] = "HistGradientBoostingRegressor"
        state["pv"]["parameters"] = self.HGBR_PARAMETERS
        return state


class Forecaster:
    """Harness-facing wrapper for the selected forecasting baseline."""

    PARAMS = "forecaster.json"

    SCAFFOLD = "scaffold"
    WEEKLY = "weekly"
    RIDGE = "ridge"
    RIDGE_DECOMPOSED = "ridge_decomposed"
    RIDGE_HGBR_DECOMPOSED = "ridge_hgbr_decomposed"

    def __init__(self, specs: SiteSpecs, model_name: str = SELECTED_MODEL) -> None:
        self.specs = specs
        self.model_name = model_name
        self.model = self._make_model(model_name)

    @staticmethod
    def _make_model(
        model_name: str,
    ) -> (
        ScaffoldBaseline
        | WeeklySeasonalBaseline
        | DirectRidgeNetLoadModel
        | DecomposedRidgeNetLoadModel
    ):
        if model_name == Forecaster.SCAFFOLD:
            return ScaffoldBaseline()
        if model_name == Forecaster.WEEKLY:
            return WeeklySeasonalBaseline()
        if model_name == Forecaster.RIDGE:
            return DirectRidgeNetLoadModel()
        if model_name == Forecaster.RIDGE_DECOMPOSED:
            return DecomposedRidgeNetLoadModel()
        if model_name == Forecaster.RIDGE_HGBR_DECOMPOSED:
            return RidgeHgbrDecomposedNetLoadModel()
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
        if isinstance(
            self.model, (DirectRidgeNetLoadModel, DecomposedRidgeNetLoadModel)
        ):
            self.model.save_artifact(path)

    @classmethod
    def load(cls, path: Path, specs: SiteSpecs) -> "Forecaster":
        params = json.loads((Path(path) / cls.PARAMS).read_text())
        self = cls(specs, model_name=params["model_name"])
        self.model.load_state(params["state"])
        if isinstance(
            self.model, (DirectRidgeNetLoadModel, DecomposedRidgeNetLoadModel)
        ):
            self.model.load_artifact(path)
        return self

    def predict(
        self,
        at_time: pd.Timestamp,
        history: pd.DataFrame,
        future_exog: pd.DataFrame,
        horizon: int,
    ) -> pd.DataFrame:
        index = future_exog.index[:horizon]
        if isinstance(self.model, DecomposedRidgeNetLoadModel):
            load, pv = self.model._predict_components(
                history, future_exog, index, at_time
            )
            return pd.DataFrame(
                {"net_kw": load - pv, "load_kw": load, "pv_kw": pv}, index=index
            )
        if isinstance(self.model, DirectRidgeNetLoadModel):
            values = self.model.predict(history, future_exog, index, at_time)
        else:
            values = self.model.predict(history, index)
        return pd.DataFrame({"net_kw": values}, index=index)
