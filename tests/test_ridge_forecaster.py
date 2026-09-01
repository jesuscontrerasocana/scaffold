from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pipeline.forecaster import Forecaster, RidgeNetLoadModel
from pipeline.specs import SiteSpecs


@pytest.fixture
def specs() -> SiteSpecs:
    return SiteSpecs.from_yaml("site.yaml")


def _data(days: int = 16) -> pd.DataFrame:
    index = pd.date_range(
        "2026-01-05", periods=days * 96, freq="15min", tz="Europe/Brussels"
    )
    slot = index.hour * 4 + index.minute // 15
    net = 40 + 12 * np.sin(2 * np.pi * slot / 96) + index.dayofweek
    return pd.DataFrame(
        {
            "grid_net_kw": net,
            "pv_production_kw": 0.0,
            "offtake_price_eur_per_mwh": 50.0,
            "injection_price_eur_per_mwh": 20.0,
            "most_recent_load_factor_forecast": net / 100,
        },
        index=index,
    )


def _predict(
    forecaster: Forecaster, history: pd.DataFrame, at_time: pd.Timestamp, horizon: int
) -> pd.DataFrame:
    index = pd.date_range(at_time, periods=horizon, freq="15min")
    future_exog = pd.DataFrame(
        {"most_recent_load_factor_forecast": 0.5}, index=index
    )
    return forecaster.predict(at_time, history, future_exog, horizon)


def test_ridge_trains_predicts_and_exposes_summary(specs: SiteSpecs) -> None:
    history = _data()
    forecaster = Forecaster(specs, model_name=Forecaster.RIDGE)
    forecaster.fit(history)
    prediction = _predict(
        forecaster, history, history.index[-1] + pd.Timedelta(minutes=15), 132
    )
    summary = forecaster.model.summary()

    assert prediction.columns.tolist() == ["net_kw"]
    assert len(prediction) == 132
    assert np.isfinite(prediction["net_kw"]).all()
    assert summary["selected_features"] == forecaster.model.feature_names
    assert set(summary["coefficients"]) == set(forecaster.model.feature_names)
    assert summary["training_samples"] == len(history)
    assert summary["train_mae"] >= 0
    assert summary["validation_rmse"] >= 0


def test_ridge_save_load_preserves_features_and_predictions(
    specs: SiteSpecs, tmp_path
) -> None:
    history = _data()
    at_time = history.index[-1] + pd.Timedelta(minutes=15)
    forecaster = Forecaster(specs, model_name=Forecaster.RIDGE)
    forecaster.fit(history)
    expected = _predict(forecaster, history, at_time, 132)
    forecaster.save(tmp_path)

    loaded = Forecaster.load(tmp_path, specs)
    actual = _predict(loaded, history, at_time, 132)
    saved = json.loads((tmp_path / Forecaster.PARAMS).read_text())

    pd.testing.assert_frame_equal(actual, expected)
    assert (tmp_path / RidgeNetLoadModel.ARTIFACT).exists()
    assert saved["state"]["selected_continuous_features"]
    assert loaded.model.selected_continuous_features == (
        forecaster.model.selected_continuous_features
    )


def test_long_horizon_does_not_use_realized_values_after_decision(
    specs: SiteSpecs,
) -> None:
    data = _data()
    forecaster = Forecaster(specs, model_name=Forecaster.RIDGE)
    forecaster.fit(data)
    at_time = data.index[-132]
    safe_history = data.loc[data.index < at_time]
    leaked_history = data.copy()
    leaked_history.loc[leaked_history.index >= at_time, "grid_net_kw"] = 1_000_000
    index = pd.date_range(at_time, periods=132, freq="15min")
    future_exog = pd.DataFrame(
        {"most_recent_load_factor_forecast": data.loc[index, "most_recent_load_factor_forecast"]},
        index=index,
    )

    safe = forecaster.predict(at_time, safe_history, future_exog, 132)
    leaked = forecaster.predict(at_time, leaked_history, future_exog, 132)

    pd.testing.assert_frame_equal(safe, leaked)
