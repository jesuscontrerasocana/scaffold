from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.forecaster import Forecaster
from pipeline.specs import SiteSpecs


@pytest.fixture
def specs() -> SiteSpecs:
    return SiteSpecs.from_yaml("site.yaml")


def _history() -> pd.DataFrame:
    index = pd.date_range(
        "2026-01-05", periods=14 * 96, freq="15min", tz="Europe/Brussels"
    )
    slots = index.dayofweek * 96 + index.hour * 4 + index.minute // 15
    return pd.DataFrame(
        {
            "grid_net_kw": slots.astype(float),
            "pv_production_kw": 0.0,
            "offtake_price_eur_per_mwh": 50.0,
            "injection_price_eur_per_mwh": 20.0,
            "most_recent_load_factor_forecast": 0.1,
        },
        index=index,
    )


def _predict(
    forecaster: Forecaster,
    history: pd.DataFrame,
    at_time: pd.Timestamp,
    horizon: int = 2,
) -> pd.DataFrame:
    index = pd.date_range(at_time, periods=horizon, freq="15min")
    return forecaster.predict(
        at_time,
        history.loc[history.index < at_time],
        pd.DataFrame(index=index),
        horizon,
    )


def test_scaffold_model_keeps_all_fallback_stages_after_save_and_load(
    specs: SiteSpecs, tmp_path
) -> None:
    history = _history()
    at_time = history.index[-1] + pd.Timedelta(minutes=15)
    history.loc[at_time - pd.Timedelta(days=1), "grid_net_kw"] = 123.0
    history.loc[
        at_time - pd.Timedelta(days=1) + pd.Timedelta(minutes=15), "grid_net_kw"
    ] = np.nan
    history.loc[
        at_time - pd.Timedelta(days=7) + pd.Timedelta(minutes=15), "grid_net_kw"
    ] = 456.0
    history.loc[
        at_time - pd.Timedelta(days=1) + pd.Timedelta(minutes=30), "grid_net_kw"
    ] = np.nan
    history.loc[
        at_time - pd.Timedelta(days=7) + pd.Timedelta(minutes=30), "grid_net_kw"
    ] = np.nan
    forecaster = Forecaster(specs, model_name=Forecaster.SCAFFOLD)
    forecaster.fit(history)
    expected = [123.0, 456.0, forecaster.fallback_kw]

    assert _predict(forecaster, history, at_time, horizon=3)["net_kw"].tolist() == expected

    forecaster.save(tmp_path)
    loaded = Forecaster.load(tmp_path, specs)

    assert _predict(loaded, history, at_time, horizon=3)["net_kw"].tolist() == expected


def test_standard_training_selects_weekly_model(specs: SiteSpecs) -> None:
    forecaster = Forecaster(specs)

    assert forecaster.model_name == Forecaster.WEEKLY


def test_weekly_model_uses_previous_week(specs: SiteSpecs) -> None:
    history = _history()
    at_time = history.index[-1] + pd.Timedelta(minutes=15)
    history.loc[at_time - pd.Timedelta(days=7), "grid_net_kw"] = 321.0
    forecaster = Forecaster(specs, model_name=Forecaster.WEEKLY)
    forecaster.fit(history)

    assert _predict(forecaster, history, at_time).iloc[0]["net_kw"] == 321.0


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_weekly_model_falls_back_to_time_of_week_median(
    specs: SiteSpecs, invalid_value: float
) -> None:
    history = _history()
    at_time = history.index[-1] + pd.Timedelta(minutes=15)
    weekly_lag = at_time - pd.Timedelta(days=7)
    expected = float(at_time.dayofweek * 96 + at_time.hour * 4 + at_time.minute // 15)
    forecaster = Forecaster(specs, model_name=Forecaster.WEEKLY)
    forecaster.fit(history)
    prediction_history = history.copy()
    prediction_history.loc[weekly_lag, "grid_net_kw"] = invalid_value

    assert _predict(forecaster, prediction_history, at_time).iloc[0]["net_kw"] == expected


def test_weekly_model_falls_back_when_weekly_observation_is_unavailable(
    specs: SiteSpecs,
) -> None:
    history = _history()
    at_time = history.index[-1] + pd.Timedelta(minutes=15)
    expected = float(at_time.dayofweek * 96 + at_time.hour * 4 + at_time.minute // 15)
    forecaster = Forecaster(specs, model_name=Forecaster.WEEKLY)
    forecaster.fit(history)
    prediction_history = history.drop(at_time - pd.Timedelta(days=7))

    assert _predict(forecaster, prediction_history, at_time).iloc[0]["net_kw"] == expected


def test_selected_model_and_state_survive_save_and_load(
    specs: SiteSpecs, tmp_path
) -> None:
    forecaster = Forecaster(specs, model_name=Forecaster.WEEKLY)
    forecaster.fit(_history())
    forecaster.save(tmp_path)

    loaded = Forecaster.load(tmp_path, specs)

    assert loaded.model_name == Forecaster.WEEKLY
    assert loaded.model.state() == forecaster.model.state()


def test_unknown_model_is_rejected(specs: SiteSpecs) -> None:
    with pytest.raises(ValueError, match="Unknown forecasting model"):
        Forecaster(specs, model_name="unknown")
