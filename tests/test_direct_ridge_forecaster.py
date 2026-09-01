from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pipeline.forecaster import DirectRidgeNetLoadModel, Forecaster
from pipeline.specs import SiteSpecs


@pytest.fixture
def specs() -> SiteSpecs:
    return SiteSpecs.from_yaml("site.yaml")


def _data(days: int = 10) -> pd.DataFrame:
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


def _future(at_time: pd.Timestamp, horizon: int = 132) -> pd.DataFrame:
    index = pd.date_range(at_time, periods=horizon, freq="15min")
    return pd.DataFrame(
        {"most_recent_load_factor_forecast": np.linspace(0.2, 0.8, horizon)},
        index=index,
    )


def test_direct_ridge_trains_and_predicts_full_horizon(specs: SiteSpecs) -> None:
    history = _data()
    model = Forecaster(specs, model_name=Forecaster.RIDGE)
    model.fit(history)
    at_time = history.index[-1] + pd.Timedelta(minutes=15)
    future = _future(at_time)

    prediction = model.predict(at_time, history, future, 132)

    assert prediction.index.equals(future.index)
    assert prediction.columns.tolist() == ["net_kw"]
    assert np.isfinite(prediction["net_kw"]).all()
    assert len(model.model.models) == 132


def test_prediction_is_direct_and_ignores_future_actuals(specs: SiteSpecs) -> None:
    data = _data()
    forecaster = Forecaster(specs, model_name=Forecaster.RIDGE)
    forecaster.fit(data)
    at_time = data.index[-132]
    future = _future(at_time)
    safe_history = data.loc[data.index < at_time]
    oversized = data.copy()
    oversized.loc[oversized.index >= at_time, "grid_net_kw"] = 1_000_000

    expected = forecaster.predict(at_time, safe_history, future, 132)
    actual = forecaster.predict(at_time, oversized, future, 132)
    later_before = actual.iloc[1:].copy()
    forecaster.model.models[0].intercept_ += 1_000_000
    changed = forecaster.predict(at_time, safe_history, future, 132)

    pd.testing.assert_frame_equal(actual, expected)
    assert changed.iloc[0, 0] != expected.iloc[0, 0]
    pd.testing.assert_series_equal(changed.iloc[1:, 0], later_before.iloc[:, 0])


def test_features_use_decision_history_and_target_time() -> None:
    data = _data()
    decisions = data.index[7 * 96 + 20 : 7 * 96 + 22]
    decision_features = DirectRidgeNetLoadModel._decision_features(
        data["grid_net_kw"], decisions
    )
    lead = 5
    features = DirectRidgeNetLoadModel._features_for_lead(
        decision_features,
        data["grid_net_kw"],
        data["most_recent_load_factor_forecast"],
        lead,
    )
    target_times = decisions + (lead - 1) * pd.Timedelta(minutes=15)

    assert features.iloc[0]["lag_15min"] == data.at[
        decisions[0] - pd.Timedelta(minutes=15), "grid_net_kw"
    ]
    assert features.iloc[0]["lag_1h"] == data.at[
        decisions[0] - pd.Timedelta(hours=1), "grid_net_kw"
    ]
    assert features.iloc[0]["most_recent_load_factor_forecast"] == data.at[
        target_times[0], "most_recent_load_factor_forecast"
    ]
    assert features.iloc[0]["target_minus_1day"] == data.at[
        target_times[0] - pd.Timedelta(days=1), "grid_net_kw"
    ]
    expected_slot = target_times[0].hour * 4 + target_times[0].minute // 15
    assert features.iloc[0]["time_of_day_sin"] == pytest.approx(
        np.sin(2 * np.pi * expected_slot / 96)
    )

    long_lead = DirectRidgeNetLoadModel._features_for_lead(
        decision_features,
        data["grid_net_kw"],
        data["most_recent_load_factor_forecast"],
        100,
    )
    assert long_lead["target_minus_1day"].isna().all()
    long_target = decisions[0] + 99 * pd.Timedelta(minutes=15)
    assert long_lead.iloc[0]["target_minus_7day"] == data.at[
        long_target - pd.Timedelta(days=7), "grid_net_kw"
    ]


def test_save_load_metadata_and_missing_fallback(
    specs: SiteSpecs, tmp_path
) -> None:
    history = _data()
    forecaster = Forecaster(specs, model_name=Forecaster.RIDGE)
    forecaster.fit(history)
    at_time = history.index[-1] + pd.Timedelta(minutes=15)
    future = _future(at_time)
    future.iloc[10, 0] = np.nan
    damaged_history = history.copy()
    damaged_history.loc[at_time - pd.Timedelta(minutes=15), "grid_net_kw"] = np.nan
    expected = forecaster.predict(at_time, damaged_history, future, 132)
    forecaster.save(tmp_path)

    loaded = Forecaster.load(tmp_path, specs)
    actual = loaded.predict(at_time, damaged_history, future, 132)
    metadata = json.loads((tmp_path / Forecaster.PARAMS).read_text())
    state = metadata["state"]

    pd.testing.assert_frame_equal(actual, expected)
    assert np.isfinite(actual["net_kw"]).all()
    assert state["max_horizon"] == 132
    assert state["feature_names"] == list(DirectRidgeNetLoadModel.FEATURE_NAMES)
    assert len(state["lead_models"]) == 132
    assert state["lead_models"][0]["lead_minutes"] == 0
    assert state["lead_models"][-1]["lead_minutes"] == 131 * 15
    assert set(state["lead_models"][0]["coefficients"]) == set(
        DirectRidgeNetLoadModel.FEATURE_NAMES
    )
    assert "intercept" in state["lead_models"][0]
    assert (tmp_path / DirectRidgeNetLoadModel.ARTIFACT).exists()


def test_prediction_builds_history_features_once(
    specs: SiteSpecs, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = _data()
    forecaster = Forecaster(specs, model_name=Forecaster.RIDGE)
    forecaster.fit(history)
    at_time = history.index[-1] + pd.Timedelta(minutes=15)
    calls = 0
    original = DirectRidgeNetLoadModel._decision_features.__func__

    def recording_features(cls, net, decisions):  # noqa: ANN001
        nonlocal calls
        calls += 1
        return original(cls, net, decisions)

    monkeypatch.setattr(
        DirectRidgeNetLoadModel,
        "_decision_features",
        classmethod(recording_features),
    )

    forecaster.predict(at_time, history, _future(at_time), 132)

    assert calls == 1
