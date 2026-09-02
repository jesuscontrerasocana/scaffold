from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pipeline.forecaster import DecomposedRidgeNetLoadModel, Forecaster
from pipeline.specs import SiteSpecs


def _data(days: int = 12) -> pd.DataFrame:
    index = pd.date_range(
        "2026-01-05", periods=days * 96, freq="15min", tz="Europe/Brussels"
    )
    slot = index.hour * 4 + index.minute // 15
    daylight = np.maximum(np.sin(np.pi * (slot - 24) / 48), 0)
    pv = 25 * daylight
    load = 55 + 8 * np.sin(2 * np.pi * (slot - 20) / 96) + index.dayofweek
    return pd.DataFrame(
        {
            "grid_net_kw": load - pv,
            "pv_production_kw": pv,
            "offtake_price_eur_per_mwh": 50.0,
            "injection_price_eur_per_mwh": 20.0,
            "most_recent_load_factor_forecast": daylight,
        },
        index=index,
    )


def _future(at_time: pd.Timestamp, horizon: int = 132) -> pd.DataFrame:
    index = pd.date_range(at_time, periods=horizon, freq="15min")
    slot = index.hour * 4 + index.minute // 15
    return pd.DataFrame(
        {"most_recent_load_factor_forecast": np.maximum(
            np.sin(np.pi * (slot - 24) / 48), 0
        )},
        index=index,
    )


@pytest.fixture(scope="module")
def trained() -> tuple[Forecaster, pd.DataFrame, SiteSpecs]:
    specs = SiteSpecs.from_yaml("site.yaml")
    history = _data()
    history.iloc[100, history.columns.get_loc("pv_production_kw")] = np.nan
    forecaster = Forecaster(specs, model_name=Forecaster.RIDGE_DECOMPOSED)
    forecaster.fit(history)
    return forecaster, history, specs


def test_full_horizon_and_decomposition_identity(trained) -> None:  # noqa: ANN001
    forecaster, history, _ = trained
    at_time = history.index[-1] + pd.Timedelta(minutes=15)
    future = _future(at_time)
    load, pv = forecaster.model._predict_components(
        history, future, future.index, at_time
    )
    prediction = forecaster.predict(at_time, history, future, 132)

    pd.testing.assert_series_equal(prediction["net_kw"], load - pv, check_names=False)
    pd.testing.assert_series_equal(prediction["load_kw"], load, check_names=False)
    pd.testing.assert_series_equal(prediction["pv_kw"], pv, check_names=False)
    assert prediction.columns.tolist() == ["net_kw", "load_kw", "pv_kw"]
    assert len(prediction) == 132
    assert np.isfinite(prediction["net_kw"]).all()
    assert (pv >= 0).all()
    assert len(forecaster.model.load_models) == 132
    assert len(forecaster.model.pv_models) == 132


def test_no_future_actual_leakage(trained) -> None:  # noqa: ANN001
    forecaster, data, _ = trained
    at_time = data.index[-132]
    future = _future(at_time)
    safe = data.loc[data.index < at_time]
    oversized = data.copy()
    oversized.loc[oversized.index >= at_time, [
        "grid_net_kw", "pv_production_kw"
    ]] = 1_000_000

    expected = forecaster.predict(at_time, safe, future, 132)
    actual = forecaster.predict(at_time, oversized, future, 132)

    pd.testing.assert_frame_equal(actual, expected)


def test_load_and_pv_feature_semantics() -> None:
    data = _data()
    decisions = data.index[7 * 96 + 20 : 7 * 96 + 22]
    load = data["grid_net_kw"] + data["pv_production_kw"]
    load_recent = DecomposedRidgeNetLoadModel._load_decision_features(
        load, decisions
    )
    pv_recent = DecomposedRidgeNetLoadModel._pv_decision_features(
        data["pv_production_kw"], decisions
    )
    load_features = DecomposedRidgeNetLoadModel._load_features_for_lead(
        load_recent, load, 100
    )
    pv_features = DecomposedRidgeNetLoadModel._pv_features_for_lead(
        pv_recent, data["most_recent_load_factor_forecast"], 5
    )
    pv_target = decisions[0] + 4 * pd.Timedelta(minutes=15)

    assert load_recent.iloc[0]["load_lag_15min"] == load.at[
        decisions[0] - pd.Timedelta(minutes=15)
    ]
    assert load_features["load_target_minus_1day"].isna().all()
    assert load_features["load_target_minus_7day"].notna().all()
    assert tuple(pv_features.columns) == DecomposedRidgeNetLoadModel.PV_FEATURE_NAMES
    assert pv_features.iloc[0]["pv_lag_1h"] == data.at[
        decisions[0] - pd.Timedelta(hours=1), "pv_production_kw"
    ]
    assert pv_features.iloc[0]["most_recent_load_factor_forecast"] == data.at[
        pv_target, "most_recent_load_factor_forecast"
    ]


def test_negative_pv_is_clipped_before_recombination(trained) -> None:  # noqa: ANN001
    forecaster, history, _ = trained
    at_time = history.index[-1] + pd.Timedelta(minutes=15)
    future = _future(at_time)
    original_intercepts = [model.intercept_ for model in forecaster.model.pv_models]
    original_coefficients = [model.coef_.copy() for model in forecaster.model.pv_models]
    try:
        for model in forecaster.model.pv_models:
            model.coef_[:] = 0
            model.intercept_ = -100.0
        load, pv = forecaster.model._predict_components(
            history, future, future.index, at_time
        )
        prediction = forecaster.predict(at_time, history, future, 132)
        assert (pv == 0).all()
        pd.testing.assert_series_equal(prediction["net_kw"], load, check_names=False)
    finally:
        for model, intercept, coefficients in zip(
            forecaster.model.pv_models,
            original_intercepts,
            original_coefficients,
            strict=True,
        ):
            model.intercept_ = intercept
            model.coef_ = coefficients


def test_save_load_and_metadata(trained, tmp_path) -> None:  # noqa: ANN001
    forecaster, history, specs = trained
    at_time = history.index[-1] + pd.Timedelta(minutes=15)
    future = _future(at_time)
    expected = forecaster.predict(at_time, history, future, 132)
    forecaster.save(tmp_path)

    loaded = Forecaster.load(tmp_path, specs)
    actual = loaded.predict(at_time, history, future, 132)
    state = json.loads((tmp_path / Forecaster.PARAMS).read_text())["state"]

    pd.testing.assert_frame_equal(actual, expected)
    assert state["load"]["feature_names"] == list(
        DecomposedRidgeNetLoadModel.LOAD_FEATURE_NAMES
    )
    assert state["pv"]["feature_names"] == list(
        DecomposedRidgeNetLoadModel.PV_FEATURE_NAMES
    )
    assert state["pv"]["prediction_clip_min_kw"] == 0.0
    assert len(state["load"]["lead_models"]) == 132
    assert len(state["pv"]["lead_models"]) == 132


def test_recent_features_are_built_once_per_prediction(
    trained, monkeypatch: pytest.MonkeyPatch
) -> None:  # noqa: ANN001
    forecaster, history, _ = trained
    at_time = history.index[-1] + pd.Timedelta(minutes=15)
    calls = {"load": 0, "pv": 0}
    load_original = DecomposedRidgeNetLoadModel._load_decision_features
    pv_original = DecomposedRidgeNetLoadModel._pv_decision_features

    def load_recording(load, decisions):  # noqa: ANN001
        calls["load"] += 1
        return load_original(load, decisions)

    def pv_recording(pv, decisions):  # noqa: ANN001
        calls["pv"] += 1
        return pv_original(pv, decisions)

    monkeypatch.setattr(
        DecomposedRidgeNetLoadModel,
        "_load_decision_features",
        staticmethod(load_recording),
    )
    monkeypatch.setattr(
        DecomposedRidgeNetLoadModel,
        "_pv_decision_features",
        staticmethod(pv_recording),
    )

    forecaster.predict(at_time, history, _future(at_time), 132)

    assert calls == {"load": 1, "pv": 1}
