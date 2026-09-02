from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pipeline.forecaster import (
    DecomposedRidgeNetLoadModel,
    DirectRidgeNetLoadModel,
    Forecaster,
    RidgeHgbrDecomposedNetLoadModel,
    _predict_ridge_leads,
    build_lag_features,
)
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


def _legacy_predict_components(
    model: DecomposedRidgeNetLoadModel,
    history: pd.DataFrame,
    future_exog: pd.DataFrame,
    index: pd.DatetimeIndex,
    at_time: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the pre-optimization inference path for parity testing."""
    known = history.loc[history.index < at_time]
    pv = pd.to_numeric(known["pv_production_kw"], errors="coerce")
    load = pd.to_numeric(known["grid_net_kw"], errors="coerce") + pv
    load_recent = model._load_decision_features(
        load, pd.DatetimeIndex([at_time])
    ).iloc[0]
    load_features = DirectRidgeNetLoadModel._target_calendar(index)
    for name, value in load_recent.items():
        load_features[name] = value
    seasonal = build_lag_features(
        load, index, (pd.Timedelta(days=1), pd.Timedelta(days=7))
    )
    load_features["load_target_minus_1day"] = seasonal["lag_1day"]
    load_features.loc[
        index - pd.Timedelta(days=1) >= at_time,
        "load_target_minus_1day",
    ] = np.nan
    load_features["load_target_minus_7day"] = seasonal["lag_7day"]
    load_features = load_features.loc[:, model.LOAD_FEATURE_NAMES].fillna(
        model.load_fill_values
    )

    pv_recent = model._pv_decision_features(
        pv, pd.DatetimeIndex([at_time])
    ).iloc[0]
    pv_features = DirectRidgeNetLoadModel._target_calendar(index).loc[
        :, ["time_of_day_sin", "time_of_day_cos"]
    ]
    for name, value in pv_recent.items():
        pv_features[name] = value
    pv_features[model.EXOG_FEATURE] = pd.to_numeric(
        future_exog[model.EXOG_FEATURE].reindex(index), errors="coerce"
    )
    pv_features = pv_features.loc[:, model.PV_FEATURE_NAMES].fillna(
        model.pv_fill_values
    )

    load_prediction = _predict_ridge_leads(
        load_features, model.load_scalers, model.load_models
    )
    pv_prediction = np.maximum(
        _predict_ridge_leads(pv_features, model.pv_scalers, model.pv_models), 0.0
    )
    return load_prediction, pv_prediction


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
    pv_target_slot = pv_target.hour * 4 + pv_target.minute // 15

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
    assert pv_features.iloc[0]["time_of_day_sin"] == pytest.approx(
        np.sin(2 * np.pi * pv_target_slot / 96)
    )
    assert pv_features.iloc[0]["time_of_day_cos"] == pytest.approx(
        np.cos(2 * np.pi * pv_target_slot / 96)
    )
    decision_slot = decisions[0].hour * 4 + decisions[0].minute // 15
    assert pv_features.iloc[0]["time_of_day_sin"] != pytest.approx(
        np.sin(2 * np.pi * decision_slot / 96)
    )


def test_negative_pv_is_clipped_before_recombination(trained) -> None:  # noqa: ANN001
    forecaster, history, _ = trained
    at_time = history.index[-1] + pd.Timedelta(minutes=15)
    future = _future(at_time)
    original_weights = forecaster.model.pv_inference_weights.copy()
    original_intercepts = forecaster.model.pv_inference_intercepts.copy()
    try:
        forecaster.model.pv_inference_weights[:] = 0.0
        forecaster.model.pv_inference_intercepts[:] = -100.0
        load, pv = forecaster.model._predict_components(
            history, future, future.index, at_time
        )
        prediction = forecaster.predict(at_time, history, future, 132)
        assert (pv == 0).all()
        pd.testing.assert_series_equal(prediction["net_kw"], load, check_names=False)
    finally:
        forecaster.model.pv_inference_weights = original_weights
        forecaster.model.pv_inference_intercepts = original_intercepts


def test_precomputed_parameters_match_sklearn_for_every_lead(trained) -> None:  # noqa: ANN001
    model = trained[0].model
    rng = np.random.default_rng(42)

    for scalers, models, weights, intercepts, feature_names in (
        (
            model.load_scalers,
            model.load_models,
            model.load_inference_weights,
            model.load_inference_intercepts,
            model.LOAD_FEATURE_NAMES,
        ),
        (
            model.pv_scalers,
            model.pv_models,
            model.pv_inference_weights,
            model.pv_inference_intercepts,
            model.PV_FEATURE_NAMES,
        ),
    ):
        features = rng.normal(size=(model.MAX_HORIZON, len(feature_names)))
        expected = np.array(
            [
                ridge.predict(
                    scaler.transform(pd.DataFrame([row], columns=feature_names))
                )[0]
                for row, scaler, ridge in zip(features, scalers, models, strict=True)
            ]
        )
        actual = np.sum(features * weights, axis=1) + intercepts

        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


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
    assert state["pv"]["feature_names"] == [
        "pv_lag_15min",
        "pv_lag_1h",
        "most_recent_load_factor_forecast",
        "time_of_day_sin",
        "time_of_day_cos",
    ]
    assert state["pv"]["prediction_clip_min_kw"] == 0.0
    assert len(state["load"]["lead_models"]) == 132
    assert len(state["pv"]["lead_models"]) == 132


@pytest.mark.parametrize("decision_position", [20, 105, 10 * 96 + 48, 12 * 96])
def test_optimized_inference_matches_previous_features(
    trained, decision_position: int
) -> None:  # noqa: ANN001
    forecaster, history, _ = trained
    at_time = history.index[0] + decision_position * pd.Timedelta(minutes=15)
    future = _future(at_time)
    expected_load, expected_pv = _legacy_predict_components(
        forecaster.model, history, future, future.index, at_time
    )

    actual_load, actual_pv = forecaster.model._predict_components(
        history, future, future.index, at_time
    )

    np.testing.assert_allclose(actual_load, expected_load, atol=1e-8, rtol=0)
    np.testing.assert_allclose(actual_pv, expected_pv, atol=1e-8, rtol=0)


def test_prediction_skips_generic_history_feature_builders(
    trained, monkeypatch: pytest.MonkeyPatch
) -> None:  # noqa: ANN001
    forecaster, history, _ = trained
    at_time = history.index[-1] + pd.Timedelta(minutes=15)

    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("Generic history feature builder used during prediction")

    monkeypatch.setattr(
        DecomposedRidgeNetLoadModel,
        "_load_decision_features",
        staticmethod(fail_if_called),
    )
    monkeypatch.setattr(
        DecomposedRidgeNetLoadModel,
        "_pv_decision_features",
        staticmethod(fail_if_called),
    )

    forecaster.predict(at_time, history, _future(at_time), 132)


@pytest.fixture(scope="module")
def hgbr_trained() -> tuple[Forecaster, pd.DataFrame, SiteSpecs]:
    specs = SiteSpecs.from_yaml("site.yaml")
    history = _data()
    history.iloc[100, history.columns.get_loc("pv_production_kw")] = np.nan
    forecaster = Forecaster(specs, model_name=Forecaster.RIDGE_HGBR_DECOMPOSED)
    forecaster.fit(history)
    return forecaster, history, specs


def test_hgbr_model_selection_preserves_ridge_decomposed_option(
    trained,
) -> None:  # noqa: ANN001
    specs = trained[2]
    ridge = Forecaster(specs, model_name=Forecaster.RIDGE_DECOMPOSED)
    hgbr = Forecaster(specs, model_name=Forecaster.RIDGE_HGBR_DECOMPOSED)

    assert type(ridge.model) is DecomposedRidgeNetLoadModel
    assert type(hgbr.model) is RidgeHgbrDecomposedNetLoadModel


def test_hgbr_full_horizon_output_and_load_models(
    trained, hgbr_trained
) -> None:  # noqa: ANN001
    ridge, history, _ = trained
    hgbr, _, _ = hgbr_trained
    at_time = history.index[-1] + pd.Timedelta(minutes=15)
    future = _future(at_time)

    prediction = hgbr.predict(at_time, history, future, 132)

    assert prediction.index.equals(future.index)
    assert prediction.columns.tolist() == ["net_kw", "load_kw", "pv_kw"]
    assert len(prediction) == 132
    assert np.isfinite(prediction).all().all()
    assert (prediction["pv_kw"] >= 0).all()
    for ridge_model, hgbr_load_model in zip(
        ridge.model.load_models, hgbr.model.load_models, strict=True
    ):
        np.testing.assert_allclose(ridge_model.coef_, hgbr_load_model.coef_)
        assert ridge_model.intercept_ == pytest.approx(hgbr_load_model.intercept_)


def test_hgbr_ignores_future_history(hgbr_trained) -> None:  # noqa: ANN001
    forecaster, data, _ = hgbr_trained
    at_time = data.index[-132]
    future = _future(at_time)
    safe = data.loc[data.index < at_time]
    oversized = data.copy()
    oversized.loc[
        oversized.index >= at_time, ["grid_net_kw", "pv_production_kw"]
    ] = 1_000_000

    expected = forecaster.predict(at_time, safe, future, 132)
    actual = forecaster.predict(at_time, oversized, future, 132)

    pd.testing.assert_frame_equal(actual, expected)


def test_hgbr_clips_negative_pv_predictions(
    hgbr_trained, monkeypatch: pytest.MonkeyPatch
) -> None:  # noqa: ANN001
    forecaster, history, _ = hgbr_trained
    at_time = history.index[-1] + pd.Timedelta(minutes=15)
    future = _future(at_time)

    for model in forecaster.model.pv_models:
        monkeypatch.setattr(model, "predict", lambda features: -np.ones(len(features)))

    prediction = forecaster.predict(at_time, history, future, 132)

    assert (prediction["pv_kw"] == 0).all()
    pd.testing.assert_series_equal(
        prediction["net_kw"], prediction["load_kw"], check_names=False
    )


def test_hgbr_save_load_forecast_equivalence(hgbr_trained, tmp_path) -> None:  # noqa: ANN001
    forecaster, history, specs = hgbr_trained
    at_time = history.index[-1] + pd.Timedelta(minutes=15)
    future = _future(at_time)
    expected = forecaster.predict(at_time, history, future, 132)
    forecaster.save(tmp_path)

    loaded = Forecaster.load(tmp_path, specs)
    actual = loaded.predict(at_time, history, future, 132)

    assert type(loaded.model) is RidgeHgbrDecomposedNetLoadModel
    pd.testing.assert_frame_equal(actual, expected)
