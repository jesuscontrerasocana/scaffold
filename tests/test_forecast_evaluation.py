from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.harness import RunConfig
from scripts.evaluate_forecast import (
    calculate_metrics,
    collect_forecasts,
    component_metrics,
    lead_metrics,
)


def test_metric_calculation_and_lead_grouping() -> None:
    comparisons = pd.DataFrame(
        {
            "lead_steps": [1, 1, 2, 2],
            "actual_kw": [1.0, 3.0, 2.0, 4.0],
            "predicted_kw": [2.0, 5.0, 1.0, 3.0],
        }
    )

    overall = calculate_metrics(comparisons["actual_kw"], comparisons["predicted_kw"])
    per_lead = lead_metrics(comparisons)

    assert overall == {
        "mae_kw": 1.25,
        "rmse_kw": np.sqrt(1.75),
        "bias_kw": 0.25,
        "nmae": 0.5,
    }
    assert per_lead.to_dict("records") == [
        {
            "component": "net",
            "lead_step": 1,
            "mae_kw": 1.5,
            "rmse_kw": np.sqrt(2.5),
            "bias_kw": 1.5,
        },
        {
            "component": "net",
            "lead_step": 2,
            "mae_kw": 1.0,
            "rmse_kw": 1.0,
            "bias_kw": -1.0,
        },
    ]


def test_metrics_reject_missing_actuals() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        calculate_metrics(pd.Series([1.0, np.nan]), pd.Series([1.0, 2.0]))


def test_rolling_forecast_alignment_and_information_boundary() -> None:
    index = pd.date_range("2026-06-01", periods=8, freq="15min", tz="Europe/Brussels")
    data = pd.DataFrame(
        {
            "grid_net_kw": np.arange(8, dtype=float),
            "most_recent_load_factor_forecast": 0.1,
            "offtake_price_eur_per_mwh": 50.0,
            "injection_price_eur_per_mwh": 20.0,
        },
        index=index,
    )

    class RecordingForecaster:
        def predict(self, at_time, history, future_exog, horizon):  # noqa: ANN001
            assert (history.index < at_time).all()
            assert list(future_exog.columns) == [
                "most_recent_load_factor_forecast",
                "offtake_price_eur_per_mwh",
                "injection_price_eur_per_mwh",
            ]
            assert len(future_exog) == horizon == 3
            return pd.DataFrame({"net_kw": data.loc[future_exog.index, "grid_net_kw"]})

    config = RunConfig(
        first_decision=index[2],
        last_decision=index[2],
        horizon_steps=3,
    )
    comparisons = collect_forecasts(data, RecordingForecaster(), config)

    assert comparisons["target_time"].tolist() == index[2:5].tolist()
    assert comparisons["lead_steps"].tolist() == [1, 2, 3]
    assert comparisons["actual_kw"].tolist() == [2.0, 3.0, 4.0]
    assert comparisons["predicted_kw"].tolist() == [2.0, 3.0, 4.0]
    assert "forecast_load_kw" not in comparisons
    assert "forecast_pv_kw" not in comparisons


def test_component_forecasts_are_included_in_comparisons() -> None:
    index = pd.date_range("2026-06-01", periods=6, freq="15min", tz="Europe/Brussels")
    data = pd.DataFrame(
        {
            "grid_net_kw": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "pv_production_kw": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "most_recent_load_factor_forecast": 0.1,
            "offtake_price_eur_per_mwh": 50.0,
            "injection_price_eur_per_mwh": 20.0,
        },
        index=index,
    )

    class ComponentForecaster:
        def predict(self, at_time, history, future_exog, horizon):  # noqa: ANN001, ARG002
            target = future_exog.index[:horizon]
            return pd.DataFrame(
                {"net_kw": 4.0, "load_kw": 7.0, "pv_kw": 3.0}, index=target
            )

    comparisons = collect_forecasts(
        data,
        ComponentForecaster(),
        RunConfig(first_decision=index[2], last_decision=index[2], horizon_steps=3),
    )

    assert comparisons["forecast_load_kw"].tolist() == [7.0, 7.0, 7.0]
    assert comparisons["forecast_pv_kw"].tolist() == [3.0, 3.0, 3.0]
    assert comparisons["actual_load_kw"].tolist() == [10.0, 12.0, 14.0]
    assert comparisons["actual_pv_kw"].tolist() == [3.0, 4.0, 5.0]
    assert comparisons["load_error_kw"].tolist() == [-3.0, -5.0, -7.0]
    assert comparisons["pv_error_kw"].tolist() == [0.0, -1.0, -2.0]

    metrics = component_metrics(comparisons)
    per_lead = lead_metrics(comparisons)
    assert list(metrics) == ["net", "load", "pv"]
    assert set(per_lead["component"]) == {"net", "load", "pv"}


def test_missing_actual_pv_is_excluded_from_component_metrics() -> None:
    comparisons = pd.DataFrame(
        {
            "lead_steps": [1, 1, 2],
            "actual_kw": [1.0, 2.0, 3.0],
            "predicted_kw": [2.0, 3.0, 4.0],
            "actual_load_kw": [5.0, np.nan, 7.0],
            "forecast_load_kw": [6.0, 100.0, 8.0],
            "actual_pv_kw": [4.0, np.nan, 4.0],
            "forecast_pv_kw": [4.0, 100.0, 5.0],
        }
    )

    metrics = component_metrics(comparisons)
    per_lead = lead_metrics(comparisons)

    assert metrics["net"]["mae_kw"] == 1.0
    assert metrics["load"]["mae_kw"] == 1.0
    assert metrics["pv"]["mae_kw"] == 0.5
    assert len(per_lead[per_lead["component"] == "net"]) == 2
    assert len(per_lead[per_lead["component"] == "load"]) == 2
    assert len(per_lead[per_lead["component"] == "pv"]) == 2


def test_net_only_comparisons_do_not_gain_component_columns() -> None:
    comparisons = pd.DataFrame(
        {"lead_steps": [1], "actual_kw": [1.0], "predicted_kw": [2.0]}
    )

    assert list(component_metrics(comparisons)) == ["net"]
    assert lead_metrics(comparisons)["component"].tolist() == ["net"]
