from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import select_forecast_model as selection


TZ = "Europe/Brussels"


def _data(start: str = "2025-12-01", end: str = "2026-07-31") -> pd.DataFrame:
    index = pd.date_range(start, end, freq="15min", tz=TZ)
    return pd.DataFrame({"grid_net_kw": 1.0}, index=index)


def test_calendar_lookbacks_are_strictly_before_validation() -> None:
    data = _data()
    validation_start = pd.Timestamp("2026-04-01", tz=TZ)

    one_month = selection.training_window(data, validation_start, "1m")
    three_months = selection.training_window(data, validation_start, "3m")
    expanding = selection.training_window(data, validation_start, "expanding")

    assert one_month.index[0] == pd.Timestamp("2026-03-01", tz=TZ)
    assert three_months.index[0] == pd.Timestamp("2026-01-01", tz=TZ)
    assert expanding.index[0] == data.index[0]
    assert one_month.index[-1] == three_months.index[-1] == expanding.index[-1]
    assert expanding.index[-1] < validation_start


def test_monthly_folds_cover_each_validation_month() -> None:
    folds = selection.monthly_folds(
        pd.Timestamp("2026-04-10", tz=TZ),
        pd.Timestamp("2026-06-20", tz=TZ),
    )

    assert [fold.strftime("%Y-%m-%d") for fold in folds] == [
        "2026-04-01",
        "2026-05-01",
        "2026-06-01",
    ]


def test_horizon_metrics_pool_comparison_rows() -> None:
    comparisons = pd.DataFrame(
        {
            "lead_steps": list(range(1, 18)),
            "actual_kw": 0.0,
            "predicted_kw": np.arange(1.0, 18.0),
        }
    )

    metrics = selection.horizon_metrics(comparisons)

    assert metrics["lead_1_mae_kw"] == 1.0
    assert metrics["first_hour_mae_kw"] == 2.5
    assert metrics["first_4_hours_mae_kw"] == 8.5
    assert metrics["full_horizon_mae_kw"] == 9.0


def test_aggregate_and_selection_are_deterministic() -> None:
    folds = pd.DataFrame(
        [
            {
                "model": model,
                "lookback": lookback,
                "forecast_seconds": 2.0,
                "n_decisions": 2,
                **{column: value for column in selection.METRIC_COLUMNS},
            }
            for model, lookback, value in [
                ("ridge", "1m", 2.0),
                ("ridge", "1m", 4.0),
                ("ridge_decomposed", "3m", 1.0),
                ("ridge_decomposed", "3m", 3.0),
            ]
        ]
    )
    aggregate = selection.aggregate_results(folds)

    selected = selection.select_configuration(aggregate)

    assert len(aggregate) == 2
    assert selected["model"] == "ridge_decomposed"
    assert selected["lookback"] == "3m"
    assert selected["net_mae_kw"] == 2.0
    assert selected["forecast_seconds_per_decision"] == 1.0


def test_run_selection_passes_model_and_writes_fold_and_aggregate_rows(
    monkeypatch,
) -> None:
    index = pd.date_range(
        "2026-03-01", "2026-04-01 00:30", freq="15min", tz=TZ
    )
    data = pd.DataFrame({"grid_net_kw": 1.0}, index=index)
    seen: dict[str, object] = {}

    class FakeForecaster:
        def __init__(self, specs, model_name):  # noqa: ANN001
            seen["model_name"] = model_name

        def fit(self, history):  # noqa: ANN001
            seen["training"] = history

    def fake_collect(data, forecaster, config):  # noqa: ANN001
        seen["first_decision"] = config.first_decision
        comparisons = pd.DataFrame(
            {
                "lead_steps": [1, 2],
                "actual_kw": [1.0, 1.0],
                "predicted_kw": [2.0, 3.0],
            }
        )
        comparisons.attrs.update(forecast_seconds=0.2, n_decisions=1)
        return comparisons

    monkeypatch.setattr(selection, "Forecaster", FakeForecaster)
    monkeypatch.setattr(selection, "collect_forecasts", fake_collect)
    start = pd.Timestamp("2026-04-01", tz=TZ)

    folds, aggregate = selection.run_selection(
        data, object(), [start], ["ridge"], ["1m"], 2, 15
    )

    assert seen["model_name"] == "ridge"
    assert seen["first_decision"] == start
    assert seen["training"].index.max() < start
    assert folds[["model", "lookback", "validation_month"]].to_dict("records") == [
        {"model": "ridge", "lookback": "1m", "validation_month": "2026-04"}
    ]
    assert aggregate[["model", "lookback", "n_folds"]].to_dict("records") == [
        {"model": "ridge", "lookback": "1m", "n_folds": 1}
    ]
