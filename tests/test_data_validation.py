from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from pipeline.forecaster import Forecaster, validate_and_clean_history
from pipeline.specs import SiteSpecs


def _history() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=4, freq="15min", tz="Europe/Brussels")
    return pd.DataFrame(
        {
            "grid_net_kw": [10.0, 0.0, 30.0, 40.0],
            "pv_production_kw": [1.0, np.nan, 3.0, 4.0],
            "offtake_price_eur_per_mwh": [50.0, -500.0, 70.0, 1000.0],
            "injection_price_eur_per_mwh": [20.0, -600.0, 30.0, 900.0],
            "most_recent_load_factor_forecast": [0.1, 0.2, 0.3, 0.4],
        },
        index=index,
    )


def test_missingness_is_reported_without_changing_values_or_dropping_rows() -> None:
    history = _history()

    cleaned, summary = validate_and_clean_history(history)

    pd.testing.assert_frame_equal(cleaned, history)
    assert len(cleaned) == len(history)
    missing = summary["missing_by_column"]
    assert set(missing) == set(history.columns)
    assert missing["pv_production_kw"] == {"count": 1, "percentage": 25.0}
    for column in history.columns.drop("pv_production_kw"):
        assert missing[column] == {"count": 0, "percentage": 0.0}


def test_fit_logs_summary_and_uses_only_grid_missingness_for_fallback(
    caplog: pytest.LogCaptureFixture, tmp_path,  # noqa: ANN001
) -> None:
    specs = SiteSpecs.from_yaml("site.yaml")
    forecaster = Forecaster(specs)

    with caplog.at_level(logging.WARNING):
        forecaster.fit(_history())

    assert forecaster.fallback_kw == 20.0
    forecaster.save(tmp_path)
    assert Forecaster.load(tmp_path, specs).fallback_kw == 20.0
