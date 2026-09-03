from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.report import calculate_dashboard_metrics, main, write_results


def _artifacts() -> tuple[pd.DataFrame, dict[str, object]]:
    index = pd.date_range("2026-06-01", periods=4, freq="15min", tz="UTC")
    schedule = pd.DataFrame({
        "forecast_net_kw": [10.0] * 4,
        "realized_net_no_bess_kw": [10.0, -4.0, 20.0, 0.0],
        "grid_net_with_bess_kw": [15.0, -2.0, 10.0, 0.0],
        "applied_charge_kw": [5.0, 2.0, 0.0, 0.0],
        "applied_discharge_kw": [0.0, 0.0, 10.0, 0.0],
        "soc": [0.5, 0.51, 0.49, 0.49],
        "offtake_price_eur_per_mwh": [100.0] * 4,
        "injection_price_eur_per_mwh": [50.0] * 4,
    }, index=index)
    summary: dict[str, object] = {
        "energy_cost_eur": 0.6, "peak_cost_eur": 30.0,
        "cycle_cost_eur": 2.0, "total_cost_eur": 32.6,
        "equivalent_cycles": 0.25,
        "peak_cost_detail": {"2026-06": {
            "peak_offtake_kw": 15.0, "month_share": 1.0,
            "peak_cost_eur": 30.0,
        }},
    }
    return schedule, summary


def test_dashboard_bill_arithmetic_reconciles() -> None:
    schedule, summary = _artifacts()
    metrics, _ = calculate_dashboard_metrics(schedule, summary)

    assert metrics["bill_without_bess_eur"] == pytest.approx(40.7)
    assert metrics["bill_with_bess_eur"] == pytest.approx(32.6)
    assert metrics["total_savings_eur"] == pytest.approx(8.1)
    assert (
        metrics["energy_cost_value_eur"]
        + metrics["peak_cost_value_eur"]
        + metrics["cycling_cost_value_eur"]
    ) == pytest.approx(metrics["total_savings_eur"])
    assert metrics["total_load_mwh"] is None
    assert metrics["pv_production_mwh"] is None


def test_write_results_and_standalone_create_dashboard(tmp_path) -> None:
    schedule, summary = _artifacts()
    schedule.attrs["forecast_log"] = pd.DataFrame(
        {"decision_time": schedule.index}, index=schedule.index
    )
    out = tmp_path / "simulation.csv"
    write_results(schedule, summary, out)
    assert (tmp_path / "simulation_dashboard.png").stat().st_size > 0

    regenerated = tmp_path / "regenerated.csv"
    assert main(["--schedule", str(out), "--summary",
        str(tmp_path / "simulation_summary.json"), "--log",
        str(tmp_path / "simulation_forecasts.csv"), "--out", str(regenerated)]) == 0
    assert (tmp_path / "regenerated_dashboard.png").exists()
    assert json.loads((tmp_path / "regenerated_summary.json").read_text()) == summary
