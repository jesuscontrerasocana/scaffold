from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from pipeline.data import HOURS_PER_STEP
from pipeline.harness import DecisionContext, MonthState
from pipeline.optimizer import Optimizer
from pipeline.specs import SiteSpecs


@pytest.fixture
def specs() -> SiteSpecs:
    return SiteSpecs.from_yaml("site.yaml")


def _inputs(
    net_kw: list[float],
    offtake: list[float],
    injection: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2026-01-05", periods=len(net_kw), freq="15min", tz="UTC")
    forecast = pd.DataFrame({"net_kw": net_kw}, index=index)
    prices = pd.DataFrame(
        {
            "offtake_price_eur_per_mwh": offtake,
            "injection_price_eur_per_mwh": injection or [0.0] * len(net_kw),
        },
        index=index,
    )
    return forecast, prices


def _context(index: pd.DatetimeIndex, initial_soc: float) -> DecisionContext:
    return DecisionContext(
        at_time=index[0],
        initial_soc=initial_soc,
        month=MonthState(peak_offtake_kw=0.0, steps_elapsed=0, steps_total=96 * 31),
        history=pd.DataFrame(index=pd.DatetimeIndex([], tz=index.tz)),
        prices_known_until=index[-1],
        state={},
    )


def _energy(schedule: pd.DataFrame, specs: SiteSpecs, initial_soc: float) -> np.ndarray:
    eta = specs.battery.one_way_efficiency
    changes = HOURS_PER_STEP * (
        eta * schedule["battery_charge_kw"]
        - schedule["battery_discharge_kw"] / eta
    )
    return initial_soc * specs.battery.capacity_kwh + changes.cumsum().to_numpy()


def test_feasible_dispatch_respects_physics_and_balance(specs: SiteSpecs) -> None:
    forecast, prices = _inputs([50.0] * 4, [60.0] * 4)
    optimizer = Optimizer(specs)
    schedule = optimizer.solve(forecast, prices, _context(forecast.index, 0.5))
    energy = _energy(schedule, specs, 0.5)
    grid_net = forecast["net_kw"] + schedule["battery_charge_kw"] - schedule[
        "battery_discharge_kw"
    ]

    assert schedule.index.equals(forecast.index)
    assert (schedule >= -1e-7).all().all()
    assert schedule["battery_charge_kw"].max() <= specs.battery.charge_power_kw + 1e-7
    assert schedule["battery_discharge_kw"].max() <= specs.battery.discharge_power_kw + 1e-7
    assert energy.min() >= specs.battery.capacity_kwh * specs.battery.min_soc - 1e-7
    assert energy.max() <= specs.battery.capacity_kwh * specs.battery.max_soc + 1e-7
    assert grid_net.max() <= specs.offtake_limit_kw + 1e-7
    assert grid_net.min() >= -specs.injection_limit_kw - 1e-7
    assert optimizer.last_summary["solver_termination"] == "optimal"


def test_profitable_price_arbitrage_charges_then_discharges(specs: SiteSpecs) -> None:
    forecast, prices = _inputs([0.0, 80.0], [0.0, 300.0])
    schedule = Optimizer(specs).solve(
        forecast, prices, _context(forecast.index, specs.battery.min_soc)
    )

    assert schedule.iloc[0]["battery_charge_kw"] > 0
    assert schedule.iloc[1]["battery_discharge_kw"] > 0


def test_negative_prices_have_correct_signs(specs: SiteSpecs) -> None:
    import_forecast, import_prices = _inputs([0.0], [-100.0], [-200.0])
    import_schedule = Optimizer(specs).solve(
        import_forecast,
        import_prices,
        _context(import_forecast.index, specs.battery.min_soc),
    )
    export_forecast, export_prices = _inputs([-50.0], [0.0], [-100.0])
    export_schedule = Optimizer(specs).solve(
        export_forecast,
        export_prices,
        _context(export_forecast.index, 0.5),
    )

    assert import_schedule.iloc[0]["battery_charge_kw"] > 0
    assert export_schedule.iloc[0]["battery_charge_kw"] > 0


def test_grid_limit_forces_discharge(specs: SiteSpecs) -> None:
    forecast, prices = _inputs([150.0], [50.0])
    schedule = Optimizer(specs).solve(
        forecast, prices, _context(forecast.index, 0.5)
    )

    assert schedule.iloc[0]["battery_discharge_kw"] >= 50.0 - 1e-7


def test_small_spread_does_not_cover_losses_and_degradation(specs: SiteSpecs) -> None:
    forecast, prices = _inputs([0.0, 50.0], [50.0, 51.0])
    schedule = Optimizer(specs).solve(
        forecast, prices, _context(forecast.index, specs.battery.min_soc)
    )

    assert schedule.to_numpy().max() == pytest.approx(0.0, abs=1e-7)


def test_unpublished_prices_are_not_invented(specs: SiteSpecs) -> None:
    forecast, prices = _inputs([0.0, 0.0], [0.0, np.nan])
    prices.iloc[1, prices.columns.get_loc("injection_price_eur_per_mwh")] = np.nan
    optimizer = Optimizer(specs)
    schedule = optimizer.solve(
        forecast, prices, _context(forecast.index, specs.battery.min_soc)
    )

    assert schedule.to_numpy().max() == pytest.approx(0.0, abs=1e-7)
    assert optimizer.last_summary["published_price_steps"] == 1


def test_model_and_mutable_inputs_are_reused_and_updated(specs: SiteSpecs) -> None:
    optimizer = Optimizer(specs)
    model = optimizer.model
    first_forecast, first_prices = _inputs(
        [10.0, 20.0], [30.0, np.nan], [5.0, np.nan]
    )
    optimizer.solve(
        first_forecast,
        first_prices,
        _context(first_forecast.index, 0.4),
    )

    assert optimizer.model is model
    assert pyo.value(model.net_kw[0]) == 10.0
    assert pyo.value(model.offtake_price[0]) == 30.0
    assert pyo.value(model.injection_price[0]) == 5.0
    assert pyo.value(model.offtake_price[1]) == 0.0
    assert pyo.value(model.injection_price[1]) == 0.0
    assert pyo.value(model.initial_energy) == pytest.approx(
        specs.battery.capacity_kwh * 0.4
    )

    second_forecast, second_prices = _inputs([70.0], [90.0], [15.0])
    optimizer.solve(
        second_forecast,
        second_prices,
        _context(second_forecast.index, 0.6),
    )

    assert optimizer.model is model
    assert pyo.value(model.net_kw[0]) == 70.0
    assert pyo.value(model.offtake_price[0]) == 90.0
    assert pyo.value(model.injection_price[0]) == 15.0
    assert pyo.value(model.initial_energy) == pytest.approx(
        specs.battery.capacity_kwh * 0.6
    )
    assert pyo.value(model.net_kw[1]) == 0.0
    assert pyo.value(model.offtake_price[1]) == 0.0
    assert pyo.value(model.injection_price[1]) == 0.0


def test_infeasible_forecast_fails_clearly(specs: SiteSpecs) -> None:
    weak_battery = dataclasses.replace(specs.battery, discharge_power_kw=10.0)
    limited_specs = dataclasses.replace(specs, battery=weak_battery)
    forecast, prices = _inputs([200.0], [50.0])

    with pytest.raises(RuntimeError, match="infeasible"):
        Optimizer(limited_specs).solve(
            forecast, prices, _context(forecast.index, weak_battery.max_soc)
        )
