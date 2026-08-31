"""End to end checks of the runner, the accounting and the planner.

These pass out of the box, against the baseline forecaster and optimizer that ship in
`pipeline/`. Once you have replaced those, they should still pass — if they stop passing,
your code is producing something the runner cannot use, and we will hit the same wall.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipeline.data import REQUIRED_COLUMNS, load_timeseries
from pipeline.forecaster import Forecaster
from pipeline.harness import RunConfig, prices_known_until, run_backtest
from pipeline.optimizer import Optimizer
from pipeline.report import write_results
from pipeline.specs import SiteSpecs
from tests.dummies import FullPowerOptimizer, IdleOptimizer


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def specs() -> SiteSpecs:
    return SiteSpecs.from_yaml(ROOT / "site.yaml")


@pytest.fixture(scope="module")
def history(specs: SiteSpecs) -> pd.DataFrame:
    df = load_timeseries(ROOT / "data" / "history.csv", timezone=specs.timezone)
    return df.iloc[: 96 * 12]  # keep the tests quick


@pytest.fixture(scope="module")
def fitted(specs: SiteSpecs, history: pd.DataFrame) -> Forecaster:
    forecaster = Forecaster(specs)
    forecaster.fit(history)
    return forecaster


def test_data_loads_on_a_regular_quarter_hourly_grid(history: pd.DataFrame) -> None:
    deltas = history.index.to_series().diff().dropna().unique()
    assert len(deltas) == 1, "index is not a regular 15 minute grid"
    assert history.index.tz is not None, "index must be timezone aware"
    assert set(REQUIRED_COLUMNS) <= set(history.columns)


def test_idle_battery_costs_exactly_the_site_bill(history, specs, fitted) -> None:  # noqa: ANN001
    schedule, summary = run_backtest(
        history, specs, fitted, IdleOptimizer(specs), RunConfig()
    )
    assert len(schedule) > 0
    assert summary["battery_throughput_kwh"] == pytest.approx(0.0)
    pd.testing.assert_series_equal(
        schedule["grid_net_with_bess_kw"],
        schedule["realized_net_no_bess_kw"],
        check_names=False,
    )
    assert summary["violations"]["offtake_limit_exceeded_steps"] == 0


def test_the_runner_clips_what_the_battery_cannot_deliver(
    history, specs, fitted
) -> None:  # noqa: ANN001
    schedule, summary = run_backtest(
        history, specs, fitted, FullPowerOptimizer(specs), RunConfig()
    )
    assert (schedule["applied_charge_kw"] <= specs.battery.charge_power_kw + 1e-6).all()
    assert summary["violations"]["clipped_by_power_limit_steps"] > 0
    assert summary["violations"]["clipped_by_soc_limit_steps"] > 0
    assert (schedule["soc"] <= specs.battery.max_soc + 1e-6).all()
    assert (schedule["soc"] >= specs.battery.min_soc - 1e-6).all()


def test_simulate_runs_with_the_shipped_baselines(
    history, specs, fitted, tmp_path
) -> None:  # noqa: ANN001
    schedule, summary = run_backtest(
        history, specs, fitted, Optimizer(specs), RunConfig()
    )
    assert summary["total_cost_eur"] > 0
    assert set(summary["violations"]) >= {"simultaneous_charge_discharge_steps"}
    out = tmp_path / "simulation.csv"
    write_results(schedule, summary, out)
    assert out.exists()
    assert out.with_name("simulation_summary.json").exists()
    assert len(pd.read_csv(out)) == len(schedule)


def test_the_scored_window_starts_where_we_say_it_does(history, specs, fitted) -> None:  # noqa: ANN001
    """Everything before --from is history the forecaster may read, and is not scored."""
    first = history.index[0] + pd.Timedelta(days=9)
    last = first + pd.Timedelta(days=2)
    schedule, summary = run_backtest(
        history,
        specs,
        fitted,
        Optimizer(specs),
        RunConfig(first_decision=first, last_decision=last),
    )
    assert schedule.index[0] == first
    assert schedule.index[-1] <= last + pd.Timedelta(minutes=15)
    assert summary["n_decisions"] == len(schedule)  # one decision per quarter hour


def test_a_month_is_billed_one_full_capacity_charge(history, specs, fitted) -> None:  # noqa: ANN001
    """The capacity charge is prorated by how much of the calendar month we covered."""
    schedule, summary = run_backtest(
        history, specs, fitted, IdleOptimizer(specs), RunConfig()
    )
    detail = summary["peak_cost_detail"]
    assert detail, "every run should bill at least one month"
    for month in detail.values():
        assert 0 < month["month_share"] <= 1.0
        assert month["peak_cost_eur"] == pytest.approx(
            month["peak_offtake_kw"]
            * specs.offtake_monthly_peak_cost_eur_per_kw
            * month["month_share"],
            rel=1e-3,
        )


def test_the_horizon_no_longer_shrinks_to_the_priced_window(
    history, specs, fitted
) -> None:  # noqa: ANN001
    """The decision horizon is the full look-ahead; only the priced part of it breathes."""
    first = history.index[0] + pd.Timedelta(days=9)
    schedule, summary = run_backtest(
        history,
        specs,
        fitted,
        Optimizer(specs),
        RunConfig(first_decision=first, last_decision=first + pd.Timedelta(days=2)),
    )
    # the published prices breathe: nine and a bit hours at 14:45, thirty-three at 15:00
    assert summary["priced_steps_min"] == 37, summary["priced_steps_min"]
    assert summary["priced_steps_max"] == 132, summary["priced_steps_max"]
    # but the horizon we hand the optimizer no longer collapses to that priced edge
    assert summary["horizon_steps_max"] == 132
    assert summary["horizon_steps_min"] > summary["priced_steps_min"]
    assert len(schedule) > 0


def test_prices_past_the_day_ahead_edge_are_masked(history, specs, fitted) -> None:  # noqa: ANN001
    """A candidate may forecast tomorrow's unpublished prices but must never be handed them."""
    seen: list[tuple[pd.Timestamp, pd.DataFrame]] = []

    class RecordingOptimizer:
        def __init__(self, specs: SiteSpecs) -> None:
            self.specs = specs

        def solve(self, forecast, prices, context):  # noqa: ANN001
            seen.append((context.prices_known_until, prices))
            return pd.DataFrame(
                {"battery_charge_kw": 0.0, "battery_discharge_kw": 0.0},
                index=forecast.index,
            )

    first = history.index[0] + pd.Timedelta(days=9)
    run_backtest(
        history,
        specs,
        fitted,
        RecordingOptimizer(specs),
        RunConfig(first_decision=first, last_decision=first + pd.Timedelta(days=2)),
    )
    assert seen
    for known_until, prices in seen:
        published = prices.index <= known_until
        assert published.any(), "the committed step is always priced"
        assert prices.loc[published].notna().all().all(), (
            "published prices must be real"
        )
        assert prices.loc[~published].isna().all().all(), (
            "unpublished prices must be masked"
        )
    assert any((prices.index > known_until).any() for known_until, prices in seen), (
        "some decisions must have an unpriced tail, or the mask is never exercised"
    )


def test_prices_known_until_follows_the_auction(specs: SiteSpecs) -> None:
    tz = specs.timezone
    before = pd.Timestamp("2025-11-05 14:45", tz=tz)
    after = pd.Timestamp("2025-11-05 15:00", tz=tz)
    assert prices_known_until(before) == pd.Timestamp("2025-11-05 23:45", tz=tz)
    assert prices_known_until(after) == pd.Timestamp("2025-11-06 23:45", tz=tz)
