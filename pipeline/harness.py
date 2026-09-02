"""The rolling decision loop, cost accounting and physical checks.

PROVIDED BY US. Do not modify this file: we run your submission with our copy of it.

What it does, once per decision time:

  1. slices the history strictly BEFORE `at_time` and hands it to your forecaster,
  2. hands your forecaster the exogenous inputs over the whole horizon, with the day-ahead
     prices masked past the point they are actually published,
  3. hands your forecast and those (masked) prices to your optimizer, together with the
     current state of charge and what has already happened this calendar month,
  4. records the whole forecast for our own analysis, then applies the first
     `decision_interval` worth of your schedule against what the site
     ACTUALLY did, clipping anything the battery physically cannot deliver and any
     charge or discharge that would drive the meter past the grid connection limits,
  5. moves the state of charge accordingly and bills the resulting grid flows.

You cannot leak the future through this loop: the slicing is done here, not by you.
So do not spend any of your time engineering around leakage.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from pipeline.data import HOURS_PER_STEP, STEP
from pipeline.specs import SiteSpecs


TOL = 1e-6

# Columns your optimizer must return, both non-negative, in kW.
CHARGE_COL = "battery_charge_kw"
DISCHARGE_COL = "battery_discharge_kw"

# Column your forecaster must return: forecast of grid_net_kw.
FORECAST_COL = "net_kw"


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """How the rolling loop is run.

    The controller re-decides every quarter hour, which is what the real one does. Each
    decision sees the meter readings up to that moment, so the forecast is rebased on what
    the site has actually just been doing, and only the next quarter hour is committed.
    """

    decision_interval_minutes: int = 15
    horizon_steps: int = (
        132  # the most we ever look ahead: 33 hours, the widest priced view
    )
    warmup_days: int = 8  # history kept aside when no explicit window is given
    first_decision: pd.Timestamp | None = None
    last_decision: pd.Timestamp | None = None
    price_publication_hour: int = 15  # when tomorrow's day-ahead prices are published

    @property
    def steps_per_decision(self) -> int:
        return self.decision_interval_minutes // 15


@dataclasses.dataclass(frozen=True)
class MonthState:
    """What has already happened in the calendar month containing `at_time`."""

    peak_offtake_kw: float  # highest quarter-hourly offtake billed so far this month
    steps_elapsed: int  # quarter hours of this month already simulated
    steps_total: int  # quarter hours in the full calendar month


@dataclasses.dataclass
class DecisionContext:
    """Everything your optimizer knows at a decision time, besides forecast and prices."""

    at_time: pd.Timestamp
    initial_soc: float
    month: MonthState
    history: (
        pd.DataFrame
    )  # every row strictly before at_time, same frame the forecaster got
    prices_known_until: (
        pd.Timestamp
    )  # last horizon step with a real price; prices are NaN after it
    state: dict[
        str, Any
    ]  # yours to use; carried unchanged from one decision to the next


class Forecaster(Protocol):
    def fit(self, history: pd.DataFrame) -> None: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path, specs: SiteSpecs) -> "Forecaster": ...
    def predict(
        self,
        at_time: pd.Timestamp,
        history: pd.DataFrame,
        future_exog: pd.DataFrame,
        horizon: int,
    ) -> pd.DataFrame: ...


class Optimizer(Protocol):
    def solve(
        self, forecast: pd.DataFrame, prices: pd.DataFrame, context: DecisionContext
    ) -> pd.DataFrame: ...


def prices_known_until(
    at_time: pd.Timestamp, publication_hour: int = 15
) -> pd.Timestamp:
    """Last quarter hour whose price is already published at `at_time`.

    Electricity is bought a day ahead. The auction for every quarter hour of tomorrow clears
    and is published at `publication_hour` today; until then you only know today's prices.

    So the priced horizon is not a fixed 24 hours — it breathes. Just before publication you
    can only see to the end of today, and at 14:45 that is nine hours. Just after, you can
    see to the end of tomorrow, which is thirty-three. A controller has to be useful at both
    ends of that.
    """
    last_day = at_time.normalize()
    if at_time.hour >= publication_hour:
        last_day = (last_day + pd.DateOffset(days=1)).normalize()
    return last_day + pd.Timedelta(hours=23, minutes=45)


def decision_times(index: pd.DatetimeIndex, config: RunConfig) -> pd.DatetimeIndex:
    """Times at which we re-forecast and re-optimize.

    Everything in the file before the first decision is history: the forecaster may use it,
    but nothing there is scored.
    """
    if config.first_decision is not None:
        first = config.first_decision
        if first < index[0]:
            raise ValueError(
                f"the run starts at {first} but the data starts at {index[0]}"
            )
    else:
        first = (index[0] + pd.Timedelta(days=config.warmup_days)).ceil(
            f"{config.decision_interval_minutes}min"
        )
    last = index[-1] - STEP * (config.steps_per_decision - 1)
    if config.last_decision is not None:
        last = min(last, config.last_decision)
    if first > last:
        raise ValueError(
            f"not enough data: need more than {config.warmup_days} days of warm-up plus "
            f"one decision interval, got {index[0]} .. {index[-1]}"
        )
    return pd.date_range(
        first, last, freq=f"{config.decision_interval_minutes}min", tz=index.tz
    )


def quarter_hours_in_month(month_start: pd.Timestamp) -> int:
    """Quarter hours in the calendar month containing `month_start`, in local time.

    Not `days_in_month * 96`: on a clock change a local month has 2,976 +/- 4 quarter hours,
    and the capacity charge is prorated by this number.
    """
    next_month = (month_start + pd.Timedelta(days=32)).normalize().replace(day=1)
    return len(
        pd.date_range(
            month_start, next_month, freq=STEP, tz=month_start.tz, inclusive="left"
        )
    )


def _month_state(realized: pd.DataFrame, at_time: pd.Timestamp) -> MonthState:
    month_start = at_time.normalize().replace(day=1)
    same_month = (
        realized.loc[realized.index >= month_start] if len(realized) else realized
    )
    peak = float(same_month["offtake_kw"].max()) if len(same_month) else 0.0
    return MonthState(
        peak_offtake_kw=0.0 if np.isnan(peak) else peak,
        steps_elapsed=len(same_month),
        steps_total=quarter_hours_in_month(month_start),
    )


def _validate_schedule(
    schedule: pd.DataFrame, expected_index: pd.DatetimeIndex
) -> pd.DataFrame:
    for col in (CHARGE_COL, DISCHARGE_COL):
        if col not in schedule.columns:
            raise ValueError(
                f"optimizer must return a '{col}' column, got {list(schedule.columns)}"
            )
    out = schedule.reindex(expected_index)
    if out[[CHARGE_COL, DISCHARGE_COL]].isna().any().any():
        raise ValueError(
            "optimizer returned a schedule that does not cover the whole horizon "
            f"({expected_index[0]} .. {expected_index[-1]})"
        )
    return out[[CHARGE_COL, DISCHARGE_COL]].astype(float)


def run_backtest(  # noqa: PLR0914
    df: pd.DataFrame,
    specs: SiteSpecs,
    forecaster: Forecaster,
    optimizer: Optimizer,
    config: RunConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the rolling loop over `df` and return (per-quarter-hour schedule, summary)."""
    config = config or RunConfig()
    battery = specs.battery

    min_energy = battery.capacity_kwh * battery.min_soc
    max_energy = battery.capacity_kwh * battery.max_soc
    eff = battery.one_way_efficiency

    energy = battery.capacity_kwh * battery.initial_soc
    state: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    forecast_log: list[pd.DataFrame] = []
    realized = pd.DataFrame(columns=["offtake_kw"], dtype=float)

    exog_cols = ["most_recent_load_factor_forecast"]
    price_cols = ["offtake_price_eur_per_mwh", "injection_price_eur_per_mwh"]

    times = decision_times(df.index, config)
    solve_seconds = 0.0
    forecast_seconds = 0.0
    priced_counts: list[int] = []

    for at_time in times:
        history = df.loc[df.index < at_time]
        # The horizon is the widest view we allow, capped only by the data we actually hold --
        # never by where the day-ahead prices happen to stop. How far to plan, and what to do
        # past the published price edge, is the candidate's decision, not ours.
        horizon_index = pd.date_range(
            at_time, periods=config.horizon_steps, freq=STEP, tz=df.index.tz
        )
        horizon_index = horizon_index[horizon_index <= df.index[-1]]
        if len(horizon_index) == 0:
            break

        # Day-ahead prices past publication are not knowable yet. We hand the whole horizon
        # over but mask those prices to NaN. `priced_until` is the last quarter hour that carries a real one.
        priced_until = prices_known_until(at_time, config.price_publication_hour)
        published = horizon_index <= priced_until
        priced_counts.append(int(published.sum()))
        prices = df.loc[horizon_index, price_cols].copy()
        prices.loc[~published] = np.nan

        future_exog = df.loc[horizon_index, exog_cols].copy()
        future_exog[price_cols] = (
            prices  # published prices ride along, masked the same way
        )

        t0 = time.perf_counter()
        forecast = forecaster.predict(
            at_time=at_time,
            history=history,
            future_exog=future_exog,
            horizon=len(horizon_index),
        )
        forecast_seconds += time.perf_counter() - t0
        if FORECAST_COL not in forecast.columns:
            raise ValueError(
                f"forecaster must return a '{FORECAST_COL}' column, got {list(forecast.columns)}"
            )
        forecast = forecast.reindex(horizon_index)
        if forecast[FORECAST_COL].isna().any():
            raise ValueError("forecaster returned NaN or an incomplete horizon")

        logged = forecast.copy()
        logged.index.name = "datetime"
        logged.insert(0, "decision_time", at_time)
        forecast_log.append(logged)

        context = DecisionContext(
            at_time=at_time,
            initial_soc=energy / battery.capacity_kwh,
            month=_month_state(realized, at_time),
            history=history,
            prices_known_until=priced_until,
            state=state,
        )

        t0 = time.perf_counter()
        schedule = optimizer.solve(forecast=forecast, prices=prices, context=context)
        solve_seconds += time.perf_counter() - t0
        schedule = _validate_schedule(schedule, horizon_index)

        applied = horizon_index[: config.steps_per_decision]
        for ts in applied:
            if ts.minute == 0 and ts.hour ==0:
                print(ts)
            planned_charge = max(float(schedule.at[ts, CHARGE_COL]), 0.0)
            planned_discharge = max(float(schedule.at[ts, DISCHARGE_COL]), 0.0)
            simultaneous = planned_charge > TOL and planned_discharge > TOL

            # --- clip to what the asset can physically do -----------------------------
            charge = min(planned_charge, battery.charge_power_kw)
            discharge = min(planned_discharge, battery.discharge_power_kw)
            clipped_power = (
                charge < planned_charge - TOL or discharge < planned_discharge - TOL
            )

            headroom_kwh = max_energy - energy
            available_kwh = energy - min_energy
            max_charge = headroom_kwh / (eff * HOURS_PER_STEP)
            max_discharge = available_kwh * eff / HOURS_PER_STEP
            clipped_soc = charge > max_charge + TOL or discharge > max_discharge + TOL
            charge = min(charge, max(max_charge, 0.0))
            discharge = min(discharge, max(max_discharge, 0.0))

            # --- clip so the battery cannot push the meter past the grid connection ---
            # Charging raises offtake, discharging raises injection. The connection cannot
            # carry more than its contractual limits, so the site curtails the battery to
            # respect them. It cannot curtail the site's own load, so a limit that the load
            # alone already breaches is still recorded as exceeded below.
            net_no_bess = float(df.at[ts, "grid_net_kw"])
            max_charge_grid = max(specs.offtake_limit_kw - net_no_bess + discharge, 0.0)
            new_charge = min(charge, max_charge_grid)
            max_discharge_grid = max(
                specs.injection_limit_kw + net_no_bess + new_charge, 0.0
            )
            new_discharge = min(discharge, max_discharge_grid)
            clipped_grid = new_charge < charge - TOL or new_discharge < discharge - TOL
            charge, discharge = new_charge, new_discharge

            # --- physics --------------------------------------------------------------
            energy = energy + (eff * charge - discharge / eff) * HOURS_PER_STEP
            energy = min(max(energy, min_energy), max_energy)

            grid_net = net_no_bess + charge - discharge
            offtake = max(grid_net, 0.0)
            injection = max(-grid_net, 0.0)

            over_offtake = max(offtake - specs.offtake_limit_kw, 0.0)
            over_injection = max(injection - specs.injection_limit_kw, 0.0)

            rows.append(
                {
                    "datetime": ts,
                    "decision_time": at_time,
                    "forecast_net_kw": float(forecast.at[ts, FORECAST_COL]),
                    "realized_net_no_bess_kw": net_no_bess,
                    "planned_charge_kw": planned_charge,
                    "planned_discharge_kw": planned_discharge,
                    "applied_charge_kw": charge,
                    "applied_discharge_kw": discharge,
                    "soc": energy / battery.capacity_kwh,
                    "grid_net_with_bess_kw": grid_net,
                    "offtake_kw": offtake,
                    "injection_kw": injection,
                    "offtake_price_eur_per_mwh": float(
                        df.at[ts, "offtake_price_eur_per_mwh"]
                    ),
                    "injection_price_eur_per_mwh": float(
                        df.at[ts, "injection_price_eur_per_mwh"]
                    ),
                    "simultaneous_charge_discharge": simultaneous,
                    "clipped_by_power_limit": clipped_power,
                    "clipped_by_soc_limit": clipped_soc,
                    "clipped_by_grid_limit": clipped_grid,
                    "offtake_limit_exceeded_kw": over_offtake,
                    "injection_limit_exceeded_kw": over_injection,
                }
            )

        realized = pd.DataFrame(rows).set_index("datetime")[["offtake_kw"]]

    schedule_df = pd.DataFrame(rows).set_index("datetime")
    schedule_df.attrs["forecast_log"] = (
        pd.concat(forecast_log) if forecast_log else pd.DataFrame()
    )
    summary = summarise(schedule_df, specs)
    summary["n_decisions"] = int(len(times))
    summary["horizon_steps_min"] = (
        int(min(len(f) for f in forecast_log)) if forecast_log else 0
    )
    summary["horizon_steps_max"] = (
        int(max(len(f) for f in forecast_log)) if forecast_log else 0
    )
    summary["priced_steps_min"] = int(min(priced_counts)) if priced_counts else 0
    summary["priced_steps_max"] = int(max(priced_counts)) if priced_counts else 0
    summary["forecast_seconds_total"] = round(forecast_seconds, 2)
    summary["solve_seconds_total"] = round(solve_seconds, 2)
    summary["solve_seconds_per_decision"] = round(solve_seconds / max(len(times), 1), 3)
    return schedule_df, summary


def monthly_peak_cost(
    offtake_kw: pd.Series, rate_eur_per_kw_month: float
) -> tuple[float, dict[str, dict[str, float]]]:
    """Capacity charge, prorated by how much of each calendar month we actually simulated.

    For every calendar month touched by the run:

        cost = max(offtake in that month) * rate * (quarter hours simulated / quarter hours in month)

    Prorating matters because a run of a few weeks would otherwise be billed a full
    month's capacity charge twice if it happens to straddle a month boundary.
    """
    total = 0.0
    detail: dict[str, dict[str, float]] = {}
    months = pd.PeriodIndex(offtake_kw.index.tz_localize(None), freq="M")
    for period, grp in offtake_kw.groupby(months):
        month_start = grp.index[0].normalize().replace(day=1)
        steps_total = quarter_hours_in_month(month_start)
        share = len(grp) / steps_total
        peak = float(grp.max())
        cost = peak * rate_eur_per_kw_month * share
        detail[str(period)] = {
            "peak_offtake_kw": round(peak, 2),
            "month_share": round(share, 4),
            "peak_cost_eur": round(cost, 2),
        }
        total += cost
    return total, detail


def summarise(schedule: pd.DataFrame, specs: SiteSpecs) -> dict[str, Any]:
    """The bill implied by a schedule, plus a count of physical rule breaches.

    We give you this so that everyone's euros are computed the same way. It bills the
    schedule that was actually applied. It does not tell you whether that bill is good.
    """
    battery = specs.battery
    energy_cost = float(
        (schedule["offtake_kw"] * schedule["offtake_price_eur_per_mwh"]).sum()
        * HOURS_PER_STEP
        / 1000
        - (schedule["injection_kw"] * schedule["injection_price_eur_per_mwh"]).sum()
        * HOURS_PER_STEP
        / 1000
    )
    peak_cost, peak_detail = monthly_peak_cost(
        schedule["offtake_kw"], specs.offtake_monthly_peak_cost_eur_per_kw
    )
    throughput_kwh = float(
        (schedule["applied_charge_kw"].sum() + schedule["applied_discharge_kw"].sum())
        * HOURS_PER_STEP
    )
    equivalent_cycles = throughput_kwh / 2 / battery.usable_capacity_kwh
    cycle_cost = equivalent_cycles * battery.cycle_cost_eur

    return {
        "from": str(schedule.index[0]),
        "to": str(schedule.index[-1]),
        "n_quarter_hours": int(len(schedule)),
        "energy_cost_eur": round(energy_cost, 2),
        "peak_cost_eur": round(peak_cost, 2),
        "peak_cost_detail": peak_detail,
        "cycle_cost_eur": round(cycle_cost, 2),
        "total_cost_eur": round(energy_cost + peak_cost + cycle_cost, 2),
        "equivalent_cycles": round(equivalent_cycles, 2),
        "battery_throughput_kwh": round(throughput_kwh, 1),
        "final_soc": round(float(schedule["soc"].iloc[-1]), 3),
        "violations": {
            "simultaneous_charge_discharge_steps": int(
                schedule["simultaneous_charge_discharge"].sum()
            ),
            "clipped_by_power_limit_steps": int(
                schedule["clipped_by_power_limit"].sum()
            ),
            "clipped_by_soc_limit_steps": int(schedule["clipped_by_soc_limit"].sum()),
            "clipped_by_grid_limit_steps": int(schedule["clipped_by_grid_limit"].sum()),
            "offtake_limit_exceeded_steps": int(
                (schedule["offtake_limit_exceeded_kw"] > TOL).sum()
            ),
            "injection_limit_exceeded_steps": int(
                (schedule["injection_limit_exceeded_kw"] > TOL).sum()
            ),
        },
    }


def write_summary(summary: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2, default=str))
