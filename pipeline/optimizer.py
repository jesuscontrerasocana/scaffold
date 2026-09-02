"""YOUR CODE GOES HERE (2 of 2).

What ships here is a **deliberately weak baseline**: a price threshold rule, not an
optimisation at all. It exists so the pipeline runs out of the box. Replace it with a
mixed-integer linear program.

Decide what the battery should do over the forecast horizon. The objective is the site's
energy bill.

Contract
--------
`simulate` constructs `Optimizer(specs)` once, then calls `solve` once per decision — every
quarter hour, so about 2,880 times over a month.

Arguments to `solve`
--------------------
forecast : DataFrame indexed by the horizon's timestamps. Whatever your forecaster returned:
           `net_kw` at minimum, plus any extra columns you chose to produce.

           **The horizon reaches up to 132 quarter hours (33 h)**, capped only near the end
           of the data. It does not shrink to where the prices stop — that is your call.
prices   : DataFrame over the same timestamps, with `offtake_price_eur_per_mwh` and
           `injection_price_eur_per_mwh` in EUR/MWh. Injection prices can be negative.

           Prices are known a day ahead, published at 15:00, so only part of the horizon has
           real prices — nine hours at 14:45, thirty-three just after 15:00. **Past that edge
           the prices are `NaN`**, and `context.prices_known_until` is the last timestamp that
           carries a real one. What you do with the unpriced tail is your decision.
context  : DecisionContext (defined in harness.py)
             .at_time            this decision time
             .initial_soc        the battery's state of charge right now, as a fraction of
                                 capacity, after everything already committed
             .prices_known_until the last horizon timestamp whose price is published; every
                                 price after it in `prices` is NaN
             .history            every observation strictly before this decision time
             .month              what has already happened this calendar month:
                                   .peak_offtake_kw  highest quarter-hourly offtake so far
                                   .steps_elapsed    quarter hours already elapsed
                                   .steps_total      quarter hours in the whole month
             .state              a dict that is yours, handed back unchanged at the next
                                 decision. Use it for anything you want to carry forward.

Return value
------------
A DataFrame indexed by exactly the horizon's timestamps, with two columns in kW:

    battery_charge_kw      power the battery draws from the site
    battery_discharge_kw   power the battery delivers to the site

Only the **next quarter hour** is committed. Then we re-forecast, with the meter reading
that has just arrived, and ask you again.

You do not have to police the physics. The harness clips anything the battery cannot
actually deliver — power beyond its rating, energy beyond its state-of-charge limits, or a
charge/discharge that would drive the meter past `offtake_limit_kw` / `injection_limit_kw` —
and records that it had to. Ask for something impossible and you will simply not get it,
which will show up in the summary and in your bill.

Where the money is
------------------
`self.specs` carries everything from `site.yaml`. Read all of it. In particular: the
capacity charge is billed on the single highest quarter hour of a **calendar month**, it
ratchets — once set it cannot come down — and a simulation begins on the 1st, when nothing
has been set yet. `context.month.peak_offtake_kw` is where it currently stands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers import Highs

from pipeline.data import HOURS_PER_STEP
from pipeline.harness import DecisionContext
from pipeline.specs import SiteSpecs


PRICE_COLUMNS = (
    "offtake_price_eur_per_mwh",
    "injection_price_eur_per_mwh",
)


def _fill_unpublished_prices(
    prices: pd.DataFrame,
    history: pd.DataFrame,
    at_time: pd.Timestamp,
) -> tuple[pd.DataFrame, int, str]:
    """Fill missing prices from the nearest earlier same-type local calendar day."""
    filled = prices.loc[:, PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    safe_history = history.loc[history.index < at_time]
    available_columns = [column for column in PRICE_COLUMNS if column in safe_history]
    if not available_columns:
        return filled, 0, "unavailable"

    historical = safe_history.loc[:, available_columns].apply(
        pd.to_numeric, errors="coerce"
    ).sort_index()
    references: set[str] = set()
    filled_steps: set[pd.Timestamp] = set()

    for target_day in filled.index.normalize().unique():
        eligible = (
            (historical.index.normalize() < target_day)
            & ((historical.index.weekday >= 5) == (target_day.weekday() >= 5))
        )
        target_timestamps = filled.index[filled.index.normalize() == target_day]
        for column in available_columns:
            candidates = historical.loc[eligible, column].dropna()
            latest_by_slot = candidates.groupby(
                [candidates.index.hour, candidates.index.minute]
            ).tail(1)
            lookup = {
                (timestamp.hour, timestamp.minute): (value, timestamp.date())
                for timestamp, value in latest_by_slot.items()
            }
            for timestamp in target_timestamps:
                if pd.notna(filled.at[timestamp, column]):
                    continue
                fallback = lookup.get((timestamp.hour, timestamp.minute))
                if fallback is None:
                    continue
                value, reference_day = fallback
                filled.at[timestamp, column] = value
                filled_steps.add(timestamp)
                references.add(str(reference_day))

    reference_days = ", ".join(sorted(references)) if references else "unavailable"
    return filled, len(filled_steps), reference_days


class ScaffoldOptimizer:
    """BASELINE — replace me. Charge when power is cheap, discharge when it is dear."""

    CHEAP_QUANTILE = 0.25
    DEAR_QUANTILE = 0.75

    def __init__(self, specs: SiteSpecs) -> None:
        self.specs = specs

    def solve(
        self,
        forecast: pd.DataFrame,
        prices: pd.DataFrame,
        context: DecisionContext,
    ) -> pd.DataFrame:
        price = prices["offtake_price_eur_per_mwh"].to_numpy()
        published = ~np.isnan(price)  # prices past the day-ahead edge arrive as NaN

        # Act only where prices are published and idle through the unpriced tail -- the
        # simplest thing that respects the day-ahead rule.
        charge = np.zeros(len(price))
        discharge = np.zeros(len(price))
        if published.any():
            cheap, dear = np.quantile(
                price[published], [self.CHEAP_QUANTILE, self.DEAR_QUANTILE]
            )
            charging = published & (price <= cheap)
            discharging = (
                published & (price >= dear) & ~charging
            )  # flat prices satisfy both
            charge[charging] = self.specs.battery.charge_power_kw
            discharge[discharging] = self.specs.battery.discharge_power_kw
        return pd.DataFrame(
            {"battery_charge_kw": charge, "battery_discharge_kw": discharge},
            index=forecast.index,
        )

class Optimizer:
    """Minimize energy and battery degradation cost over the forecast horizon.

    Positive ``net_kw`` is site consumption. Charging increases grid demand and
    discharging reduces it, so import - export = net + charge - discharge.
    """

    MAX_HORIZON = 132

    def __init__(
        self,
        specs: SiteSpecs,
        enforce_negative_price_exclusivity: bool = False,
    ) -> None:
        self.specs = specs
        self.enforce_negative_price_exclusivity = (
            enforce_negative_price_exclusivity
        )
        self.last_summary: dict[str, float | int | str] = {}
        self.model = self._build_model(self.MAX_HORIZON)
        self.lp_solver = Highs()
        self.milp_solver = Highs()
        self.lp_solver.config.load_solution = False
        self.milp_solver.config.load_solution = False

    def _build_model(
        self,
        horizon_steps: int,
        *,
        negative_steps: list[int] | None = None,
    ) -> pyo.ConcreteModel:
        battery = self.specs.battery
        eta = battery.one_way_efficiency
        minimum_energy = battery.capacity_kwh * battery.min_soc
        maximum_energy = battery.capacity_kwh * battery.max_soc

        model = pyo.ConcreteModel()
        model.steps = pyo.RangeSet(0, horizon_steps - 1)
        model.net_kw = pyo.Param(model.steps, mutable=True, initialize=0.0)
        model.offtake_price = pyo.Param(model.steps, mutable=True, initialize=0.0)
        model.injection_price = pyo.Param(model.steps, mutable=True, initialize=0.0)
        model.current_month_step = pyo.Param(
            model.steps, mutable=True, within=pyo.Binary, initialize=0
        )
        model.initial_energy = pyo.Param(mutable=True, initialize=0.0)
        model.past_month_peak = pyo.Param(mutable=True, initialize=0.0)
        model.charge = pyo.Var(
            model.steps, bounds=(0.0, battery.charge_power_kw)
        )
        model.discharge = pyo.Var(
            model.steps, bounds=(0.0, battery.discharge_power_kw)
        )
        model.energy = pyo.Var(
            model.steps, bounds=(minimum_energy, maximum_energy)
        )
        model.grid_import = pyo.Var(
            model.steps, bounds=(0.0, self.specs.offtake_limit_kw)
        )
        model.grid_export = pyo.Var(
            model.steps, bounds=(0.0, self.specs.injection_limit_kw)
        )
        model.planned_peak = pyo.Var(bounds=(0.0, self.specs.offtake_limit_kw))

        def energy_balance(m: pyo.ConcreteModel, step: int) -> pyo.Constraint:
            previous = m.initial_energy if step == 0 else m.energy[step - 1]
            return m.energy[step] == previous + HOURS_PER_STEP * (
                eta * m.charge[step] - m.discharge[step] / eta
            )

        model.energy_balance = pyo.Constraint(model.steps, rule=energy_balance)

        model.terminal_energy = pyo.Constraint(expr=model.energy[horizon_steps - 1] >= 0.5 * maximum_energy)
        model.grid_balance = pyo.Constraint(
            model.steps,
            rule=lambda m, step: m.grid_import[step] - m.grid_export[step]
            == m.net_kw[step] + m.charge[step] - m.discharge[step],
        )
        model.past_peak_limit = pyo.Constraint(
            expr=model.planned_peak >= model.past_month_peak
        )
        model.horizon_peak_limit = pyo.Constraint(
            model.steps,
            rule=lambda m, step: m.planned_peak
            >= m.grid_import[step]
            - self.specs.offtake_limit_kw * (1 - m.current_month_step[step]),
        )
        if negative_steps:
            model.negative_steps = pyo.Set(initialize=negative_steps)
            model.charge_mode = pyo.Var(model.negative_steps, domain=pyo.Binary)
            model.charge_exclusive = pyo.Constraint(
                model.negative_steps,
                rule=lambda m, step: m.charge[step]
                <= battery.charge_power_kw * m.charge_mode[step],
            )
            model.discharge_exclusive = pyo.Constraint(
                model.negative_steps,
                rule=lambda m, step: m.discharge[step]
                <= battery.discharge_power_kw * (1 - m.charge_mode[step]),
            )

        degradation_per_internal_kwh = battery.cycle_cost_eur / (
            2 * battery.usable_capacity_kwh
        )
        model.energy_cost = pyo.Expression(
            expr=sum(
                HOURS_PER_STEP
                / 1000
                * (
                    model.grid_import[step] * model.offtake_price[step]
                    - model.grid_export[step] * model.injection_price[step]
                )
                for step in model.steps
            )
        )
        model.degradation_cost = pyo.Expression(
            expr=sum(
                HOURS_PER_STEP
                * degradation_per_internal_kwh
                * (
                    eta * model.charge[step]
                    + model.discharge[step] / eta
                )
                for step in model.steps
            )
        )
        model.incremental_peak_cost = pyo.Expression(
            expr=self.specs.offtake_monthly_peak_cost_eur_per_kw
            * (model.planned_peak - model.past_month_peak)
        )
        model.objective = pyo.Objective(
            expr=(
                model.energy_cost
                + model.degradation_cost
                + model.incremental_peak_cost
            ),
            sense=pyo.minimize,
        )
        return model

    def solve(
        self,
        forecast: pd.DataFrame,
        prices: pd.DataFrame,
        context: DecisionContext,
    ) -> pd.DataFrame:
        if not forecast.index.equals(prices.index):
            raise ValueError("Forecast and price indexes must match")
        if "net_kw" not in forecast or not {
            "offtake_price_eur_per_mwh",
            "injection_price_eur_per_mwh",
        } <= set(prices.columns):
            raise ValueError("Forecast or prices are missing required columns")
        if len(forecast) == 0:
            self.last_summary = {
                "solver_termination": "empty horizon",
                "solver_mode": "lp",
                "negative_price_exclusivity_enabled": (
                    self.enforce_negative_price_exclusivity
                ),
            }
            return pd.DataFrame(
                columns=["battery_charge_kw", "battery_discharge_kw"],
                index=forecast.index,
                dtype=float,
            )
        if len(forecast) > self.MAX_HORIZON:
            raise ValueError(f"Forecast horizon cannot exceed {self.MAX_HORIZON} steps")

        net = pd.to_numeric(forecast["net_kw"], errors="coerce").to_numpy()
        if not np.isfinite(net).all():
            raise ValueError("Forecast net_kw must contain only finite values")

        numeric_prices = prices.loc[:, PRICE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )
        published_price_steps = int(numeric_prices.notna().all(axis=1).sum())
        filled_prices, filled_price_steps, reference_days = _fill_unpublished_prices(
            numeric_prices, context.history, context.at_time
        )
        # Missing historical values retain the previous zero-value economic fallback.
        filled_prices = filled_prices.fillna(0.0)
        offtake_price = filled_prices[PRICE_COLUMNS[0]].to_numpy(dtype=float)
        injection_price = filled_prices[PRICE_COLUMNS[1]].to_numpy(dtype=float)
        negative_steps = [
            step
            for step in range(len(forecast))
            if injection_price[step] < 0
        ]
        use_milp = self.enforce_negative_price_exclusivity and bool(negative_steps)

        battery = self.specs.battery
        eta = battery.one_way_efficiency
        minimum_energy = battery.capacity_kwh * battery.min_soc
        maximum_energy = battery.capacity_kwh * battery.max_soc
        initial_energy = battery.capacity_kwh * context.initial_soc
        if not minimum_energy <= initial_energy <= maximum_energy:
            raise ValueError("Initial state of charge is outside battery bounds")
        past_month_peak = float(context.month.peak_offtake_kw)

        model = (
            self._build_model(len(forecast), negative_steps=negative_steps)
            if use_milp
            else self.model
        )
        for step in model.steps:
            model.net_kw[step] = 0.0
            model.offtake_price[step] = 0.0
            model.injection_price[step] = 0.0
            model.current_month_step[step] = 0
        for step in range(len(forecast)):
            model.net_kw[step] = net[step]
            model.offtake_price[step] = offtake_price[step]
            model.injection_price[step] = injection_price[step]
            timestamp = forecast.index[step]
            model.current_month_step[step] = int(
                timestamp.year == context.at_time.year
                and timestamp.month == context.at_time.month
            )
        model.initial_energy.set_value(initial_energy)
        model.past_month_peak.set_value(past_month_peak)

        solver = self.milp_solver if use_milp else self.lp_solver
        result = solver.solve(model)
        termination = result.termination_condition
        if termination != TerminationCondition.optimal:
            raise RuntimeError(f"Battery optimization failed: {termination}")
        result.solution_loader.load_vars()

        active_steps = range(len(forecast))
        charge = np.array([pyo.value(model.charge[t]) for t in active_steps])
        discharge = np.array([pyo.value(model.discharge[t]) for t in active_steps])
        energy = np.array([pyo.value(model.energy[t]) for t in active_steps])
        planned_peak = float(pyo.value(model.planned_peak))

        self.last_summary = {
            "solver_termination": termination.name,
            "solver_mode": "milp" if use_milp else "lp",
            "objective_eur": float(pyo.value(model.objective)),
            "energy_cost_eur": float(pyo.value(model.energy_cost)),
            "degradation_cost_eur": float(pyo.value(model.degradation_cost)),
            "modeled_peak_cost_eur": float(
                pyo.value(model.incremental_peak_cost)
            ),
            "past_month_peak_kw": past_month_peak,
            "planned_peak_kw": planned_peak,
            "initial_soc": float(context.initial_soc),
            "final_soc": float(energy[-1] / battery.capacity_kwh),
            "published_price_steps": published_price_steps,
            "filled_price_steps": filled_price_steps,
            "price_fallback_reference_day": reference_days,
        }
        return pd.DataFrame(
            {
                "battery_charge_kw": charge,
                "battery_discharge_kw": discharge,
            },
            index=forecast.index,
        )
