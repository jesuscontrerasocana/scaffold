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
from pyomo.contrib.solver.common.util import NoFeasibleSolutionError
from pyomo.opt import TerminationCondition

from pipeline.data import HOURS_PER_STEP
from pipeline.harness import DecisionContext
from pipeline.specs import SiteSpecs


class Optimizer:
    """Minimize energy and battery degradation cost over the forecast horizon.

    Positive ``net_kw`` is site consumption. Charging increases grid demand and
    discharging reduces it, so import - export = net + charge - discharge.
    """

    MAX_HORIZON = 132

    def __init__(self, specs: SiteSpecs) -> None:
        self.specs = specs
        self.last_summary: dict[str, float | int | str] = {}
        self.model = self._build_model()

    def _build_model(self) -> pyo.ConcreteModel:
        battery = self.specs.battery
        eta = battery.one_way_efficiency
        minimum_energy = battery.capacity_kwh * battery.min_soc
        maximum_energy = battery.capacity_kwh * battery.max_soc

        model = pyo.ConcreteModel()
        model.steps = pyo.RangeSet(0, self.MAX_HORIZON - 1)
        model.net_kw = pyo.Param(model.steps, mutable=True, initialize=0.0)
        model.offtake_price = pyo.Param(model.steps, mutable=True, initialize=0.0)
        model.injection_price = pyo.Param(model.steps, mutable=True, initialize=0.0)
        model.initial_energy = pyo.Param(mutable=True, initialize=0.0)
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

        def energy_balance(m: pyo.ConcreteModel, step: int) -> pyo.Constraint:
            previous = m.initial_energy if step == 0 else m.energy[step - 1]
            return m.energy[step] == previous + HOURS_PER_STEP * (
                eta * m.charge[step] - m.discharge[step] / eta
            )

        model.energy_balance = pyo.Constraint(model.steps, rule=energy_balance)
        model.grid_balance = pyo.Constraint(
            model.steps,
            rule=lambda m, step: m.grid_import[step] - m.grid_export[step]
            == m.net_kw[step] + m.charge[step] - m.discharge[step],
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
        model.objective = pyo.Objective(
            expr=model.energy_cost + model.degradation_cost,
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
            self.last_summary = {"solver_termination": "empty horizon"}
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

        offtake_price = pd.to_numeric(
            prices["offtake_price_eur_per_mwh"], errors="coerce"
        ).to_numpy(dtype=float)
        injection_price = pd.to_numeric(
            prices["injection_price_eur_per_mwh"], errors="coerce"
        ).to_numpy(dtype=float)
        published = np.isfinite(offtake_price) & np.isfinite(injection_price)
        # The unpublished tail has no assumed energy value. It remains in the model so
        # battery physics and grid limits are still enforced without inventing prices.
        offtake_price = np.where(published, offtake_price, 0.0)
        injection_price = np.where(published, injection_price, 0.0)

        battery = self.specs.battery
        eta = battery.one_way_efficiency
        minimum_energy = battery.capacity_kwh * battery.min_soc
        maximum_energy = battery.capacity_kwh * battery.max_soc
        initial_energy = battery.capacity_kwh * context.initial_soc
        if not minimum_energy <= initial_energy <= maximum_energy:
            raise ValueError("Initial state of charge is outside battery bounds")

        model = self.model
        for step in model.steps:
            model.net_kw[step] = 0.0
            model.offtake_price[step] = 0.0
            model.injection_price[step] = 0.0
        for step in range(len(forecast)):
            model.net_kw[step] = net[step]
            model.offtake_price[step] = offtake_price[step]
            model.injection_price[step] = injection_price[step]
        model.initial_energy.set_value(initial_energy)

        try:
            result = pyo.SolverFactory("highs").solve(model)
        except NoFeasibleSolutionError as error:
            raise RuntimeError("Battery optimization failed: infeasible") from error
        termination = result.solver.termination_condition
        if termination != TerminationCondition.optimal:
            raise RuntimeError(f"Battery optimization failed: {termination}")

        active_steps = range(len(forecast))
        charge = np.array([pyo.value(model.charge[t]) for t in active_steps])
        discharge = np.array([pyo.value(model.discharge[t]) for t in active_steps])
        energy = np.array([pyo.value(model.energy[t]) for t in active_steps])
        grid_import = np.array([pyo.value(model.grid_import[t]) for t in active_steps])
        grid_export = np.array([pyo.value(model.grid_export[t]) for t in active_steps])

        self.last_summary = {
            "solver_termination": str(termination),
            "objective_eur": float(pyo.value(model.objective)),
            "energy_cost_eur": float(pyo.value(model.energy_cost)),
            "injection_value_eur": float(
                HOURS_PER_STEP / 1000 * np.sum(grid_export * injection_price)
            ),
            "degradation_cost_eur": float(pyo.value(model.degradation_cost)),
            "total_import_kwh": float(HOURS_PER_STEP * grid_import.sum()),
            "total_export_kwh": float(HOURS_PER_STEP * grid_export.sum()),
            "published_price_steps": int(published.sum()),
            "initial_soc": float(context.initial_soc),
            "final_soc": float(energy[-1] / battery.capacity_kwh),
            "minimum_soc": float(energy.min() / battery.capacity_kwh),
            "maximum_soc": float(energy.max() / battery.capacity_kwh),
            "maximum_charge_kw": float(charge.max()),
            "maximum_discharge_kw": float(discharge.max()),
        }
        return pd.DataFrame(
            {
                "battery_charge_kw": charge,
                "battery_discharge_kw": discharge,
            },
            index=forecast.index,
        )
