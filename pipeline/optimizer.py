"""Transparent linear battery scheduling with Pyomo and HiGHS."""

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

    def __init__(self, specs: SiteSpecs) -> None:
        self.specs = specs
        self.last_summary: dict[str, float | int | str] = {}

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

        model = pyo.ConcreteModel()
        model.steps = pyo.RangeSet(0, len(forecast) - 1)
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
            previous = initial_energy if step == 0 else m.energy[step - 1]
            return m.energy[step] == previous + HOURS_PER_STEP * (
                eta * m.charge[step] - m.discharge[step] / eta
            )

        model.energy_balance = pyo.Constraint(model.steps, rule=energy_balance)
        model.grid_balance = pyo.Constraint(
            model.steps,
            rule=lambda m, step: m.grid_import[step] - m.grid_export[step]
            == net[step] + m.charge[step] - m.discharge[step],
        )

        degradation_per_internal_kwh = battery.cycle_cost_eur / (
            2 * battery.usable_capacity_kwh
        )
        model.energy_cost = pyo.Expression(
            expr=sum(
                HOURS_PER_STEP
                / 1000
                * (
                    model.grid_import[step] * offtake_price[step]
                    - model.grid_export[step] * injection_price[step]
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

        try:
            result = pyo.SolverFactory("highs").solve(model)
        except NoFeasibleSolutionError as error:
            raise RuntimeError("Battery optimization failed: infeasible") from error
        termination = result.solver.termination_condition
        if termination != TerminationCondition.optimal:
            raise RuntimeError(f"Battery optimization failed: {termination}")

        charge = np.array([pyo.value(model.charge[t]) for t in model.steps])
        discharge = np.array([pyo.value(model.discharge[t]) for t in model.steps])
        energy = np.array([pyo.value(model.energy[t]) for t in model.steps])
        grid_import = np.array([pyo.value(model.grid_import[t]) for t in model.steps])
        grid_export = np.array([pyo.value(model.grid_export[t]) for t in model.steps])

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
