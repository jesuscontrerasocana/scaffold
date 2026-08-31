"""Site and battery specifications, loaded from site.yaml.

Provided by us. You should not need to change this file.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class BatterySpecs:
    capacity_kwh: float
    charge_power_kw: float
    discharge_power_kw: float
    min_soc: float
    max_soc: float
    round_trip_efficiency: float
    n_cycles_per_year: int
    capex_eur_per_kwh: float
    years_on_warranty: int
    initial_soc: float

    @property
    def usable_capacity_kwh(self) -> float:
        """Energy available between min_soc and max_soc."""
        return self.capacity_kwh * (self.max_soc - self.min_soc)

    @property
    def one_way_efficiency(self) -> float:
        """Efficiency applied on each leg, so that charge * discharge = round_trip."""
        return self.round_trip_efficiency**0.5

    @property
    def cycle_cost_eur(self) -> float:
        """Cost of one full equivalent cycle, amortising capex over the warranty envelope.

        A "cycle" here means moving `usable_capacity_kwh` through the battery once.
        """
        total_capex = self.capex_eur_per_kwh * self.capacity_kwh
        total_cycles = self.years_on_warranty * self.n_cycles_per_year
        return total_capex / total_cycles


@dataclasses.dataclass(frozen=True)
class SiteSpecs:
    site_name: str
    country_code: str
    timezone: str
    offtake_limit_kw: float
    injection_limit_kw: float
    offtake_monthly_peak_cost_eur_per_kw: float
    battery: BatterySpecs

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SiteSpecs":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        battery = BatterySpecs(**raw.pop("battery"))
        return cls(battery=battery, **raw)
