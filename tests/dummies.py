"""Stand-ins used by the tests to poke at edge cases the shipped baselines do not reach."""

from __future__ import annotations

import pandas as pd

from pipeline.specs import SiteSpecs


class IdleOptimizer:
    """Never touches the battery."""

    def __init__(self, specs: SiteSpecs) -> None:
        self.specs = specs

    def solve(self, forecast: pd.DataFrame, prices: pd.DataFrame, context) -> pd.DataFrame:  # noqa: ANN001, ARG002
        return pd.DataFrame({"battery_charge_kw": 0.0, "battery_discharge_kw": 0.0}, index=forecast.index)


class FullPowerOptimizer:
    """Asks for more than the battery can give, so we can check the runner clips it."""

    def __init__(self, specs: SiteSpecs) -> None:
        self.specs = specs

    def solve(self, forecast: pd.DataFrame, prices: pd.DataFrame, context) -> pd.DataFrame:  # noqa: ANN001, ARG002
        return pd.DataFrame(
            {"battery_charge_kw": self.specs.battery.charge_power_kw * 2, "battery_discharge_kw": 0.0},
            index=forecast.index,
        )
