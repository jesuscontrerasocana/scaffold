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

from pipeline.harness import DecisionContext
from pipeline.specs import SiteSpecs


class Optimizer:
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
