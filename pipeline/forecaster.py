"""YOUR CODE GOES HERE (1 of 2).

What ships here is a **deliberately weak baseline**: whatever the site did at this time
yesterday. It exists so the pipeline runs the moment you unzip it, and so you have
something to beat. It is not a starting point for your model — replace it.

Forecast what the site will do over the horizon, at quarter-hourly resolution.

Contract
--------
`train` calls:     Forecaster(specs) -> fit(history) -> save(model_dir)
`simulate` calls:  Forecaster.load(model_dir, specs) -> predict(...) once per decision

A decision is taken every quarter hour, so `predict` is called about 2,880 times over a
month and the whole run must finish in under fifteen minutes. Fit expensive things in `fit`,
persist them in `save`, keep `predict` cheap. Do not retrain inside `predict`.

Arguments to `predict`
----------------------
at_time      : the decision time. You know nothing at or after it.
history      : every row strictly before `at_time`, with all the columns from `data.py`.
               Metered values included — this is the past.
future_exog  : rows from `at_time` onwards, restricted to what is knowable in advance:
               `most_recent_load_factor_forecast` over the whole horizon, and
               `offtake_price_eur_per_mwh` and `injection_price_eur_per_mwh` only as far as
               the day-ahead auction has published them — NaN past that edge, since
               tomorrow's prices appear at 15:00 today.
horizon      : how many quarter hours to forecast, counting from `at_time`.

Return value
------------
A DataFrame indexed by exactly `future_exog.index[:horizon]`, containing at least:

    net_kw   your forecast of `grid_net_kw` over those timestamps
             (positive = the site imports, negative = the site exports)

No NaNs, and the index must cover the whole horizon — the harness rejects both.

Any extra column you add is passed straight through to your optimizer untouched. If you
want to hand the control layer more than a point forecast, that is how it travels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline.specs import SiteSpecs


class Forecaster:
    """BASELINE — replace me. Persistence: what happened at this time yesterday."""

    PARAMS = "forecaster.json"

    def __init__(self, specs: SiteSpecs) -> None:
        self.specs = specs
        self.fallback_kw = 0.0

    def fit(self, history: pd.DataFrame) -> None:
        self.fallback_kw = float(history["grid_net_kw"].median())

    def save(self, path: Path) -> None:
        (Path(path) / self.PARAMS).write_text(
            json.dumps({"fallback_kw": self.fallback_kw})
        )

    @classmethod
    def load(cls, path: Path, specs: SiteSpecs) -> "Forecaster":
        self = cls(specs)
        self.fallback_kw = json.loads((Path(path) / cls.PARAMS).read_text())[
            "fallback_kw"
        ]
        return self

    def predict(
        self,
        at_time: pd.Timestamp,
        history: pd.DataFrame,
        future_exog: pd.DataFrame,
        horizon: int,
    ) -> pd.DataFrame:
        index = future_exog.index[:horizon]
        net = history["grid_net_kw"]
        yesterday = net.reindex(index - pd.Timedelta(days=1))
        last_week = net.reindex(index - pd.Timedelta(days=7))
        values = yesterday.to_numpy()
        values = pd.Series(values, index=index).fillna(
            pd.Series(last_week.to_numpy(), index=index)
        )
        return pd.DataFrame({"net_kw": values.fillna(self.fallback_kw)}, index=index)
