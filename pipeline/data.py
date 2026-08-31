"""Loading the site CSV. Provided by us — you should not need to change this file.

## The columns

Every row is one quarter hour. The file is semicolon-separated with a comma decimal mark
(European CSV).

| column | meaning |
|---|---|
| `datetime_utc` | timestamp in **UTC**, e.g. `2025-06-30 22:45:00`. Quarter-hourly. Loaded and converted to local Belgian time; because it is UTC there is no ambiguous hour, so the two clock changes resolve cleanly. |
| `grid_net_kw` | what the grid meter reads, **excluding** the battery. Positive means the site is importing, negative means it is exporting. This is the thing your optimizer has to work against. |
| `pv_production_kw` | PV production, in kW. Always `>= 0`. |
| `offtake_price_eur_per_mwh` | EUR/MWh paid on every kWh imported. |
| `injection_price_eur_per_mwh` | EUR/MWh received on every kWh exported. **Can be negative** — there are hours when the site pays to export. |
| `most_recent_load_factor_forecast` | a third-party PV proxy in `[0, 1]`: the most recent Elia day-ahead solar-production forecast for Belgium, divided by the total monitored PV capacity. Published ahead of time, so it is available across the horizon you are forecasting — and imperfect. |

Prices and the load-factor forecast are known in advance. The two metered columns
(`grid_net_kw`, `pv_production_kw`) are not: at any decision time you only have them for the
past, and the harness enforces that.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


STEP = pd.Timedelta(minutes=15)
STEPS_PER_HOUR = 4
HOURS_PER_STEP = 0.25

REQUIRED_COLUMNS = [
    "grid_net_kw",
    "pv_production_kw",
    "offtake_price_eur_per_mwh",
    "injection_price_eur_per_mwh",
    "most_recent_load_factor_forecast",
]


def load_timeseries(
    path: str | Path, timezone: str = "Europe/Brussels"
) -> pd.DataFrame:
    """Read a site CSV into a DataFrame the rest of the pipeline can rely on.

    Args:
        path: the CSV to read. Semicolon-separated with a comma decimal mark. It must have a
            `datetime_utc` column and the five data columns listed in `REQUIRED_COLUMNS`;
            anything else it carries is passed through untouched.
        timezone: the site's local timezone. Timestamps are read as UTC and converted to it,
            so a clock change is represented correctly rather than as a duplicated or missing
            label.

    Returns:
        A DataFrame indexed by a timezone-aware `DatetimeIndex` named `datetime`, sorted,
        with no duplicate timestamps, on a **strictly regular 15-minute grid** from the
        first row to the last. Columns are those in `REQUIRED_COLUMNS`.

        Because the index is made regular, any quarter hour missing from the file appears
        as a row of `NaN` rather than as an invisible jump in time. If you rely on lags,
        that is the behaviour you want — but it does mean you should expect `NaN` and
        decide what to do about it.

    Raises:
        ValueError: if `datetime_utc` or any required column is absent.
    """
    with open(path, encoding="utf-8") as handle:
        header = handle.readline()
    european = header.count(";") > header.count(",")
    df = pd.read_csv(
        path, sep=";" if european else ",", decimal="," if european else "."
    )

    if "datetime_utc" not in df.columns:
        raise ValueError("CSV must contain a 'datetime_utc' column")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    index = pd.to_datetime(
        df["datetime_utc"], utc=True, format="ISO8601"
    ).dt.tz_convert(timezone)
    df = df.drop(columns=["datetime_utc"]).set_index(index).sort_index()
    df.index.name = "datetime"

    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]

    full_index = pd.date_range(df.index[0], df.index[-1], freq=STEP, tz=df.index.tz)
    df = df.reindex(full_index)
    df.index.name = "datetime"
    return df
