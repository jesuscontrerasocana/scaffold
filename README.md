# Site forecasting and battery dispatch — starter repo

One commercial site in Belgium: quarter-hourly metering, rooftop PV, a grid connection with
a capacity charge, and a battery. Minimise the site's energy bill.

**Everything here runs already.** A baseline forecaster and a baseline optimizer ship in the
box, so within five minutes you can simulate a month, see a bill and look at a plot. They
are deliberately not ideal. Your job is to replace them.

---

## 1. Getting started (2 minutes)

Make sure **Python ≥ 3.11** and **uv** are installed on your laptop, then install
dependencies:

```bash
uv sync --extra dev                        # or: python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

Point your IDE at the virtual environment uv creates in `.venv`. Then check it all runs:

```bash
uv run --extra dev python -m pytest        # 12 tests, all should pass immediately
```

Add any dependency you like as long as it `pip install`s or `uv sync`s correctly.

## 2. Run the baseline implementation (3 minutes)

```bash
uv run python -m pipeline train --data data/history.csv --model-dir models/

uv run python -m pipeline simulate --data data/history.csv --model-dir models/ \
    --from 2026-06-01 --to 2026-06-30 --out out/june.csv
```

That rolls your pipeline forward **every quarter hour** across June. At each step it
gives your forecaster the meter readings up to that moment and nothing after, asks your
optimizer what the battery should do, commits the next quarter hour against what the site
really did, and bills it. Rows before `--from` are history your forecaster may read; they
are not scored.

It writes:

| file | what it is |
|---|---|
| `out/june.csv` | one row per quarter hour: forecast, reality, what the battery was asked to do, what it did, state of charge, grid flows, prices, and a flag for every physical rule that had to be bent |
| `out/june_summary.json` | the bill: energy, capacity charge, cycle cost, and those flags counted up |
| `out/june_plot.png` | **look at this first** — the meter with and without the battery, forecast against reality, charge/discharge and state of charge, and the prices |

Open the plot. The baseline never looks at the load: it charges at full power whenever
electricity is cheap, straight through the site's busiest hours, and pushes the monthly peak
far above where it would have been with no battery at all. Its capacity charge alone more
than eats every euro it earns from arbitrage.

### Evaluate forecast accuracy only

After training a model, evaluate its forecasts without running the optimizer, battery
simulation, billing, or reporting pipeline:

```bash
uv run python scripts/evaluate_forecast.py \
    --data data/history.csv \
    --model-dir models/ \
    --from 2026-06-01 \
    --to 2026-06-30
```

The model type comes from the persisted model in `--model-dir`, so the same command works
for scaffold and weekly models. The script rolls through the requested decision times with
the same 33-hour forecast horizon and information boundary as the normal harness. Forecasts
may extend beyond `--to` when later realized data is available.

It prints overall MAE, RMSE, mean error (bias), and nMAE. By default, it writes two files
beside the persisted model:

| file | contents |
|---|---|
| `<model-dir>/forecast_lead_metrics.csv` | MAE, RMSE, and bias for each forecast lead step |
| `<model-dir>/forecast_comparisons.csv` | every forecast and actual value side by side, with its decision time, target time, and lead step |

Use `--out` to choose another path for the lead-metrics CSV; the comparisons CSV is written
beside it. Use `--horizon-steps` or `--decision-interval-minutes` for a smaller diagnostic
run:

```bash
uv run python scripts/evaluate_forecast.py \
    --data data/history.csv \
    --model-dir models/ \
    --from 2026-06-01 \
    --to 2026-06-02 \
    --horizon-steps 32 \
    --decision-interval-minutes 60 \
    --out out/june_forecast_metrics.csv
```

## 3. How the site works

If you have not worked on an energy system before, this is everything you need. If you have,
skim it — the sign conventions and the day-ahead rule are the parts worth checking.

### Power flows

At any quarter hour, four things move power around:

```
            PV panels                    the grid
        pv_production_kw                    |
                |                           |
                v                           v
        +-------+------------+------+  <-- the meter measures HERE
        |                    |      |
        v                    |      |
   the building's            |      |
     consumption             |      |
                             |      |
                        +----+------+----+
                        |    the battery |
                        +----------------+
```

The meter sits between the site and the grid, so it sees the sum of everything behind it:

```
grid_net_kw            =  consumption  -  PV production        (given to you, before the battery)

grid_net_with_bess_kw  =  grid_net_kw  +  charge  -  discharge  (what you control; what the
                                                                meter finally sees)
```

- **positive** `grid_net_with_bess_kw` is **offtake**: the site imports, and pays
  `offtake_price_eur_per_mwh` on every kWh.
- **negative** is **injection**: the site exports, and receives `injection_price_eur_per_mwh`
  on every kWh — which is sometimes negative, meaning it pays to export.
- both are capped: `offtake_limit_kw` and `injection_limit_kw` in `site.yaml`.


### The battery

`site.yaml` describes it. In quarter-hourly terms:

| | |
|---|---|
| `capacity_kwh` | total size. Only the band between `min_soc` and `max_soc` is usable, so a 400 kWh battery with 0.05–0.95 limits gives you 360 kWh to play with. |
| `charge_power_kw`, `discharge_power_kw` | how fast, in either direction. |
| `round_trip_efficiency` | 0.9 means a kWh in and back out again returns 0.9 kWh. Split evenly, each leg is `sqrt(0.9) = 0.949`. |
| `n_cycles_per_year`, `capex_eur_per_kwh`, `years_on_warranty` | the battery wears out. `BatterySpecs.cycle_cost_eur` in `pipeline/specs.py` turns those into a price per full cycle, and the bill charges it. Cycling for less than that loses money. |

Energy moves like this, with `dt = 0.25` hours:

```
E[t+1]  =  E[t]  +  ( efficiency x charge[t]  -  discharge[t] / efficiency ) x dt
```

so charging costs you more grid energy than the battery stores, and discharging drains more
than it delivers. The harness applies exactly this, and clips anything that would take the
battery outside its limits.

### The bill

Three parts, all in `site.yaml`, all computed for you.

**Energy.** `offtake_kw x offtake_price_eur_per_mwh x 0.25 / 1000` per quarter hour, minus the same for
injection. Prices are in EUR/MWh and move hour to hour, which is where arbitrage comes from:
charge when power is cheap, discharge when it is dear, and keep the difference minus your
losses and your cycle cost.

**The capacity charge.** `max(offtake_kw over the calendar month) x
offtake_monthly_peak_cost_eur_per_kw`. Read that again — it is a maximum, over a whole
month, and at €4.25/kW a single careless quarter hour is expensive. It is also *ratcheting*:
once the month's peak is set it cannot come down, so the damage is permanent until the 1st.
A simulation starts on the 1st, when nothing has been set yet.

**The cycle cost.** As above. It is what stops you cycling the battery for a two-euro spread.

### When you know the prices

Electricity here is bought on **day ahead**. The auction for every quarter hour of tomorrow
clears and is available in our database at **15:00 today**. Before 15:00 you only know today's prices; after
15:00 you know today's and tomorrow's.

The harness enforces this. The stretch of horizon you have *real* prices for is not fixed —
it breathes:

| decision time | prices published until | priced hours ahead |
|---|---|---|
| 00:00 | end of today | 24 h |
| 14:45 | end of today | **9.25 h** |
| 15:00 | end of tomorrow | **33 h** |
| 23:45 | end of tomorrow | 24.25 h |

But the harness always hands you the **full 33-hour horizon**; the prices past the published
edge simply arrive as `NaN`, and `context.prices_known_until` tells you where that edge is.
What you do past it is your call. What the harness will never do is let you read a price that
has not been published.

## 4. Write your two files

| File | Yours? |
|---|---|
| `pipeline/forecaster.py` | **yes** — forecast the site over the horizon |
| `pipeline/optimizer.py` | **yes** — decide what the battery does |
| `pipeline/report.py` | **optional** — generate a report on the battery's performance (not mandatory to edit) |
| everything else in `pipeline/` | no, we run our own copy |
| `site.yaml` | no, we run our own copy |

Both files document their contract at the top. Read them. Add modules, tests and
dependencies freely; don't change the two commands, the two interfaces, or the meaning of
anything in `site.yaml`.

Loop: edit → `train` → `simulate` → read the summary, open the plot → repeat.

**Simulate a full calendar month, starting on the 1st.** The capacity charge is billed on
the highest quarter hour of a calendar month, so a shorter window bills only part of it and
will mislead you about what your controller is worth.

Pick a month with data behind it as well as in front of it. `history.csv` ends on
31 July, so simulating July leaves the last day with no prices to look ahead to and an
artificially short horizon. June is a better test bed; the month we hand you at the
interview has a tail for exactly this reason.

---

## Data

Every column is documented at the top of `pipeline/data.py`, along with what
`load_timeseries` returns and how it handles gaps and clock changes.

## Two things the harness does for you

**It stops you seeing the future.** At each decision the history is sliced in our code, not
yours. Don't spend time engineering around leakage.

**It enforces the physics.** Your optimizer may ask for anything; the harness clips what the
battery cannot deliver — power beyond its rating, energy beyond its state-of-charge limits,
or a charge/discharge that would push the meter past `offtake_limit_kw` / `injection_limit_kw`
— and records that it had to.
