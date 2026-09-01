# IMPLEMENTATION_TASKS.md

Concrete implementation checklist for the Octave.energy technical assignment.

The goal is to keep the solution simple, modular, defensible, and easy to iterate on.

## 1. Repository setup

- [x] Add `AGENTS.md`
  - Keep coding simple and explicit
  - Avoid unnecessary abstractions
  - Do not let coding agents make major modeling decisions
  - Preserve scaffold interfaces
- [x] Add `MODELING_NOTES.md`
  - Record open modeling questions, assumptions, candidate models, experiment results, and decisions
- [ ] Keep this task list updated as work progresses

## 2. Data validation and minimal cleaning

- [x] Add a lightweight validation / cleaning helper
- [x] Check timestamp ordering, duplicates, 15-minute cadence, missing values
- [x] Print or log a short validation summary during training

## 3. Development split / leakage control

- [ ] Use January-May for development training
- [ ] Use June as the main holdout/backtest month
- [ ] Add a simple way to train with a date cutoff for development
- [ ] After model selection, retrain the final forecaster on all available history through July
- [ ] Keep final commands compatible with the scaffold

## 4. Forecasting baseline

Implement a simple seasonal baseline first.

Candidate logic:

1. Use the same quarter-hour from the previous week.
2. If that value is unavailable or invalid, fall back to the historical time-of-week median.

The time-of-week fallback uses historical observations for the same weekday and quarter-hour, making it robust to isolated telemetry gaps.

Tasks:
- [ ] Refactor current code to keep the current forecaster.
- [ ] Implement baseline
- [ ] Add focused tests for missing-lag fallback
- [ ] Evaluate on June

## 5. Forecast Model A — direct net-load forecast

Forecast `grid_net_kw` directly.

Candidate models:
1. Ridge regression
2. `HistGradientBoostingRegressor`

Potential features:

### Calendar
- quarter of day / time of day
- day of week
- weekend
- simple seasonal/month feature if useful

### Lag / recent load
- previous quarter(s)
- same time previous day
- same time previous week
- recent rolling mean / recent level

### Known future information
- `most_recent_load_factor_forecast`

Tasks:
- [ ] Implement common leakage-safe feature builder
- [ ] Implement Ridge model
- [ ] Implement HistGradientBoosting model
- [ ] Compare both against the seasonal baseline on June
- [ ] Keep extra complexity only if it provides measurable benefit

## 6. Forecast Model B — consumption / PV decomposition

Use:

`consumption = grid_net_kw + pv_production_kw`

Then:

`net_forecast = consumption_forecast - pv_forecast`

### Consumption forecast

Potential inputs:
- calendar features
- recent consumption
- previous-day lag
- previous-week lag
- rolling consumption statistics

### PV forecast

Use the Belgian PV load-factor forecast as the main forward signal and calibrate it to the site using recent realized PV.

Potential features:
- future `most_recent_load_factor_forecast`
- quarter of day
- month / seasonal information
- latest realized site PV
- recent PV lags
- recent site-PV / national-proxy scaling ratio
- recent PV forecast-error correction

Simple first candidate:

`PV_hat = alpha_recent * load_factor_forecast`

where `alpha_recent` is estimated robustly from recent daylight observations.

Alternative later:
- small Ridge or HistGradientBoosting PV model

Tasks:
- [ ] Implement simple PV calibration first
- [ ] Implement consumption model
- [ ] Recombine into net-load forecast
- [ ] Compare Model B against direct Model A
- [ ] Keep decomposition only if it materially helps

## 7. Forecast evaluation

Candidate metrics:
- MAE
- RMSE
- bias
- MAE by forecast horizon

Potential slices:
- high-load periods
- high-PV periods
- peak-risk periods

Tasks:
- [ ] Compare baseline, Model A, and Model B
- [ ] Select one final forecasting approach
- [ ] Record results and rationale in `MODELING_NOTES.md`

## 8. Optimization baseline formulation

Implement a transparent LP/MILP using Pyomo including:
- battery charging/discharging
- SoC evolution
- one-way efficiency
- charge/discharge power limits
- SoC limits
- grid import/export limits
- energy offtake cost
- injection revenue/cost
- battery degradation cost

Tasks:
- [ ] Implement physical constraints
- [ ] Implement energy cost
- [ ] Implement degradation cost
- [ ] Verify feasibility
- [ ] Add a few high-value optimizer tests

Avoid relying on harness clipping during normal operation.

## 9. Monthly peak-charge treatment

The optimizer sees only 33 hours, while the peak charge is monthly.

Use a small interchangeable peak-model class.

Candidate models:
- `TimeWeightedPeakModel`
- `AchievablePeakTargetModel`

### Time-weighted peak value

Scale the marginal peak cost by elapsed fraction of the month.

Tasks:
- [ ] Implement this first
- [ ] Evaluate June bill and peak behavior

### Dynamic achievable peak target

Use a coarse estimate of a monthly billed peak that should be realistically achievable with battery control.

Within the 33-hour optimization:
- peaks below the target receive little/no incremental peak penalty
- peaks above the target are penalized

Tasks:
- [ ] Add only if the simple time-weighted treatment leaves obvious value on the table
- [ ] Keep target estimation deliberately simple
- [ ] Record limitations in `MODELING_NOTES.md`

## 10. Unknown-price tail

The 33-hour horizon can extend beyond published day-ahead prices.

Candidate simple treatments:
- optimize only against known prices plus a terminal SoC/value condition
- use a simple historical/expected price for the unknown tail

Tasks:
- [ ] Implement one simple defensible treatment
- [ ] Check for end-of-horizon battery emptying/filling artifacts
- [ ] Keep the method easy to explain

## 11. Terminal battery value

- [ ] First run without special terminal treatment
- [ ] Inspect SoC behavior near the horizon boundary
- [ ] Add a simple terminal SoC target or terminal energy value only if needed

## 12. Economic evaluation

Compare at least:
1. no battery
2. scaffold baseline controller
3. improved controller

Candidate outputs:
- total bill
- total savings
- energy cost
- capacity charge
- degradation cost
- maximum offtake
- equivalent battery cycles
- clipping / violation counts
- runtime

Tasks:
- [ ] Produce a compact results table
- [ ] Identify where savings come from
- [ ] Separate forecast improvements from optimizer improvements where possible

## 13. Robustness checks

Test a few important cases:
- missing lag values
- missing PV telemetry
- negative electricity prices
- high load near grid import limit
- battery near min/max SoC
- candidate new monthly peak
- beginning vs end of month
- unknown-price horizon tail

- [ ] Add only high-value tests

## 14. Final model selection

- [ ] Choose forecasting model from June validation
- [ ] Choose peak-value model
- [ ] Choose unknown-price-tail treatment
- [ ] Confirm optimizer rarely relies on clipping
- [ ] Confirm runtime is comfortably within live-run limits
- [ ] Retrain final forecaster on all history through July
- [ ] Persist final model to `models/`

## 15. Interview preparation

- [ ] Rehearse the exact live-run flow using June as the analogue
- [ ] Verify clean checkout + dependency installation
- [ ] Verify model persistence/reload
- [ ] Verify no hidden local files are required
- [ ] Prepare presentation with placeholders for August holdout results

Be ready to explain:
- data-quality decisions
- temporal validation / leakage handling
- forecast model progression
- why the final model was selected
- battery objective and constraints
- monthly peak approximation
- unknown-price treatment
- economic value vs forecast accuracy
- limitations and next steps

## Suggested implementation order

1. Data validation / cleaning
2. Temporal development split
3. Seasonal forecast baseline
4. Direct Ridge / HistGradientBoosting forecast
5. Simple LP/MILP optimizer
6. Time-weighted monthly peak model
7. Economic backtest
8. PV/load decomposition model
9. Alternative peak-target model
10. Tail / terminal refinements
11. Final model selection and retraining
12. Interview rehearsal

Later items are optional if the simpler models already perform well.
