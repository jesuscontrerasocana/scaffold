

### Added

- AGENTS.md
- Validate quarter-hourly training data and report field-level missingness without altering raw measurements.
- Add persisted scaffold and weekly seasonal baselines, with the weekly model selected by default.
- Add forecast-only rolling evaluation with summary, per-lead, and forecast-versus-actual outputs saved beside the model by default.
- Add explicit, leakage-safe calendar, lag, and rolling time-series feature helpers.
- Add a persisted direct multi-horizon Ridge net-load forecaster with one model per lead.
- Align Ridge lead-minute metadata with the harness target-index convention.
- Add leakage-safe target-minus-one-day and target-minus-seven-day Ridge features.
- Remove unconditional progress output from forecast evaluation.
- Add a direct decomposed Ridge model that forecasts site load and non-negative PV separately.
- Expose decomposed load/PV forecasts in model output and forecast comparisons.
- Evaluate available net, load, and PV forecasts with aggregate and per-lead metrics.
- Add target time-of-day sine/cosine features to the decomposed PV Ridge model.
- Add a transparent Pyomo/HiGHS battery optimizer with physical, grid, energy-cost, and degradation constraints.
- Reuse a fixed-size Pyomo optimizer model with mutable forecast, price, and initial-energy inputs.
- Use a temporary MILP to prevent simultaneous battery charging and discharging only when published injection prices are negative.
- Allow negative-price exclusivity to be disabled so all solves use the reusable LP.
- Add deterministic incremental monthly peak-capacity cost for current-month horizon steps.
- Fill unpublished optimizer prices from the nearest historical same-type calendar day.
- Limit decomposed Ridge inference to required recent and seasonal history observations.
- Add a selectable decomposed Ridge-load and histogram-gradient-boosted PV forecaster.
- Precompute original-feature Ridge weights and intercepts for forecast inference.
- Reuse APPSI HiGHS solver instances for continuous and temporary mixed-integer optimizer solves.
- Add terminal energy constraint.
- Add rolling-origin selection across forecast models and calendar lookbacks.
- Add an opt-in soft penalty when the committed import exceeds a safe grid threshold.
