

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
