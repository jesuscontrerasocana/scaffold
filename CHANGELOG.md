

### Added

- AGENTS.md
- Validate quarter-hourly training data and report field-level missingness without altering raw measurements.
- Add persisted scaffold and weekly seasonal baselines, with the weekly model selected by default.
- Add forecast-only rolling evaluation with summary, per-lead, and forecast-versus-actual outputs saved beside the model by default.
- Add explicit, leakage-safe calendar, lag, and rolling time-series feature helpers.
- Add a persisted Ridge net-load forecaster with time-series-safe feature selection and inspectable metrics.
- Keep Ridge fit diagnostics training-only; recursive validation remains in the forecast evaluator.
