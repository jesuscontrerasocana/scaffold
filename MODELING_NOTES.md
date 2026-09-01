# MODELING_NOTES.md

Living notes for the forecasting and battery-dispatch assignment.

This file is for:

* modeling questions
* assumptions
* possible approaches
* experiment results
* decisions and their rationale
* unresolved issues
* ideas worth revisiting

It is not intended to describe final architecture.

---

# 1. Problem framing

Goal:

Forecast site behavior and dispatch the battery to minimize the site's total electricity bill.

Main economic components:

* energy offtake cost
* injection revenue/cost
* monthly peak capacity charge
* battery degradation/cycling cost

The controller operates every 15 minutes using a rolling horizon.

Only the first action of each optimization is executed before the problem is solved again.

---

# 2. Data observations

## General quality

Initial inspection suggests the dataset is generally clean.

Observed:

* timestamps are regular at 15-minute resolution
* no duplicate timestamps
* most columns are complete

Potential telemetry issue:

* 164 missing `pv_production_kw` observations
* these coincide with `grid_net_kw == 0`
* they occur in a small number of contiguous blocks
* likely telemetry outages rather than genuine zero consumption

Current inclination:

Treat these periods as unavailable observations rather than valid zero-load observations.

Avoid aggressive interpolation of long gaps.

## Price structure

Observed:

`offtake_price - injection_price` is piecewise constant rather than independently varying.

Two observed regimes:

* approximately €47.54/MWh
* approximately €61.29/MWh

Possible interpretation:

The two prices may share the same underlying market component with different fixed tariffs/adders.

For implementation, retain both supplied price columns rather than reconstructing one from the other.

---

# 3. Validation strategy

Important distinction:

The rolling simulation harness prevents prediction-time lookahead, but training on the entire historical dataset and then evaluating June would still make June in-sample.

Development validation should therefore use temporal splits.

Possible structure:

* train on data before June
* evaluate on June
* after methodology is selected, retrain on all available history through July
* final interview evaluation occurs on August

Need to decide how many validation periods are useful given the assignment time limit.

---

# 4. Forecasting

## Target

Primary forecast required by the optimizer:

`grid_net_kw`

Possible alternatives to investigate:

### Direct net-load forecast

Forecast `grid_net_kw` directly.

Advantages:

* simplest
* directly matches optimization input

Questions:

* how well does it capture PV-driven variation?
* does the PV forecast proxy materially improve it?

### Component forecast

Infer site consumption:

`consumption = grid_net_kw + pv_production_kw`

Forecast consumption and PV separately, then:

`net = consumption_forecast - pv_forecast`

Potential advantage:

The provided forward PV load-factor forecast may be easier to exploit explicitly.

Potential disadvantage:

More moving parts and additional forecast error.

Decision pending experimentation.

### Month peak load

The capacity charge is based on the maximum quarter-hourly offtake over the full month, while the optimizer only sees a 33-hour horizon. The short-horizon optimization therefore needs some approximation of the remaining month's peak risk.

#### Time-weighted peak value

Scale the marginal peak cost by how far through the month we are:

$$
\text{effective peak cost}
=
c_{\text{peak}}
\cdot
\frac{\text{elapsed time at end of horizon}}{\text{total time in month}}
$$

Early in the month, a candidate peak is penalized less because there is still substantial time for a higher peak to occur later. Near month-end, the full capacity charge becomes increasingly relevant.

This is intentionally a simple heuristic and does not account for the magnitude or distribution of future loads.

#### Dynamic achievable peak target

Estimate a monthly offtake peak that should be realistically achievable with the battery, \(P_{\text{target}}\).

Within the 33-hour optimization, increases in offtake below this target carry little or no incremental capacity cost, while peaks above it are penalized:

$$
\text{peak penalty}
=
c_{\text{peak}}
\max(P_{\text{candidate}} - P_{\text{target}}, 0)
$$

The target represents the expected **billed peak after battery control**, rather than the site's uncontrolled future maximum. It can be estimated from historical load behavior and realistic battery peak-shaving capability, and potentially updated as the month progresses.

This separates the long-term question — *what monthly peak is realistically achievable?* — from the short-term MPC problem of deciding how to operate the battery over the next 33 hours.


---

# 5. Forecast features / candidate signals

Potentially useful information:

* recent net load
* same time previous day
* same time previous week
* time of day
* day of week
* recent rolling statistics
* PV production history
* forward load-factor forecast

Need to decide the simplest model that provides adequate forecast quality.

Prefer measurable improvements over additional complexity.

---

# 6. Forecast horizon

The forecaster receives a full 33-hour horizon.

Forecast quality may differ substantially by horizon.

Possible evaluation:

* overall MAE
* short-horizon MAE
* day-ahead MAE
* bias
* errors during high-load periods

Need to determine which forecast errors matter most economically for battery dispatch.

---

# 7. Price availability

Day-ahead prices are not known across the entire 33-hour horizon at every decision time.

Known-price horizon varies during the day.

Unknown future prices arrive as `NaN`.

Open modeling question:

How should the optimizer value the unpriced tail?

Possible directions:

* ignore the unknown-price tail
* use an expected/historical price
* use a terminal battery value / terminal SoC target
* combine a simple tail-price estimate with a terminal condition

Important:

Do not accidentally treat unpublished prices as known.

---

# 8. Battery optimization

The optimizer needs to balance several competing objectives:

* energy arbitrage
* avoiding expensive exports when injection prices are negative
* peak shaving
* degradation cost
* preserving useful SoC for future periods

Battery:

* 430 kWh total capacity
* 5–95% SoC
* 387 kWh usable
* 200 kW charge
* 200 kW discharge
* 90% round-trip efficiency
* approximately €16.125/full equivalent cycle degradation cost

Grid:

* 100 kW offtake limit
* 150 kW injection limit

Important consequence:

Battery charging power exceeds the site's allowable grid offtake, so grid constraints can frequently bind.

---

# 9. Capacity charge

Monthly peak charge:

€4.25/kW applied to the maximum quarter-hourly monthly offtake.

This is path-dependent.

Once a peak occurs, reducing demand later cannot undo the charge.

The optimizer receives the already-realized monthly peak.

Questions to resolve:

* exact formulation of incremental peak cost
* whether forecast uncertainty around future peaks should affect aggressiveness
* how much arbitrage value justifies setting a new peak

Peak behavior is likely one of the most economically important parts of the assignment.

---

# 10. Forecast uncertainty

Current inclination:

Start deterministic.

Forecast uncertainty may matter particularly for:

* peak shaving
* preserving SoC
* avoiding grid-limit clipping

Potential simple robustness measures could be considered later if deterministic performance reveals a problem.

Avoid introducing stochastic optimization unless there is clear evidence it is worth the complexity.

---

# 11. Terminal behavior

Rolling-horizon optimization creates an end-of-horizon problem.

Without some representation of future battery value, the optimizer may empty or fill the battery unnaturally near the horizon boundary.

Need to investigate:

* terminal SoC target
* terminal energy value
* interaction with unknown future prices

Keep treatment simple and economically interpretable.

---

# 12. Evaluation

Need to evaluate both forecasting and economic performance.

Potential forecast metrics:

* MAE
* RMSE
* bias
* error by forecast horizon
* error during high-demand periods

Potential economic metrics:

* total site bill
* savings versus no battery
* savings versus scaffold baseline
* energy cost
* peak-capacity cost
* degradation cost
* maximum offtake
* battery cycles
* physical clipping counts

Need to select a compact final set suitable for the presentation.

---

# 13. Baselines

Useful comparisons may include:

* no battery
* provided baseline controller
* improved forecast + simple optimizer
* final controller

Potential value:

Helps separate where improvement comes from:

* forecasting
* peak-aware optimization
* arbitrage logic
* other modeling choices

---

# 14. Open questions

Maintain unresolved modeling questions here.

* What is the best simple forecasting formulation?
* Direct net-load forecast or load/PV decomposition?
* Which forecast metric correlates best with economic value?
* How should unknown future prices be handled?
* What terminal value/SoC treatment is appropriate?
* How aggressively should the controller protect against new monthly peaks?
* Is explicit forecast uncertainty worth introducing?
* What should be used as the main development validation month?
* Which results are most compelling for the interview presentation?

---

# 15. Decisions

Record decisions once made.

Format:

### Decision: <name>

**Choice:**
...

**Reason:**
...

**Evidence:**
...

**Alternatives considered:**
...

**Date:**
...

---

# 16. Experiment log

Record experiments briefly.

### Experiment: <name>

**Change:**
...

**Validation period:**
...

**Forecast result:**
...

**Economic result:**
...

**Conclusion:**
...
