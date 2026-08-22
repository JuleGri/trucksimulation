# Receive-versus-delivery utilisation analysis

## Scope and semantics

This analysis compares `RMG_receive` and `RMG_delivery`. The dependent variable is the
operational response duration from `ALP_ZEITPUNKT_BEREITMELDUNG` to completion. It
contains resource waiting and physical handling; it is **not** a pure crane service-time
measurement. `target_utilization` is an event-log context feature, not a randomized or
continuously updated treatment.

## Primary training-only model

The transparent OLS model was fitted only on the temporal training split. It adjusts for
target demand, visit complexity, container count, hour, weekday and target-area fixed
effects. Primary uncertainty is CR1-clustered by calendar day (41 training
days); HC3 heteroskedasticity-robust intervals remain in the CSV as a sensitivity check.

| Contrast | Effect [min] | 95% CI | p |
|---|---:|---:|---:|
| receive: +10pp utilisation | 0.417 | [0.038, 0.797] | 0.03182 |
| delivery: +10pp utilisation | -0.273 | [-0.570, 0.025] | 0.07113 |
| delivery minus receive utilisation slope | -0.690 | [-0.920, -0.460] | 3.981e-07 |

Training-period result: **delivery response duration is less positively associated with utilisation than receive**. Each slope is the adjusted change associated
with a 10-percentage-point utilisation increase. Crucially, **the receive-delivery slope difference is not statistically replicated in the temporal holdout**; the holdout
differential is -0.144 min per +10 pp (95% CI
[-0.598, 0.310]).

## Temporal holdout

| Model | Holdout MAE [min] | Holdout RMSE [min] | Holdout R2 |
|---|---:|---:|---:|
| Training activity mean | 7.600 | 12.825 | -0.0287 |
| Adjusted utilisation model | 7.192 | 12.498 | 0.0230 |

The descriptive split-wise interaction file must be used to assess whether the slope
direction survives the temporal boundary. Predictive improvement, statistical
association and operational causation are distinct questions.

## Defensible interpretation

This design can identify a conditional association in the observed event log. It cannot
by itself prove an operational receive-priority policy: assignment is observational,
the start timestamp is a proxy, and unobserved block state, job urgency or dispatch logic
may confound the relation. A causal prioritisation claim would require dispatch-rule
data, a quasi-experiment or a controlled simulation intervention whose rule is explicitly
changed.
