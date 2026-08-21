# Calibrated ProSiT validation analysis

Run date: 2026-08-21

## Scope

- Discovery parameters: `prosit_discovery_workload_sequential_calibrated/prosit_params.pkl`
- Real holdout: `data/processed/CTB/s6_test.csv`
- Single simulation: 17,892 cases
- Monte-Carlo validation: 10 replications, seeds 42 through 51, 17,892 cases per replication
- Calibration was estimated from the training split only; the holdout was not used for fitting.

## Main results

| Metric | Estimate |
|---|---:|
| Service-time EMD, mean across activities | 2.581 min (95% CI 2.447-2.715) |
| Case-turnaround EMD | 9.400 min (95% CI 9.296-9.504) |
| Inter-arrival EMD | 0.072 min (95% CI 0.067-0.077) |
| Simulated simultaneous-arrival share | 54.221% (95% CI 54.035%-54.407%) |
| Real holdout simultaneous-arrival share | 56.660% |
| Service events above 24 h | 0 in every replication |
| Turnaround cases above 24 h | 0 in every replication |

All ten replications produced zero:

- gate-only cases;
- cases with an invalid Gate In/Gate Out boundary;
- overlaps between activities within the same truck case;
- decreasing completion timestamps; and
- cases in which Gate Out precedes the final yard activity.

## Routing interpretation

The mean L1 error of yard-activity rates against the temporal holdout is 0.186
(95% CI 0.183-0.189). This is not primarily a failure of the calibration:
the empirical train-to-test L1 drift is already 0.178, whereas the calibrated
single simulation differs from the training rates by only 0.018. The largest
holdout shifts are:

- `LL_delivery`: 0.134 events/case in training versus 0.202 in the holdout;
- `RMG_receive`: 0.458 versus 0.401;
- `LL_receive`: 0.027 versus 0.008.

Accordingly, the remaining routing discrepancy is evidence of temporal
distribution shift/generalisation error. Tuning routing to the test split
would reduce the reported error but would invalidate the held-out evaluation.

## Critical qualifications

- Waiting-time EMD is intentionally unavailable. The real holdout has no
  model-derived `enabled:timestamp`, so a like-for-like waiting-time target
  cannot be calculated without inventing information.
- The turnaround EMD is stable across simulation seeds, but its approximately
  9.4-minute level is systematic rather than Monte-Carlo noise. The simulated
  mean turnaround is about 30.34 minutes. This should be discussed as residual
  duration/resource-model bias, not hidden by the strong process-contract
  results.
- Rare activities have volatile relative errors. For example, `LL_mixed`
  occurs only 11 times in the holdout; absolute rates and sample counts are
  more defensible than percentage error for such activities.
- These confidence intervals quantify simulation sampling uncertainty for one
  discovered model. They do not quantify uncertainty from repeating discovery
  on different training samples.

## Reproducibility

The full command sequence and resolved input paths are stored in
`pipeline_execution.json`. Per-seed values are in the sibling directory
`../prosit_sequential_calibrated_vs_holdout_ci/mc_replications.csv`, and the
aggregated confidence intervals are in `mc_summary.csv`.
