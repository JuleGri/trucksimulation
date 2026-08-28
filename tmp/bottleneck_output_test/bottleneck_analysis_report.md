# Held-out bottleneck analysis

This analysis ranks where observed truck time accumulates. It does not equate
pre-service delay with queueing: the measure also includes movement,
traffic/transfer and dispatching that are not separately timestamped.

Mean held-out turnaround is 39.742 min. The
additive decomposition attributes 14.755
min to recorded handling service and
24.987 min to composite
pre-service delay. The maximum case-level reconstruction error is
0.000000 min.

The highest frequency-weighted total-burden activities are:
| concept:name | total_observed_burden_min | burden_share |
| --- | --- | --- |
| Gate Out | 201690 | 0.283642 |
| RMG_receive | 199013 | 0.279877 |
| RMG_delivery | 109857 | 0.154495 |
| LL_delivery | 61519 | 0.0865159 |
| HO2_receive | 44605 | 0.0627292 |

The highest composite pre-service burdens are:
| concept:name | total_pre_service_min | pre_service_mean_min | pre_service_p90_min |
| --- | --- | --- | --- |
| Gate Out | 201690 | 11.2726 | 21 |
| RMG_receive | 100205 | 13.9736 | 33 |
| RMG_delivery | 56829 | 12.1404 | 26 |
| HO2_delivery | 26416 | 18.2305 | 34 |
| LL_delivery | 21651 | 5.98425 | 8 |

These are bottleneck candidates, not causal diagnoses. A physical queueing claim
requires spatially sticky resource assignment and explicit movement/queue state.
