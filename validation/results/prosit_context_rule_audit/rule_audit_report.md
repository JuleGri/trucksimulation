# Frozen ProSiT context-rule audit

Source SHA-256: `11a982f4cb36de57648e4eb729145b7a9726459e8f190220d6d51cdece075698`

## Rule-layer inventory

| Layer | Models | Split nodes | Utilisation | Demand | Queue | Area |
|---|---:|---:|---:|---:|---:|---:|
| execution_time | 11 | 0 | 0 | 0 | 0 | 0 |
| waiting_time | 26 | 3 | 0 | 0 | 3 | 0 |
| resource_selection | 26 | 22 | 0 | 0 | 0 | 22 |
| transition_routing | 144 | 144 | 6 | 4 | 0 | 14 |

## Focused operation-response models

| Activity | Root value [min] | Distribution | Training samples | Splits |
|---|---:|---|---:|---:|
| RMG_receive | 11.591 | lognorm | 32794 | 0 |
| HO2_receive | 16.468 | lognorm | 4188 | 0 |
| HO2_delivery | 10.117 | lognorm | 5775 | 0 |
| RMG_delivery | 8.992 | lognorm | 17823 | 0 |

The focused execution-time models are single-leaf distributions when their split count
is zero. In that case, ProSiT did not retain utilisation, demand, operation type, block or
another context feature for the activity's execution-duration parameter.

## Where utilisation or demand actually occurs

| Transition/model | Target label | Process prefix | Feature | Threshold |
|---|---|---|---|---:|
| yard_038 | LL_receive | RMG_delivery | mt_demand | 1.635 |
| gate_out_001 | Gate Out | HO2_mixed | mt_demand | 25.235 |
| yard_016 | RMG_receive | HO2_mixed | vc_utilization | 0.549 |
| yard_015 | RMG_delivery | HO2_mixed | target_utilization | 0.255 |
| yard_055 | RMG_receive | LL_delivery > HO2_mixed | vc_utilization | 0.557 |
| gate_out_023 | Gate Out | LL_delivery > HO2_mixed | vc_utilization | 0.557 |
| yard_071 | HO2_receive | LL_delivery > HO2_receive > RMG_receive | vc_utilization | 0.464 |
| gate_out_053 | Gate Out | LL_delivery > HO2_receive > RMG_receive | vc_utilization | 0.464 |
| gate_out_029 | Gate Out | LL_receive > HO2_delivery | mt_demand | 27.975 |
| yard_060 | RMG_receive | LL_receive > HO2_delivery | mt_demand | 27.975 |

These rows must be interpreted according to their layer. In particular, a split in
`transition_routing` changes the relative next-transition weight at a specific process
prefix. It does **not** change the ready-to-completion duration, prove that a crane
prioritises receive jobs, or establish a physical capacity response.

### Focused context-rule leaf paths

| Model | Target | Prefix | Complete path | Relative leaf weight |
|---|---|---|---|---:|
| yard_016 | RMG_receive | HO2_mixed | primary_target_area = VC > 0.5 | 5.74142e-09 |
| yard_016 | RMG_receive | HO2_mixed | primary_target_area = VC <= 0.5 AND vc_utilization <= 0.549 | 5.74142 |
| yard_016 | RMG_receive | HO2_mixed | primary_target_area = VC <= 0.5 AND vc_utilization > 0.549 | 5.74142e-09 |
| yard_015 | RMG_delivery | HO2_mixed | target_utilization <= 0.255 | 4.0534 |
| yard_015 | RMG_delivery | HO2_mixed | target_utilization > 0.255 | 4.0534e-09 |
| yard_055 | RMG_receive | LL_delivery > HO2_mixed | vc_utilization <= 0.557 | 1e-09 |
| yard_055 | RMG_receive | LL_delivery > HO2_mixed | vc_utilization > 0.557 | 1 |
| yard_071 | HO2_receive | LL_delivery > HO2_receive > RMG_receive | vc_utilization <= 0.464 | 1e-09 |
| yard_071 | HO2_receive | LL_delivery > HO2_receive > RMG_receive | vc_utilization > 0.464 | 1 |
| yard_060 | RMG_receive | LL_receive > HO2_delivery | mt_demand <= 27.975 | 1e-09 |
| yard_060 | RMG_receive | LL_receive > HO2_delivery | mt_demand > 27.975 | 1 |

For transition routing, these are raw **relative weights**, not normalized probabilities.
The probability of a next step also depends on all other transitions enabled at that
prefix. ProSiT follows the `<=` branch when the stated condition is true.

## Defensible conclusion

The audit provides exact white-box provenance for every selected feature. Absence of a
feature from the final trees means only that the configured discovery procedure did not
retain it in that model layer; it is not proof that the operational effect does not exist.
The separate receive-versus-delivery analysis tests the observational association more
directly and keeps that evidence distinct from the frozen simulator's learned rules.
