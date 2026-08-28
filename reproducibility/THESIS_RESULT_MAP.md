# Thesis result map

| Manuscript claim | Configuration or test | Reproduction evidence |
|---|---|---|
| Contextual-expressiveness ablation | no rules → rules only → rules + workload | `thesis_results/historical/*/mc_replications.csv`; `thesis_results/ablation/` |
| Gate-only bypass and precision repair | discovered versus restricted Petri net | `thesis_results/structural_repair/` |
| Train-to-hold-out drift | chronological observed split | `thesis_results/temporal_transfer/` |
| Service and composite pre-service burden | held-out component decomposition | `thesis_results/bottleneck/` |
| T22 removal and +20% demand | workload-aware cap-three models | `models/params_*`; `expected_results/`; `thesis_results/scenarios/rules_workload/` |
| Scenario robustness to state abstraction | workload-aware versus workload-blind rules | `thesis_results/scenarios/rules_only/`; `thesis_results/scenario_state_ablation/` |
| RMG mean-capacity pressure | held-out block audit, capacity three | `thesis_results/capacity_pressure/` |

The notebook reruns the saved workload-aware scenario family exactly. Claims requiring the confidential CTB log are reconstructed from frozen per-seed or derived outputs and cannot be regenerated from raw terminal records in this shareable package.
