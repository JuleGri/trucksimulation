# Validation pipeline

This folder implements the held-out validation and diagnostic audits for the
CTB simulation. It covers train/test rigor, distributional validation,
configuration ablation, temporal transfer, structural repair, bottleneck
analysis, and sensitivity of the transition-baseline choice.

## Purpose

The authoritative discovery scripts fit ProSiT parameters on the frozen
temporal training partition. The validation pipeline in this folder does six
things:

1. Split the real event log at **case level, chronologically** into a
   training portion (used for discovery) and a held-out testing portion
   (used only for validation).
2. Compare simulated event logs against the held-out real log using
   distributional metrics (Wasserstein/EMD, two-sample
   Kolmogorov–Smirnov, MAE/RMSE of summary statistics) at both the
   activity level and the case level.
3. Produce publication-ready CDF, box and QQ plots for the thesis.
4. Enforce the CTB case contract: exactly one Gate In, at least one Yard
   activity, exactly one Gate Out, and strictly sequential simulated events.
5. Separate observed train-to-test change from simulation mismatch and compare
   the three contextual-expressiveness levels with paired seeds.
6. Audit bottleneck candidates and a precision-oriented Petri-net repair
   without silently replacing the automatic discovery result.

An additional script performs a sensitivity analysis on the 5th
percentile transition baseline used for enabled-timestamp
reconstruction.

## Scripts

| Order | Script | What it does |
|-------|--------|--------------|
| 1 | `01_train_test_split.py` | Case-level temporal 80/20 split of the s6 event log. Writes `s6_train.csv`, `s6_test.csv` and a `split_manifest.json` recording cutoff, coverage and per-activity counts. |
| 2 | `02_validate_simulation.py` | Compares a simulated log against the held-out real log. Reports raw and robust (up to 24 h) EMD/KS side by side. Service-time fidelity is additionally exported for yard activities only: per activity, as an unweighted mean and weighted by real held-out activity frequency; Gate In/Out are excluded. |
| 3 | `03_validation_plots.py` | Generates CDF overlays, box plots and QQ plots per activity + a case-level turnaround CDF. Reads the metrics file produced by step 2. |
| 4 | `04_baseline_percentile_sensitivity.py` | Recomputes the transition baseline at the 5th, 10th, 25th and 50th percentile and reports the impact on discovered waiting times and the data-aware duration R² / MAE. Addresses the "circular reasoning" concern. |
| 6 | `06_multi_seed_ci.py` | Repeats simulation for multiple seeds, reports raw/robust confidence intervals (including yard-only unweighted, real-frequency-weighted and per-activity EMD) and rejects any seed that violates the sequential case contract. |
| 8 | `08_validate_eventlog_contract.py` | Hard structural/causality gate for the real and simulated logs; also writes routing-rate and duration-outlier audits. |
| 9 | `09_multi_seed_scenarios.py` | Runs the final cap-3 effective baseline, T22 resource-pool reallocation and demand increase with matched seeds. It reports T22 effects separately for RMG receive and delivery work. `--no-rmg-cap` reproduces the freely discovered overlap capacities for diagnosis only. |
| 10 | `10_compare_rmg_capacity_sensitivity.py` | Joins the original and capped scenario replications by seed, reports the direct cap effect and the cap-by-scenario difference-in-differences with 95% paired intervals. |
| 11 | `11_receive_delivery_utilization_analysis.py` | Tests the association between target-area utilisation and RMG receive/delivery operational response time. Fits an adjusted model on the temporal training split, clusters uncertainty by calendar day, evaluates predictions on the holdout and checks whether the receive-delivery slope difference replicates. |
| 12 | `12_audit_prosit_context_rules.py` | Audits every rule in the frozen ProSiT pickle and separates execution-time, waiting-time, resource-selection and transition-routing effects. Reconstructs each transition's process prefix and lists all retained utilisation/demand splits without mutating the model. |
| 13 | `13_bottleneck_analysis.py` | Decomposes held-out turnaround into recorded service and composite pre-service delay, then ranks frequency-weighted accumulation points without treating them as causal queues. |
| 14 | `14_temporal_transfer_audit.py` | Compares the observed training and held-out periods: turnaround, yard-service distributions, case mix, target areas, demand and utilisation proxies. |
| 15 | `15_three_configuration_ablation.py` | Joins identical seeds for no-rules, workload-blind rules and rules+workload and reports paired incremental effects with 95% Student-t intervals. |
| 16 | `16_gate_only_structural_repair.py` | Exports a separate PNML and ProSiT bundle that removes only the silent gate-only bypass and compares before/after conformance and structure. |
| 17 | `17_compare_scenario_state_ablation.py` | Compares matched-seed T22 and demand responses between workload-blind rules and rules+workload, including a paired difference-in-differences. |
| 18 | `18_rmg_capacity_pressure_audit.py` | Tests whether the 20 percent demand intervention approaches the stated capacity of three concurrent RMG tasks per block using held-out offered load and minute-grid overlap diagnostics. |

### Layered validity audits

After the three ten-seed configuration runs exist, reproduce the additional
audits from the repository root with:

```powershell
python validation/13_bottleneck_analysis.py
python validation/14_temporal_transfer_audit.py
python validation/15_three_configuration_ablation.py
python validation/16_gate_only_structural_repair.py
python validation/17_compare_scenario_state_ablation.py
python validation/18_rmg_capacity_pressure_audit.py
```

The structural-repair pickle is a robustness model, not a replacement for the
primary baseline. It must undergo the same simulation and scenario validation
before it is used operationally. The temporal-transfer audit is descriptive and
never refits a model on the hold-out.

`run_full_validation.py` orchestrates the current core workflow (split
consistency check, order-preserving XES export, hard case-contract check,
steps 2 and 3, and multi-seed confidence intervals from step 6). It records
every executed command and duration in
`validation/results/<label>/pipeline_execution.json`.  The default split check
validates the immutable train/test snapshot actually used for discovery (case
disjointness and chronological arrivals) and refreshes its manifest without
replacing either CSV.  `--verify-against-full-log` additionally reproduces the
split from the current full log and stops if that later data state differs.
For Monte-Carlo replications, the orchestrator passes the simulation start
timestamp recorded by the discovery run explicitly to step 6, so a refreshed
manifest cannot silently change the experimental window.

The percentile-sensitivity script is retained as **descriptive evidence** for
the physical-transition component; it is not part of ProSiT parameter
discovery.  Run comparison and thesis-figure export are optional because they
require additional current result bundles.

### One-command validation

```powershell
python validation/run_full_validation.py `
    --label prosit_sequential_workload_vs_holdout `
    --n-seeds 10
```

Inspect the complete plan without writing files:

```powershell
python validation/run_full_validation.py --dry-run
```

Optional stages:

```powershell
python validation/run_full_validation.py `
    --include-sensitivity `
    --compare-run-a prosit_20260820_noworkload_vs_holdout `
    --include-thesis-figures
```

## Reproducing the validation run

```powershell
# From the trucksimulation/ workspace root
python validation/01_train_test_split.py --test-size 0.20 --seed 42
python baseline/07_run_prosit_discovery.py --enable-multitasking --write-xes
python validation/08_validate_eventlog_contract.py `
    --real data/processed/CTB/s6_test.csv `
    --sim baseline/discovery_params/params_20260816_214403_train80/prosit_discovery_workload_sequential/sim_baseline_train80_workload_sequential.csv `
    --label sequential_vs_holdout
python validation/02_validate_simulation.py `
    --real  data/processed/CTB/s6_test.csv `
    --sim   data/processed/CTB/prosit_simulations/what_if_T22_closed/baseline_reference_sim_log.csv `
    --label baseline_vs_holdout
python validation/03_validation_plots.py --run-label baseline_vs_holdout
python validation/04_baseline_percentile_sensitivity.py
```

All outputs are written under `validation/results/<run-label>/` so
different discovery configurations can be compared side by side.

### Final domain-constrained RMG baseline

The freely discovered source model represents the 22 RMG block identifiers
with an aggregate effective maximum concurrency of 100 because timestamp
overlaps yield per-block values of four or five. The physical upper bound is
three: CTB has three RMG cranes and three truck waiting lanes at a block. The
scenario workflow therefore deep-copies the discovery bundle, clamps every RMG
resource to at most three, and uses that effective bundle as the final baseline.
This does not overwrite or mutate the freely discovered source pickle:

```powershell
python validation/09_multi_seed_scenarios.py `
    --label prosit_sequential_calibrated_scenarios_rmg_cap3_ci `
    --rmg-max-concurrency 3 `
    --n-seeds 10 `
    --base-seed 42
```

The effective cap-3 baseline and both derived scenario bundles are serialized
inside the labelled result directory. `scenario_parameter_changes.json`
records the discovered and effective capacities, while
`scenario_run_summary.json` records both parameter hashes and the contract
status. The cap is an upper bound, not a claim that all three cranes serve
landside trucks simultaneously: actual waterside/landside allocation is absent
from the data and remains outside model scope.

For the diagnostic comparison only, reproduce the unconstrained overlap model
with `--no-rmg-cap`. After both runs are complete, generate the paired capacity
comparison and interaction analysis:

```powershell
python validation/10_compare_rmg_capacity_sensitivity.py
```

The comparison refuses to run if source-model hashes, seeds, trace counts,
start time, timestamp resolution or demand settings differ, or if either run
failed a case or duration contract.

### Receive/delivery utilisation and frozen-rule audit

Run the empirical holdout analysis and inspect the exact white-box rules with:

```powershell
python validation/11_receive_delivery_utilization_analysis.py
python validation/12_audit_prosit_context_rules.py
```

The first script interprets `ALP_ZEITPUNKT_BEREITMELDUNG` to completion as an
**operational response duration** containing both resource waiting and physical
handling, not as pure service time. Its utilisation feature is observational, so
the analysis can support conditional associations but not a causal prioritisation
claim. Utilisation bands, covariate scaling and category levels are learned from
the training period and reused unchanged for the temporal holdout.

The second script reads the frozen pickle and records its SHA-256 hash. A retained
utilisation split in a transition-weight tree is a data-aware routing choice at a
specific process prefix; it must not be described as an execution-duration or
physical-capacity rule. Outputs are written to
`validation/results/receive_delivery_utilisation/` and
`validation/results/prosit_context_rule_audit/`.

## Notes on framing

- The split is **temporal, not random**: the first 80 % of cases (by
  arrival timestamp) go to train, the last 20 % to test. This mirrors an
  operational forecasting setting and prevents leakage from
  aggregated features (utilization, demand) that were engineered from
  the whole log.
- Metrics on the held-out set are treated as the primary evidence in
  Chapter 4. Metrics on the full log are still reported for context but
  clearly labelled as *training* metrics.
- The comparison is distributional: ProSiT simulates a fresh time
  window, so a sequence-level comparison is not meaningful. EMD and KS
  compare marginal distributions of durations and inter-arrival times;
  MAE/RMSE compare summary statistics (mean, p50, p90) of the KPIs.
- Resource multitasking and case concurrency are different model layers.
  ProSiT may learn `max_concurrency > 1` for a crane/resource from overlapping
  work on different trucks. The control-flow net still carries only one token
  per truck case, so Yard activities inside that case execute sequentially.
- Observed timestamps are not altered to manufacture causality. The explicit
  `case:event:order` governs control flow; timestamp overlaps in the real log
  are reported separately. Generated simulation timestamps must be strictly
  causal and are a hard validation gate.
- ProSiT receives an explicit case-attribute allowlist. Auxiliary event-level
  fields such as `ocr_timestamp`, technical ordering columns and PM4Py helper
  attributes remain available in the canonical CSV when appropriate but are
  excluded from discovery/XES features. This prevents high-cardinality
  timestamps from being one-hot encoded as categorical routing predictors.
