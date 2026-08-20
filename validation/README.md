# Validation pipeline

This folder implements the held-out validation stage that was missing from
the baseline discovery pipeline. It is the response to the Tier 1 review
items on train/test rigor, distributional validation and sensitivity of
the transition-baseline choice.

## Purpose

The scripts under `baseline/` fit ProSiT-style simulation parameters
directly on the full CTB event log
(`data/processed/CTB/s6_eventlog_target_rank_features.csv`). The
validation pipeline in this folder does three things:

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

An additional script performs a sensitivity analysis on the 10th
percentile transition baseline used for enabled-timestamp
reconstruction.

## Scripts

| Order | Script | What it does |
|-------|--------|--------------|
| 1 | `01_train_test_split.py` | Case-level temporal 80/20 split of the s6 event log. Writes `s6_train.csv`, `s6_test.csv` and a `split_manifest.json` recording cutoff, coverage and per-activity counts. |
| 2 | `02_validate_simulation.py` | Compares a simulated log against the held-out real log. Reports raw and robust (up to 24 h) EMD/KS side by side, so long-tail failures are never silently discarded. |
| 3 | `03_validation_plots.py` | Generates CDF overlays, box plots and QQ plots per activity + a case-level turnaround CDF. Reads the metrics file produced by step 2. |
| 4 | `04_baseline_percentile_sensitivity.py` | Recomputes the transition baseline at the 5th, 10th, 25th and 50th percentile and reports the impact on discovered waiting times and the data-aware duration R² / MAE. Addresses the "circular reasoning" concern. |
| 6 | `06_multi_seed_ci.py` | Repeats simulation for multiple seeds, reports raw/robust confidence intervals and rejects any seed that violates the sequential case contract. |
| 8 | `08_validate_eventlog_contract.py` | Hard structural/causality gate for the real and simulated logs; also writes routing-rate and duration-outlier audits. |

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
