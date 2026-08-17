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

An additional script performs a sensitivity analysis on the 10th
percentile transition baseline used for enabled-timestamp
reconstruction.

## Scripts

| Order | Script | What it does |
|-------|--------|--------------|
| 1 | `01_train_test_split.py` | Case-level temporal 80/20 split of the s6 event log. Writes `s6_train.csv`, `s6_test.csv` and a `split_manifest.json` recording cutoff, coverage and per-activity counts. |
| 2 | `02_validate_simulation.py` | Compares a simulated log against the held-out real log. Computes per-activity EMD, KS statistic (with p-value), MAE/RMSE of summary stats for service times, waiting times, inter-arrival times and case turnaround times. |
| 3 | `03_validation_plots.py` | Generates CDF overlays, box plots and QQ plots per activity + a case-level turnaround CDF. Reads the metrics file produced by step 2. |
| 4 | `04_baseline_percentile_sensitivity.py` | Recomputes the transition baseline at the 5th, 10th, 25th and 50th percentile and reports the impact on discovered waiting times and the data-aware duration R² / MAE. Addresses the "circular reasoning" concern. |

## Reproducing the validation run

```powershell
# From the trucksimulation/ workspace root
python validation/01_train_test_split.py --test-size 0.20 --seed 42
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
