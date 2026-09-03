#!/usr/bin/env python3
"""Compare the cleaned static-state sensitivity with visit-only ProSiT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "validation/results"
METRICS = (
    "case_turnaround_emd_min",
    "yard_service_time_emd_frequency_weighted_min",
    "yard_activity_rate_l1_error",
    "inter_arrival_emd_min",
    "gate_only_cases",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visit-only",
        type=Path,
        default=RESULTS
        / "feature_factorial_20260829_visit_only_common_vs_holdout_ci"
        / "mc_replications.csv",
    )
    parser.add_argument(
        "--clean-static",
        type=Path,
        default=RESULTS
        / "feature_clean_static_20260829_vs_holdout_ci"
        / "mc_replications.csv",
    )
    parser.add_argument(
        "--ngd",
        type=Path,
        default=RESULTS
        / "feature_clean_static_ngd_20260829"
        / "ngd_replications.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "feature_clean_static_sensitivity_20260829",
    )
    return parser.parse_args()


def mean_ci(values: pd.Series) -> tuple[int, float, float, float, float]:
    sample = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(sample) == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    mean = float(sample.mean())
    sd = float(sample.std(ddof=1)) if len(sample) > 1 else 0.0
    half = (
        float(stats.t.ppf(0.975, len(sample) - 1) * sd / np.sqrt(len(sample)))
        if len(sample) > 1
        else 0.0
    )
    return len(sample), mean, sd, mean - half, mean + half


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = {
        "visit_only": pd.read_csv(args.visit_only).set_index("seed"),
        "clean_static": pd.read_csv(args.clean_static).set_index("seed"),
    }
    if not frames["visit_only"].index.equals(frames["clean_static"].index):
        raise ValueError("Visit-only and clean-static seed sets/order differ")

    metrics = list(METRICS)
    if args.ngd.exists():
        ngd = pd.read_csv(args.ngd)
        for configuration, frame in frames.items():
            addition = ngd.loc[ngd["configuration"].eq(configuration)].set_index("seed")
            for metric in ("ngd_completion_order", "ngd_explicit_order"):
                frame[metric] = addition.loc[frame.index, metric]
        metrics.extend(["ngd_completion_order", "ngd_explicit_order"])

    summaries = []
    effects = []
    for configuration, frame in frames.items():
        for metric in metrics:
            n, mean, sd, lo, hi = mean_ci(frame[metric])
            summaries.append(
                {
                    "configuration": configuration,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "sd": sd,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                }
            )

    for metric in metrics:
        difference = frames["clean_static"][metric] - frames["visit_only"][metric]
        n, mean, sd, lo, hi = mean_ci(difference)
        effects.append(
            {
                "contrast": "clean_static_minus_visit_only",
                "metric": metric,
                "n_paired_seeds": n,
                "mean_difference": mean,
                "sd_difference": sd,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "interval_excludes_zero": bool(n and (lo > 0 or hi < 0)),
            }
        )

    summary = pd.DataFrame(summaries)
    effect = pd.DataFrame(effects)
    summary.to_csv(args.output / "configuration_summary.csv", index=False)
    effect.to_csv(args.output / "paired_effects.csv", index=False)
    manifest = {
        "schema": "ctb-clean-static-sensitivity-1",
        "classification": (
            "Post-hoc, pre-specified sensitivity analysis; not an independent "
            "model-selection test after inspection of the temporal hold-out."
        ),
        "design": "Paired seeds 42--51; clean-static minus visit-only.",
        "feature_policy": {
            "kept_static_state": [
                "rmg_demand", "vc_demand", "mt_demand",
                "rmg_utilization", "vc_utilization", "mt_utilization",
            ],
            "removed_redundant_or_derived": [
                "gate_demand", "gate_utilization", "target_demand",
                "target_utilization", "target_demand_bin",
                "target_utilization_bin", "target_rank", "target_rank_group",
            ],
            "native_dynamic_state": False,
        },
        "sources": {
            "visit_only": str(args.visit_only),
            "clean_static": str(args.clean_static),
            "ngd": str(args.ngd),
        },
    }
    with (args.output / "sensitivity_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(effect.to_string(index=False))


if __name__ == "__main__":
    main()
