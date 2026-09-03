#!/usr/bin/env python3
"""Analyse the paired 2x2 CTB feature-source factorial experiment.

Visit/process attributes are present in every cell. Factor S toggles the
manually engineered demand/utilisation proxies; factor N toggles ProSiT's
native dynamic workload and queue-length features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "validation/results"
DEFAULT_OUTPUT = RESULTS / "feature_source_factorial_standardized_20260829"

DEFAULT_FOLDERS = {
    "visit_only": "feature_factorial_20260829_visit_only_common_vs_holdout_ci",
    "static_only": "feature_factorial_20260829_static_only_vs_holdout_ci",
    "native_only": "feature_factorial_20260829_native_only_vs_holdout_ci",
    "both": "feature_factorial_20260829_both_common_vs_holdout_ci",
}

METRICS = (
    "case_turnaround_emd_min",
    "case_turnaround_sim_mean",
    "case_turnaround_sim_p90",
    "yard_service_time_emd_frequency_weighted_min",
    "mean_waiting_time_emd_min",
    "yard_activity_rate_l1_error",
    "inter_arrival_emd_min",
    "gate_only_cases",
)

EFFECTS = {
    "static_effect_native_off": {"static_only": 1, "visit_only": -1},
    "static_effect_native_on": {"both": 1, "native_only": -1},
    "native_effect_static_off": {"native_only": 1, "visit_only": -1},
    "native_effect_static_on": {"both": 1, "static_only": -1},
    "static_by_native_interaction": {
        "both": 1,
        "static_only": -1,
        "native_only": -1,
        "visit_only": 1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--visit-only", default=DEFAULT_FOLDERS["visit_only"])
    parser.add_argument("--static-only", default=DEFAULT_FOLDERS["static_only"])
    parser.add_argument("--native-only", default=DEFAULT_FOLDERS["native_only"])
    parser.add_argument("--both", default=DEFAULT_FOLDERS["both"])
    parser.add_argument(
        "--ngd-replications",
        type=Path,
        default=RESULTS / "feature_source_factorial_standardized_ngd_20260829/ngd_replications.csv",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def mean_ci(values: pd.Series) -> tuple[int, float, float, float, float]:
    sample = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(sample) == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    mean = float(sample.mean())
    sd = float(sample.std(ddof=1)) if len(sample) > 1 else 0.0
    half = (
        float(stats.t.ppf(0.975, len(sample) - 1) * sd / np.sqrt(len(sample)))
        if len(sample) > 1 else 0.0
    )
    return len(sample), mean, sd, mean - half, mean + half


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    folders = {
        "visit_only": args.visit_only,
        "static_only": args.static_only,
        "native_only": args.native_only,
        "both": args.both,
    }
    frames = {
        name: pd.read_csv(args.results / folder / "mc_replications.csv").set_index("seed")
        for name, folder in folders.items()
    }
    seeds = frames["visit_only"].index
    for name, frame in frames.items():
        if not seeds.equals(frame.index):
            raise ValueError(f"Seed set/order differs for {name}")

    if args.ngd_replications.exists():
        ngd = pd.read_csv(args.ngd_replications)
        for name in frames:
            addition = ngd.loc[ngd["configuration"].eq(name)].set_index("seed")
            for metric in ("ngd_completion_order", "ngd_explicit_order"):
                frames[name][metric] = addition.loc[frames[name].index, metric]

    metrics = list(METRICS)
    if "ngd_completion_order" in frames["visit_only"].columns:
        metrics.extend(["ngd_completion_order", "ngd_explicit_order"])

    summary_rows = []
    for configuration, frame in frames.items():
        for metric in metrics:
            n, mean, sd, lo, hi = mean_ci(frame[metric])
            summary_rows.append({
                "configuration": configuration,
                "static_proxies": configuration in {"static_only", "both"},
                "native_dynamic": configuration in {"native_only", "both"},
                "metric": metric,
                "n": n,
                "mean": mean,
                "sd": sd,
                "ci95_lo": lo,
                "ci95_hi": hi,
            })

    effect_rows = []
    seed_rows = []
    for effect, coefficients in EFFECTS.items():
        for metric in metrics:
            values = sum(
                coefficient * frames[configuration][metric]
                for configuration, coefficient in coefficients.items()
            )
            n, mean, sd, lo, hi = mean_ci(values)
            effect_rows.append({
                "effect": effect,
                "metric": metric,
                "n_paired_seeds": n,
                "mean_effect": mean,
                "sd_effect": sd,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "interval_excludes_zero": bool(lo > 0 or hi < 0),
            })
            seed_rows.extend(
                {"effect": effect, "metric": metric, "seed": int(seed), "value": float(value)}
                for seed, value in values.items()
            )

    summary = pd.DataFrame(summary_rows)
    effects = pd.DataFrame(effect_rows)
    summary.to_csv(args.output / "factorial_configuration_summary.csv", index=False)
    effects.to_csv(args.output / "factorial_effects.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(args.output / "factorial_effects_by_seed.csv", index=False)
    with (args.output / "factorial_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "schema": "ctb-feature-source-factorial-1",
            "design": {
                "common_features": "visit/process, container, and location attributes",
                "static_factor": "manual demand, utilisation, and derived rank/bin proxies",
                "native_factor": "ProSiT workload and queue_length",
                "paired_seeds": [int(seed) for seed in seeds],
            },
            "configuration_folders": folders,
            "ngd_replications": str(args.ngd_replications),
            "interaction": "both - static_only - native_only + visit_only",
        }, handle, indent=2)
    print(effects.to_string(index=False))
    print(f"[factorial] Results written to {args.output}")


if __name__ == "__main__":
    main()
