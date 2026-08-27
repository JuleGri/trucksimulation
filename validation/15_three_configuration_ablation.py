"""Paired three-level ablation of CTB ProSiT contextual expressiveness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO / "validation/results"

CONFIGS = {
    "no_rules": "prosit_no_rules_inductive_calibrated_vs_holdout_ci",
    "rules_only": "prosit_rules_only_workload_blind_inductive_calibrated_vs_holdout_ci",
    "rules_workload": "prosit_inductive_calibrated_vs_holdout_ci",
}
METRICS = [
    "case_turnaround_emd_min",
    "case_turnaround_sim_mean",
    "case_turnaround_sim_p90",
    "yard_service_time_emd_frequency_weighted_min",
    "yard_activity_rate_l1_error",
    "inter_arrival_emd_min",
    "gate_only_cases",
]
CONTRASTS = [
    ("rules_only", "no_rules"),
    ("rules_workload", "rules_only"),
    ("rules_workload", "no_rules"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS / "three_configuration_ablation",
    )
    return parser.parse_args()


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    sample = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    mean = float(sample.mean())
    sd = float(sample.std(ddof=1)) if len(sample) > 1 else 0.0
    half_width = (
        float(stats.t.ppf(0.975, len(sample) - 1) * sd / np.sqrt(len(sample)))
        if len(sample) > 1
        else 0.0
    )
    return mean, sd, mean - half_width, mean + half_width


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = {}
    for label, folder in CONFIGS.items():
        source = args.results / folder / "mc_replications.csv"
        frames[label] = pd.read_csv(source).set_index("seed")

    summary_rows = []
    for label, frame in frames.items():
        for metric in METRICS:
            mean, sd, lo, hi = mean_ci(frame[metric])
            summary_rows.append(
                {
                    "configuration": label,
                    "metric": metric,
                    "n": int(frame[metric].notna().sum()),
                    "mean": mean,
                    "sd": sd,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                }
            )

    contrast_rows = []
    for left, right in CONTRASTS:
        common_seeds = frames[left].index.intersection(frames[right].index)
        if len(common_seeds) != len(frames[left]) or len(common_seeds) != len(frames[right]):
            raise ValueError(f"Unpaired seed sets for {left} and {right}.")
        for metric in METRICS:
            differences = (
                frames[left].loc[common_seeds, metric]
                - frames[right].loc[common_seeds, metric]
            )
            mean, sd, lo, hi = mean_ci(differences)
            contrast_rows.append(
                {
                    "contrast": f"{left}_minus_{right}",
                    "metric": metric,
                    "n_paired_seeds": int(differences.notna().sum()),
                    "mean_difference": mean,
                    "sd_difference": sd,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "interval_excludes_zero": bool(lo > 0 or hi < 0),
                }
            )

    summary = pd.DataFrame(summary_rows)
    contrasts = pd.DataFrame(contrast_rows)
    summary.to_csv(args.output / "configuration_summary.csv", index=False)
    contrasts.to_csv(args.output / "paired_contrasts.csv", index=False)
    payload = {
        "design": "paired comparison over identical simulation seeds 42--51",
        "configuration_folders": CONFIGS,
        "summary": summary.to_dict(orient="records"),
        "paired_contrasts": contrasts.to_dict(orient="records"),
    }
    with (args.output / "ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
