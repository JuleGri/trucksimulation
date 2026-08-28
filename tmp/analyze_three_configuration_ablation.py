from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


RESULTS = Path(
    r"C:\Users\Jule\Documents\Master\Masterthesis\trucksimulation\validation\results"
)
OUT = Path(__file__).resolve().parent / "analysis" / "three_configuration_ablation"

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


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    if len(x) > 1:
        half = float(stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x)))
    else:
        half = 0.0
    return mean, sd, mean - half, mean + half


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {}
    for label, folder in CONFIGS.items():
        path = RESULTS / folder / "mc_replications.csv"
        frame = pd.read_csv(path).set_index("seed")
        frames[label] = frame

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
        common = frames[left].index.intersection(frames[right].index)
        for metric in METRICS:
            differences = frames[left].loc[common, metric] - frames[right].loc[common, metric]
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
    summary.to_csv(OUT / "configuration_summary.csv", index=False)
    contrasts.to_csv(OUT / "paired_contrasts.csv", index=False)
    payload = {
        "design": "paired comparison over identical simulation seeds 42--51",
        "configuration_folders": CONFIGS,
        "summary": summary.to_dict(orient="records"),
        "paired_contrasts": contrasts.to_dict(orient="records"),
    }
    (OUT / "ablation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
