"""Compare scenario responses between workload-blind and workload-aware rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "validation/results"
DEFAULT_CONFIGS = {
    "rules_only": RESULTS / "revised_20260827_rules_only_scenarios_cap3_ci",
    "rules_workload": RESULTS / "final_deterministic_20260829_rules_workload_scenarios_cap3_ci",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules-only", type=Path, default=DEFAULT_CONFIGS["rules_only"])
    parser.add_argument(
        "--rules-workload", type=Path, default=DEFAULT_CONFIGS["rules_workload"]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "final_deterministic_20260829_scenario_state_ablation",
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
    folders = {"rules_only": args.rules_only, "rules_workload": args.rules_workload}
    frames = {
        label: pd.read_csv(folder / "scenario_paired_deltas.csv").set_index(
            ["seed", "scenario"]
        )
        for label, folder in folders.items()
    }
    common = frames["rules_only"].index.intersection(frames["rules_workload"].index)
    if len(common) != len(frames["rules_only"]) or len(common) != len(frames["rules_workload"]):
        raise ValueError("Scenario and seed sets are not identical across configurations.")

    metrics = sorted(
        column
        for column in frames["rules_only"].columns
        if column.startswith("delta_") and not column.startswith("delta_pct_")
    )
    response_rows = []
    difference_rows = []
    scenarios = sorted({scenario for _, scenario in common})
    for scenario in scenarios:
        scenario_index = [index for index in common if index[1] == scenario]
        for metric in metrics:
            for label, frame in frames.items():
                mean, sd, lo, hi = mean_ci(frame.loc[scenario_index, metric])
                response_rows.append(
                    {
                        "scenario": scenario,
                        "metric": metric.removeprefix("delta_"),
                        "configuration": label,
                        "n_paired_seeds": len(scenario_index),
                        "mean_scenario_delta": mean,
                        "sd_scenario_delta": sd,
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                        "interval_excludes_zero": bool(lo > 0 or hi < 0),
                    }
                )

            difference = (
                frames["rules_workload"].loc[scenario_index, metric]
                - frames["rules_only"].loc[scenario_index, metric]
            )
            mean, sd, lo, hi = mean_ci(difference)
            difference_rows.append(
                {
                    "scenario": scenario,
                    "metric": metric.removeprefix("delta_"),
                    "contrast": "rules_workload_response_minus_rules_only_response",
                    "n_paired_seeds": len(scenario_index),
                    "difference_in_differences": mean,
                    "sd_difference": sd,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "interval_excludes_zero": bool(lo > 0 or hi < 0),
                }
            )

    responses = pd.DataFrame(response_rows)
    differences = pd.DataFrame(difference_rows)
    responses.to_csv(args.output / "scenario_response_by_configuration.csv", index=False)
    differences.to_csv(args.output / "scenario_response_difference_in_differences.csv", index=False)
    payload = {
        "design": (
            "matched-seed comparison of each intervention response between the "
            "workload-blind rules and rules+workload parameter bundles"
        ),
        "configuration_folders": {key: str(value) for key, value in folders.items()},
        "responses": responses.to_dict(orient="records"),
        "difference_in_differences": differences.to_dict(orient="records"),
    }
    with (args.output / "scenario_state_ablation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)

    headline = differences[
        differences["metric"].isin(
            [
                "mean_turnaround_min",
                "mean_rmg_service_min",
                "mean_rmg_pre_service_min",
                "mean_rmg_receive_service_min",
                "mean_rmg_delivery_service_min",
            ]
        )
    ]
    print(headline.to_string(index=False))


if __name__ == "__main__":
    main()
