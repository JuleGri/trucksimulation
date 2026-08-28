"""Audit temporal transfer between the CTB discovery and hold-out periods.

This diagnostic separates mismatch already present in the observed process
from simulator error.  It never uses the hold-out to refit the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[1]
CASE = "case:concept:name"
ACTIVITY = "concept:name"
START = "start:timestamp"
COMPLETE = "time:timestamp"

CATEGORICAL_CASE_ATTRIBUTES = ["process_flow_type", "primary_target_area"]
NUMERIC_CASE_ATTRIBUTES = [
    "n_stops",
    "n_deliveries",
    "n_receives",
    "gate_utilization",
    "rmg_utilization",
    "vc_utilization",
    "mt_utilization",
    "gate_demand",
    "rmg_demand",
    "vc_demand",
    "mt_demand",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=REPO / "data/processed/CTB/s6_train.csv",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=REPO / "data/processed/CTB/s6_test.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "validation/results/temporal_transfer_audit",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {CASE, ACTIVITY, START, COMPLETE}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing required columns: {sorted(missing)}")
    frame[START] = pd.to_datetime(frame[START], errors="coerce")
    frame[COMPLETE] = pd.to_datetime(frame[COMPLETE], errors="coerce")
    frame["service_time_min"] = (
        (frame[COMPLETE] - frame[START]).dt.total_seconds() / 60.0
    )
    return frame


def case_table(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(CASE, sort=False)
    result = grouped.first().reset_index()
    result["turnaround_min"] = (
        grouped[COMPLETE].max().to_numpy() - grouped[START].min().to_numpy()
    ).astype("timedelta64[s]").astype(float) / 60.0
    result["n_events_observed"] = grouped.size().to_numpy()
    return result


def mean_difference_ci(test: np.ndarray, train: np.ndarray) -> dict[str, float]:
    difference = float(np.mean(test) - np.mean(train))
    test_var = np.var(test, ddof=1) / len(test)
    train_var = np.var(train, ddof=1) / len(train)
    standard_error = np.sqrt(test_var + train_var)
    degrees = (test_var + train_var) ** 2 / (
        test_var**2 / (len(test) - 1) + train_var**2 / (len(train) - 1)
    )
    half_width = float(stats.t.ppf(0.975, degrees) * standard_error)
    return {
        "difference_test_minus_train": difference,
        "ci95_lo": difference - half_width,
        "ci95_hi": difference + half_width,
        "welch_df": float(degrees),
    }


def bootstrap_quantile_difference(
    test: np.ndarray,
    train: np.ndarray,
    quantile: float,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        differences[index] = np.quantile(
            rng.choice(test, size=len(test), replace=True), quantile
        ) - np.quantile(rng.choice(train, size=len(train), replace=True), quantile)
    return {
        "difference_test_minus_train": float(
            np.quantile(test, quantile) - np.quantile(train, quantile)
        ),
        "bootstrap_repetitions": repetitions,
        "ci95_lo": float(np.quantile(differences, 0.025)),
        "ci95_hi": float(np.quantile(differences, 0.975)),
    }


def categorical_total_variation(train: pd.Series, test: pd.Series) -> dict[str, float | str]:
    levels = sorted(set(train.dropna().astype(str)) | set(test.dropna().astype(str)))
    train_share = train.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0.0)
    test_share = test.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0.0)
    shift = test_share - train_share
    largest = shift.abs().idxmax()
    return {
        "total_variation_distance": float(0.5 * shift.abs().sum()),
        "largest_share_shift_level": str(largest),
        "largest_share_shift_pp": float(100.0 * shift.loc[largest]),
    }


def numeric_drift(train: pd.Series, test: pd.Series) -> dict[str, float]:
    train_values = pd.to_numeric(train, errors="coerce").dropna().to_numpy(float)
    test_values = pd.to_numeric(test, errors="coerce").dropna().to_numpy(float)
    pooled_sd = np.sqrt(
        (
            (len(train_values) - 1) * np.var(train_values, ddof=1)
            + (len(test_values) - 1) * np.var(test_values, ddof=1)
        )
        / (len(train_values) + len(test_values) - 2)
    )
    mean_shift = float(np.mean(test_values) - np.mean(train_values))
    return {
        "train_mean": float(np.mean(train_values)),
        "test_mean": float(np.mean(test_values)),
        "mean_shift_test_minus_train": mean_shift,
        "standardized_mean_shift": mean_shift / pooled_sd if pooled_sd > 0 else 0.0,
        "wasserstein_distance": float(
            stats.wasserstein_distance(train_values, test_values)
        ),
    }


def activity_service_drift(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for activity in sorted(set(train[ACTIVITY]) | set(test[ACTIVITY])):
        if activity in {"Gate In", "Gate Out"}:
            continue
        train_values = train.loc[
            train[ACTIVITY].eq(activity), "service_time_min"
        ].dropna().to_numpy(float)
        test_values = test.loc[
            test[ACTIVITY].eq(activity), "service_time_min"
        ].dropna().to_numpy(float)
        rows.append(
            {
                "activity": activity,
                "train_n": len(train_values),
                "test_n": len(test_values),
                "train_mean_min": float(np.mean(train_values)),
                "test_mean_min": float(np.mean(test_values)),
                "mean_shift_min": float(np.mean(test_values) - np.mean(train_values)),
                "train_p90_min": float(np.quantile(train_values, 0.9)),
                "test_p90_min": float(np.quantile(test_values, 0.9)),
                "wasserstein_min": float(
                    stats.wasserstein_distance(train_values, test_values)
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["test_frequency_weight"] = result["test_n"] / result["test_n"].sum()
    return result


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = load(args.train)
    test = load(args.test)
    train_cases = case_table(train)
    test_cases = case_table(test)
    train_turnaround = train_cases["turnaround_min"].to_numpy(float)
    test_turnaround = test_cases["turnaround_min"].to_numpy(float)

    turnaround = {
        "train_n": len(train_turnaround),
        "test_n": len(test_turnaround),
        "train_mean_min": float(np.mean(train_turnaround)),
        "test_mean_min": float(np.mean(test_turnaround)),
        "train_median_min": float(np.median(train_turnaround)),
        "test_median_min": float(np.median(test_turnaround)),
        "train_p90_min": float(np.quantile(train_turnaround, 0.9)),
        "test_p90_min": float(np.quantile(test_turnaround, 0.9)),
        "wasserstein_min": float(
            stats.wasserstein_distance(train_turnaround, test_turnaround)
        ),
        "mean_shift": mean_difference_ci(test_turnaround, train_turnaround),
        "p90_shift": bootstrap_quantile_difference(
            test_turnaround,
            train_turnaround,
            0.9,
            args.bootstrap_repetitions,
            args.seed,
        ),
    }

    categorical_rows = []
    for attribute in CATEGORICAL_CASE_ATTRIBUTES:
        row = {"attribute": attribute}
        row.update(categorical_total_variation(train_cases[attribute], test_cases[attribute]))
        categorical_rows.append(row)
    categorical = pd.DataFrame(categorical_rows)

    numeric_rows = []
    for attribute in NUMERIC_CASE_ATTRIBUTES:
        row = {"attribute": attribute}
        row.update(numeric_drift(train_cases[attribute], test_cases[attribute]))
        numeric_rows.append(row)
    numeric = pd.DataFrame(numeric_rows)

    service = activity_service_drift(train, test)
    weighted_service_emd = float(
        np.average(service["wasserstein_min"], weights=service["test_n"])
    )
    weighted_service_mean_shift = float(
        np.average(service["mean_shift_min"], weights=service["test_n"])
    )

    categorical.to_csv(args.output / "categorical_case_drift.csv", index=False)
    numeric.to_csv(args.output / "numeric_case_drift.csv", index=False)
    service.to_csv(args.output / "activity_service_drift.csv", index=False)
    payload = {
        "design": "observed training period versus chronologically held-out period",
        "train": str(args.train),
        "test": str(args.test),
        "turnaround": turnaround,
        "yard_service_frequency_weighted_wasserstein_min": weighted_service_emd,
        "yard_service_frequency_weighted_mean_shift_min": weighted_service_mean_shift,
        "categorical_case_drift": categorical.to_dict(orient="records"),
        "numeric_case_drift": numeric.to_dict(orient="records"),
    }
    with (args.output / "temporal_transfer_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
