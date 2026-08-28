from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(r"C:\Users\Jule\Documents\Master\Masterthesis\trucksimulation")
TRAIN = REPO / "data" / "processed" / "CTB" / "s6_train.csv"
TEST = REPO / "data" / "processed" / "CTB" / "s6_test.csv"
OUT = Path(__file__).resolve().parent / "analysis" / "temporal_transfer_audit"

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


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[START] = pd.to_datetime(df[START], errors="coerce")
    df[COMPLETE] = pd.to_datetime(df[COMPLETE], errors="coerce")
    df["service_time_min"] = (df[COMPLETE] - df[START]).dt.total_seconds() / 60.0
    return df


def case_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(CASE, sort=False)
    result = grouped.first().reset_index()
    result["turnaround_min"] = (
        grouped[COMPLETE].max().to_numpy() - grouped[START].min().to_numpy()
    ).astype("timedelta64[s]").astype(float) / 60.0
    result["n_events_observed"] = grouped.size().to_numpy()
    return result


def mean_difference_ci(test: np.ndarray, train: np.ndarray) -> dict:
    difference = float(np.mean(test) - np.mean(train))
    se = np.sqrt(np.var(test, ddof=1) / len(test) + np.var(train, ddof=1) / len(train))
    df_num = (np.var(test, ddof=1) / len(test) + np.var(train, ddof=1) / len(train)) ** 2
    df_den = (
        (np.var(test, ddof=1) / len(test)) ** 2 / (len(test) - 1)
        + (np.var(train, ddof=1) / len(train)) ** 2 / (len(train) - 1)
    )
    dof = df_num / df_den
    half = float(stats.t.ppf(0.975, dof) * se)
    return {
        "difference_test_minus_train": difference,
        "ci95_lo": difference - half,
        "ci95_hi": difference + half,
        "welch_df": float(dof),
    }


def bootstrap_quantile_difference(
    test: np.ndarray,
    train: np.ndarray,
    quantile: float,
    repetitions: int = 1000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    observed = float(np.quantile(test, quantile) - np.quantile(train, quantile))
    differences = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        test_draw = rng.choice(test, size=len(test), replace=True)
        train_draw = rng.choice(train, size=len(train), replace=True)
        differences[index] = np.quantile(test_draw, quantile) - np.quantile(
            train_draw, quantile
        )
    return {
        "difference_test_minus_train": observed,
        "bootstrap_repetitions": repetitions,
        "ci95_lo": float(np.quantile(differences, 0.025)),
        "ci95_hi": float(np.quantile(differences, 0.975)),
    }


def categorical_total_variation(train: pd.Series, test: pd.Series) -> dict:
    levels = sorted(set(train.dropna().astype(str)) | set(test.dropna().astype(str)))
    train_share = train.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0.0)
    test_share = test.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0.0)
    return {
        "total_variation_distance": float(0.5 * np.abs(test_share - train_share).sum()),
        "largest_share_shift_level": str((test_share - train_share).abs().idxmax()),
        "largest_share_shift_pp": float(100.0 * (test_share - train_share).loc[(test_share - train_share).abs().idxmax()]),
    }


def numeric_drift(train: pd.Series, test: pd.Series) -> dict:
    a = pd.to_numeric(train, errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(test, errors="coerce").dropna().to_numpy(float)
    pooled_sd = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    return {
        "train_mean": float(np.mean(a)),
        "test_mean": float(np.mean(b)),
        "mean_shift_test_minus_train": float(np.mean(b) - np.mean(a)),
        "standardized_mean_shift": float((np.mean(b) - np.mean(a)) / pooled_sd) if pooled_sd > 0 else 0.0,
        "wasserstein_distance": float(stats.wasserstein_distance(a, b)),
    }


def activity_service_drift(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for activity in sorted(set(train[ACTIVITY]) | set(test[ACTIVITY])):
        if activity in {"Gate In", "Gate Out"}:
            continue
        a = train.loc[train[ACTIVITY] == activity, "service_time_min"].dropna().to_numpy(float)
        b = test.loc[test[ACTIVITY] == activity, "service_time_min"].dropna().to_numpy(float)
        rows.append(
            {
                "activity": activity,
                "train_n": len(a),
                "test_n": len(b),
                "train_mean_min": float(np.mean(a)),
                "test_mean_min": float(np.mean(b)),
                "mean_shift_min": float(np.mean(b) - np.mean(a)),
                "train_p90_min": float(np.quantile(a, 0.9)),
                "test_p90_min": float(np.quantile(b, 0.9)),
                "wasserstein_min": float(stats.wasserstein_distance(a, b)),
            }
        )
    result = pd.DataFrame(rows)
    result["test_frequency_weight"] = result["test_n"] / result["test_n"].sum()
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = load(TRAIN)
    test = load(TEST)
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
        "wasserstein_min": float(stats.wasserstein_distance(train_turnaround, test_turnaround)),
        "mean_shift": mean_difference_ci(test_turnaround, train_turnaround),
        "p90_shift": bootstrap_quantile_difference(test_turnaround, train_turnaround, 0.9),
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

    categorical.to_csv(OUT / "categorical_case_drift.csv", index=False)
    numeric.to_csv(OUT / "numeric_case_drift.csv", index=False)
    service.to_csv(OUT / "activity_service_drift.csv", index=False)
    payload = {
        "design": "observed training period versus chronologically held-out period",
        "turnaround": turnaround,
        "yard_service_frequency_weighted_wasserstein_min": weighted_service_emd,
        "yard_service_frequency_weighted_mean_shift_min": weighted_service_mean_shift,
        "categorical_case_drift": categorical.to_dict(orient="records"),
        "numeric_case_drift": numeric.to_dict(orient="records"),
    }
    (OUT / "temporal_transfer_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
