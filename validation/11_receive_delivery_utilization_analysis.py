#!/usr/bin/env python3
"""Focused RMG receive-versus-delivery utilisation analysis.

The operational response duration is measured from
ALP_ZEITPUNKT_BEREITMELDUNG (the closest available operation-start proxy) to
completion.  It therefore includes resource waiting plus physical handling;
it is not interpreted as pure crane service time.

The explanatory model is fitted on the frozen temporal training split only.
Its predictive performance is evaluated on the untouched temporal holdout.
All utilisation bands and preprocessing parameters are learned from training
data and then reused for the holdout.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "processed" / "CTB" / "s6_train.csv"
DEFAULT_TEST = ROOT / "data" / "processed" / "CTB" / "s6_test.csv"
DEFAULT_RESULTS = ROOT / "validation" / "results"
FOCUS_ACTIVITIES = ("RMG_receive", "RMG_delivery")
ACTIVITY_LABELS = {"RMG_receive": "Receive", "RMG_delivery": "Delivery"}
UTILISATION_TERMS = ("utilisation_10pp", "delivery_x_utilisation_10pp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument(
        "--label", default="receive_delivery_utilisation", help="Result subdirectory name."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def _column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"None of the required columns exists: {candidates}")


def load_focus(path: Path, split: str) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(path, low_memory=False)
    activity_col = _column(raw, ("concept:name", "activity"))
    start_col = _column(raw, ("start:timestamp", "start_timestamp", "time:start_timestamp"))
    complete_col = _column(raw, ("time:timestamp", "complete_timestamp"))

    required = {
        "target_utilization",
        "target_demand",
        "target_area",
        "visit_complexity",
        "n_containers",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")

    focus = raw.loc[raw[activity_col].isin(FOCUS_ACTIVITIES)].copy()
    start = pd.to_datetime(focus[start_col], errors="coerce", utc=True)
    complete = pd.to_datetime(focus[complete_col], errors="coerce", utc=True)
    focus["operational_response_min"] = (complete - start).dt.total_seconds() / 60.0
    focus["activity"] = focus[activity_col].astype(str)
    focus["event_hour"] = start.dt.hour
    focus["event_weekday"] = start.dt.weekday
    focus["event_date"] = start.dt.strftime("%Y-%m-%d")
    focus["split"] = split

    numeric_columns = (
        "target_utilization",
        "target_demand",
        "visit_complexity",
        "n_containers",
        "operational_response_min",
        "event_hour",
        "event_weekday",
    )
    for column in numeric_columns:
        focus[column] = pd.to_numeric(focus[column], errors="coerce")

    counts = {
        "input_rows": int(len(raw)),
        "focus_rows": int(len(focus)),
        "missing_or_nonfinite_required": 0,
        "negative_response_duration": int((focus["operational_response_min"] < 0).sum()),
    }
    finite_required = [
        "operational_response_min",
        "target_utilization",
        "target_demand",
        "visit_complexity",
        "n_containers",
        "event_hour",
        "event_weekday",
    ]
    mask = np.isfinite(focus[finite_required].to_numpy(dtype=float)).all(axis=1)
    mask &= focus["target_area"].notna()
    counts["missing_or_nonfinite_required"] = int((~mask).sum())
    mask &= focus["operational_response_min"].ge(0)
    focus = focus.loc[mask].copy()
    counts["analysis_rows"] = int(len(focus))
    return focus, counts


def train_quartile_edges(train: pd.DataFrame) -> np.ndarray:
    values = train["target_utilization"].to_numpy(dtype=float)
    edges = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    edges[0] = -np.inf
    edges[-1] = np.inf
    for index in range(1, len(edges) - 1):
        if edges[index] <= edges[index - 1]:
            edges[index] = np.nextafter(edges[index - 1], np.inf)
    return edges


def add_bands(df: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    result = df.copy()
    labels = ("Q1 low", "Q2", "Q3", "Q4 high")
    result["utilisation_band"] = pd.cut(
        result["target_utilization"], bins=edges, labels=labels, include_lowest=True
    )
    return result


def descriptive_table(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, activity, band), group in combined.groupby(
        ["split", "activity", "utilisation_band"], observed=False
    ):
        values = group["operational_response_min"].dropna().to_numpy(dtype=float)
        n = len(values)
        mean = float(np.mean(values)) if n else math.nan
        sem = float(stats.sem(values)) if n > 1 else math.nan
        calendar_days = int(group["event_date"].nunique()) if n else 0
        if calendar_days > 1:
            cluster_scores = group.groupby("event_date")["operational_response_min"].apply(
                lambda series: float((series - mean).sum())
            )
            cluster_variance = (
                calendar_days / (calendar_days - 1.0)
                * float(np.square(cluster_scores).sum())
                / (n * n)
            )
            cluster_sem = math.sqrt(max(cluster_variance, 0.0))
            cluster_critical = float(stats.t.ppf(0.975, calendar_days - 1))
        else:
            cluster_sem = math.nan
            cluster_critical = math.nan
        rows.append(
            {
                "split": split,
                "activity": activity,
                "operation_type": ACTIVITY_LABELS.get(activity, activity),
                "utilisation_band": str(band),
                "n": n,
                "n_calendar_days": calendar_days,
                "utilisation_mean": float(group["target_utilization"].mean()) if n else math.nan,
                "response_mean_min": mean,
                "response_median_min": float(np.median(values)) if n else math.nan,
                "response_p90_min": float(np.quantile(values, 0.90)) if n else math.nan,
                "response_std_min": float(np.std(values, ddof=1)) if n > 1 else math.nan,
                "naive_mean_ci95_low_min": mean - 1.96 * sem if n > 1 else math.nan,
                "naive_mean_ci95_high_min": mean + 1.96 * sem if n > 1 else math.nan,
                "day_cluster_mean_se_min": cluster_sem,
                "mean_ci95_low_min": mean - cluster_critical * cluster_sem,
                "mean_ci95_high_min": mean + cluster_critical * cluster_sem,
            }
        )
    return pd.DataFrame(rows)


def preprocessing_spec(train: pd.DataFrame) -> dict[str, Any]:
    continuous = ("target_demand", "visit_complexity", "n_containers")
    means = {column: float(train[column].mean()) for column in continuous}
    scales = {
        column: float(train[column].std(ddof=0)) if train[column].std(ddof=0) > 0 else 1.0
        for column in continuous
    }
    return {
        "utilisation_mean": float(train["target_utilization"].mean()),
        "continuous_means": means,
        "continuous_scales": scales,
        "hours": sorted(int(value) for value in train["event_hour"].unique()),
        "weekdays": sorted(int(value) for value in train["event_weekday"].unique()),
        "target_areas": sorted(str(value) for value in train["target_area"].unique()),
    }


def design_matrix(df: pd.DataFrame, spec: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    n = len(df)
    columns: list[np.ndarray] = [np.ones(n, dtype=float)]
    names = ["intercept"]

    delivery = (df["activity"] == "RMG_delivery").to_numpy(dtype=float)
    utilisation = (
        df["target_utilization"].to_numpy(dtype=float) - spec["utilisation_mean"]
    ) / 0.10
    columns.extend((delivery, utilisation, delivery * utilisation))
    names.extend(("delivery", "utilisation_10pp", "delivery_x_utilisation_10pp"))

    for column in ("target_demand", "visit_complexity", "n_containers"):
        standardized = (
            df[column].to_numpy(dtype=float) - spec["continuous_means"][column]
        ) / spec["continuous_scales"][column]
        columns.append(standardized)
        names.append(f"{column}_z")

    categorical = (
        ("event_hour", spec["hours"], "hour"),
        ("event_weekday", spec["weekdays"], "weekday"),
        ("target_area", spec["target_areas"], "target_area"),
    )
    for column, levels, prefix in categorical:
        for level in levels[1:]:
            if column == "target_area":
                values = (df[column].astype(str) == str(level)).to_numpy(dtype=float)
            else:
                values = (df[column] == level).to_numpy(dtype=float)
            columns.append(values)
            names.append(f"{prefix}={level}")

    return np.column_stack(columns), names


def fit_hc3(
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    clusters: pd.Series | np.ndarray | None = None,
) -> dict[str, Any]:
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residual = y - fitted
    xtx_inv = np.linalg.pinv(x.T @ x)
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    denominator = np.maximum(1.0 - leverage, 1e-8)
    adjusted_sq = np.square(residual / denominator)
    meat = x.T @ (x * adjusted_sq[:, None])
    hc3_covariance = xtx_inv @ meat @ xtx_inv
    hc3_standard_error = np.sqrt(np.maximum(np.diag(hc3_covariance), 0.0))
    hc3_df = max(int(len(y) - rank), 1)

    robust_method = "HC3"
    covariance = hc3_covariance
    degrees_freedom = hc3_df
    n_clusters = None
    if clusters is not None:
        cluster_values = pd.Series(clusters).reset_index(drop=True)
        if len(cluster_values) != len(y):
            raise ValueError("Cluster labels and design matrix have different lengths.")
        codes, unique = pd.factorize(cluster_values, sort=True)
        n_clusters = int(len(unique))
        if n_clusters < 2:
            raise ValueError("Cluster-robust covariance requires at least two clusters.")
        cluster_meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
        for code in range(n_clusters):
            mask = codes == code
            score = x[mask].T @ residual[mask]
            cluster_meat += np.outer(score, score)
        correction = (n_clusters / (n_clusters - 1.0)) * ((len(y) - 1.0) / (len(y) - rank))
        covariance = correction * (xtx_inv @ cluster_meat @ xtx_inv)
        degrees_freedom = n_clusters - 1
        robust_method = "calendar-day cluster CR1"

    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    critical = float(stats.t.ppf(0.975, degrees_freedom))
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = beta / standard_error
    p_values = 2.0 * stats.t.sf(np.abs(statistic), degrees_freedom)
    coefficients = pd.DataFrame(
        {
            "term": names,
            "estimate_min": beta,
            "robust_se_min": standard_error,
            "ci95_low_min": beta - critical * standard_error,
            "ci95_high_min": beta + critical * standard_error,
            "t": statistic,
            "p_value": p_values,
            "hc3_se_min": hc3_standard_error,
        }
    )
    return {
        "beta": beta,
        "covariance": covariance,
        "hc3_covariance": hc3_covariance,
        "fitted": fitted,
        "rank": int(rank),
        "df_resid": degrees_freedom,
        "hc3_df_resid": hc3_df,
        "robust_method": robust_method,
        "n_clusters": n_clusters,
        "coefficients": coefficients,
    }


def linear_contrast(
    fit: dict[str, Any], names: list[str], weights: dict[str, float], label: str
) -> dict[str, Any]:
    vector = np.zeros(len(names), dtype=float)
    for term, weight in weights.items():
        vector[names.index(term)] = weight
    estimate = float(vector @ fit["beta"])
    variance = float(vector @ fit["covariance"] @ vector)
    standard_error = math.sqrt(max(variance, 0.0))
    critical = float(stats.t.ppf(0.975, fit["df_resid"]))
    statistic = estimate / standard_error if standard_error else math.nan
    p_value = float(2.0 * stats.t.sf(abs(statistic), fit["df_resid"])) if standard_error else math.nan
    hc3_variance = float(vector @ fit["hc3_covariance"] @ vector)
    hc3_standard_error = math.sqrt(max(hc3_variance, 0.0))
    hc3_critical = float(stats.t.ppf(0.975, fit["hc3_df_resid"]))
    return {
        "contrast": label,
        "estimate_min": estimate,
        "robust_method": fit["robust_method"],
        "robust_se_min": standard_error,
        "ci95_low_min": estimate - critical * standard_error,
        "ci95_high_min": estimate + critical * standard_error,
        "p_value": p_value,
        "hc3_se_min": hc3_standard_error,
        "hc3_ci95_low_min": estimate - hc3_critical * hc3_standard_error,
        "hc3_ci95_high_min": estimate + hc3_critical * hc3_standard_error,
    }


def model_metrics(y: np.ndarray, prediction: np.ndarray, split: str, model: str) -> dict[str, Any]:
    error = y - prediction
    total = float(np.sum(np.square(y - np.mean(y))))
    return {
        "split": split,
        "model": model,
        "n": int(len(y)),
        "mae_min": float(np.mean(np.abs(error))),
        "rmse_min": float(np.sqrt(np.mean(np.square(error)))),
        "r2": 1.0 - float(np.sum(np.square(error))) / total if total > 0 else math.nan,
    }


def plot_descriptives(table: pd.DataFrame, output: Path) -> None:
    bands = ["Q1 low", "Q2", "Q3", "Q4 high"]
    colors = {"RMG_receive": "#1f77b4", "RMG_delivery": "#d95f02"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for axis, split in zip(axes, ("train", "test")):
        subset = table[table["split"] == split]
        for activity in FOCUS_ACTIVITIES:
            line = subset[subset["activity"] == activity].set_index("utilisation_band").reindex(bands)
            mean = line["response_mean_min"].to_numpy(dtype=float)
            lower = mean - line["mean_ci95_low_min"].to_numpy(dtype=float)
            upper = line["mean_ci95_high_min"].to_numpy(dtype=float) - mean
            axis.errorbar(
                range(len(bands)), mean, yerr=np.vstack((lower, upper)), marker="o",
                capsize=3, linewidth=1.7, color=colors[activity], label=ACTIVITY_LABELS[activity]
            )
        axis.set_title("Training period" if split == "train" else "Temporal holdout")
        axis.set_xticks(range(len(bands)), bands)
        axis.set_xlabel("Target-area utilisation band (training cut points)")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean operational response duration [min]")
    axes[1].legend(frameon=False)
    fig.suptitle("RMG receive versus delivery under target-area utilisation", y=1.01)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.results_root.resolve() / args.label
    output.mkdir(parents=True, exist_ok=True)

    train, train_counts = load_focus(args.train.resolve(), "train")
    test, test_counts = load_focus(args.test.resolve(), "test")
    edges = train_quartile_edges(train)
    train = add_bands(train, edges)
    test = add_bands(test, edges)
    combined = pd.concat((train, test), ignore_index=True)

    descriptives = descriptive_table(combined)
    descriptives.to_csv(output / "utilisation_band_descriptives.csv", index=False)
    plot_descriptives(descriptives, output / "receive_delivery_utilisation.png")

    spec = preprocessing_spec(train)
    x_train, names = design_matrix(train, spec)
    x_test, test_names = design_matrix(test, spec)
    if names != test_names:
        raise RuntimeError("Training and holdout design matrices are not aligned.")
    y_train = train["operational_response_min"].to_numpy(dtype=float)
    y_test = test["operational_response_min"].to_numpy(dtype=float)
    fit = fit_hc3(x_train, y_train, names, clusters=train["event_date"])
    fit["coefficients"].to_csv(output / "adjusted_train_model_coefficients.csv", index=False)

    contrasts = [
        linear_contrast(fit, names, {"utilisation_10pp": 1.0}, "receive: +10pp utilisation"),
        linear_contrast(
            fit, names,
            {"utilisation_10pp": 1.0, "delivery_x_utilisation_10pp": 1.0},
            "delivery: +10pp utilisation",
        ),
        linear_contrast(
            fit, names, {"delivery_x_utilisation_10pp": 1.0},
            "delivery minus receive utilisation slope",
        ),
    ]
    contrast_table = pd.DataFrame(contrasts)
    contrast_table.to_csv(output / "adjusted_utilisation_contrasts.csv", index=False)

    train_means = train.groupby("activity")["operational_response_min"].mean().to_dict()
    baseline_train = train["activity"].map(train_means).to_numpy(dtype=float)
    baseline_test = test["activity"].map(train_means).to_numpy(dtype=float)
    metrics = pd.DataFrame(
        [
            model_metrics(y_train, baseline_train, "train", "training activity mean"),
            model_metrics(y_train, fit["fitted"], "train", "adjusted utilisation model"),
            model_metrics(y_test, baseline_test, "test", "training activity mean"),
            model_metrics(y_test, x_test @ fit["beta"], "test", "adjusted utilisation model"),
        ]
    )
    metrics.to_csv(output / "temporal_holdout_metrics.csv", index=False)

    # A small, explicitly descriptive split-wise interaction model documents
    # whether the unadjusted direction is stable across the temporal boundary.
    split_rows: list[dict[str, Any]] = []
    for split, frame in (("train", train), ("test", test)):
        delivery = (frame["activity"] == "RMG_delivery").to_numpy(dtype=float)
        util = (frame["target_utilization"].to_numpy(dtype=float) - spec["utilisation_mean"]) / 0.10
        x_simple = np.column_stack((np.ones(len(frame)), delivery, util, delivery * util))
        simple_names = ["intercept", "delivery", *UTILISATION_TERMS]
        simple_fit = fit_hc3(
            x_simple,
            frame["operational_response_min"].to_numpy(dtype=float),
            simple_names,
            clusters=frame["event_date"],
        )
        for row in (
            linear_contrast(simple_fit, simple_names, {"utilisation_10pp": 1.0}, "receive slope"),
            linear_contrast(
                simple_fit, simple_names,
                {"utilisation_10pp": 1.0, "delivery_x_utilisation_10pp": 1.0},
                "delivery slope",
            ),
            linear_contrast(
                simple_fit, simple_names, {"delivery_x_utilisation_10pp": 1.0},
                "delivery minus receive slope",
            ),
        ):
            split_rows.append({"split": split, **row})
    split_interactions = pd.DataFrame(split_rows)
    split_interactions.to_csv(output / "descriptive_split_interactions.csv", index=False)

    differential = contrasts[2]
    if differential["ci95_low_min"] > 0:
        inference = "delivery response duration is more positively associated with utilisation than receive"
    elif differential["ci95_high_min"] < 0:
        inference = "delivery response duration is less positively associated with utilisation than receive"
    else:
        inference = "no statistically resolved receive-delivery difference in the utilisation slope"

    holdout_differential = next(
        row for row in split_rows
        if row["split"] == "test" and row["contrast"] == "delivery minus receive slope"
    )
    holdout_resolved = not (
        holdout_differential["ci95_low_min"] <= 0 <= holdout_differential["ci95_high_min"]
    )
    same_direction = np.sign(holdout_differential["estimate_min"]) == np.sign(differential["estimate_min"])
    if holdout_resolved and same_direction:
        stability = "the receive-delivery slope difference is directionally replicated in the temporal holdout"
    else:
        stability = "the receive-delivery slope difference is not statistically replicated in the temporal holdout"

    summary = {
        "analysis_scope": list(FOCUS_ACTIVITIES),
        "duration_semantics": (
            "ALP_ZEITPUNKT_BEREITMELDUNG-to-completion operational response duration; "
            "includes resource waiting and physical handling and is not pure service time"
        ),
        "utilisation_semantics": (
            "target-area contextual utilisation attached to the event log; not a continuously "
            "updated causal treatment variable"
        ),
        "train_source": str(args.train.resolve()),
        "test_source": str(args.test.resolve()),
        "row_audit": {"train": train_counts, "test": test_counts},
        "training_utilisation_quartile_edges": [
            None if not np.isfinite(value) else float(value) for value in edges
        ],
        "preprocessing": spec,
        "model_rank": fit["rank"],
        "model_columns": len(names),
        "robust_inference": {
            "method": fit["robust_method"],
            "calendar_day_clusters": fit["n_clusters"],
            "hc3_reported_as_sensitivity": True,
        },
        "adjusted_contrasts": contrasts,
        "training_interpretation": inference,
        "temporal_stability": stability,
        "causal_claim_permitted": False,
        "prioritisation_claim_permitted": False,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    test_metrics = metrics[metrics["split"] == "test"].set_index("model")
    report = f"""# Receive-versus-delivery utilisation analysis

## Scope and semantics

This analysis compares `RMG_receive` and `RMG_delivery`. The dependent variable is the
operational response duration from `ALP_ZEITPUNKT_BEREITMELDUNG` to completion. It
contains resource waiting and physical handling; it is **not** a pure crane service-time
measurement. `target_utilization` is an event-log context feature, not a randomized or
continuously updated treatment.

## Primary training-only model

The transparent OLS model was fitted only on the temporal training split. It adjusts for
target demand, visit complexity, container count, hour, weekday and target-area fixed
effects. Primary uncertainty is CR1-clustered by calendar day ({fit['n_clusters']} training
days); HC3 heteroskedasticity-robust intervals remain in the CSV as a sensitivity check.

| Contrast | Effect [min] | 95% CI | p |
|---|---:|---:|---:|
{chr(10).join(f"| {row['contrast']} | {row['estimate_min']:.3f} | [{row['ci95_low_min']:.3f}, {row['ci95_high_min']:.3f}] | {row['p_value']:.4g} |" for row in contrasts)}

Training-period result: **{inference}**. Each slope is the adjusted change associated
with a 10-percentage-point utilisation increase. Crucially, **{stability}**; the holdout
differential is {holdout_differential['estimate_min']:.3f} min per +10 pp (95% CI
[{holdout_differential['ci95_low_min']:.3f}, {holdout_differential['ci95_high_min']:.3f}]).

## Temporal holdout

| Model | Holdout MAE [min] | Holdout RMSE [min] | Holdout R2 |
|---|---:|---:|---:|
| Training activity mean | {test_metrics.loc['training activity mean', 'mae_min']:.3f} | {test_metrics.loc['training activity mean', 'rmse_min']:.3f} | {test_metrics.loc['training activity mean', 'r2']:.4f} |
| Adjusted utilisation model | {test_metrics.loc['adjusted utilisation model', 'mae_min']:.3f} | {test_metrics.loc['adjusted utilisation model', 'rmse_min']:.3f} | {test_metrics.loc['adjusted utilisation model', 'r2']:.4f} |

The descriptive split-wise interaction file must be used to assess whether the slope
direction survives the temporal boundary. Predictive improvement, statistical
association and operational causation are distinct questions.

## Defensible interpretation

This design can identify a conditional association in the observed event log. It cannot
by itself prove an operational receive-priority policy: assignment is observational,
the start timestamp is a proxy, and unobserved block state, job urgency or dispatch logic
may confound the relation. A causal prioritisation claim would require dispatch-rule
data, a quasi-experiment or a controlled simulation intervention whose rule is explicitly
changed.
"""
    (output / "analysis_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote focused utilisation analysis to: {output}")


if __name__ == "__main__":
    main()
