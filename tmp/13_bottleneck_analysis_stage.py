"""Held-out CTB bottleneck analysis with explicit delay semantics.

The analysis distinguishes observed handling duration from the inter-activity
gap before an event.  The latter is deliberately called a composite
pre-service delay: it may contain physical movement, traffic/transfer,
dispatching, and queueing, so it must not be interpreted as a pure queue.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CASE = "case:concept:name"
ACTIVITY = "concept:name"
RESOURCE = "org:resource"
START = "start:timestamp"
COMPLETE = "time:timestamp"
ORDER = "case:event:order"
RMG_ACTIVITIES = {"RMG_receive", "RMG_delivery", "RMG_mixed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Held-out CTB bottleneck analysis.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/CTB/s6_test.csv"),
        help="Temporally held-out CTB event CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/results/bottleneck_analysis"),
    )
    parser.add_argument(
        "--rmg-cap",
        type=int,
        default=3,
        help="Physical comparison cap per RMG block; used only for an overlap audit.",
    )
    return parser.parse_args()


def prepare_events(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {CASE, ACTIVITY, RESOURCE, START, COMPLETE}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    for column in (START, COMPLETE):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame = frame.dropna(subset=[CASE, ACTIVITY, START, COMPLETE]).copy()
    sort_columns = [CASE, ORDER] if ORDER in frame.columns else [CASE, START, COMPLETE]
    frame = frame.sort_values(sort_columns, kind="stable")
    frame["previous_complete"] = frame.groupby(CASE)[COMPLETE].shift()
    frame["service_min"] = (
        (frame[COMPLETE] - frame[START]).dt.total_seconds().div(60).clip(lower=0)
    )
    frame["composite_pre_service_min"] = (
        (frame[START] - frame["previous_complete"])
        .dt.total_seconds()
        .div(60)
        .clip(lower=0)
        .fillna(0)
    )
    frame["observed_time_burden_min"] = (
        frame["service_min"] + frame["composite_pre_service_min"]
    )
    return frame


def activity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(ACTIVITY, as_index=False)
        .agg(
            n_events=(ACTIVITY, "size"),
            n_cases=(CASE, "nunique"),
            service_mean_min=("service_min", "mean"),
            service_median_min=("service_min", "median"),
            service_p90_min=("service_min", lambda values: values.quantile(0.90)),
            pre_service_mean_min=("composite_pre_service_min", "mean"),
            pre_service_median_min=("composite_pre_service_min", "median"),
            pre_service_p90_min=("composite_pre_service_min", lambda values: values.quantile(0.90)),
            total_service_min=("service_min", "sum"),
            total_pre_service_min=("composite_pre_service_min", "sum"),
            total_observed_burden_min=("observed_time_burden_min", "sum"),
        )
    )
    total = float(summary["total_observed_burden_min"].sum())
    summary["burden_share"] = summary["total_observed_burden_min"] / total if total else 0.0
    summary["service_burden_rank"] = summary["total_service_min"].rank(
        method="min", ascending=False
    ).astype(int)
    summary["pre_service_burden_rank"] = summary["total_pre_service_min"].rank(
        method="min", ascending=False
    ).astype(int)
    summary["total_burden_rank"] = summary["total_observed_burden_min"].rank(
        method="min", ascending=False
    ).astype(int)
    return summary.sort_values("total_burden_rank")


def resource_summary(frame: pd.DataFrame, rmg_cap: int) -> pd.DataFrame:
    timed = frame[frame[COMPLETE].gt(frame[START])].copy()
    timed["concurrency_at_start"] = 0
    for resource, index in timed.groupby(RESOURCE, sort=False).groups.items():
        group = timed.loc[index]
        starts = np.sort(group[START].astype("int64").to_numpy())
        ends = np.sort(group[COMPLETE].astype("int64").to_numpy())
        query = group[START].astype("int64").to_numpy()
        # All events starting at the same instant are included. Events ending
        # at that instant are no longer active.
        concurrency = np.searchsorted(starts, query, side="right") - np.searchsorted(
            ends, query, side="right"
        )
        timed.loc[index, "concurrency_at_start"] = concurrency

    timed["is_rmg"] = timed[ACTIVITY].isin(RMG_ACTIVITIES)
    timed["at_or_above_rmg_cap"] = timed["is_rmg"] & timed["concurrency_at_start"].ge(rmg_cap)
    timed["above_rmg_cap"] = timed["is_rmg"] & timed["concurrency_at_start"].gt(rmg_cap)
    return (
        timed.groupby(RESOURCE, as_index=False)
        .agg(
            n_events=(ACTIVITY, "size"),
            n_cases=(CASE, "nunique"),
            service_mean_min=("service_min", "mean"),
            pre_service_mean_min=("composite_pre_service_min", "mean"),
            total_service_min=("service_min", "sum"),
            total_pre_service_min=("composite_pre_service_min", "sum"),
            concurrency_at_start_p90=("concurrency_at_start", lambda values: values.quantile(0.90)),
            concurrency_at_start_max=("concurrency_at_start", "max"),
            starts_at_or_above_rmg_cap=("at_or_above_rmg_cap", "sum"),
            starts_above_rmg_cap=("above_rmg_cap", "sum"),
        )
        .sort_values(["total_pre_service_min", "total_service_min"], ascending=False)
    )


def case_decomposition(frame: pd.DataFrame) -> dict[str, float | int]:
    grouped = frame.groupby(CASE)
    turnaround = (grouped[COMPLETE].max() - grouped[START].min()).dt.total_seconds() / 60
    decomposed = grouped["observed_time_burden_min"].sum()
    error = (turnaround - decomposed).abs()
    return {
        "n_cases": int(grouped.ngroups),
        "turnaround_mean_min": float(turnaround.mean()),
        "turnaround_median_min": float(turnaround.median()),
        "service_contribution_mean_min": float(grouped["service_min"].sum().mean()),
        "composite_pre_service_contribution_mean_min": float(
            grouped["composite_pre_service_min"].sum().mean()
        ),
        "maximum_decomposition_error_min": float(error.max()),
    }


def plot_activity_burden(summary: pd.DataFrame, path: Path) -> None:
    plotted = summary.sort_values("total_observed_burden_min").copy()
    fig, axis = plt.subplots(figsize=(8.6, 5.4))
    axis.barh(plotted[ACTIVITY], plotted["total_service_min"], label="Handling service")
    axis.barh(
        plotted[ACTIVITY],
        plotted["total_pre_service_min"],
        left=plotted["total_service_min"],
        label="Composite pre-service delay",
    )
    axis.set_xlabel("Accumulated minutes in the held-out period")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.6g}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    events = prepare_events(args.input)
    activities = activity_summary(events)
    resources = resource_summary(events, args.rmg_cap)
    decomposition = case_decomposition(events)

    activities.to_csv(args.output / "activity_bottleneck_summary.csv", index=False)
    resources.to_csv(args.output / "resource_bottleneck_summary.csv", index=False)
    plot_activity_burden(activities, args.output / "activity_time_burden.png")

    top_total = activities.nsmallest(5, "total_burden_rank")
    top_wait = activities.nsmallest(5, "pre_service_burden_rank")
    summary = {
        "source": str(args.input),
        "temporal_role": "held_out_test_split",
        "delay_semantics": (
            "Composite pre-service delay is start time minus the previous event's "
            "completion within the same sequential truck case. It aggregates movement, "
            "traffic/transfer, dispatching and queueing; it is not a pure queue measure."
        ),
        "bottleneck_definition": (
            "A bottleneck candidate is an activity or resource at which a large amount "
            "of held-out case time accumulates. Frequency-weighted totals are primary; "
            "mean and p90 values describe severity. Causal attribution requires an "
            "intervention or a richer physical model."
        ),
        "rmg_cap_for_overlap_audit": args.rmg_cap,
        "case_decomposition": decomposition,
        "top_total_burden_activities": top_total[ACTIVITY].tolist(),
        "top_pre_service_burden_activities": top_wait[ACTIVITY].tolist(),
        "outputs": {
            "activity_summary": str(args.output / "activity_bottleneck_summary.csv"),
            "resource_summary": str(args.output / "resource_bottleneck_summary.csv"),
            "figure": str(args.output / "activity_time_burden.png"),
        },
    }
    with (args.output / "bottleneck_analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    report = f"""# Held-out bottleneck analysis

This analysis ranks where observed truck time accumulates. It does not equate
pre-service delay with queueing: the measure also includes movement,
traffic/transfer and dispatching that are not separately timestamped.

Mean held-out turnaround is {decomposition['turnaround_mean_min']:.3f} min. The
additive decomposition attributes {decomposition['service_contribution_mean_min']:.3f}
min to recorded handling service and
{decomposition['composite_pre_service_contribution_mean_min']:.3f} min to composite
pre-service delay. The maximum case-level reconstruction error is
{decomposition['maximum_decomposition_error_min']:.6f} min.

The highest frequency-weighted total-burden activities are:
{markdown_table(top_total[[ACTIVITY, 'total_observed_burden_min', 'burden_share']])}

The highest composite pre-service burdens are:
{markdown_table(top_wait[[ACTIVITY, 'total_pre_service_min', 'pre_service_mean_min', 'pre_service_p90_min']])}

These are bottleneck candidates, not causal diagnoses. A physical queueing claim
requires spatially sticky resource assignment and explicit movement/queue state.
"""
    (args.output / "bottleneck_analysis_report.md").write_text(report, encoding="utf-8")

    print(activities.head(8).to_string(index=False))
    print(json.dumps(decomposition, indent=2))
    print(f"Wrote bottleneck analysis to {args.output}")


if __name__ == "__main__":
    main()
