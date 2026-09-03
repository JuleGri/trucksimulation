"""Audit the structural position and time burden of LL_mixed visits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CASE = "case:concept:name"
ACTIVITY = "concept:name"
START = "start:timestamp"
COMPLETE = "time:timestamp"
ORDER = "case:event:order"
BOUNDARIES = {"Gate In", "Gate Out"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/CTB/s6_train.csv"))
    parser.add_argument("--holdout", type=Path, default=Path("data/processed/CTB/s6_test.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/results/ll_mixed_sequence_bottleneck_audit"),
    )
    return parser.parse_args()


def prepare(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {CASE, ACTIVITY, START, COMPLETE, ORDER}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns in {path}: {sorted(missing)}")
    frame[ORDER] = pd.to_numeric(frame[ORDER], errors="raise")
    frame[START] = pd.to_datetime(frame[START], errors="raise")
    frame[COMPLETE] = pd.to_datetime(frame[COMPLETE], errors="raise")
    frame = frame.sort_values([CASE, ORDER], kind="stable").copy()
    grouped = frame.groupby(CASE, sort=False)
    frame["split"] = split
    frame["previous_activity"] = grouped[ACTIVITY].shift(1)
    frame["previous_complete"] = grouped[COMPLETE].shift(1)
    frame["next_activity"] = grouped[ACTIVITY].shift(-1)
    frame["next_start"] = grouped[START].shift(-1)
    frame["service_min"] = (frame[COMPLETE] - frame[START]).dt.total_seconds().div(60)
    frame["composite_pre_service_min"] = (
        (frame[START] - frame["previous_complete"])
        .dt.total_seconds()
        .div(60)
        .clip(lower=0)
        .fillna(0)
    )
    frame["gap_to_next_min"] = (
        (frame["next_start"] - frame[COMPLETE])
        .dt.total_seconds()
        .div(60)
        .clip(lower=0)
    )
    return frame


def quantile90(values: pd.Series) -> float:
    return float(values.quantile(0.90))


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = [prepare(args.train, "train"), prepare(args.holdout, "holdout")]
    combined = pd.concat(frames, ignore_index=True)

    yard = combined[~combined[ACTIVITY].isin(BOUNDARIES)].copy()
    yard["yard_position"] = yard.groupby(["split", CASE]).cumcount() + 1
    yard["yard_count"] = yard.groupby(["split", CASE])[ACTIVITY].transform("size")
    ll = yard[yard[ACTIVITY].eq("LL_mixed")].copy()

    split_summaries: list[dict[str, object]] = []
    for split in ("train", "holdout", "combined"):
        events = ll if split == "combined" else ll[ll["split"].eq(split)]
        multiple = events[events["yard_count"].gt(1)]
        split_summaries.append(
            {
                "split": split,
                "ll_mixed_events": int(len(events)),
                "ll_mixed_cases": int(events[CASE].nunique()),
                "ll_mixed_only_yard_cases": int(events["yard_count"].eq(1).sum()),
                "ll_mixed_multi_yard_cases": int(events["yard_count"].gt(1).sum()),
                "ll_mixed_first_in_multi_yard": int(multiple["yard_position"].eq(1).sum()),
                "ll_mixed_last_in_multi_yard": int(
                    multiple["yard_position"].eq(multiple["yard_count"]).sum()
                ),
                "service_mean_min": float(events["service_min"].mean()),
                "service_median_min": float(events["service_min"].median()),
                "service_total_min": float(events["service_min"].sum()),
                "pre_service_mean_min": float(events["composite_pre_service_min"].mean()),
                "pre_service_median_min": float(events["composite_pre_service_min"].median()),
                "pre_service_total_min": float(events["composite_pre_service_min"].sum()),
            }
        )
    summaries = pd.DataFrame(split_summaries)

    successor_rows: list[dict[str, object]] = []
    for split, events in ll.groupby("split", sort=False):
        for successor, group in events.groupby("next_activity", dropna=False, sort=False):
            successor_rows.append(
                {
                    "split": split,
                    "successor": successor,
                    "n": int(len(group)),
                    "share_within_ll_mixed": float(len(group) / len(events)),
                    "gap_mean_min": float(group["gap_to_next_min"].mean()),
                    "gap_median_min": float(group["gap_to_next_min"].median()),
                    "gap_p90_min": quantile90(group["gap_to_next_min"]),
                    "gap_total_min": float(group["gap_to_next_min"].sum()),
                }
            )
    successors = pd.DataFrame(successor_rows)

    sequence_rows: list[dict[str, object]] = []
    multi_case_keys = ll.loc[ll["yard_count"].gt(1), ["split", CASE]].drop_duplicates()
    for row in multi_case_keys.itertuples(index=False):
        split, case = row
        events = combined[combined["split"].eq(split) & combined[CASE].eq(case)]
        events = events.sort_values(ORDER, kind="stable")
        sequence_rows.append(
            {
                "split": split,
                CASE: case,
                "activity_sequence": " -> ".join(events[ACTIVITY].astype(str)),
            }
        )
    sequences = pd.DataFrame(sequence_rows)

    comparison_rows: list[dict[str, object]] = []
    ll_yard_successors = ll.loc[
        ll["yard_count"].gt(1) & ~ll["next_activity"].eq("Gate Out"),
        ["split", "next_activity"],
    ].drop_duplicates()
    for row in ll_yard_successors.itertuples(index=False):
        split, successor = row
        events = combined[combined["split"].eq(split)]
        after_ll = events[
            events[ACTIVITY].eq(successor) & events["previous_activity"].eq("LL_mixed")
        ]["composite_pre_service_min"]
        after_other = events[
            events[ACTIVITY].eq(successor) & ~events["previous_activity"].eq("LL_mixed")
        ]["composite_pre_service_min"]
        comparison_rows.append(
            {
                "split": split,
                "successor": successor,
                "n_after_ll_mixed": int(len(after_ll)),
                "mean_after_ll_mixed_min": float(after_ll.mean()),
                "median_after_ll_mixed_min": float(after_ll.median()),
                "p90_after_ll_mixed_min": quantile90(after_ll),
                "n_after_other": int(len(after_other)),
                "mean_after_other_min": float(after_other.mean()),
                "median_after_other_min": float(after_other.median()),
                "p90_after_other_min": quantile90(after_other),
                "mean_difference_min": float(after_ll.mean() - after_other.mean()),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)

    holdout = frames[1]
    turnaround_total = (
        holdout.groupby(CASE)[COMPLETE].max() - holdout.groupby(CASE)[START].min()
    ).dt.total_seconds().sum() / 60
    holdout_ll = ll[ll["split"].eq("holdout")]
    holdout_ll_burden = float(
        (holdout_ll["service_min"] + holdout_ll["composite_pre_service_min"]).sum()
    )

    summaries.to_csv(args.output / "ll_mixed_summary.csv", index=False)
    successors.to_csv(args.output / "ll_mixed_successors.csv", index=False)
    sequences.to_csv(args.output / "ll_mixed_multi_yard_sequences.csv", index=False)
    comparisons.to_csv(args.output / "ll_mixed_successor_gap_comparison.csv", index=False)
    manifest = {
        "train_source": str(args.train),
        "holdout_source": str(args.holdout),
        "ordering": ORDER,
        "delay_semantics": (
            "Composite pre-service delay is the non-negative gap between the previous "
            "event completion and current event start. It is not interpreted as pure queueing."
        ),
        "holdout_ll_mixed_service_plus_pre_service_min": holdout_ll_burden,
        "holdout_total_turnaround_min": float(turnaround_total),
        "holdout_ll_mixed_burden_share": float(holdout_ll_burden / turnaround_total),
        "interpretation_boundary": (
            "The audit describes sequence frequency and accumulated time. The seven multi-yard "
            "LL_mixed cases are too sparse to establish a mandatory routing rule or a causal bottleneck."
        ),
    }
    with (args.output / "ll_mixed_audit_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(summaries.to_string(index=False))
    print(successors.to_string(index=False))
    print(sequences.to_string(index=False))
    print(comparisons.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
