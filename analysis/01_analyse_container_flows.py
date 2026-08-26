"""Analyse truck-related container flows by inbound and outbound carrier type.

The CTB container master contains truck-related moves only:

* DELIVERY: the inbound carrier is the truck and the outbound carrier is the
  next transport mode.
* RECEIVE: the inbound carrier is the previous transport mode and the outbound
  carrier is the truck.

Carrier identifiers are classified without exporting their raw values:

* identifiers starting with ``CTB`` are rail services;
* digit-only identifiers are vessels;
* anonymised hexadecimal keys and all remaining non-empty identifiers are
  trucks (the latter covers non-anonymised licence plates);
* empty identifiers remain missing.

The script writes aggregate CSV/JSON results and a two-panel PNG. It never
prints or writes individual carrier identifiers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Pattern

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "container_id",
    "flow_type",
    "Inbound_Carrier",
    "Outbound_Carrier",
}
FLOW_TYPES = ("DELIVERY", "RECEIVE")
CARRIER_ORDER = ("Vessel", "Rail", "Truck", "Missing")
MISSING_TOKENS = frozenset({"", "NAN", "NONE", "NULL", "<NA>"})
ANONYMISED_TRUCK_PATTERNS = (
    re.compile(r"^[A-F0-9]{64}$"),
    re.compile(r"^[A-F0-9]{32}$"),
    re.compile(
        r"^[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-"
        r"[A-F0-9]{4}-[A-F0-9]{12}$"
    ),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Classify CTB container-master inbound/outbound carriers and "
            "summarise truck-vessel, truck-rail, and truck-truck flows."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "data" / "raw" / "CTB" / "container_master_0304.csv",
        help="Semicolon-separated container master.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "analysis" / "CTB" / "container_flows",
        help="Directory for aggregate CSV, JSON, and PNG outputs.",
    )
    parser.add_argument("--separator", default=";", help="Input CSV separator.")
    parser.add_argument(
        "--rail-pattern",
        default=r"^CTB",
        help="Case-insensitive regular expression for rail identifiers.",
    )
    parser.add_argument(
        "--vessel-pattern",
        default=r"^\d+$",
        help="Regular expression for vessel identifiers.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Return a non-zero exit code if a known flow has a non-truck "
            "carrier on its expected truck side or if unknown flow types occur."
        ),
    )
    return parser.parse_args()


def normalise_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    return "" if text in MISSING_TOKENS else text


def classify_identifier(
    value: object,
    rail_pattern: Pattern[str],
    vessel_pattern: Pattern[str],
) -> tuple[str, str]:
    """Return a privacy-safe carrier type and the applied classification rule."""

    identifier = normalise_identifier(value)
    if not identifier:
        return "Missing", "missing"
    if rail_pattern.search(identifier):
        return "Rail", "rail_pattern"
    if any(pattern.fullmatch(identifier) for pattern in ANONYMISED_TRUCK_PATTERNS):
        return "Truck", "anonymised_truck_key"
    if vessel_pattern.fullmatch(identifier):
        return "Vessel", "vessel_pattern"
    return "Truck", "plate_or_other_truck_key"


def classify_series(
    series: pd.Series,
    rail_pattern: Pattern[str],
    vessel_pattern: Pattern[str],
) -> pd.DataFrame:
    classified = series.map(
        lambda value: classify_identifier(value, rail_pattern, vessel_pattern)
    )
    return pd.DataFrame(
        classified.tolist(),
        columns=["carrier_type", "classification_rule"],
        index=series.index,
    )


def validate_classifier(
    rail_pattern: Pattern[str], vessel_pattern: Pattern[str]
) -> None:
    examples = {
        "CTB_RAIL_SERVICE": "Rail",
        "22533123": "Vessel",
        "A" * 64: "Truck",
        "HH-AB 123": "Truck",
        None: "Missing",
    }
    for identifier, expected in examples.items():
        actual, _ = classify_identifier(identifier, rail_pattern, vessel_pattern)
        if actual != expected:
            raise AssertionError(
                f"Carrier classifier self-check failed: expected {expected}, got {actual}."
            )


def add_flow_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["flow_type"] = result["flow_type"].fillna("").str.strip().str.upper()

    delivery = result["flow_type"].eq("DELIVERY")
    receive = result["flow_type"].eq("RECEIVE")

    result["traffic_direction"] = "Unknown"
    result.loc[delivery, "traffic_direction"] = "Truck to terminal"
    result.loc[receive, "traffic_direction"] = "Terminal to truck"

    result["counterparty_carrier_type"] = "Missing"
    result.loc[delivery, "counterparty_carrier_type"] = result.loc[
        delivery, "outbound_carrier_type"
    ]
    result.loc[receive, "counterparty_carrier_type"] = result.loc[
        receive, "inbound_carrier_type"
    ]

    result["container_flow"] = "Unknown"
    result.loc[delivery, "container_flow"] = (
        "Truck -> " + result.loc[delivery, "counterparty_carrier_type"]
    )
    result.loc[receive, "container_flow"] = (
        result.loc[receive, "counterparty_carrier_type"] + " -> Truck"
    )
    return result


def build_flow_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby(
            [
                "flow_type",
                "traffic_direction",
                "container_flow",
                "inbound_carrier_type",
                "outbound_carrier_type",
            ],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    totals = summary.groupby("flow_type")["count"].transform("sum")
    summary["share_within_flow_type"] = summary["count"] / totals
    summary["share_overall"] = summary["count"] / len(data)
    return summary.sort_values(
        ["flow_type", "count", "container_flow"],
        ascending=[True, False, True],
    )


def build_counterparty_summary(data: pd.DataFrame) -> pd.DataFrame:
    known = data[data["flow_type"].isin(FLOW_TYPES)].copy()
    counts = known.groupby(
        ["flow_type", "counterparty_carrier_type"], dropna=False
    ).size()
    complete_index = pd.MultiIndex.from_product(
        [FLOW_TYPES, CARRIER_ORDER],
        names=["flow_type", "counterparty_carrier_type"],
    )
    summary = counts.reindex(complete_index, fill_value=0).rename("count").reset_index()
    summary["traffic_direction"] = summary["flow_type"].map(
        {"DELIVERY": "Truck to terminal", "RECEIVE": "Terminal to truck"}
    )
    totals = summary.groupby("flow_type")["count"].transform("sum")
    summary["share_within_flow_type"] = summary["count"] / totals
    return summary[
        [
            "flow_type",
            "traffic_direction",
            "counterparty_carrier_type",
            "count",
            "share_within_flow_type",
        ]
    ].sort_values(["flow_type", "count"], ascending=[True, False])


def build_carrier_matrix(data: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.crosstab(
        data["inbound_carrier_type"],
        data["outbound_carrier_type"],
        margins=True,
        margins_name="Total",
        dropna=False,
    )
    return matrix.reindex(
        index=[*CARRIER_ORDER, "Total"],
        columns=[*CARRIER_ORDER, "Total"],
        fill_value=0,
    )


def build_rule_audit(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for side in ("inbound", "outbound"):
        group = (
            data.groupby(
                [f"{side}_carrier_type", f"{side}_classification_rule"],
                dropna=False,
            )
            .size()
            .rename("count")
            .reset_index()
            .rename(
                columns={
                    f"{side}_carrier_type": "carrier_type",
                    f"{side}_classification_rule": "classification_rule",
                }
            )
        )
        group.insert(0, "carrier_side", side)
        group["share_within_side"] = group["count"] / len(data)
        rows.append(group)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["carrier_side", "count"], ascending=[True, False]
    )


def build_quality_report(data: pd.DataFrame, args: argparse.Namespace) -> dict:
    delivery = data["flow_type"].eq("DELIVERY")
    receive = data["flow_type"].eq("RECEIVE")
    expected_truck_side_missing = int(
        (delivery & data["inbound_carrier_type"].eq("Missing")).sum()
        + (receive & data["outbound_carrier_type"].eq("Missing")).sum()
    )
    expected_truck_side_nontruck = int(
        (
            delivery
            & ~data["inbound_carrier_type"].isin(["Truck", "Missing"])
        ).sum()
        + (
            receive
            & ~data["outbound_carrier_type"].isin(["Truck", "Missing"])
        ).sum()
    )
    return {
        "input_file": str(args.input.resolve()),
        "rows": int(len(data)),
        "counting_unit": "container-master movement records; no deduplication applied",
        "unique_container_flow_keys": int(
            data[["container_id", "flow_type"]].drop_duplicates().shape[0]
        ),
        "repeated_container_flow_key_rows": int(
            data.duplicated(["container_id", "flow_type"], keep=False).sum()
        ),
        "flow_type_counts": {
            str(key): int(value)
            for key, value in data["flow_type"].value_counts(dropna=False).items()
        },
        "unknown_flow_type_rows": int((~data["flow_type"].isin(FLOW_TYPES)).sum()),
        "expected_truck_side_missing_rows": expected_truck_side_missing,
        "expected_truck_side_nontruck_rows": expected_truck_side_nontruck,
        "classification": {
            "rail_pattern": args.rail_pattern,
            "vessel_pattern": args.vessel_pattern,
            "anonymised_truck_keys": "32/64 hexadecimal or UUID",
            "remaining_nonempty_identifiers": "truck plate or other truck key",
            "raw_carrier_identifiers_exported": False,
        },
    }


def plot_counterparty_flows(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=False)
    panels = (
        ("DELIVERY", "Inbound by truck", "Truck -> counterparty"),
        ("RECEIVE", "Outbound by truck", "Counterparty -> Truck"),
    )
    colours = {
        "Vessel": "#163A70",
        "Rail": "#D00028",
        "Truck": "#777777",
        "Missing": "#C8C8C8",
    }

    for axis, (flow_type, title, direction) in zip(axes, panels, strict=True):
        panel = (
            summary[summary["flow_type"].eq(flow_type)]
            .set_index("counterparty_carrier_type")
            .reindex(CARRIER_ORDER, fill_value=0)
        )
        counts = panel["count"].astype(int)
        total = int(counts.sum())
        labels = list(CARRIER_ORDER)
        bars = axis.barh(
            labels,
            counts.values,
            color=[colours[label] for label in labels],
        )
        axis.invert_yaxis()
        axis.set_title(f"{title}\n{direction}")
        axis.set_xlabel("Container movement records")
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        for bar, count in zip(bars, counts.values, strict=True):
            share = (100.0 * count / total) if total else 0.0
            axis.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"  {count:,} ({share:.1f}%)",
                va="center",
                fontsize=9,
            )
        axis.margins(x=0.28)

    fig.suptitle("CTB truck-related container flows by counterparty carrier")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rail_pattern = re.compile(args.rail_pattern, flags=re.IGNORECASE)
    vessel_pattern = re.compile(args.vessel_pattern)
    validate_classifier(rail_pattern, vessel_pattern)

    if not args.input.exists():
        raise FileNotFoundError(f"Container master not found: {args.input}")
    data = pd.read_csv(args.input, sep=args.separator, dtype=str)
    data.columns = data.columns.str.strip()
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise KeyError(f"Container master is missing columns: {sorted(missing_columns)}")

    inbound = classify_series(data["Inbound_Carrier"], rail_pattern, vessel_pattern)
    outbound = classify_series(data["Outbound_Carrier"], rail_pattern, vessel_pattern)
    data["inbound_carrier_type"] = inbound["carrier_type"]
    data["inbound_classification_rule"] = inbound["classification_rule"]
    data["outbound_carrier_type"] = outbound["carrier_type"]
    data["outbound_classification_rule"] = outbound["classification_rule"]
    data = add_flow_columns(data)

    flow_summary = build_flow_summary(data)
    counterparty_summary = build_counterparty_summary(data)
    carrier_matrix = build_carrier_matrix(data)
    rule_audit = build_rule_audit(data)
    quality = build_quality_report(data, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    flow_summary.to_csv(args.output_dir / "container_flow_summary.csv", index=False)
    counterparty_summary.to_csv(
        args.output_dir / "counterparty_carrier_summary.csv", index=False
    )
    carrier_matrix.to_csv(args.output_dir / "inbound_outbound_carrier_matrix.csv")
    rule_audit.to_csv(args.output_dir / "carrier_classification_audit.csv", index=False)
    with (args.output_dir / "container_flow_quality.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(quality, stream, indent=2, ensure_ascii=False)
    plot_counterparty_flows(
        counterparty_summary,
        args.output_dir / "container_flows_by_direction.png",
    )

    print(f"Analysed {len(data):,} container movement records.")
    print(counterparty_summary.to_string(index=False))
    print(f"Aggregate outputs: {args.output_dir.resolve()}")
    print("Raw carrier identifiers were not printed or exported.")

    strict_failure = bool(
        quality["unknown_flow_type_rows"]
        or quality["expected_truck_side_nontruck_rows"]
    )
    if args.strict and strict_failure:
        print("Strict carrier-flow validation failed; see quality JSON.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
