import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
from _eventlog_contract import (  # noqa: E402
    canonicalize_case_order,
    validate_eventlog_contract,
)

# ==========================================================
# CONFIG
# ==========================================================

EVENTLOG_FILE = (
    "data/processed/CTB/"
    "s2_eventlog_with_demand_features.csv"
)

YARD_FILE = (
    "data/interim/CTB/"
    "yard_aggregated_s2_interpolated_1min.csv"
)

OUTPUT_FILE = (
    "data/processed/CTB/"
    "s3_eventlog_with_utilization_features.csv"
)

# ==========================================================
# LOAD
# ==========================================================

print("Loading eventlog...")

eventlog = pd.read_csv(
    EVENTLOG_FILE
)

print("Loading yard state...")

yard = pd.read_csv(
    YARD_FILE
)

# ==========================================================
# DATETIME
# ==========================================================

eventlog["start:timestamp"] = pd.to_datetime(
    eventlog["start:timestamp"]
)

yard["ZEITPUNKT"] = pd.to_datetime(
    yard["ZEITPUNKT"]
)

# ==========================================================
# GATE IN EVENTS
# ==========================================================

gate_in = eventlog[
    eventlog["concept:name"] == "Gate In"
].copy()

gate_in = gate_in[
    [
        "case:concept:name",
        "start:timestamp"
    ]
]

# ==========================================================
# HELPER
# ==========================================================

def add_util_feature(
    gate_df,
    yard_df,
    area,
    feature_name
):

    tmp = yard_df[
        yard_df["AREA"] == area
    ].copy()

    tmp = tmp.sort_values(
        "ZEITPUNKT"
    )

    merged = gate_df.merge(
        tmp[
            [
                "ZEITPUNKT",
                "AUSLASTUNG_%"
            ]
        ],
        left_on="start:timestamp",
        right_on="ZEITPUNKT",
        how="left",
        validate="many_to_one",
    )

    return merged[
        [
            "case:concept:name",
            "AUSLASTUNG_%"
        ]
    ].rename(
        columns={
            "AUSLASTUNG_%":
                feature_name
        }
    )

# ==========================================================
# BUILD FEATURES
# ==========================================================

print("Matching terminal utilization...")

gate_util = add_util_feature(
    gate_in,
    yard,
    "total",
    "gate_utilization"
)

rmg_util = add_util_feature(
    gate_in,
    yard,
    "RMG",
    "rmg_utilization"
)

vc_util = add_util_feature(
    gate_in,
    yard,
    "VC",
    "vc_utilization"
)

mt_util = add_util_feature(
    gate_in,
    yard,
    "MT",
    "mt_utilization"
)

# ==========================================================
# CASE FEATURE TABLE
# ==========================================================

case_features = (
    gate_util
    .merge(
        rmg_util,
        on="case:concept:name",
        how="left"
    )
    .merge(
        vc_util,
        on="case:concept:name",
        how="left"
    )
    .merge(
        mt_util,
        on="case:concept:name",
        how="left"
    )
)

# ==========================================================
# MERGE BACK
# ==========================================================

eventlog = eventlog.merge(
    case_features,
    on="case:concept:name",
    how="left"
)

# ==========================================================
# QUALITY REPORT
# ==========================================================

print("\n" + "=" * 70)
print("YARD UTILIZATION FEATURE REPORT")
print("=" * 70)

for col in [
    "gate_utilization",
    "rmg_utilization",
    "vc_utilization",
    "mt_utilization"
]:

    print(
        f"{col:20s}: "
        f"{eventlog[col].notna().sum():,} events"
    )

print()

print(
    eventlog[
        [
            "gate_utilization",
            "rmg_utilization",
            "vc_utilization",
            "mt_utilization"
        ]
    ].describe()
)

print("=" * 70)

# ==========================================================
# SAVE
# ==========================================================

eventlog = canonicalize_case_order(eventlog)
contract = validate_eventlog_contract(eventlog, label="stage 3 real event log")
print(
    "CTB case contract: "
    f"gate-only cases={contract['gate_only_cases']}, "
    f"minimum yard events/case={contract['yard_events_per_case']['min']}"
)

eventlog.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)
