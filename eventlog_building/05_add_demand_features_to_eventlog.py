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
    "s1_eventlog_with_case_features.csv"
)

TRUCK_DEMAND_FILE = (
    "data/interim/CTB/"
    "truck_demand_interpolated_1min.csv"
)

OUTPUT_FILE = (
    "data/processed/CTB/"
    "s2_eventlog_with_demand_features.csv"
)

# ==========================================================
# LOAD
# ==========================================================

print("Loading eventlog...")

eventlog = pd.read_csv(
    EVENTLOG_FILE
)

print("Loading demand table...")

demand = pd.read_csv(
    TRUCK_DEMAND_FILE
)

# ==========================================================
# DATETIME
# ==========================================================

eventlog["start:timestamp"] = pd.to_datetime(
    eventlog["start:timestamp"]
)

demand["ZEITPUNKT"] = pd.to_datetime(
    demand["ZEITPUNKT"]
)

# ==========================================================
# GATE IN PER CASE
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

def add_demand_feature(
    gate_df,
    demand_df,
    area,
    feature_name
):

    tmp = demand_df[
        demand_df["AREA"] == area
    ].copy()

    tmp = tmp.sort_values(
        "ZEITPUNKT"
    )

    merged = gate_df.merge(
        tmp[
            [
                "ZEITPUNKT",
                "TRUCK_DEMAND_LAST_1H"
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
            "TRUCK_DEMAND_LAST_1H"
        ]
    ].rename(
        columns={
            "TRUCK_DEMAND_LAST_1H":
                feature_name
        }
    )

# ==========================================================
# CREATE DEMAND FEATURES
# ==========================================================

print("Matching Gate Demand...")

gate_demand = add_demand_feature(
    gate_in,
    demand,
    "total",
    "gate_demand"
)

print("Matching RMG Demand...")

rmg_demand = add_demand_feature(
    gate_in,
    demand,
    "RMG",
    "rmg_demand"
)

print("Matching VC Demand...")

vc_demand = add_demand_feature(
    gate_in,
    demand,
    "VC",
    "vc_demand"
)

print("Matching MT Demand...")

mt_demand = add_demand_feature(
    gate_in,
    demand,
    "MT",
    "mt_demand"
)

# ==========================================================
# BUILD CASE FEATURE TABLE
# ==========================================================

case_features = gate_demand.merge(
    rmg_demand,
    on="case:concept:name",
    how="left"
)

case_features = case_features.merge(
    vc_demand,
    on="case:concept:name",
    how="left"
)

case_features = case_features.merge(
    mt_demand,
    on="case:concept:name",
    how="left"
)

# ==========================================================
# MERGE BACK TO EVENTLOG
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
print("TRUCK DEMAND FEATURE REPORT")
print("=" * 70)

print(
    f"Cases: "
    f"{eventlog['case:concept:name'].nunique():,}"
)

print(
    f"Events: "
    f"{len(eventlog):,}"
)

print()

for col in [
    "gate_demand",
    "rmg_demand",
    "vc_demand",
    "mt_demand"
]:

    matched = (
        eventlog[col]
        .notna()
        .sum()
    )

    print(
        f"{col:15s}: "
        f"{matched:,} events"
    )

print()

print(
    eventlog[
        [
            "gate_demand",
            "rmg_demand",
            "vc_demand",
            "mt_demand"
        ]
    ]
    .describe()
)

print("=" * 70)

# ==========================================================
# SAVE
# ==========================================================

eventlog = canonicalize_case_order(eventlog)
contract = validate_eventlog_contract(eventlog, label="stage 2 real event log")
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
