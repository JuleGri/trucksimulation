import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
from _eventlog_contract import (  # noqa: E402
    canonicalize_case_order,
    validate_eventlog_contract,
)

# =====================================================
# INPUTS
# =====================================================

INPUT = (
    "data/processed/CTB/"
    "s3_eventlog_with_utilization_features.csv"
)

OUTPUT = (
    "data/processed/CTB/"
    "s4_eventlog_rmg_generic_and_target_areas.csv"
)

# =====================================================
# LOAD
# =====================================================

log = pd.read_csv(INPUT)

print(f"Rows: {len(log):,}")
print(
    f"Cases: "
    f"{log['case:concept:name'].nunique():,}"
)

# =====================================================
# KEEP ORIGINAL ACTIVITY
# =====================================================

log["original_activity"] = log["concept:name"]

# =====================================================
# TARGET AREA FUNCTION
# =====================================================

def extract_area(activity):

    if not isinstance(activity, str):
        return None

    if activity.startswith("T"):
        return activity.split("_")[0]

    if activity.startswith("LL"):
        return "MT"

    if activity.startswith("HO2"):
        return "VC"

    return None

# =====================================================
# REMOVE UNKNOWN CASES
# =====================================================

unknown_cases = set(

    log.loc[
        log["concept:name"]
        .str.contains(
            "unknown",
            case=False,
            na=False
        ),
        "case:concept:name"
    ]

)

print()
print(
    f"Cases containing unknown activities: "
    f"{len(unknown_cases):,}"
)

log = log[
    ~log["case:concept:name"]
    .isin(unknown_cases)
].copy()

# =====================================================
# DETERMINE TARGET AREAS
# =====================================================

yard_events = log.copy()

yard_events["target_area"] = (
    yard_events["original_activity"]
    .apply(extract_area)
)

yard_events = yard_events[
    yard_events["target_area"].notna()
].copy()

# =====================================================
# CASE ANALYSIS
# =====================================================

case_analysis = []

for case_id, grp in yard_events.groupby(
    "case:concept:name"
):

    areas = set(grp["target_area"])

    rmg_blocks = {
        a
        for a in areas
        if a.startswith("T")
    }

    case_analysis.append({

        "case:concept:name": case_id,

        "areas": sorted(list(areas)),

        "rmg_blocks": sorted(list(rmg_blocks)),

        "n_rmg_blocks": len(rmg_blocks),

        "has_mt": "MT" in areas,

        "has_vc": "VC" in areas

    })

case_analysis = pd.DataFrame(case_analysis)

# =====================================================
# REPORT
# =====================================================

print()
print("=" * 60)
print("TARGET AREA ANALYSIS")
print("=" * 60)

print(
    f"Total cases: "
    f"{len(case_analysis):,}"
)

print(
    f"Exactly one RMG block: "
    f"{(case_analysis['n_rmg_blocks'] == 1).sum():,}"
)

print(
    f"Multiple RMG blocks: "
    f"{(case_analysis['n_rmg_blocks'] > 1).sum():,}"
)

print(
    f"MT only: "
    f"{((case_analysis['n_rmg_blocks']==0) & (case_analysis['has_mt']) & (~case_analysis['has_vc'])).sum():,}"
)

print(
    f"VC only: "
    f"{((case_analysis['n_rmg_blocks']==0) & (~case_analysis['has_mt']) & (case_analysis['has_vc'])).sum():,}"
)

# =====================================================
# PRIMARY TARGET AREA
# =====================================================

target_mapping = {}

for _, row in case_analysis.iterrows():

    rmgs = row["rmg_blocks"]

    # exactly one RMG block
    if len(rmgs) == 1:

        target_mapping[
            row["case:concept:name"]
        ] = rmgs[0]

    # no RMG block
    elif len(rmgs) == 0:

        if row["has_mt"]:

            target_mapping[
                row["case:concept:name"]
            ] = "MT"

        elif row["has_vc"]:

            target_mapping[
                row["case:concept:name"]
            ] = "VC"

        else:

            target_mapping[
                row["case:concept:name"]
            ] = None

    # multi-RMG
    else:

        target_mapping[
            row["case:concept:name"]
        ] = None

# =====================================================
# FILTER MULTI-RMG
# =====================================================

valid_cases = set(

    case_analysis[
        case_analysis["n_rmg_blocks"] <= 1
    ]["case:concept:name"]

)

cases_before = (
    log["case:concept:name"]
    .nunique()
)

log = log[
    log["case:concept:name"]
    .isin(valid_cases)
].copy()

# =====================================================
# ATTACH PRIMARY TARGET AREA
# =====================================================

log["primary_target_area"] = (
    log["case:concept:name"]
    .map(target_mapping)
)

# =====================================================
# MAP ACTIVITIES TO GENERIC RMG ACTIVITIES
# =====================================================

def map_activity(activity):

    if isinstance(activity, str) and activity.startswith("T"):

        if activity.endswith("_receive"):
            return "RMG_receive"

        if activity.endswith("_delivery"):
            return "RMG_delivery"

        if activity.endswith("_mixed"):
            return "RMG_mixed"

    return activity

log["concept:name"] = (
    log["concept:name"]
    .apply(map_activity)
)

# =====================================================
# FINAL REPORT
# =====================================================

print()
print("=" * 60)
print("FILTER RESULT")
print("=" * 60)

print(
    f"Cases before: "
    f"{cases_before:,}"
)

print(
    f"Cases after: "
    f"{log['case:concept:name'].nunique():,}"
)

print(
    f"Rows after: "
    f"{len(log):,}"
)

print()
print(
    log["primary_target_area"]
    .value_counts(dropna=False)
)

# =====================================================
# CLEANUP
# =====================================================

log = log.drop(
    columns=["original_activity"],
    errors="ignore"
)

# =====================================================
# SAVE
# =====================================================

log = canonicalize_case_order(log)
contract = validate_eventlog_contract(log, label="stage 4 real event log")
print(
    "CTB case contract: "
    f"gate-only cases={contract['gate_only_cases']}, "
    f"minimum yard events/case={contract['yard_events_per_case']['min']}"
)

log.to_csv(
    OUTPUT,
    index=False
)

print()
print(
    f"Saved: {OUTPUT}"
)
