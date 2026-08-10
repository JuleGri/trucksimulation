import pandas as pd

# =====================================================
# INPUTS
# =====================================================

INPUT = (
    "data/processed/CTB/"
    "s3_eventlog_with_utilization_features.csv"
)

OUTPUT = (
    "data/processed/CTB/"
    "s4_eventlog_rmg_generic_activities.csv"
)

# =====================================================
# LOAD
# =====================================================

log = pd.read_csv(INPUT)

print(
    f"Rows: {len(log):,}"
)

print(
    f"Cases: "
    f"{log['case:concept:name'].nunique():,}"
)

# =====================================================
# MAP ACTIVITIES
# =====================================================

def map_activity(activity):

    # only storage blocks T06-T27 etc.

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
# REPORT
# =====================================================

print("\nActivities after mapping:\n")

print(
    log["concept:name"]
    .value_counts()
)

print()

print(
    "Unique activities:",
    log["concept:name"]
    .nunique()
)
#============= REMOVE UNKOWN
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

print(
    f"Cases containing unknown activities: "
    f"{len(unknown_cases):,}"
)
rows_before = len(log)

cases_before = (
    log["case:concept:name"]
    .nunique()
)

log = log[
    ~log["case:concept:name"]
    .isin(unknown_cases)
].copy()

rows_after = len(log)

cases_after = (
    log["case:concept:name"]
    .nunique()
)

print()

print(
    f"Rows removed: "
    f"{rows_before - rows_after:,}"
)

print(
    f"Cases removed: "
    f"{cases_before - cases_after:,}"
)

print()

print(
    f"Remaining rows: "
    f"{rows_after:,}"
)

print(
    f"Remaining cases: "
    f"{cases_after:,}"
)

remaining_unknowns = (

    log["concept:name"]

    .str.contains(
        "unknown",
        case=False,
        na=False
    )

    .sum()

)

print(
    f"Remaining unknown events: "
    f"{remaining_unknowns}"
)
# =====================================================
# SAVE
# =====================================================

log.to_csv(
    OUTPUT,
    index=False
)

print()

print(
    f"Saved: {OUTPUT}"
)
