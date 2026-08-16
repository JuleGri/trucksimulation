import pandas as pd

# =====================================================
# INPUT
# =====================================================

INPUT = (
    "data/processed/CTB/"
    "s4_eventlog_rmg_generic_activities_v2.csv"
)

OUTPUT = (
    "data/processed/CTB/"
    "s4_single_blocks_eventlog_v2.csv"
)

# =====================================================
# LOAD
# =====================================================

log = pd.read_csv(INPUT)

print(
    f"Original rows  : {len(log):,}"
)

print(
    f"Original cases : "
    f"{log['case:concept:name'].nunique():,}"
)

# =====================================================
# RMG ACTIVITY LIST
# =====================================================

rmg_activities = [

    "RMG_receive",

    "RMG_delivery",

    "RMG_mixed"

]

# =====================================================
# IDENTIFY VALID CASES
# =====================================================

valid_cases = []

for case_id, grp in log.groupby(
    "case:concept:name"
):

    activities = list(
        grp["concept:name"]
    )

    rmg_count = sum(
        a in rmg_activities
        for a in activities
    )

    # exactly one RMG event
    if rmg_count != 1:
        continue

    # exactly 3 events total
    if len(activities) != 3:
        continue

    # must contain Gate In + Gate Out
    if "Gate In" not in activities:
        continue

    if "Gate Out" not in activities:
        continue

    valid_cases.append(case_id)

# =====================================================
# FILTER
# =====================================================

single_block_log = log[
    log["case:concept:name"]
    .isin(valid_cases)
].copy()

# =====================================================
# REPORT
# =====================================================

print()

print(
    f"Valid cases: "
    f"{len(valid_cases):,}"
)

print()

print(
    f"Remaining rows: "
    f"{len(single_block_log):,}"
)

print(
    f"Remaining cases: "
    f"{single_block_log['case:concept:name'].nunique():,}"
)

print()

print(
    single_block_log[
        "concept:name"
    ]
    .value_counts()
)

# =====================================================
# SAFETY CHECK
# =====================================================

non_rmg = single_block_log[
    ~single_block_log["concept:name"]
    .isin(
        [
            "Gate In",
            "Gate Out",
            "RMG_receive",
            "RMG_delivery",
            "RMG_mixed"
        ]
    )
]

print()

print(
    "Non-RMG activities remaining:"
)

print(
    non_rmg["concept:name"]
    .unique()
)

# =====================================================
# SAVE
# =====================================================

single_block_log.to_csv(
    OUTPUT,
    index=False
)

print()

print(
    f"Saved: {OUTPUT}"
)