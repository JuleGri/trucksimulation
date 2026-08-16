import pandas as pd

# =====================================================
# INPUT
# =====================================================

INPUT = (
    "data/processed/CTB/"
    "s6_eventlog_target_rank_features.csv"
)

OUTPUT = (
    "data/processed/CTB/"
    "s7_eventlog_with_performance_targets_v2.csv"
)

# =====================================================
# LOAD
# =====================================================

log = pd.read_csv(INPUT)

# =====================================================
# TIMESTAMPS
# =====================================================

for col in [

    "enabled:timestamp",

    "start:timestamp",

    "time:timestamp"

]:

    log[col] = pd.to_datetime(
        log[col],
        errors="coerce"
    )

# =====================================================
# WAITING TIME
# =====================================================

log["waiting_time"] = (

    log["start:timestamp"]

    -

    log["enabled:timestamp"]

).dt.total_seconds() / 60

# =====================================================
# SERVICE TIME
# =====================================================

log["service_time"] = (

    log["time:timestamp"]

    -

    log["start:timestamp"]

).dt.total_seconds() / 60

# =====================================================
# TURNAROUND TIME
# =====================================================

case_turnaround = (

    log

    .groupby(
        "case:concept:name"
    )

    .agg(

        start=(
            "start:timestamp",
            "min"
        ),

        end=(
            "time:timestamp",
            "max"
        )

    )

)

case_turnaround["turnaround_time"] = (

    case_turnaround["end"]

    -

    case_turnaround["start"]

).dt.total_seconds() / 60

case_turnaround = case_turnaround[
    ["turnaround_time"]
]

# =====================================================
# MERGE BACK
# =====================================================

log = log.merge(

    case_turnaround,

    left_on="case:concept:name",

    right_index=True,

    how="left"

)

# =====================================================
# REPORT
# =====================================================

print()

print(
    log[
        [

            "waiting_time",

            "service_time",

            "turnaround_time"

        ]

    ].describe()
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