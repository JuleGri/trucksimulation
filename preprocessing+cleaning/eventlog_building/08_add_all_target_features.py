import pandas as pd

# =====================================================
# INPUTS
# =====================================================

EVENTLOG = (
    "data/processed/CTB/"
    "s4_eventlog_rmg_generic_activities.csv"
)

YARD_FILE = (
    "data/interim/CTB/"
    "yard_aggregated_s2_interpolated_1min.csv"
)

DEMAND_FILE = (
    "data/interim/CTB/"
    "truck_demand_interpolated_1min.csv"
)

OUTPUT = (
    "data/processed/CTB/"
    "s6_eventlog_target_rank_features.csv"
)

# =====================================================
# LOAD
# =====================================================

log = pd.read_csv(EVENTLOG)

yard = pd.read_csv(YARD_FILE)

demand = pd.read_csv(DEMAND_FILE)

# =====================================================
# TIMESTAMPS
# =====================================================

for col in [

    "enabled:timestamp",
    "start:timestamp",
    "time:timestamp"

]:

    log[col] = pd.to_datetime(
        log[col]
    )

yard["ZEITPUNKT"] = pd.to_datetime(
    yard["ZEITPUNKT"]
)

demand["ZEITPUNKT"] = pd.to_datetime(
    demand["ZEITPUNKT"]
)

# =====================================================
# REMOVE UNKNOWN
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

log = log[
    ~log["case:concept:name"]
    .isin(unknown_cases)
].copy()

# =====================================================
# MAP RMG ACTIVITIES
# =====================================================

def map_activity(activity):

    if isinstance(activity, str):

        if activity.startswith("T"):

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
# DETERMINE TARGET BLOCK
# =====================================================

rmg_events = log[

    log["concept:name"]

    .isin(

        [

            "RMG_receive",
            "RMG_delivery",
            "RMG_mixed"

        ]

    )

].copy()

# =====================================================
# KEEP ONLY SINGLE BLOCK CASES
# =====================================================

rmg_count = (

    rmg_events

    .groupby(
        "case:concept:name"
    )

    .size()

)

single_cases = set(

    rmg_count[
        rmg_count == 1
    ].index

)

log = log[
    log["case:concept:name"]
    .isin(single_cases)
].copy()

rmg_events = rmg_events[
    rmg_events["case:concept:name"]
    .isin(single_cases)
].copy()

# =====================================================
# TARGET BLOCK
# =====================================================

target_block = (

    rmg_events

    .groupby(
        "case:concept:name"
    )

    ["org:resource"]

    .first()

    .rename(
        "target_block"
    )

)

# =====================================================
# GATE IN TIMESTAMP
# =====================================================

gate_in = (

    log[
        log["concept:name"]
        == "Gate In"
    ]

    .groupby(
        "case:concept:name"
    )

    ["start:timestamp"]

    .min()

    .rename(
        "gate_in_ts"
    )

)

case_info = pd.concat(

    [

        target_block,

        gate_in

    ],

    axis=1

).reset_index()

# =====================================================
# UTILIZATION LOOKUP
# =====================================================

yard = yard.rename(

    columns={

        "ZEITPUNKT":
            "gate_in_ts",

        "AREA":
            "target_block",

        "AUSLASTUNG_%":
            "target_utilization"

    }

)
# =====================================================
# UTILIZATION RANK
# =====================================================

yard["target_rank"] = (

    yard

    .groupby("gate_in_ts")

    ["target_utilization"]

    .rank(
        ascending=False,
        method="dense"
    )

)
yard = yard.sort_values(
    "gate_in_ts"
)

case_info = case_info.sort_values(
    "gate_in_ts"
)

case_info = pd.merge_asof(

    case_info,

    yard[
        [

            "gate_in_ts",
            "target_block",
            "target_utilization",
            "target_rank"

        ]
    ],

    on="gate_in_ts",

    by="target_block",

    direction="nearest"

)

# =====================================================
# DEMAND LOOKUP
# =====================================================

demand = demand.rename(

    columns={

        "ZEITPUNKT":
            "gate_in_ts",

        "AREA":
            "target_block",

        "TRUCK_DEMAND_LAST_1H":
            "target_demand"

    }

)

demand = demand.sort_values(
    "gate_in_ts"
)

case_info = pd.merge_asof(

    case_info,

    demand[
        [

            "gate_in_ts",
            "target_block",
            "target_demand"

        ]
    ],

    on="gate_in_ts",

    by="target_block",

    direction="nearest"

)
# =====================================================
# TARGET RANK GROUP
# =====================================================

def rank_group(rank):

    if rank <= 3:
        return "top_3"

    elif rank <= 10:
        return "mid"

    return "low"


case_info["target_rank_group"] = (
    case_info["target_rank"]
    .apply(rank_group)
)
# =====================================================
# BINNING
# =====================================================

print("Creating utilization bins...")

case_info["target_utilization_bin"] = pd.qcut(
    case_info["target_utilization"],
    q=4,
    labels=[
        "low",
        "medium",
        "high",
        "very_high"
    ],
    duplicates="drop"
)

print("Creating demand bins...")

case_info["target_demand_bin"] = pd.qcut(
    case_info["target_demand"],
    q=4,
    labels=[
        "low",
        "medium",
        "high",
        "very_high"
    ],
    duplicates="drop"
)

# =====================================================
# BIN SUMMARY
# =====================================================

print("\nUtilization Bin Distribution")

print(
    case_info["target_utilization_bin"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nDemand Bin Distribution")

print(
    case_info["target_demand_bin"]
    .value_counts(dropna=False)
    .sort_index()
)
print("\nTarget Rank Distribution")

print(
    case_info["target_rank"]
    .describe()
)

print("\nTarget Rank Group Distribution")

print(
    case_info["target_rank_group"]
    .value_counts(dropna=False)
)

# =====================================================
# MERGE BACK
# =====================================================

log = log.merge(

    case_info[
        [
            "case:concept:name",
            "target_block",
            "target_utilization",
            "target_demand",
            "target_utilization_bin",
            "target_demand_bin",
            "target_rank",
            "target_rank_group"
        ]
    ],

    on="case:concept:name",

    how="left"

)
# =====================================================
# RESOURCE = TARGET BLOCK
# =====================================================

mask = (

    log["concept:name"]

    .isin(

        [

            "RMG_receive",
            "RMG_delivery",
            "RMG_mixed"

        ]

    )

)

log.loc[
    mask,
    "org:resource"
] = log.loc[
    mask,
    "target_block"
]

# =====================================================
# REPORT
# =====================================================

print()

print(
    "Cases:",
    log["case:concept:name"]
    .nunique()
)

print(
    "Rows:",
    len(log)
)

print()

print(
    log[
        [

            "target_block",
            "target_utilization",
            "target_demand"

        ]

    ]

    .describe()
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