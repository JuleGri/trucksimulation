import pandas as pd

INPUT = "data/raw/CTB/ctb_fahrplan_lkw_0304.csv"
OUTPUT = "data/interim/CTB/s1_its_fahrplan_cleaned_0304.csv"

# -------------------------------------------------
# LOAD
# -------------------------------------------------

df = pd.read_csv(INPUT, sep=";")

# -------------------------------------------------
# PARSE TIMESTAMPS
# -------------------------------------------------

time_cols = [
    "FP_ZEITPUNKT_ERSTELLUNG",
    "ALP_ZEITPUNKT_BEREITMELDUNG",
    "ALP_ZEITPUNKT_ERFUELLUNG",
    "FP_ZEITPUNKT_GATEOUT"
]

for col in time_cols:

    df[col] = pd.to_datetime(
        df[col],
        format="%d.%m.%Y %H:%M",
        errors="coerce"
    )
# -------------------------------------------------
# 0) MISSING CRITICAL TIMESTAMPS
# -------------------------------------------------

reasons = []

critical_timestamp_cases = set()

timestamp_mask = (

    df["FP_ZEITPUNKT_ERSTELLUNG"].isna()

    |

    df["ALP_ZEITPUNKT_BEREITMELDUNG"].isna()

    |

    df["ALP_ZEITPUNKT_ERFUELLUNG"].isna()

)

critical_timestamp_cases = set(
    df.loc[
        timestamp_mask,
        "FAHRPLAN_UID"
    ]
)

for case in critical_timestamp_cases:

    reasons.append(
        {
            "FAHRPLAN_UID": case,
            "reason": "missing_critical_timestamp"
        }
    )

print(
    f"Cases with missing timestamps: "
    f"{len(critical_timestamp_cases)}"
)

print(
    f"Affected rows: "
    f"{timestamp_mask.sum()}"
)
# -------------------------------------------------
# 1) NEGATIVE DURATIONS
# -------------------------------------------------
#reasons = []
negative_rows = df[
    (
        df["ALP_ZEITPUNKT_ERFUELLUNG"]
        <
        df["ALP_ZEITPUNKT_BEREITMELDUNG"]
    )
]

negative_cases = set(
    negative_rows["FAHRPLAN_UID"]
)

for case in negative_cases:
    reasons.append(
        {
            "FAHRPLAN_UID": case,
            "reason": "negative_duration"
        }
  
    )
print(
    f"Negative duration cases: "
    f"{len(negative_cases)}"
)

# -------------------------------------------------
# 2) OVERLAPPING ACTIVITIES
# -------------------------------------------------
#reasons = []
overlap_cases = set()

for case_id, grp in df.groupby(
    "FAHRPLAN_UID"
):

    grp = grp.sort_values(
        "ALP_ZEITPUNKT_BEREITMELDUNG"
    )

    prev_end = None

    for _, row in grp.iterrows():

        current_start = row[
            "ALP_ZEITPUNKT_BEREITMELDUNG"
        ]

        current_end = row[
            "ALP_ZEITPUNKT_ERFUELLUNG"
        ]

        if pd.isna(current_start):
            continue

        if pd.isna(current_end):
            continue

        if (
            prev_end is not None
            and
            current_start < prev_end
        ):
            overlap_cases.add(case_id)
            break

        prev_end = current_end

for case in overlap_cases:
    reasons.append(
        {
            "FAHRPLAN_UID": case,
            "reason": "overlapping_activities"
        }
    )

print(
    f"Overlap cases: "
    f"{len(overlap_cases)}"
)

# -------------------------------------------------
# 3) EXTREME GATE OUT DELAY
# -------------------------------------------------
#reasons = []
gateout_cases = set()

last_stops = (
    df.groupby("FAHRPLAN_UID")
    .agg(
        last_completion=(
            "ALP_ZEITPUNKT_ERFUELLUNG",
            "max"
        ),
        gate_out=(
            "FP_ZEITPUNKT_GATEOUT",
            "max"
        )
    )
)

last_stops["gate_out_delay_sec"] = (
    last_stops["gate_out"]
    -
    last_stops["last_completion"]
).dt.total_seconds()

MAX_GATE_OUT_DELAY = 10 * 3600

gateout_cases = set(
    last_stops[
        last_stops["gate_out_delay_sec"]
        > MAX_GATE_OUT_DELAY
    ].index
)


for case in gateout_cases:
    reasons.append(
        {
            "FAHRPLAN_UID": case,
            "reason": "gate_out_delay_gt_10h"
        }
    )

print(
    f"Extreme gate-out cases (>10h): "
    f"{len(gateout_cases)}"
)
# -------------------------------------------------
# 4) EXTREME TOTAL CASE DURATION
# -------------------------------------------------
#reasons = []
runtime_cases = set()

case_runtime = (
    df.groupby("FAHRPLAN_UID")
    .agg(
        gate_in=(
            "FP_ZEITPUNKT_ERSTELLUNG",
            "min"
        ),
        gate_out=(
            "FP_ZEITPUNKT_GATEOUT",
            "max"
        )
    )
)

case_runtime["runtime_sec"] = (
    case_runtime["gate_out"]
    -
    case_runtime["gate_in"]
).dt.total_seconds()

MAX_RUNTIME = 12 * 3600

runtime_cases = set(
    case_runtime[
        case_runtime["runtime_sec"]
        > MAX_RUNTIME
    ].index
)

for case in runtime_cases:
    reasons.append(
        {
            "FAHRPLAN_UID": case,
            "reason": "runtime_gt_12h"
        }
    )

print(
    f"Cases >12h runtime: "
    f"{len(runtime_cases)}"
)

print("\nCase Runtime Statistics:")

print(
    case_runtime["runtime_sec"]
    .describe()
)

print(
    case_runtime
    .sort_values(
        "runtime_sec",
        ascending=False
    )
    .head(5)
)

# -------------------------------------------------
# UNION
# -------------------------------------------------


bad_cases = (
    critical_timestamp_cases
    |
    negative_cases
    |
    overlap_cases
    |
    gateout_cases
    |
    runtime_cases
)


print(
    f"Total bad cases: "
    f"{len(bad_cases)}"
)

# -------------------------------------------------
# REMOVE BAD CASES
# -------------------------------------------------

df_clean = df[
    ~df["FAHRPLAN_UID"].isin(
        bad_cases
    )
].copy()

# -------------------------------------------------
# OPTIONAL OUTLIER FILTER
# -------------------------------------------------

duration_sec = (
    df_clean["ALP_ZEITPUNKT_ERFUELLUNG"]
    -
    df_clean["ALP_ZEITPUNKT_BEREITMELDUNG"]
).dt.total_seconds()

extreme_cases = set(
    df_clean.loc[
        duration_sec > 12 * 3600,
        "FAHRPLAN_UID"
    ]
)

print(
    f"Extreme duration cases (>12h): "
    f"{len(extreme_cases)}"
)

df_clean = df_clean[
    ~df_clean["FAHRPLAN_UID"].isin(
        extreme_cases
    )
]
total_cases = df["FAHRPLAN_UID"].nunique()
total_rows = len(df)

#===========
# Rates Calculation
#  ================
missing_timestamp_rate = (

    len(critical_timestamp_cases)

    / total_cases

    * 100

)
negative_rate = (
    len(negative_cases)
    / total_cases
    * 100
)

overlap_rate = (
    len(overlap_cases)
    / total_cases
    * 100
)

total_bad_rate = (
    len(bad_cases)
    / total_cases
    * 100
)

gateout_rate = (
    len(gateout_cases)
    / total_cases
    * 100
)

runtime_rate = (
    len(runtime_cases)
    / total_cases
    * 100
)


print("\nGate-Out Delay Statistics:")

print(
    last_stops.loc[
        last_stops.index.isin(gateout_cases),
        "gate_out_delay_sec"
    ].describe()
)

print("\n========== DATA QUALITY REPORT ==========")

print(f"Rows                    : {total_rows:,}")
print(f"Cases                   : {total_cases:,}")

print()

print(
    f"Missing timestamp cases : "
    f"{len(critical_timestamp_cases):,} "
    f"({missing_timestamp_rate:.4f}%)"
)
print(
    f"Negative duration cases : "
    f"{len(negative_cases):,} "
    f"({negative_rate:.4f}%)"
)

print(
    f"Overlap cases           : "
    f"{len(overlap_cases):,} "
    f"({overlap_rate:.4f}%)"
)

print(
    f"Extreme gate-out cases : "
    f"{len(gateout_cases):,} "
    f"({gateout_rate:.4f}%)"
)

print(
    f"Runtime >12h cases     : "
    f"{len(runtime_cases):,} "
    f"({runtime_rate:.4f}%)"
)



print(
    f"Total invalid cases     : "
    f"{len(bad_cases):,} "
    f"({total_bad_rate:.4f}%)"
)

print()
# -------------------------------------------------
# SAVE
# -------------------------------------------------


reasons_df = (
    pd.DataFrame(reasons)
    .groupby("FAHRPLAN_UID")["reason"]
    .agg(lambda x: ", ".join(sorted(set(x))))
    .reset_index()
)

reasons_df.to_csv(
    "removed_cases_reasons.csv",
    sep=";",
    index=False
)

print(
    f"Removed case report written: "
    f"{len(reasons_df)} entries"
)

df_clean.to_csv(
    OUTPUT,
    sep=";",
    index=False
)

print()
print(
    f"Original rows : {len(df)}"
)
print(
    f"Clean rows    : {len(df_clean)}"
)
print(
    f"Removed rows  : {len(df)-len(df_clean)}"
)
print()
print(
    f"Saved: {OUTPUT}"
)