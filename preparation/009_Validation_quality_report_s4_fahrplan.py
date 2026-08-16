import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

INPUT_FILE = (
    "data/interim/CTB/"
    "s4_its_fahrplan_with_case_features.csv"
)

# ==========================================================
# LOAD
# ==========================================================

print("Loading schedule...")

df = pd.read_csv(
    INPUT_FILE,
    sep=";"
)

# ==========================================================
# BASIC STATS
# ==========================================================

print("\n" + "=" * 80)
print("GENERAL STATISTICS")
print("=" * 80)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Truck Visits: "
    f"{df['FAHRPLAN_UID'].nunique():,}"
)

print(
    f"Stops: "
    f"{df['ANLAUFPUNKT_UID'].nunique():,}"
)

# ==========================================================
# DUPLICATE STOP CHECK
# ==========================================================

print("\n" + "=" * 80)
print("STOP DUPLICATE CHECK")
print("=" * 80)

dup_check = (
    df.groupby(
        [
            "FAHRPLAN_UID",
            "ANLAUFPUNKT_UID"
        ]
    )
    .size()
)

print(
    dup_check.value_counts()
    .sort_index()
)

duplicates = (
    dup_check > 1
).sum()

print(
    f"\nDuplicated Stops: "
    f"{duplicates:,}"
)

# ==========================================================
# FLOW TYPE DISTRIBUTION
# ==========================================================

print("\n" + "=" * 80)
print("PROCESS FLOW TYPES")
print("=" * 80)

print(
    df[
        [
            "FAHRPLAN_UID",
            "process_flow_type"
        ]
    ]
    .drop_duplicates()
    ["process_flow_type"]
    .value_counts()
)

# ==========================================================
# STOP FLOW DISTRIBUTION
# ==========================================================

print("\n" + "=" * 80)
print("TOP 30 STOP FLOWS")
print("=" * 80)

print(
    df["stop_flow"]
    .value_counts()
    .head(30)
)

# ==========================================================
# UNKNOWN FLOWS
# ==========================================================

print("\n" + "=" * 80)
print("UNKNOWN FLOW ANALYSIS")
print("=" * 80)

unknown_cases = (
    df[
        [
            "FAHRPLAN_UID",
            "process_flow_type"
        ]
    ]
    .drop_duplicates()
)

unknown_cases = unknown_cases[
    unknown_cases[
        "process_flow_type"
    ] == "unknown"
]

print(
    f"Unknown Cases: "
    f"{len(unknown_cases):,}"
)

# ==========================================================
# CONTAINER MATCH QUALITY
# ==========================================================

print("\n" + "=" * 80)
print("CONTAINER MATCHING")
print("=" * 80)

container_stats = []

for uid, grp in df.groupby(
    "FAHRPLAN_UID"
):

    unique_ids = (
        grp["container_id"]
        .dropna()
        .nunique()
    )

    declared = (
        grp[
            "ANLAUFPUNKT_ANZAHL_CONTAINER"
        ]
        .fillna(0)
        .sum()
    )

    container_stats.append({

        "uid": uid,

        "unique_ids":
            unique_ids,

        "declared":
            declared,

        "difference":
            declared - unique_ids
    })

container_stats = pd.DataFrame(
    container_stats
)

print(
    container_stats[
        "difference"
    ]
    .value_counts()
    .sort_index()
    .head(20)
)

print()

print(
    f"Cases with mismatch: "
    f"{(container_stats['difference'] != 0).sum():,}"
)

# ==========================================================
# MISSING CONTAINERS
# ==========================================================

print("\n" + "=" * 80)
print("MISSING CONTAINER IDS")
print("=" * 80)

missing_container_cases = (
    container_stats[
        container_stats[
            "unique_ids"
        ] == 0
    ]
)

print(
    f"Cases without container ids: "
    f"{len(missing_container_cases):,}"
)

# ==========================================================
# HAZARDOUS / REEFER
# ==========================================================

print("\n" + "=" * 80)
print("SPECIAL CONTAINERS")
print("=" * 80)

case_features = (
    df[
        [
            "FAHRPLAN_UID",
            "has_hazardous",
            "has_reefer"
        ]
    ]
    .drop_duplicates()
)

print(
    f"Hazardous Visits: "
    f"{case_features['has_hazardous'].sum():,}"
)

print(
    f"Reefer Visits: "
    f"{case_features['has_reefer'].sum():,}"
)

# ==========================================================
# VISIT COMPLEXITY
# ==========================================================

print("\n" + "=" * 80)
print("VISIT COMPLEXITY")
print("=" * 80)

complexity = (
    df[
        [
            "FAHRPLAN_UID",
            "visit_complexity"
        ]
    ]
    .drop_duplicates()
)

print(
    complexity[
        "visit_complexity"
    ]
    .describe()
)

# ==========================================================
# CASE FEATURE QUALITY
# ==========================================================

print("\n" + "=" * 80)
print("CASE FEATURE DISTRIBUTIONS")
print("=" * 80)

case_df = (
    df[
        [
            "FAHRPLAN_UID",
            "n_containers",
            "n_stops",
            "n_deliveries",
            "n_receives",
            "full_ratio"
        ]
    ]
    .drop_duplicates()
)

print(case_df.describe())

# ==========================================================
# MIXED STOP ANALYSIS
# ==========================================================

print("\n" + "=" * 80)
print("TOP MIXED STOPS")
print("=" * 80)

mixed = df[
    df["stop_flow"]
    .str.contains(
        "_mixed",
        na=False
    )
]

print(
    mixed["ANLAUFPUNKT_HALTESTELLE"]
    .value_counts()
    .head(20)
)

# ==========================================================
# STOP DURATION ANALYSIS
# ==========================================================

print("\n" + "=" * 80)
print("AVERAGE EXECUTION TIME PER STOP FLOW")
print("=" * 80)

duration = (
    df.groupby("stop_flow")
    [
        "ALP_DAUER_ABFERTIGUNG_SEK"
    ]
    .mean()
    .sort_values(
        ascending=False
    )
)

print(
    duration.head(30)
)

print("\n" + "=" * 80)
print("QUALITY REPORT FINISHED")
print("=" * 80)