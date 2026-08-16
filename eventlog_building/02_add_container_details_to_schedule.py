import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

FAHRPLAN_FILE = (
    "data/interim/CTB/"
    "s2_its_fahrplan_cleaned_with_container_ids.csv"
)

CONTAINER_MASTER_FILE = (
    "data/raw/CTB/"
    "container_master_0304.csv"
)

OUTPUT_FILE = (
    "data/interim/CTB/"
    "s3_its_fahrplan_container_details.csv"
)

# ==========================================================
# LOAD
# ==========================================================

print("Loading Fahrplan...")

fahrplan = pd.read_csv(
    FAHRPLAN_FILE,
    sep=";"
)

fahrplan.columns = (
    fahrplan.columns.str.strip()
)

print("Loading Container Master...")

container_master = pd.read_csv(
    CONTAINER_MASTER_FILE,
    sep=";"
)

container_master.columns = (
    container_master.columns.str.strip()
)

# ==========================================================
# DATETIME
# ==========================================================

fahrplan["FP_ZEITPUNKT_ERSTELLUNG"] = pd.to_datetime(
    fahrplan["FP_ZEITPUNKT_ERSTELLUNG"],
    errors="coerce"
)

fahrplan["FP_ZEITPUNKT_GATEOUT"] = pd.to_datetime(
    fahrplan["FP_ZEITPUNKT_GATEOUT"],
    errors="coerce"
)

container_master["reference_timestamp"] = pd.to_datetime(
    container_master["reference_timestamp"],
    dayfirst=True,
    errors="coerce"
)

# ==========================================================
# PREPARE MASTER
# ==========================================================

container_master["flow_type"] = (
    container_master["flow_type"]
    .astype(str)
    .str.upper()
)

# DELIVERY -> Truck = Inbound Carrier

delivery = container_master[
    container_master["flow_type"] == "DELIVERY"
].copy()

delivery["truck_license_match"] = (
    delivery["Inbound_Carrier"]
)

# RECEIVE -> Truck = Outbound Carrier

receive = container_master[
    container_master["flow_type"] == "RECEIVE"
].copy()

receive["truck_license_match"] = (
    receive["Outbound_Carrier"]
)

container_master_match = pd.concat(
    [delivery, receive],
    ignore_index=True
)

# ==========================================================
# KEEP ATTRIBUTES
# ==========================================================

keep_cols = [
    "container_id",
    "flow_type",
    "truck_license_match",
    "reference_timestamp",
    "container_type",
    "container_size",
    "full_empty",
    "is_full",
    "imco",
    "is_hazardous",
    "is_reefer",
    "Inbound_Carrier",
    "Outbound_Carrier",
    "CTR_TYP_LANG",
    "ORT",
    "REED",
    "container_complexity"
]

container_master_match = (
    container_master_match[keep_cols]
    .drop_duplicates()
)

# ==========================================================
# UNIQUE ROW ID
# ==========================================================

fahrplan["_row_id"] = range(
    len(fahrplan)
)

# ==========================================================
# MATCH
# ==========================================================

fahrplan = fahrplan.merge(
    container_master_match,
    left_on=[
        "container_id",
        "LKW_KENNZEICHEN"
    ],
    right_on=[
        "container_id",
        "truck_license_match"
    ],
    how="left"
)

# ==========================================================
# RENAME MASTER FLOW TYPE
# ==========================================================

fahrplan = fahrplan.rename(
    columns={
        "flow_type": "flow_type_master"
    }
)

# ==========================================================
# BEST TEMPORAL MATCH
# ==========================================================

fahrplan["matching_timestamp"] = pd.NaT

delivery_mask = (
    fahrplan["flow_type_master"]
    .eq("DELIVERY")
)

receive_mask = (
    fahrplan["flow_type_master"]
    .eq("RECEIVE")
)

# truck delivers a container to terminal
fahrplan.loc[
    delivery_mask,
    "matching_timestamp"
] = fahrplan.loc[
    delivery_mask,
    "FP_ZEITPUNKT_ERSTELLUNG"
]

# truck receives a container from terminal
fahrplan.loc[
    receive_mask,
    "matching_timestamp"
] = fahrplan.loc[
    receive_mask,
    "FP_ZEITPUNKT_GATEOUT"
]

fahrplan["time_distance_sec"] = (
    fahrplan["reference_timestamp"]
    -
    fahrplan["matching_timestamp"]
).abs().dt.total_seconds()

candidate_rows = len(fahrplan)

fahrplan = (
    fahrplan
    .sort_values(
        [
            "_row_id",
            "time_distance_sec"
        ],
        na_position="last"
    )
    .drop_duplicates(
        subset="_row_id",
        keep="first"
    )
    .copy()
)

removed_candidates = (
    candidate_rows - len(fahrplan)
)

print()
print(
    f"Ambiguous container matches resolved: "
    f"{removed_candidates:,}"
)

# ==========================================================
# QUALITY REPORT
# ==========================================================

matched_rows = (
    fahrplan["container_type"]
    .notna()
    .sum()
)

rows_with_container = (
    fahrplan["container_id"]
    .notna()
    .sum()
)

print("\n" + "=" * 70)
print("CONTAINER MASTER MATCHING REPORT")
print("=" * 70)

print(
    f"Rows in Fahrplan:               "
    f"{len(fahrplan):,}"
)

print(
    f"Rows with container id:         "
    f"{rows_with_container:,}"
)

print(
    f"Master matches:                "
    f"{matched_rows:,}"
)

if rows_with_container > 0:

    print(
        f"Container match rate:          "
        f"{100 * matched_rows / rows_with_container:.2f}%"
    )

print(
    f"Ambiguous matches resolved:    "
    f"{removed_candidates:,}"
)

print()
print("=" * 70)

# ==========================================================
# FLOW TYPE REPORT
# ==========================================================

print("\nFLOW TYPE COMPARISON")

comparison = (
    fahrplan[
        fahrplan["flow_type_master"]
        .notna()
    ]
    .groupby(
        "flow_type_master"
    )
    .size()
    .reset_index(name="count")
    .sort_values(
        "count",
        ascending=False
    )
)

print(
    comparison.head(20)
)

# ==========================================================
# CLEANUP
# ==========================================================

fahrplan = fahrplan.drop(
    columns=[
        "truck_license_match",
        "_row_id",
        "matching_timestamp",
        "time_distance_sec"
    ],
    errors="ignore"
)

# ==========================================================
# SAVE
# ==========================================================

fahrplan.to_csv(
    OUTPUT_FILE,
    sep=";",
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)