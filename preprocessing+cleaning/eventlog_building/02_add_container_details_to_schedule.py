import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

FAHRPLAN_FILE = (
    "data/interim/CTB/"
    "s2_its_fahrplan_cleaned_with_container_ids.csv"
)

CONTAINER_MASTER_FILE = (
    "data/raw/CTB/container_master_0304.csv"
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
# PRESERVE REPORT FLOW TYPE
# ==========================================================

""" fahrplan = fahrplan.rename(
    columns={
        "flow_type": "flow_type_report"
    }
) """

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
# FLOW TYPE COMPARISON
# ==========================================================

""" fahrplan["flow_match"] = (
    fahrplan["flow_type_report"]
    .astype(str)
    .str.upper()
    ==
    fahrplan["flow_type_master"]
    .astype(str)
    .str.upper()
) """

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

flow_comparable = (
    fahrplan["flow_type_master"]
    .notna()
).sum()

""" flow_equal = (
    fahrplan["flow_match"]
).sum()
 """
""" flow_different = (
    flow_comparable
    - flow_equal
) """

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

print()









print("=" * 70)

# ==========================================================
# FLOW TYPE CONFUSION
# ==========================================================

print("\nFLOW TYPE COMPARISON")

comparison = (
    fahrplan[
        fahrplan["flow_type_master"]
        .notna()
    ]
    .groupby(
        [
            #"flow_type_report",
            "flow_type_master"
        ]
    )
    .size()
    .reset_index(name="count")
    .sort_values(
        "count",
        ascending=False
    )
)

print(comparison.head(20))

# ==========================================================
# CLEANUP
# ==========================================================

fahrplan = fahrplan.drop(
    columns=[
        "truck_license_match"
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