import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

FAHRPLAN_FILE = (
    "data/interim/CTB/s1_its_fahrplan_cleaned_0304.csv"
)

REPORT_FILE = (
    "data/interim/CTB/Report_0304_mapped.csv"
)

OUTPUT_FILE = (
    "data/interim/CTB/s2_its_fahrplan_cleaned_with_container_ids.csv"
)

# ==========================================================
# LOAD FAHRPLAN
# ==========================================================

print("Loading Fahrplan...")

fahrplan = pd.read_csv(
    FAHRPLAN_FILE,
    sep=";",
    low_memory=False
)

fahrplan.columns = fahrplan.columns.str.strip()

fahrplan["FP_ZEITPUNKT_ERSTELLUNG"] = pd.to_datetime(
    fahrplan["FP_ZEITPUNKT_ERSTELLUNG"],
    errors="coerce"
)

# ==========================================================
# LOAD REPORT
# ==========================================================

print("Loading Report...")

report = pd.read_csv(
    REPORT_FILE,
    sep=";",
    low_memory=False
)

report.columns = report.columns.str.strip()

report["CallupTime"] = pd.to_datetime(
    report["CallupTime"],
    dayfirst=True,
    errors="coerce"
)

# ==========================================================
# TRUCK VISIT MATCH
# ==========================================================

tv_mapping = (
    report[
        [
            "TruckLicenseNbr",
            "CallupTime",
            "TruckVisitGkey"
        ]
    ]
    .drop_duplicates()
)

fahrplan_tv = fahrplan.merge(
    tv_mapping,
    left_on=[
        "LKW_KENNZEICHEN",
        "FP_ZEITPUNKT_ERSTELLUNG"
    ],
    right_on=[
        "TruckLicenseNbr",
        "CallupTime"
    ],
    how="left"
)

# ==========================================================
# CONTAINER MATCH
# ==========================================================

container_mapping = (
    report[
        [
            "TruckVisitGkey",
            "MappedExchangeArea",
            "TranCtrNbr"
        ]
    ]
    .drop_duplicates()
)

fahrplan_tv = fahrplan_tv.merge(
    container_mapping,
    left_on=[
        "TruckVisitGkey",
        "ANLAUFPUNKT_HALTESTELLE"
    ],
    right_on=[
        "TruckVisitGkey",
        "MappedExchangeArea"
    ],
    how="left"
)

# ==========================================================
# RENAME
# ==========================================================

fahrplan_tv.rename(
    columns={
        "TranCtrNbr": "container_id"
    },
    inplace=True
)

# ==========================================================
# FALLBACK MATCHING
# ==========================================================

print("\nSearching for fallback matches...")

yard_counts = (

    fahrplan_tv

    .groupby("FAHRPLAN_UID")

    ["ANLAUFPUNKT_HALTESTELLE"]

    .nunique()

)

single_yard_cases = set(
    yard_counts[
        yard_counts == 1
    ].index
)

print(
    f"Single-yard cases: "
    f"{len(single_yard_cases):,}"
)

fallback_rows = (

    fahrplan_tv["FAHRPLAN_UID"]
    .isin(single_yard_cases)

    &

    fahrplan_tv["container_id"].isna()

    &

    fahrplan_tv["TruckVisitGkey"].notna()

    &

    fahrplan_tv["ANLAUFPUNKT_HALTESTELLE"].notna()

)

print(
    f"Rows eligible for fallback: "
    f"{fallback_rows.sum():,}"
)

# ----------------------------------------------------------
# keep only TruckVisits with exactly ONE container
# ----------------------------------------------------------

fallback_mapping = (

    report[
        [
            "TruckVisitGkey",
            "TranCtrNbr"
        ]
    ]

    .dropna(
        subset=["TranCtrNbr"]
    )

)

tv_container_count = (

    fallback_mapping

    .groupby("TruckVisitGkey")

    ["TranCtrNbr"]

    .nunique()

)

single_container_visits = set(

    tv_container_count[
        tv_container_count == 1
    ].index

)

print(
    f"TruckVisits with exactly one container: "
    f"{len(single_container_visits):,}"
)

fallback_mapping = (

    fallback_mapping[
        fallback_mapping[
            "TruckVisitGkey"
        ].isin(
            single_container_visits
        )
    ]

    .groupby("TruckVisitGkey")

    ["TranCtrNbr"]

    .first()

    .reset_index()

)
before_matches = (
    fahrplan_tv["container_id"]
    .notna()
    .sum()
)

fallback_match = (

    fahrplan_tv.loc[
        fallback_rows,
        ["TruckVisitGkey"]
    ]

    .merge(
        fallback_mapping,
        on="TruckVisitGkey",
        how="left"
    )

)

fahrplan_tv.loc[
    fallback_rows,
    "container_id"
] = (
    fallback_match["TranCtrNbr"]
    .values
)

after_matches = (
    fahrplan_tv["container_id"]
    .notna()
    .sum()
)

print(
    f"Additional container matches: "
    f"{after_matches-before_matches:,}"
)

# ==========================================================
# CLEANUP
# ==========================================================

fahrplan_tv.drop(
    columns=[
        "TruckLicenseNbr",
        "CallupTime",
        "MappedExchangeArea"
    ],
    errors="ignore",
    inplace=True
)

# ==========================================================
# MATCHING REPORT
# ==========================================================

truckvisit_matches = (
    fahrplan_tv["TruckVisitGkey"]
    .notna()
    .sum()
)

container_matches = (
    fahrplan_tv["container_id"]
    .notna()
    .sum()
)

print("\n" + "=" * 60)

print(
    f"Rows: {len(fahrplan_tv):,}"
)

print(
    f"TruckVisit matches: "
    f"{truckvisit_matches:,}"
)

print(
    f"Container matches: "
    f"{container_matches:,}"
)

print(
    f"TruckVisit Match Rate: "
    f"{100 * truckvisit_matches / len(fahrplan_tv):.2f}%"
)

print(
    f"Container Match Rate: "
    f"{100 * container_matches / len(fahrplan_tv):.2f}%"
)

print("=" * 60)

# ==========================================================
# REMAINING UNMATCHED ANALYSIS
# ==========================================================

missing_rows = (
    fahrplan_tv["container_id"]
    .isna()
)

missing_cases = (

    fahrplan_tv.loc[
        missing_rows,
        "FAHRPLAN_UID"
    ]

    .nunique()

)

print("\n" + "=" * 60)
print("UNMATCHED CONTAINER ANALYSIS")
print("=" * 60)

print(
    f"Rows without container: "
    f"{missing_rows.sum():,}"
)

print(
    f"Cases without container: "
    f"{missing_cases:,}"
)

print(
    f"Container assignment rate: "
    f"{100 * fahrplan_tv['container_id'].notna().mean():.2f}%"
)

print("\nRows without TruckVisit:")

print(
    fahrplan_tv[
        "TruckVisitGkey"
    ].isna().sum()
)

print("\nRows with TruckVisit but no Container:")

print(
    (
        fahrplan_tv[
            "TruckVisitGkey"
        ].notna()

        &

        fahrplan_tv[
            "container_id"
        ].isna()
    ).sum()
)

print("\nTop affected Yard Areas:")

print(
    fahrplan_tv.loc[
        missing_rows,
        "ANLAUFPUNKT_HALTESTELLE"
    ]
    .value_counts()
    .head(20)
)

# ==========================================================
# EXPORT REMAINING PROBLEM CASES
# ==========================================================

unmatched_cases = fahrplan_tv.loc[
    missing_rows
].copy()

unmatched_cases.to_csv(
    "containerless_cases.csv",
    sep=";",
    index=False
)

print(
    "\nWritten: containerless_cases.csv"
)
print("\n" + "=" * 60)
print("REMOVIN CASES WITH MITSSING CONTAINER IDS")
# ==========================================================
# REMOVE CASES WITHOUT CONTAINER
# ==========================================================

rows_before = len(fahrplan_tv)
cases_before = fahrplan_tv["FAHRPLAN_UID"].nunique()

missing_container_mask = (
    fahrplan_tv["container_id"].isna()
)

missing_container_cases = set(
    fahrplan_tv.loc[
        missing_container_mask,
        "FAHRPLAN_UID"
    ]
)

print("\n" + "=" * 60)
print("REMOVE CONTAINERLESS VISITS")
print("=" * 60)

print(
    f"Rows without container: "
    f"{missing_container_mask.sum():,}"
)

print(
    f"Cases without container: "
    f"{len(missing_container_cases):,}"
)

# komplette Anläufe entfernen

fahrplan_tv = fahrplan_tv[
    ~fahrplan_tv["FAHRPLAN_UID"].isin(
        missing_container_cases
    )
].copy()

rows_after = len(fahrplan_tv)
cases_after = fahrplan_tv["FAHRPLAN_UID"].nunique()

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

#============== ANALYSE
unmatched_ho2 = fahrplan_tv[
    (fahrplan_tv["container_id"].isna())
    &
    (
        fahrplan_tv["ANLAUFPUNKT_HALTESTELLE"]
        == "HO2"
    )
]

print(len(unmatched_ho2))

print(
    unmatched_ho2[
        "TruckVisitGkey"
    ]
    .isna()
    .sum()
)

print(
    unmatched_ho2[
        "TruckVisitGkey"
    ]
    .notna()
    .sum()
)
#====================



# ==========================================================
# SAVE
# ==========================================================

fahrplan_tv.to_csv(
    OUTPUT_FILE,
    sep=";",
    index=False
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)