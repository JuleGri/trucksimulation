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
    sep=";"
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
    sep=";"
)

report.columns = report.columns.str.strip()

report["CallupTime"] = pd.to_datetime(
    report["CallupTime"],
    dayfirst=True,
    errors="coerce"
)

# ==========================================================
# ONLY ENTRY RECORDS
# ==========================================================

report_in = report.copy()

# ==========================================================
# TRUCK VISIT MATCH
# ==========================================================

tv_mapping = (
    report_in[
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
# CONTAINER + FLOWTYPE
# ==========================================================

container_mapping = (
    report[
        [
            "TruckVisitGkey",
            "MappedExchangeArea",
            "TranCtrNbr"
            #"FlowType"
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
        #"FlowType": "flow_type"
    },
    inplace=True
)

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
# REPORT
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

""" flow_matches = (
    fahrplan_tv["flow_type"]
    .notna()
    .sum()
) """

print("\n" + "=" * 60)

print(
    f"Rows: {len(fahrplan_tv):,}"
)

print(
    f"TruckVisit matches: {truckvisit_matches:,}"
)

print(
    f"Container matches: {container_matches:,}"
)

""" print(
    f"FlowType matches: {flow_matches:,}"
) """

print(
    f"TruckVisit Match Rate: "
    f"{100 * truckvisit_matches / len(fahrplan_tv):.2f}%"
)

print(
    f"Container Match Rate: "
    f"{100 * container_matches / len(fahrplan_tv):.2f}%"
)

""" print(
    f"FlowType Match Rate: "
    f"{100 * flow_matches / len(fahrplan_tv):.2f}%"
) """

print("=" * 60)

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