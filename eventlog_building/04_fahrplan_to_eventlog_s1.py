"""
04_fahrplan_to_eventlog_s1.py

Constructs the stage-1 event log from the cleaned Fahrplan, Report lookup
and case-level features.

Methodological note (2026-08 revision):
    The canonical CTB event log contains only *observed* temporal information:
      - start:timestamp  (operational start of the activity)
      - time:timestamp   (operational completion of the activity)
    Activity enablement is a process-model concept and is NOT reconstructed
    as an input-log column.  ProSiT derives enablement internally via A*
    alignment against the discovered Petri net.

    The OCR/entry timestamp from the Report system is retained as the
    auxiliary column ``ocr_timestamp`` for Gate In events where available.
    It is NOT mapped to ``enabled:timestamp``.

    The transition_baseline.csv is no longer loaded or used by this script.
"""

import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

FAHRPLAN_FILE = (
    "data/interim/CTB/"
    "s4_its_fahrplan_with_case_features.csv"
)

REPORT_FILE = (
    "data/interim/CTB/"
    "Report_0304_mapped.csv"
)

OUTPUT_FILE = (
    "data/processed/CTB/s1_eventlog_with_case_features.csv"
)

# ==========================================================
# LOAD
# ==========================================================

print("Loading Fahrplan...")

fp = pd.read_csv(
    FAHRPLAN_FILE,
    sep=";"
)

print("Loading Report...")

report = pd.read_csv(
    REPORT_FILE,
    sep=";"
)

# ==========================================================
# DATETIME
# ==========================================================

for col in [
    "FP_ZEITPUNKT_ERSTELLUNG",
    "ALP_ZEITPUNKT_BEREITMELDUNG",
    "ALP_ZEITPUNKT_ERFUELLUNG",
    "FP_ZEITPUNKT_GATEOUT"
]:
    fp[col] = pd.to_datetime(
        fp[col],
        errors="coerce"
    )

report["CallupTime"] = pd.to_datetime(
    report["CallupTime"],
    dayfirst=True,
    errors="coerce"
)

report["StageStartTime"] = pd.to_datetime(
    report["StageStartTime"],
    errors="coerce"
)

# ==========================================================
# BUILD REPORT ENTRY LOOKUP
# ==========================================================

entry_lookup = report[
    report["StageId"]
    .astype(str)
    .str.lower()
    .eq("entry")
].copy()

entry_lookup = entry_lookup[
    [
        "TruckLicenseNbr",
        "CallupTime",
        "StageStartTime"
    ]
].drop_duplicates()

# ==========================================================
# MERGE ENTRY TIME
# ==========================================================

fp = fp.merge(
    entry_lookup,
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
# EVENT CREATION
# ==========================================================

events = []

missing_gate_enabled = 0

for case_id, case_df in fp.groupby(
    "FAHRPLAN_UID"
):

    case_df = case_df.sort_values(
        "ALP_ZEITPUNKT_BEREITMELDUNG"
    )

    first = case_df.iloc[0]

    # OCR/entry timestamp retained as auxiliary (not as enablement)
    ocr_timestamp = first["StageStartTime"]

    gate_start = first[
        "FP_ZEITPUNKT_ERSTELLUNG"
    ]

    if pd.isna(ocr_timestamp):
        missing_gate_enabled += 1

    case_features = {

        "process_flow_type":
            first["process_flow_type"],

        "n_containers":
            first["n_containers"],

        "n_stops":
            first["n_stops"],

        "n_deliveries":
            first["n_deliveries"],

        "n_receives":
            first["n_receives"],

        "has_hazardous":
            first["has_hazardous"],

        "has_reefer":
            first["has_reefer"],

        "full_ratio":
            first["full_ratio"],

        "visit_complexity":
            first["visit_complexity"]
    }

    # ------------------------------------------------------
    # GATE IN
    # ------------------------------------------------------

    events.append({

        "case:concept:name":
            case_id,

        "concept:name":
            "Gate In",

        "org:resource":
            "Res.GateIn",

        "start:timestamp":
            gate_start,

        "time:timestamp":
            gate_start,

        "ocr_timestamp":
            ocr_timestamp if pd.notna(ocr_timestamp) else pd.NaT,

        **case_features
    })

    # ------------------------------------------------------
    # STOPS
    # ------------------------------------------------------

    for _, row in case_df.iterrows():

        stop = row[
            "ANLAUFPUNKT_HALTESTELLE"
        ]

        activity = row[
            "stop_flow"
        ]

        start_ts = row[
            "ALP_ZEITPUNKT_BEREITMELDUNG"
        ]

        complete_ts = row[
            "ALP_ZEITPUNKT_ERFUELLUNG"
        ]

        events.append({

            "case:concept:name":
                case_id,

            "concept:name":
                activity,

            "org:resource":
                stop,

            "start:timestamp":
                start_ts,

            "time:timestamp":
                complete_ts,

            **case_features
        })

    # ------------------------------------------------------
    # GATE OUT
    # ------------------------------------------------------

    gate_out_time = (
        case_df.iloc[-1]
        ["FP_ZEITPUNKT_GATEOUT"]
    )

    events.append({

        "case:concept:name":
            case_id,

        "concept:name":
            "Gate Out",

        "org:resource":
            "Res.GateOut",

        "start:timestamp":
            gate_out_time,

        "time:timestamp":
            gate_out_time,

        **case_features
    })

# ==========================================================
# EVENTLOG
# ==========================================================

eventlog = pd.DataFrame(events)

eventlog = eventlog.sort_values(
    [
        "case:concept:name",
        "start:timestamp"
    ]
)

# ==========================================================
# QUALITY REPORT
# ==========================================================

print("\n" + "=" * 70)
print("EVENTLOG BUILD REPORT")
print("=" * 70)

print(
    f"Cases: "
    f"{eventlog['case:concept:name'].nunique():,}"
)

print(
    f"Events: "
    f"{len(eventlog):,}"
)

print()

print(
    f"Missing GateIn OCR/entry timestamps: "
    f"{missing_gate_enabled:,}"
)

print()

print(
    eventlog["concept:name"]
    .value_counts()
    .head(30)
)

print("=" * 70)

# ==========================================================
# AUTOMATED ASSERTIONS
# ==========================================================

assert "enabled:timestamp" not in eventlog.columns, \
    "enabled:timestamp must NOT be in the canonical event log"

assert "start:timestamp" in eventlog.columns
assert "time:timestamp" in eventlog.columns

eventlog["start:timestamp"] = pd.to_datetime(eventlog["start:timestamp"], errors="coerce")
eventlog["time:timestamp"] = pd.to_datetime(eventlog["time:timestamp"], errors="coerce")

yard_mask = ~eventlog["concept:name"].isin(["Gate In", "Gate Out"])
svc = (eventlog.loc[yard_mask, "time:timestamp"] - eventlog.loc[yard_mask, "start:timestamp"]).dt.total_seconds()
n_negative = (svc < 0).sum()
if n_negative > 0:
    print(f"WARNING: {n_negative} yard events have negative service time (complete < start)")
else:
    print("OK: All yard-activity service times are non-negative")

n_dup = eventlog.duplicated().sum()
assert n_dup == 0, f"Found {n_dup} duplicate rows in the event log"

print("All assertions passed.")

# ==========================================================
# SAVE
# ==========================================================

eventlog.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)