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

BASELINE_FILE = (
    "data/raw/CTB/"
    "transition_baseline.csv"
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

print("Loading Baselines...")

baseline = pd.read_csv(
    BASELINE_FILE
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
# BASELINE LOOKUP
# ==========================================================

baseline_lookup = {}

for _, row in baseline.iterrows():

    baseline_lookup[
        (
            str(row["activity"]),
            str(row["next_activity"])
        )
    ] = float(
        row["baseline_duration_sec"]
    )

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
missing_baselines = 0

for case_id, case_df in fp.groupby(
    "FAHRPLAN_UID"
):

    case_df = case_df.sort_values(
        "ALP_ZEITPUNKT_BEREITMELDUNG"
    )

    first = case_df.iloc[0]

    gate_enabled = first[
        "StageStartTime"
    ]

    gate_start = first[
        "FP_ZEITPUNKT_ERSTELLUNG"
    ]

    if pd.isna(gate_enabled):
        missing_gate_enabled += 1
        gate_enabled = gate_start

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

        "enabled:timestamp":
            gate_enabled,

        "start:timestamp":
            gate_start,

        "time:timestamp":
            gate_start,

        **case_features
    })

    previous_activity = "Gate In"

    previous_complete = gate_start

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

        baseline_sec = baseline_lookup.get(
            (
                previous_activity,
                stop
            ),
            None
        )

        if baseline_sec is None:

            missing_baselines += 1
            baseline_sec = 0

        enabled_ts = (
            previous_complete
            +
            pd.to_timedelta(
                baseline_sec,
                unit="s"
            )
        )

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

            "enabled:timestamp":
                enabled_ts,

            "start:timestamp":
                start_ts,

            "time:timestamp":
                complete_ts,

            **case_features
        })

        previous_activity = stop
        previous_complete = complete_ts

    # ------------------------------------------------------
    # GATE OUT
    # ------------------------------------------------------

    gate_out_baseline = baseline_lookup.get(
        (
            previous_activity,
            "Gate Out"
        ),
        None
    )

    if gate_out_baseline is None:

        missing_baselines += 1
        gate_out_baseline = 0

    gate_out_enabled = (
        previous_complete
        +
        pd.to_timedelta(
            gate_out_baseline,
            unit="s"
        )
    )

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

        "enabled:timestamp":
            gate_out_enabled,

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
    f"Missing GateIn entry timestamps: "
    f"{missing_gate_enabled:,}"
)

print(
    f"Missing baseline transitions: "
    f"{missing_baselines:,}"
)

print()

print(
    eventlog["concept:name"]
    .value_counts()
    .head(30)
)

print("=" * 70)

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