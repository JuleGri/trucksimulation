import pandas as pd
import pm4py

# ==========================================================
# CONFIG
# ==========================================================

INPUT_CSV = (
    "data/processed/CTB/"
    "s6_eventlog_target_rank_features.csv"
)

OUTPUT_XES = (
    "data/processed/CTB/xes_files/"
    "s6_eventlog_target_rank_features.xes"
)

# ==========================================================
# LOAD
# ==========================================================

print("Loading CSV...")

df = pd.read_csv(
    INPUT_CSV
)

# ==========================================================
# TIMESTAMP CONVERSION
# ==========================================================

for col in [
    "enabled:timestamp",
    "start:timestamp",
    "time:timestamp"
]:
    df[col] = pd.to_datetime(
        df[col],
        errors="coerce"
    )
# ==========================================================
# ARRIVAL CONTEXT FEATURES
# ==========================================================

# erster Startzeitpunkt des Cases
case_arrival = (
    df.groupby(
        "case:concept:name"
    )["start:timestamp"]
    .transform("min")
)

# Prosit-Style Features
df["@@hour"] = (
    case_arrival.dt.hour
)

df["@@weekday"] = (
    case_arrival.dt.dayofweek
)

print(
    df[
        [
            "@@hour",
            "@@weekday"
        ]
    ].head()
)
# ==========================================================
# PM4PY FORMAT
# ==========================================================

df = pm4py.format_dataframe(
    df,
    case_id="case:concept:name",
    activity_key="concept:name",
    timestamp_key="time:timestamp"
)

# Anzahl vor dem Filter
n_before = len(df)

# Unknown Events zählen
n_unknown = (
    df["concept:name"]
    .str.endswith(
        "_unknown",
        na=False
    )
    .sum()
)

# Entfernen
df = df[
    ~df["concept:name"]
    .str.endswith(
        "_unknown",
        na=False
    )
]

# Anzahl nach dem Filter
n_after = len(df)

print(f"Unknown events removed: {n_unknown:,}")
print(f"Events before: {n_before:,}")
print(f"Events after:  {n_after:,}")
# ==========================================================
# CONVERT TO EVENT LOG
# ==========================================================

event_log = pm4py.convert_to_event_log(
    df
)

# ==========================================================
# EXPORT XES
# ==========================================================

pm4py.write_xes(
    event_log,
    OUTPUT_XES
)

# ==========================================================
# REPORT
# ==========================================================

print("\n" + "=" * 60)

print(
    f"Cases: "
    f"{df['case:concept:name'].nunique():,}"
)

print(
    f"Events: "
    f"{len(df):,}"
)

print(
    f"\nSaved:\n{OUTPUT_XES}"
)

print("=" * 60)