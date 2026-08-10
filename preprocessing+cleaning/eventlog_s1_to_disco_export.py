import pandas as pd

# ==========================================================
# LOAD LIFECYCLE EVENT LOG
# ==========================================================

INPUT_FILE = "data/interim/CTB/its_eventlog_s1.csv"
OUTPUT_FILE = "data/processed/CTB/disco_eventlog_area_split.csv"

# ==========================================================
# LOAD EVENT LOG
# ==========================================================

print("Loading event log...")

df = pd.read_csv(INPUT_FILE)

df["time:timestamp"] = pd.to_datetime(
    df["time:timestamp"]
)

# ==========================================================
# ADJUST AREA LABELS FOR DISCO
# ==========================================================

df.loc[
    df["concept:name"] == "Gate In",
    "area"
] = "Gate In"

df.loc[
    df["concept:name"] == "Gate Out",
    "area"
] = "Gate Out"

# ==========================================================
# SPLIT START / COMPLETE
# ==========================================================

starts = df[
    df["lifecycle:transition"] == "start"
].copy()

completes = df[
    df["lifecycle:transition"] == "complete"
].copy()

# ==========================================================
# MERGE USING ACTIVITY INSTANCE
# ==========================================================

merged = pd.merge(
    completes,
    starts[
        [
            "activity_instance",
            "time:timestamp"
        ]
    ],
    on="activity_instance",
    how="left",
    suffixes=("_complete", "_start")
)

# ==========================================================
# CALCULATE DURATION
# ==========================================================

merged["duration_seconds"] = (
    merged["time:timestamp_complete"]
    - merged["time:timestamp_start"]
).dt.total_seconds()

# ==========================================================
# BUILD DISCO EXPORT
# ==========================================================

disco_df = pd.DataFrame({

    "Case ID":
        merged["case:concept:name"],

    "Activity":
        merged["concept:name"],

    "Timestamp":
        merged["time:timestamp_start"],

    "Duration (seconds)":
        merged["duration_seconds"],

    "Area":
        merged["area"],

    "Hour":
        merged["hour"],

    "Weekday":
        merged["weekday"]

})

# ==========================================================
# SORT
# ==========================================================

disco_df = disco_df.sort_values(
    ["Case ID", "Timestamp"]
)

# ==========================================================
# SAVE
# ==========================================================

disco_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Exported {len(disco_df):,} events")
print(f"Saved to {OUTPUT_FILE}")