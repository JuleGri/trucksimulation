import sys
from pathlib import Path

import pandas as pd
import pm4py

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
from _eventlog_contract import select_prosit_dataframe, to_pm4py_event_log  # noqa: E402

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
    if col in df.columns:
        df[col] = pd.to_datetime(
            df[col],
            errors="coerce"
        )

# Anzahl vor dem Filter
n_before = len(df)

# Unknown cases count.  Remove the complete case; deleting only the unknown
# event could manufacture an invalid Gate-only process.
unknown_mask = (
    df["concept:name"]
    .str.endswith(
        "_unknown",
        na=False
    )
)
n_unknown = int(unknown_mask.sum())
unknown_cases = set(df.loc[unknown_mask, "case:concept:name"])

# Entfernen
df = df[
    ~df["case:concept:name"].isin(unknown_cases)
]

# Anzahl nach dem Filter
n_after = len(df)

print(f"Unknown events found: {n_unknown:,}")
print(f"Complete cases removed: {len(unknown_cases):,}")
print(f"Events before: {n_before:,}")
print(f"Events after:  {n_after:,}")
# ==========================================================
# CONVERT TO EVENT LOG
# ==========================================================

df, dropped = select_prosit_dataframe(df, label="full s6 XES export")
if dropped:
    print(f"Non-ProSiT attributes excluded from XES: {', '.join(dropped)}")
event_log = to_pm4py_event_log(df, label="full s6 XES export")

# ==========================================================
# EXPORT XES
# ==========================================================

Path(OUTPUT_XES).parent.mkdir(parents=True, exist_ok=True)
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
