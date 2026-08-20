import sys
from pathlib import Path

import pandas as pd
import pm4py

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
from _eventlog_contract import (  # noqa: E402
    canonicalize_case_order,
    select_prosit_dataframe,
    to_pm4py_event_log,
    validate_eventlog_contract,
)

INPUT_CSV = (
    "data/processed/CTB/s6_eventlog_target_rank_features.csv"
)

OUTPUT_XES = (
    "data/processed/CTB/xes_files/"
    "s6_sample_50.000_eventlog_target_rank_features.xes"
)

OUTPUT_CSV = (
    "data/processed/CTB/sampled_real_eventlogs/"
    "s6_sample_50.000_eventlog_target_rank_features.csv"
)

df = pd.read_csv(INPUT_CSV)

for col in [
    "enabled:timestamp",
    "start:timestamp",
    "time:timestamp"
]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])


# =====================================================
# RANDOM SAMPLE
# =====================================================

sample_cases = (
    df["case:concept:name"]
    .drop_duplicates()
    .sample(
        n=50000,
        random_state=42
    )
)

df_small = df[
    df["case:concept:name"]
    .isin(sample_cases)
].copy()

# =====================================================
# XES
# =====================================================

df_small = canonicalize_case_order(df_small)
validate_eventlog_contract(df_small, label="sampled real event log")
df_small, dropped = select_prosit_dataframe(df_small, label="sampled real event log")
if dropped:
    print(f"Non-ProSiT attributes excluded from sample: {', '.join(dropped)}")
log_small = to_pm4py_event_log(df_small, label="sampled real event log")

Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
df_small.to_csv(
    OUTPUT_CSV,
    index=False
)

Path(OUTPUT_XES).parent.mkdir(parents=True, exist_ok=True)
pm4py.write_xes(
    log_small,
    OUTPUT_XES
)

print(
    f"Cases: {df_small['case:concept:name'].nunique()}"
)

print(
    f"Events: {len(df_small)}"
)
