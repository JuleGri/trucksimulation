import pandas as pd
import pm4py

INPUT_CSV = (
    "data/processed/CTB/s6_eventlog_target_rank_features.csv"
)

OUTPUT_XES = (
    "data/processed/CTB/xes_files/"
    "s5_sample_30.000_eventlog_one_block_target_features.xes"
)

OUTPUT_CSV = (
    "data/processed/CTB/sampled_real_eventlogs/"
    "s6_sample_24.000_eventlog_target_rank_features.csv"
)

df = pd.read_csv(INPUT_CSV)

for col in [
    "enabled:timestamp",
    "start:timestamp",
    "time:timestamp"
]:
    df[col] = pd.to_datetime(df[col])


# =====================================================
# RANDOM SAMPLE
# =====================================================

sample_cases = (
    df["case:concept:name"]
    .drop_duplicates()
    .sample(
        n=24000,
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

df_small = pm4py.format_dataframe(
    df_small,
    case_id="case:concept:name",
    activity_key="concept:name",
    timestamp_key="time:timestamp"
)

df_small = df_small.drop(
    columns=[
        "@@index",
        "@@case_index"
    ],
    errors="ignore"
)
log_small = pm4py.convert_to_event_log(
    df_small
)

df_small.to_csv(
    OUTPUT_CSV,
    index=False
)

""" pm4py.write_xes(
    log_small,
    OUTPUT_XES
) """

print(
    f"Cases: {df_small['case:concept:name'].nunique()}"
)

print(
    f"Events: {len(df_small)}"
)