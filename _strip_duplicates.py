"""Strip duplicate event rows from the final event log and train/test splits."""
import pandas as pd

for path in [
    "data/processed/CTB/s6_eventlog_target_rank_features.csv",
    "data/processed/CTB/s6_train.csv",
    "data/processed/CTB/s6_test.csv",
]:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"  {path}: not found, skipping")
        continue

    n_before = len(df)
    # Deduplicate on the identity columns (same event = same case+activity+resource+start+complete)
    dedup_cols = ["case:concept:name", "concept:name", "org:resource", "start:timestamp", "time:timestamp"]
    df = df.drop_duplicates(subset=dedup_cols, keep="first")
    n_after = len(df)
    dropped = n_before - n_after

    if dropped > 0:
        df.to_csv(path, index=False)
        print(f"  {path}: {n_before:,} -> {n_after:,} events ({dropped:,} duplicates removed)")
    else:
        print(f"  {path}: no duplicates found ({n_before:,} events)")

    # Also report cases and activity distribution
    print(f"    Cases: {df['case:concept:name'].nunique():,}")
    print(f"    Activities: {df['concept:name'].value_counts().to_dict()}")
    print()
