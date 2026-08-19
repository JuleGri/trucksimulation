"""Regenerate clean XES files from the train/test CSVs without enabled:timestamp."""
import pandas as pd
import pm4py
from pathlib import Path

XES_DIR = Path("data/processed/CTB/xes_files")
XES_DIR.mkdir(parents=True, exist_ok=True)

for split in ("s6_train", "s6_test"):
    csv_path = Path(f"data/processed/CTB/{split}.csv")
    xes_path = XES_DIR / f"{split}.xes"
    
    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    
    # Drop enabled:timestamp if still present
    if "enabled:timestamp" in df.columns:
        df = df.drop(columns=["enabled:timestamp"])
        print(f"  Dropped enabled:timestamp")
    
    # Parse timestamps
    df["start:timestamp"] = pd.to_datetime(df["start:timestamp"], errors="coerce")
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")
    
    # Format for pm4py
    formatted = pm4py.format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )
    log = pm4py.convert_to_event_log(formatted)
    
    print(f"  {len(log)} traces -> {xes_path}")
    pm4py.write_xes(log, str(xes_path))
    print(f"  Done.")

print("\nClean XES files written (no enabled:timestamp).")
