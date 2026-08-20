"""Regenerate XES while preserving CTB's explicit within-case order."""
import sys
from pathlib import Path

import pandas as pd
import pm4py

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from _eventlog_contract import (  # noqa: E402
    select_prosit_dataframe,
    to_pm4py_event_log,
)

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

    df, dropped = select_prosit_dataframe(df, label=split)
    if dropped:
        print(f"  Excluded non-ProSiT attributes: {', '.join(dropped)}")
    
    # Do not use pm4py.format_dataframe here: it sorts by completion time and
    # can move Gate Out ahead of an overlapping yard event.  The shared
    # builder uses case:event:order (or migrates legacy row order) and removes
    # that technical column from the XES event attributes.
    log = to_pm4py_event_log(df, label=split)
    
    print(f"  {len(log)} traces -> {xes_path}")
    pm4py.write_xes(log, str(xes_path))
    print(f"  Done.")

print("\nClean XES files written (no enabled:timestamp).")
